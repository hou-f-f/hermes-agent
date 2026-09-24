"""【模块概述 / Architecture Overview】
跨进程会话排他轮次租约机制（Durable Multi-Process Turn Lease），为 `TurnFacadeMixin.run_conversation` 提供底层并发安全保障。

【核心架构背景与痛点 / Context & Invariant】
在 Hermes 现代架构中，同一个会话（Session）可能被多个不同界面的进程并发访问：
例如：用户在 Electron 桌面端打开了该会话，同时在终端执行 `hermes -c` 进行会话恢复，或者 Telegram 网关正在接收用户发送的新消息。
如果没有任何跨进程协调，两个进程同时读取 SQLite state.db 并发起 API 调用与工具写操作，将导致：
1. 会话转录本（Transcript）严重分叉与交替污染；
2. 工具产生不可控的并发写副作用（覆盖同一个文件）；
3. 提示词前缀缓存（Prompt Caching）因历史消息分叉彻底失效。

为了坚守【架构不变量 2：Durable Multi-Process Turn Lease】：
- 同一时刻、同一个 session_id 仅允许一个进程持有行租约（Row Lease）执行 `load -> run -> flush` 完整生命周期；
- `admit_durable_turn_lease` 负责以行级排他锁形式向 SQLite 申请该租约；若被其他进程占用，会在终端输出“等待其他进程完成”的进度通知；
- `DurableTurnLease` 负责后台周期性心跳续约（默认 60 秒刷新一次，TTL 300 秒），防止进程异常退出导致死锁；
- 集成轮次活跃度看门狗（TurnLivenessWatchdog），用于检测工具或网络卡死并在真停滞时实施熔断中断；
- 两个定时任务均复用统一的周期调度器（`periodic_scheduler`），避免单轮创建大量冗余操作系统线程。

Durable cross-process session turn lease for ``TurnFacadeMixin.run_conversation``.

One process at a time may load -> run -> flush a session shared through state.db (Desktop, CLI
resume, gateway, background delivery). ``admit_durable_turn_lease`` acquires the row lease (or
returns the early result the façade must hand back); ``DurableTurnLease`` owns the periodic
refresher, the turn-liveness watchdog wiring, and the lease-loss / stall interrupt plumbing. Both
timers run via the shared scheduler (``agent/periodic_scheduler.py``; timer thread orders,
bodies run on per-handle workers), not per-turn threads.
"""
import logging
import os
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# 与源模块保持相同的 logger 名称，使得日志记录器与 caplog 过滤行为完全一致。
# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("run_agent")

# 【租约丢失中断原因 / Lease Lost Attribution】
# 当周期性续约失败或被其他进程强行抢占时，将停机原因归咎于“租约丢失”而非用户主动取消（参见 issue #112647）。
# ``tool_reason`` for a lost session turn lease: attributes the stop to the lease, not the user (#112647).
_REASON_LEASE_LOST = "session turn lease lost"

LEASE_TTL_SECONDS = 300.0   # 租约在数据库中的存活超时时间（5分钟无心跳则判定持有者崩溃）
LEASE_WAIT_SECONDS = 1800.0 # 等待其他并发进程释放租约的最大阻塞容忍时间（30分钟）


class DurableTurnLease:
    """【持久化轮次租约对象 / Admitted Turn Lease Container】
    管理已准入的会话排他租约实体，包含周期性心跳刷新器与停滞监控看门狗。

    `stop` 事件由刷新器与看门狗共享；`turn_active` 状态作为所有中断触发的守卫，
    杜绝因迟到的刷新失败而错误打断下一个崭新的轮次。两者均仅在 `_lock` 临界区内读写。

    An admitted session turn lease plus the periodic timers that keep it alive and watch the turn.

    ``stop`` is shared by the refresher and the liveness watchdog; ``turn_active`` gates every
    interrupt so a late refresher miss can never hard-interrupt the NEXT turn. Both are read and
    written only under ``_lock``.
    """

    def __init__(self, agent, db, session_id: str, holder: str) -> None:
        self.agent = agent
        self.db = db
        self.session_id = session_id  # 准入时的会话 ID；后续释放租约始终针对该行
        self.holder = holder          # 租约持有者标识（包含 pid、relay_turn_id 与 platform）
        self.stop = threading.Event()
        self.refresh_interval = float(getattr(agent, "_session_turn_lease_refresh_interval", 60.0))
        self._lock = threading.Lock()
        self.turn_active = False
        self.interrupt_message: Optional[str] = None
        self.watchdog = None  # 配置了活跃度监控时的 TurnLivenessWatchdog 实例
        self.timer_handles: list = []  # periodic_scheduler 的句柄列表，在 join_threads 中统一取消

    def _current_session_id(self) -> str:
        """获取当前有效的 session_id（会话轮转时优先读取 agent 的实时 id）。"""
        return getattr(self.agent, "session_id", None) or self.session_id

    def build_threads(self) -> None:
        """【构建看门狗实例 / Build Watchdog】
        若系统开启了活跃度配置，创建（但暂不启动调度）TurnLivenessWatchdog 看门狗。
        【关键洞察】：租约续约成功绝不等于有业务进展！如果一个工具陷入死循环或静默卡死，
        租约刷新线程仍然会按时心跳续约，导致任务永远挂死。因此必须由看门狗基于 activity_clock 独立检测停滞。
        
        Create (not schedule) the liveness watchdog when configured: lease renewal is NOT
        evidence of progress; a silently stalled turn would renew forever.
        """
        try:
            from hermes_cli.config import load_config_readonly

            liveness_config = load_config_readonly() or {}
        except Exception:
            liveness_config = {}
        from agent import turn_liveness

        timeout_s, poll_s = turn_liveness.resolve_turn_liveness_settings(liveness_config)
        if timeout_s is not None:
            self.watchdog = turn_liveness.TurnLivenessWatchdog(
                self.agent, session_id=self._current_session_id(), timeout_s=timeout_s,
                poll_s=poll_s, stop_event=self.stop,
                activity_lock=self.agent._liveness_activity_lock(),
                is_turn_active=self.is_turn_active, commit_abort=self.commit_liveness_abort,
                deactivate_turn=self.stop_refresher,
            )

    def start(self) -> None:
        """【启动租约与定时器 / Activate Lease Timers】
        标记本轮次进入活跃状态，重置活动时钟，并向周期调度器注册心跳任务与看门狗。
        """
        with self._lock:
            self.turn_active = True
        # 在轮次进入点盖上活跃时钟戳：由于 `_last_activity_ts` 跨轮次持久存在，
        # 如果不在此处更新，看门狗可能会基于上一轮的结束时间误判当前轮次一启动就超时
        self.agent._touch_activity("starting new turn")
        from agent.periodic_scheduler import schedule

        self.timer_handles.append(schedule(self.refresh_tick, self.refresh_interval))
        if self.watchdog is not None:
            self.timer_handles.append(self.watchdog.schedule())

    def stop_refresher(self) -> None:
        """【停止心跳与停用轮次 / Deactivate Turn】
        停止租约自动续期，标记本轮对话已脱离活跃状态。
        同时作为看门狗熔断时的回调：若硬中断无法解开底层卡死的底层调用，
        绝不能让心跳线程无限期维持租约；停止刷新后，300 秒 TTL 到期会自动让孤儿行被回收。

        Stop renewal and deactivate the turn. Also the watchdog's deactivate callback: a wedge the
        hard interrupt cannot unwind must not keep the lease alive forever; TTL expiry lets
        stale-turn cleanup reclaim the row.
        """
        with self._lock:
            self.turn_active = False
            self.stop.set()

    deactivate_after_liveness_abort = stop_refresher

    def join_threads(self, timeout: float = 1.0) -> None:
        """【等待定时任务退出 / Cancel Timers & Join】
        取消心跳与看门狗定时句柄，确保正在执行的回调完全退出，再执行后续的 `clear_interrupt`。
        
        Cancel both timers; ``wait=timeout`` mirrors the old ``thread.join(timeout)`` so an
        in-flight tick finishes before ``clear_interrupt`` runs.
        """
        for handle in self.timer_handles:
            handle.cancel(wait=timeout)

    def release(self) -> None:
        """【释放数据库租约行 / Release DB Row Lease】
        在 SQLite state.db 中安全释放租约行，并清空 agent 上的持有者属性。
        
        Release the row and drop the agent's holder attrs (only if they still name this lease).
        """
        agent = self.agent
        try:
            self.db.release_session_turn_lease(self.session_id, self.holder)
        except Exception:
            logger.error("Failed to release session turn lease: %s", self.session_id, exc_info=True)
        if getattr(agent, "_active_session_turn_lease_holder", None) == self.holder:
            agent._active_session_turn_lease_holder = None
            agent._active_session_turn_lease_ttl_seconds = None

    def is_turn_active(self) -> bool:
        """查询本轮次当前是否处于活跃中（受锁保护）。"""
        with self._lock:
            return self.turn_active

    def _interrupt_turn(self, message: str) -> None:
        """【租约丢失无条件硬中断 / Unconditional Lease-Loss Interrupt】
        租约丢失产生的中断是无条件触发的（无需世代声明 generation claim）：
        租约一旦在数据库层面丢失，意味着其他外部进程已经接管了该会话，当前进程必须以最高优先级立刻停机，
        坚决阻止本进程向 state.db 写入任何可能分叉的数据。
        
        Lease-loss interrupts fire UNCONDITIONALLY (no generation claim): a lost lease means
        this process no longer owns the session. Only the watchdog's stalls can be spuriously stale.
        """
        with self._lock:
            if self.stop.is_set() or not self.turn_active:
                return
            self.interrupt_message = message
            try:
                self.agent.interrupt(message, hard_cancel=True, tool_reason=_REASON_LEASE_LOST)
            except Exception:
                self.agent._interrupt_requested = True
                self.agent._interrupt_message = message
                self.agent._tool_interrupt_reason = _REASON_LEASE_LOST

    def commit_liveness_abort(self, snapshot, message: str) -> bool:
        """【看门狗停滞熔断提交点 / Commit Watchdog Stall Observation】
        看门狗观测到系统长时间无进展时的提议提交点。
        【防误杀机制 / Safe Guard】：
        在与 `_touch_activity` 相同的锁（`_liveness_activity_lock`）下重新校验 `(generation, timestamp)`。
        如果在看门狗记录日志到发起中断的微小时间差内，Agent 刚好收到了新的 Token 流或工具输出（generation 发生递增），
        则证明任务已自行复苏，此时绝不执行强杀；
        若确认停滞，则调用 `interrupt(require_generation=current_generation)` 在单次原子操作中触发熔断。
        
        Commit point for the watchdog's stall observation.

        Revalidates the observed ``(generation, timestamp)`` under the SAME lock ``_touch_activity``
        uses, so a turn that resumed while the stall was logged is never hard-cancelled; the
        revalidated generation is consumed by ``interrupt(require_generation=...)`` with the first
        publication in ONE critical section. If ``interrupt`` raises, the abort declines FAIL-CLOSED.
        Returns False when stale or already winding down."""
        agent = self.agent
        with agent._liveness_activity_lock():
            current_generation = getattr(agent, "_turn_liveness_activity_generation", 0)
            if (current_generation, getattr(agent, "_last_activity_ts", None)) != (
                snapshot.generation, snapshot.activity_ts
            ):
                return False
        with self._lock:
            if self.stop.is_set() or not self.turn_active:
                return False
        try:
            published = agent.interrupt(
                message, hard_cancel=True, tool_reason="turn liveness watchdog",
                require_generation=current_generation,
            )
        except Exception:
            logger.debug("Turn liveness abort interrupt raised; declining the abort", exc_info=True)
            published = False
        if published is False:
            # 声明已失效：在重新校验和真正下发杀进程之间，系统确实产生了真实进展
            # Claim went stale between revalidation and the hammer: real progress landed.
            return False
        with self._lock:
            self.interrupt_message = message
        return True

    def clear_interrupt(self) -> None:
        """【清除租约看门狗产生的局部中断标志 / Clear Lease Interrupt】
        仅清除由本租约刷新器或看门狗引发的中断标志。必须在 join_threads() 之后执行。
        
        Clear only the interrupt admitted by this lease's refresher/watchdog. Run AFTER join."""
        message = self.interrupt_message
        if not message:
            return
        agent = self.agent
        from tools.interrupt import set_interrupt as _set_interrupt

        with getattr(agent, "_pending_redirect_lock", None) or nullcontext():
            if getattr(agent, "_interrupt_message", None) != message:
                return
            agent._interrupt_requested = False
            agent._interrupt_message = None
            getattr(agent, "_hard_interrupt_requested", threading.Event()).clear()
            agent._interrupt_thread_signal_pending = False
            if agent._execution_thread_id is not None:
                _set_interrupt(False, agent._execution_thread_id)

    def refresh_tick(self):
        """【周期性续租滴答 / Periodic Lease Renewal Tick】
        由周期调度器定期触发的心跳续期方法（默认每 60 秒触发一次）。
        若心跳失败（数据库被锁或租约被抢夺），立即主动掐断当前 turn。
        返回 False 时调度器会自动停止该定时器。

        One periodic renewal (every ``refresh_interval`` via the shared scheduler); a miss or
        error interrupts the turn. Returning False stops the timer.

        The holder-qualified UPDATE fences a late refresher from a successor lease. The façade's
        finally sets ``stop`` before releasing, so a holder-fenced miss observed after stop is not
        a loss."""
        if self.stop.is_set():
            return False
        try:
            if self.db.refresh_session_turn_lease(
                self._current_session_id(), self.holder, ttl_seconds=LEASE_TTL_SECONDS
            ):
                return None
            if self.stop.is_set():
                return False
            logger.error(
                "Lost session turn lease while turn is active: %s", self._current_session_id()
            )
            self._interrupt_turn("Session turn lease lost; stopping to protect the transcript.")
        except Exception:
            if self.stop.is_set():
                return False
            logger.warning(
                "Failed to refresh session turn lease: %s", self._current_session_id(), exc_info=True,
            )
            self._interrupt_turn(
                "Session turn lease could not be refreshed; stopping to protect the transcript."
            )
        return False


@dataclass
class TurnLeaseAdmission:
    """【准入结果包装数据类 / Turn Lease Admission Result】
    `admit_durable_turn_lease` 的返回结果：
    `lease`（成功准入）与 `early_result`（提前失败或中断返回）二者有且仅有一个会被赋值。
    
    Outcome of ``admit_durable_turn_lease``: exactly one of ``lease`` / ``early_result`` may be set."""

    lease: Optional[DurableTurnLease] = None
    early_result: Optional[Dict[str, Any]] = None
    conversation_history: Optional[List[Dict[str, Any]]] = None


def _durable_session_exists(db, session_id: str) -> bool:
    """【持久化会话存在性检查 / Fail-Closed Probe】
    探测 SQLite state.db 中是否存在该 session_id 的行。
    【安全设计 / Fail-Closed Invariant】：
    遇到数据库被锁或非 WAL 并发读取异常时，探测失败绝不能当成“全新未落盘会话”（fail-open 会导致并发竞争）；
    必须采取 fail-closed 策略，强制尝试申请租约，避免在未同步序列化的情况下并发运行（参见 issue #84234）。
    """
    try:
        return db.get_session(session_id) is not None
    except Exception:
        # A locked / non-WAL read is not proof the row is absent; treating probe failure as "fresh"
        # ran fail-open at the exact contention point. Acquire, or fail closed.
        logger.warning(
            # Acquire (or fail closed if acquire itself cannot) rather than start load/run/flush
            # unsynchronized. get_session returns None — it does not raise — when the row is missing. See
            # #84234.
            "Could not check durable session before turn lease; "
            "will acquire rather than run without serialization",
            exc_info=True,
        )
        return True


def admit_durable_turn_lease(
    agent, *, session_id: str, relay_turn_id: str, task_context: Dict[str, Any],
    conversation_history: Optional[List[Dict[str, Any]]],
) -> TurnLeaseAdmission:
    """【执行跨进程持久化租约准入 / Admit Durable Turn Lease】
    核心准入控制逻辑：
    1. 快速旁路检查：若持久化被显式禁用（`_persist_disabled`，如单测或后台审查临时分支），或会话行尚不存在，直接放行无租约运行；
    2. 构造持有者标识符 holder（`pid:turn_id:platform`）；
    3. 调用数据库 `acquire_session_turn_lease` 申请排他行级锁（带 wait_seconds=1800 秒超时容忍）：
       - 若遇到竞争，触发 `_on_wait` 友好回调，在终端展示“另一个 Hermes 进程正在使用此会话，等待其完成...”；
    4. 【前缀缓存保活与转录本重载（Prompt Cache Invariant）】：
       - 若发生了等待（waited=True），意味着排在前面的进程可能执行了上下文微压缩（Micro-compaction）或会话轮转；
         准入成功后必须且仅在此时重新从数据库加载最新的消息历史（get_messages_as_conversation）；
       - 若无需等待（即时获取到租约），坚决不重新读库，确保继续复用内存中的消息前缀，保持大模型 Prompt 缓存完全命中；
    5. 构建并启动看门狗线程句柄。

    Acquire the session turn lease when the session is durable; build (not start) its threads.

    Mutates ``task_context["session_id"]`` and ``agent.session_id`` when the wait forced a resume-id
    reload. Returns an ``early_result`` (interrupted / timed out) instead of a lease when admission
    fails; the caller returns it verbatim."""
    db = getattr(agent, "_session_db", None)
    admission = TurnLeaseAdmission(conversation_history=conversation_history)
    if db is None or not session_id:
        return admission
    # 崭新的 session_id 尚无数据库历史竞争，且调用方可能在会话行创建前提供了内存初始种子；
    # 盲目重载会抹掉种子。对于 MagicMock 单测伪造对象，未实现 acquire_session_turn_lease 则安全跳过。
    # A fresh session id has no durable transcript to race over, and callers may supply an
    # in-memory seed before the row exists — reloading would erase it. Check the concrete type:
    # MagicMock-style shims accept any attribute without the protocol.
    if (
        getattr(agent, "_persist_disabled", False)
        or not _durable_session_exists(db, session_id)
        or not callable(getattr(type(db), "acquire_session_turn_lease", None))
    ):
        return admission
    # 确认会话行已在数据库存在，避免后续多余的 create_session 调用
    # Row proven to exist — suppress the redundant create attempt.
    agent._session_db_created = True
    holder = (
        f"pid={os.getpid()}:turn={relay_turn_id}:platform={task_context['platform'] or 'unknown'}"
    )
    waited = False

    def _on_wait(elapsed: float) -> None:
        """排队等待其他外部进程释放锁时的状态回调（在 UI/CLI 上给用户打出明确等待提示）。"""
        nonlocal waited
        waited = True
        agent._emit_status(
            "⏳ Another Hermes process is using this session; "
            "waiting for it to finish before starting your turn..."
            if elapsed < 1.0 else
            f"⏳ Still waiting for the other Hermes process on this session ({int(elapsed)}s)..."
        )

    if not db.acquire_session_turn_lease(
        session_id, holder, ttl_seconds=LEASE_TTL_SECONDS, wait_seconds=LEASE_WAIT_SECONDS,
        on_wait=_on_wait, should_abort=lambda: getattr(agent, "_interrupt_requested", False),
    ):
        admission.early_result = _lease_not_acquired_result(agent, session_id, conversation_history)
        return admission

    # 准入成功后方可给 agent 赋值 holder：确保 finally 绝不会错误释放不属于自己的租约
    # Assign only after admission so the finally cannot release a holder that never owned the
    # row; persist paths read the agent attr so a late flush is fenced in the same transaction.
    lease = DurableTurnLease(agent, db, session_id, holder)
    agent._active_session_turn_lease_holder = holder
    agent._active_session_turn_lease_ttl_seconds = LEASE_TTL_SECONDS
    try:
        if waited:
            agent._emit_status("Session is free; loading the latest transcript...")
            # 持有者在排队期间可能完成了上下文压缩或切分会话，此时按需更新最新 session_id
            # The holder may have compressed/rotated the session while we waited: reload only
            # AFTER admission; an immediate acquisition skips this (needless prompt-cache miss).
            latest_session_id = db.resolve_resume_session_id(session_id)
            if latest_session_id:
                agent.session_id = latest_session_id
                task_context["session_id"] = latest_session_id
            reloaded = db.get_messages_as_conversation(
                agent.session_id, repair_alternation=True, include_row_ids=True
            )
            # 在等待期间若有因中断暂存在内存中未持久化的消息，在此重新拼装回历史
            # A follow-up that aborted an earlier wait carries that turn's never-persisted input
            # only in memory (see carry_unadmitted_user_message); the reload would drop it.
            from agent.session_persistence import _PERSIST_AFTER_ADMISSION_INTERRUPT
            reloaded.extend(
                m for m in (conversation_history or [])
                if isinstance(m, dict) and m.get(_PERSIST_AFTER_ADMISSION_INTERRUPT)
                and "_row_id" not in m
            )
            admission.conversation_history = reloaded
        lease.build_threads()
    except BaseException:
        # 异常防护：若准入成功后发生异常，立即释放租约防止死锁
        # The façade never saw this lease; release here so an admitted row is not leaked.
        lease.release()
        raise
    admission.lease = lease
    return admission


def carry_unadmitted_user_message(
    early_result: Dict[str, Any], user_message: Any, persist_user_message: Any, *,
    timestamp: Optional[float], display_kind: Optional[str], display_metadata: Optional[Dict[str, Any]],
    platform_id: Optional[str],
) -> None:
    """【未准入用户消息保全传递 / Carry Unadmitted User Input】
    当用户发起的新消息打断了正在等待租约的过程时，该消息绝不能被直接丢弃或视作已消费：
    将其以带 `_PERSIST_AFTER_ADMISSION_INTERRUPT` 标记的格式追加到 early_result 的历史中，
    使得后续轮次在真正获得租约准入时能够读取并进行落盘持久化。
    若属于强行中止命令（如 `/stop`），则直接取消不予传递。

    A follow-up that interrupted the lease wait must not consume the accepted input: append it to
    the early result's history so the follow-up turn sees it and persists it (the flush honours
    ``_PERSIST_AFTER_ADMISSION_INTERRUPT`` because this turn never owned the lease). A hard stop
    (``/stop``) cancels the input instead."""
    hard_interrupted = early_result.pop("_hard_interrupted", False)
    if hard_interrupted or not early_result.get("interrupted") or user_message in (None, ""):
        return
    from agent.message_metadata import append_message
    from agent.session_persistence import _PERSIST_AFTER_ADMISSION_INTERRUPT

    durable_content = user_message
    if persist_user_message is not None and (
        not isinstance(user_message, list) or isinstance(persist_user_message, list)
    ):
        durable_content = persist_user_message
    deferred_user: Dict[str, Any] = {
        "role": "user", "content": durable_content, _PERSIST_AFTER_ADMISSION_INTERRUPT: True,
    }
    if isinstance(user_message, str) and user_message != durable_content:
        deferred_user["api_content"] = user_message
    if display_kind:
        deferred_user["display_kind"] = display_kind
    if display_metadata:
        deferred_user["display_metadata"] = display_metadata
    if platform_id is not None:
        deferred_user["platform_message_id"] = platform_id
    append_message(early_result["messages"], deferred_user, timestamp=timestamp)


def _lease_not_acquired_result(agent, session_id: str, conversation_history) -> Dict[str, Any]:
    """【未获取到租约时的返回体封装 / Early Exit Result】
    分为两种情况：
    1. 用户主动中断：返回 interrupted=True，并清理 agent 上的中断状态，防止污染下次调用；
    2. 等待超时（30 分钟）：向用户发出提示“其他 Hermes 进程占用该会话过久，请稍后重发”，
       标记 failure_reason="session_busy" 和 failure_retryable=True。
    """
    base = {"messages": list(conversation_history or []), "api_calls": 0, "completed": False}
    if getattr(agent, "_interrupt_requested", False):
        logger.info("session turn lease wait aborted by interrupt: %s", session_id)
        hard_event = getattr(agent, "_hard_interrupt_requested", None)
        hard_interrupted = bool(
            callable(getattr(hard_event, "is_set", None)) and hard_event.is_set()
        )
        result = {
            "final_response": (
                "Stopped waiting for another Hermes process on this session. "
                "Your message was not processed."
            ),
            **base,
            "interrupted": True,
        }
        if hard_interrupted:
            result["_hard_interrupted"] = True
        if getattr(agent, "_interrupt_message", None):
            result["interrupt_message"] = agent._interrupt_message
        # The finalizer never runs on this early return; clear so a cached agent doesn't
        # fail-close the next turn.
        try:
            agent.clear_interrupt()
        except Exception:
            agent._interrupt_requested = False
            agent._interrupt_message = None
        return result
    # Fail closed like gateway TurnLeaseTimeoutError: surface a resend notice, not a bare TimeoutError.
    timeout_msg = (
        "⏳ Another Hermes process kept this session busy too long. Your message was not "
        "processed - wait for the other process to finish, then send it again."
    )
    logger.error("session turn lease wait timed out for %s", session_id)
    try:
        agent._emit_warning(timeout_msg)
    except Exception:
        logger.debug("Failed to emit session turn lease timeout warning", exc_info=True)
    # Stamped so Desktop/TUI show "session busy, send again" instead of code="unknown".
    return {
        "final_response": timeout_msg,
        **base,
        "failed": True,
        "error": f"session_turn_lease_timeout:{session_id}",
        "failure_reason": "session_busy",
        "failure_retryable": True,
    }
