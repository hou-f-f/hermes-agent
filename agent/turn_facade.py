"""【模块概述 / Architecture Overview】
AIAgent 的 `run_conversation` 与 `chat` 顶层门面（Facade）。

本模块是单轮对话生命周期的统一准入网关（Turn Admission）：
1. 跨进程持久化轮次租约（Durable Turn Lease）：通过 `admit_durable_turn_lease` 申请数据库排他租约与看门狗心跳刷新线程（`turn_facade_lease`），杜绝多客户端或多进程并发写入同一会话；
2. 后台审查抢占（Background Review Preemption）：在用户新消息进入时，优先通过 `cancel_background_review_for_live_turn` 掐断正在后台提炼记忆/技能的低优先级任务；
3. 会话全链路可观测性与记账范围（Relay / Accounting / Portal Scopes）：通过 ContextVar 设定 Portal 根对话 ID 标签、会话亲和性范围与辅助 Token 记账句柄；
4. 结构化收尾不变量（Symmetric Lifecycle Invariant）：严密嵌套的 `try...finally` 块确保即使被中断或抛出异常，租约释放、线程 Join、Token 还原与空闲队列计数绝不泄漏。
从 `run_agent.py` 拆分提取；所有方法通过 `AIAgent` 的方法解析顺序（MRO）透明继承。

``AIAgent.run_conversation`` / ``chat`` façade.

Turn admission around ``conversation_loop.run_conversation``: durable cross-process session turn lease +
refresher thread and liveness watchdog (``agent.turn_facade_lease``), relay/accounting/portal scopes, and
balanced start/finish marks. Extracted from ``run_agent.py``; every method resolves through ``AIAgent``'s
MRO unchanged.
"""
import logging
import uuid
from contextlib import suppress
from typing import Any, Dict, List, Optional

from agent.lazy_forward import forward as _forward

# 与源模块保持相同的 logger 名称，使得日志记录器与 caplog 过滤行为完全一致。
# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("run_agent")


class TurnFacadeMixin:
    """【对话门面混合类 / Turn Facade Mixin】
    定义 AIAgent 的核心公开调用接口 `run_conversation()` 和 `chat()`。
    
    run_conversation()/chat() (see module docstring).
    """

    def run_conversation(
        self, user_message: Any, system_message: str=None,
        conversation_history: List[Dict[str, Any]]=None, task_id: str=None,
        stream_callback: Optional[callable]=None, persist_user_message: Optional[Any]=None,
        persist_user_timestamp: Optional[float]=None, persist_user_display_kind: Optional[str]=None,
        persist_user_display_metadata: Optional[Dict[str, Any]]=None,
        persist_user_platform_id: Optional[str]=None, moa_config: Optional[dict[str, Any]]=None,
        turn_author: Optional[Dict[str, Any]] = None,
        relay_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """【执行单轮完整对话 / Execute Single Turn Conversation】
        负责前置准入校验、排他租约竞拍、上下文范围绑定，并委托 `agent.conversation_loop.run_conversation` 驱动核心推理循环。
        
        Forwarder — see ``agent.conversation_loop.run_conversation``.
        """
        # 【后台反思让路机制 / Review Preemption】
        # 后台提炼任务与当前会话共享相同的 session_id 以维持前缀缓存一致性。
        # 当人类用户发起新的实时交互时，必须立即建立隔离栅栏阻止后台任务启动，
        # 或中断已准入的请求并等待其退出，再开启前台对话仪表盘；
        # 若后台任务在超时时间内未响应中断，前台强制保留最高优先执行权（见 issue #84423）。
        # A review shares this session_id for cache parity: fence review startup or interrupt
        # an admitted request and await its exit before opening live-turn instrumentation.
        # Foreground priority is retained if the review does not acknowledge within the bounded deadline
        # (#84423).
        from agent.background_review import cancel_background_review_for_live_turn

        cancel_background_review_for_live_turn(self)

        from agent import relay_runtime
        from agent.aux_accounting import reset_accounting_context, set_accounting_context
        from agent.auxiliary_client import scoped_runtime_main
        from agent.conversation_loop import run_conversation
        from agent.portal_tags import (
            reset_affinity_scope, reset_conversation_context, set_affinity_scope,
            set_conversation_context,
        )
        from agent.prompt_cache_scope import declared_conversation_scope_safe
        from agent.review_idle_queue import QUEUE as _review_queue
        from agent.subagent_lifecycle import bind_subagent_parent
        from agent.interrupt_scope import track_in_interrupt_scope
        from agent.turn_facade_lease import admit_durable_turn_lease, carry_unadmitted_user_message
        from hermes_cli.observability.relay_shared_metrics import finish_task_run, start_task_run

        effective_task_id = task_id or str(uuid.uuid4())
        session_id = str(getattr(self, "session_id", None) or "")
        task_context = {
            "session_id": session_id,
            "task_id": effective_task_id,
            "platform": getattr(self, "platform", None) or "",
        }
        relay_turn_id = f"{session_id or 'session'}:{effective_task_id}:{uuid.uuid4().hex[:8]}"
        self._relay_pending_turn_id = relay_turn_id
        relay_parent_session_id = (
            str(getattr(self, "_parent_session_id", None) or "")
            if task_context["platform"] == "subagent"
            else ""
        )
        relay_lease = relay_turn = lease = None
        # 【范围 Token 初始化】
        # 初始设为 None：若在 set_*() 前发生提前返回，finally 块能安全无条件重置已注册的 Token。
        # Scope tokens start None: early returns leave the try before the set_*() calls and
        # the finally resets each one unconditionally.
        token = affinity_token = acct_token = None
        task_started = task_finished = False
        relay_outcome = "failed"

        try:
            # 放在 try 块的最首行：确保 finally 中的 note_turn_finished 能够精确配对每一次进出，避免计数器泄漏
            # First statement of the try so the finally's note_turn_finished balances every exit.
            _review_queue.note_turn_started()
            # 【申请持久化轮次租约 / Durable Multi-Process Turn Lease】
            # 防止多终端（桌面、CLI、移动网关）并发打入相同 session_id 造成状态竞争破坏
            admission = admit_durable_turn_lease(
                self, session_id=session_id, relay_turn_id=relay_turn_id, task_context=task_context,
                conversation_history=conversation_history,
            )
            # 若租约抢占失败或前置检查被阻断（如并发超时、外部中断）：
            if admission.early_result is not None:
                carry_unadmitted_user_message(
                    admission.early_result, user_message, persist_user_message,
                    timestamp=persist_user_timestamp, display_kind=persist_user_display_kind,
                    display_metadata=persist_user_display_metadata, platform_id=persist_user_platform_id,
                )
                relay_outcome = (
                    "cancelled" if admission.early_result.get("interrupted") else "timed_out"
                )
                return admission.early_result
            lease = admission.lease
            conversation_history = admission.conversation_history

            # 协调器登记会话租约
            relay_lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
                profile_key=relay_runtime.current_profile_key(),
                session_id=task_context["session_id"], platform=task_context["platform"],
                parent_session_id=relay_parent_session_id,
                model=str(getattr(self, "model", None) or ""),
            )
            relay_turn_kwargs: Dict[str, Any] = {
                "turn_id": relay_turn_id,
                "task_id": effective_task_id,
            }
            if relay_metadata:
                relay_turn_kwargs["metadata"] = relay_metadata
            relay_turn = relay_runtime.SESSION_COORDINATOR.begin_turn(
                relay_lease, **relay_turn_kwargs
            )
            # 极简 relay-runtime 垫片可能缺少 opt-out 标记：默认开启指标上报
            # Minimal relay-runtime shims may lack the opt-out flag: default enabled.
            if getattr(relay_turn, "relay_enabled", True):
                start_task_run(
                    **task_context,
                    parent_session_id=getattr(self, "_parent_session_id", None) or "",
                )
                task_started = True
            # 【环境级 Nous Portal 标签继承 / Ambient Portal Tagging】
            # 当前轮次中的所有大模型调用（主循环、压缩、多模态视觉、MoA、审查分支）全部继承 `conversation=<root>`；
            # 宿主声明的亲和范围（affinity scope）回退至此；记账句柄将辅助用量关联至该会话。
            # Ambient Nous Portal tagging: every LLM call in this turn (loop, compression,
            # vision, MoA, review forks) inherits `conversation=<root>`; host-declared
            # affinity scope falls back to it; accounting handles route aux usage to the session.
            token = set_conversation_context(self._conversation_root_id())
            affinity_token = set_affinity_scope(declared_conversation_scope_safe(self))
            # 【辅助用量统一记账 / Auxiliary Token Accounting (issue #23270)】
            # 以相同方式发布会话记账句柄，使所有辅助调用（如压缩、摘要）将 Token 记录到 session_model_usage 表中
            acct_token = set_accounting_context(
                getattr(self, "_session_db", None), getattr(self, "session_id", None)
            )

            # 保持 ContextVar 范围的线程局部性；宿主控制线程可跨线程安全中断当前轮次
            # Keep the ContextVar scope local (agent tokens may be observed from another thread).
            # A host that owns this thread (Hermes Console) may cancel the turn cross-thread.
            with bind_subagent_parent(self), scoped_runtime_main({}), track_in_interrupt_scope(self):
                try:
                    if lease is not None:
                        lease.start()
                    # 真正进入核心对话与工具执行驱动循环
                    result = run_conversation(
                        self, user_message, system_message, conversation_history, effective_task_id,
                        stream_callback, persist_user_message,
                        persist_user_timestamp=persist_user_timestamp,
                        persist_user_display_kind=persist_user_display_kind,
                        persist_user_display_metadata=persist_user_display_metadata,
                        persist_user_platform_id=persist_user_platform_id, moa_config=moa_config,
                        turn_author=turn_author,
                    )
                finally:
                    # 循环后阶段绝不能接收到迟到的心跳刷新中断信号；中断清除自身等待外层 finally 的线程 join
                    # Post-loop relay/task finalization must not receive a late refresh interrupt;
                    # the interrupt clear itself waits for the thread join in the outer finally.
                    if lease is not None:
                        lease.stop_refresher()
            terminal = result if isinstance(result, dict) else {}
            relay_outcome = (
                "cancelled" if terminal.get("interrupted") is True
                else "failed" if terminal.get("failed") is True
                else "success"
            )
            relay_runtime.SESSION_COORDINATOR.finish_logical_calls(relay_turn, outcome=relay_outcome)
            if task_started:
                task_finished = True
                finish_task_run(**task_context, result=result)
            return result
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, InterruptedError)) or (
                type(exc).__name__ == "CancelledError"
            ):
                relay_outcome = "cancelled"
            elif isinstance(exc, TimeoutError):
                relay_outcome = "timed_out"
            if relay_turn is not None:
                relay_runtime.SESSION_COORDINATOR.finish_logical_calls(
                    relay_turn, outcome=relay_outcome
                )
            if task_started and not task_finished:
                task_finished = True
                finish_task_run(**task_context, error=exc)
            raise
        finally:
            try:
                if relay_turn is not None:
                    relay_runtime.SESSION_COORDINATOR.end_turn(relay_turn, outcome=relay_outcome)
            finally:
                try:
                    if relay_lease is not None:
                        relay_runtime.SESSION_COORDINATOR.release_conversation(relay_lease)
                finally:
                    if lease is not None:
                        lease.stop_refresher()
                        lease.join_threads()
                        lease.clear_interrupt()  # 刷新线程停止并 join 之后再清除中断标志
                        lease.release()
                    # 轮次退出时始终重置轮中活动标签（包括跳过 finalize_turn 的提前中断分支），保留时间戳
                    # Always clear mid-turn labels on exit — including interrupted early returns
                    # that skip finalize_turn. Keep ts.
                    with suppress(Exception):
                        self._reset_activity_labels_after_turn()
                    if getattr(self, "_relay_pending_turn_id", None) == relay_turn_id:
                        self._relay_pending_turn_id = None
                    if acct_token is not None:
                        reset_accounting_context(acct_token)
                    if token is not None:
                        reset_conversation_context(token)
                    if affinity_token is not None:
                        reset_affinity_scope(affinity_token)
                    # 严格配对 note_turn_started，确保空闲队列中的活跃轮次计数器绝不泄漏
                    # Balance note_turn_started so the idle queue's live-turn count cannot leak.
                    with suppress(Exception):
                        _review_queue.note_turn_finished()

    def chat(self, message: str, stream_callback: Optional[callable] = None) -> str:
        """【单轮对话精简调用 / Convenience Chat Method】
        单轮对话的同步字符串返回值接口；流式回调 `stream_callback` 接收每个增量文本分块。
        
        Final response string of one turn; ``stream_callback`` receives each text delta.
        """
        return self.run_conversation(message, stream_callback=stream_callback)["final_response"]

    _run_codex_app_server_turn = _forward("agent.codex_runtime", "run_codex_app_server_turn")

