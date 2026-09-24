"""【轮次迭代外层状态管理与前后准备阶段 / Outer-iteration Bookkeeping for Turn Loop】
本模块负责对话轮次主循环中单次迭代（Iteration）的簿记管理与前后处理，按调用时序包含四大关键阶段：
1. begin_iteration（Phase 1 开启迭代）：处理用户动态插入的重定向（/redirect）、打断检查、
   后台复盘输入预算拦截以及迭代上限看门狗。
2. prepare_iteration（Phase 2 迭代准备）：触发 agent:step 网关回调、技能沉淀计数器递增、
   在最新 tool 结果后排空 /steer 引导、运行时间预算收尾通知、工具调用 JSON 参数校验清洗、
   剔除旧版打断脚手架幽灵行、以及严格角色交替修复（并重新锚定 current_turn_user_idx）。
3. announce_api_call（Phase 5 请求宣发）：终端 Verbose 模式打印详细用量或启动安静模式打字机/思考加载指示器（Thinking Spinner）。
4. apply_retry_restarts（Phase 7 重试重启分发）：消费 TurnRetryState 的各类重启标志
   （重定向重启、压缩后重启、故障转移模型重建、超长截断续写），并在无有效输出时优雅返还迭代预算。
本模块在模块级别绝对禁止导入 agent.conversation_loop，从根本上规避循环导入死锁。

Outer-iteration bookkeeping for the conversation turn loop, in call order:
``begin_iteration`` (pending redirect, interrupt / review-budget / iteration-budget exits),
``prepare_iteration`` (``agent:step`` callback, skill-nudge counter, pre-API ``/steer`` drain as a
standalone user row after the newest tool result, run-budget wrap-up notice, tool_call
argument sanitization, interrupt-scaffold ghost-row drop, role-alternation repair),
``announce_api_call`` (verbose summary / quiet spinner) and, after the retry loop,
``apply_retry_restarts`` (consumes the ``TurnRetryState`` restart flags). Nothing here
imports ``agent.conversation_loop`` at module level (cycle)."""

from __future__ import annotations

import logging
import random
import sys
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Dict

from agent.display import KawaiiSpinner
from agent.interrupt_control import interrupt_issuer
from agent.turn_context_compaction import _reanchor

logger = logging.getLogger("agent.conversation_loop")


def _anchors_current_turn(messages: Any, idx: Any, user_message: Any) -> bool:
    """【判断指定索引是否为当前轮次真实用户行 / Check Turn Anchor】
    若 messages[idx] 确系当前轮次用户消息（字面量精确匹配或经由压缩还原的原始用户视图），则返回 True。
    用于在角色交替修复或上下文压缩后，验证 current_turn_user_idx 锚点是否依然有效。

    True when ``messages[idx]`` is this turn's user row (verbatim, or its user-originated view)."""
    if not isinstance(idx, int) or not 0 <= idx < len(messages):
        return False
    msg = messages[idx]
    if not (isinstance(msg, dict) and msg.get("role") == "user"):
        return False
    if msg.get("content") == user_message:
        return True
    from agent.context_compressor import user_originated_turn_view

    view = user_originated_turn_view(msg)
    return view is not None and view.get("content") == user_message

# 【迭代预算进度检查点通知模板】
ITERATION_BUDGET_WARNING_TEMPLATE = (
    "[SYSTEM NOTICE — iteration budget checkpoint] You have used {used} of {maximum} "
    "iterations. Checkpoint durable progress now, then continue the task; do not stop "
    "solely because of this warning."
)


def _maybe_inject_iteration_budget_warning(agent: Any, messages: Any) -> bool:
    """【向最新工具结果追加迭代预算预警通知】
    机制与缓存安全设计：
    1. 动态预警阈值：当迭代消耗达到设定的比例阈值（如 Kanban 任务自动设定为 90% 阈值）时触发；
    2. 坚守前缀缓存不变量：绝不作为独立 user/system 消息注入，而是追加在当前轮次最新的未落盘 tool 消息尾部，
       命令模型立即落盘阶段性成果，同时告知模型“不要因为此警告就草率停工，必须继续完成任务”；
    3. 打断状态屏蔽：若已收到打断请求，则跳过追加，避免催促更多无意义的工作。

    Append the opt-in one-shot warning to the newest tool result."""
    import os
    from agent.delegation_context import is_dispatcher_owned_worker_context

    # Cancellation results still need persistence, but must not urge more work.
    if getattr(agent, "_interrupt_requested", False):
        return False

    ratio = getattr(agent, "budget_warning_ratio", None)
    kanban_worker = (
        bool(os.environ.get("HERMES_KANBAN_TASK"))
        and is_dispatcher_owned_worker_context()
        and "kanban_complete" in getattr(agent, "valid_tool_names", ())
    )
    if ratio is None and kanban_worker:
        ratio = 0.9
    budget = getattr(agent, "iteration_budget", None)
    if (
        ratio is None
        or budget is None
        or budget.max_total <= 1
        or budget.max_total >= sys.maxsize
        or getattr(agent, "_iteration_budget_warning_injected", False)
        or budget.used < min(ratio * budget.max_total, budget.max_total - 1)
    ):
        return False
    notice = ITERATION_BUDGET_WARNING_TEMPLATE.format(
        used=budget.used, maximum=budget.max_total
    )
    if kanban_worker:
        notice += (
            " While tools are still available, call kanban_complete only if all task "
            "requirements are verified, or kanban_request_review if it is ready for "
            "review; otherwise persist a kanban_comment handoff and "
            "continue. A diff or commit alone is not completion evidence."
        )
    # Only the current tool-result tail is mutable; an older turn may already be cached.
    from agent.context_compressor import _DB_PERSISTED_MARKER
    if (not messages or messages[-1].get("role") != "tool"
            or messages[-1].get(_DB_PERSISTED_MARKER)):
        return False
    message = messages[-1]
    content = message.get("content", "")
    if isinstance(content, str):
        message["content"] = content + f"\n\n{notice}"
    elif isinstance(content, list) or content is None:
        message["content"] = [*(content or []), {"type": "text", "text": notice}]
    else:
        return False
    agent._iteration_budget_warning_injected = True
    return True


@dataclass
class IterationPrep:
    """Always ``action == "fallthrough"``. ``messages`` is the (possibly filtered) transcript
    and ``request_logger`` the per-request logger the caller keeps using."""

    action: str
    messages: Any
    request_logger: Any
    current_turn_user_idx: Any


def prepare_iteration(
    agent: Any, *, messages: Any, api_call_count: Any, user_message: Any = None, current_turn_user_idx: Any = None,
) -> IterationPrep:
    """【准备当前迭代消息历史（Phase 2 核心预处理流水线）】
    按严格时序对 messages 进行 API 发送前的就地安全修整。
    
    前缀缓存不变量与关键设计保障（Cache-safe by Construction）：
    1. 动态线路切换：若上一轮响应决定了 Anthropic 协议线路切换（nous.anthropic_wire），在本次请求构建前静默生效；
    2. /steer 引导独立追加：将上一轮模型思考期间用户发来的 /steer 引导文本，作为独立的 user 行追加在最新 tool 结果之后；
       ⚠️ 绝对不能拼贴到 tool 消息中（因为已落盘的 tool 消息不可变，篡改会导致回放时前缀漂移破坏提示词缓存）；
    3. 工具参数校验游标（_sanitize_args_cursor）：基于对象标识记忆已校验项，避免每次迭代全量重复反序列化 JSON；
    4. 剔除打断脚手架幽灵行（Scaffold Ghost Drop）：剔除旧版残留的隐藏 assistant 占位行，防止模型在后续回放中复读自我复制；
    5. 严格角色交替修复与锚点重校（Role Alternation & Re-anchoring）：
       若历史存在连续两条 user 或 tool 结尾等破坏交替性的坏数据，进行序列折叠修复；
       由于折叠收缩了列表长度，必须通过 _reanchor 重新校准 current_turn_user_idx，
       防止外挂记忆预取将上下文错误注入到历史轮次或导致 WebUI 消息顺序错乱。

    Prepare ``messages`` for this iteration in the original order. Every mutation here is
    cache-safe by construction: steer text is appended as a new (not yet persisted) user row, the ghost-row
    filter only drops hidden scaffold placeholders, and repair runs BEFORE the request build."""
    from agent.conversation_loop import (
        _INTERRUPT_SCAFFOLD_MARKER, _maybe_inject_run_budget_wrapup
    )

    # 【动态线路切换：在迭代开始前静默应用】
    # 由上一轮响应决策的线路切换（如切换至 Anthropic 原生 wire 协议）在此处应用，此时无网络请求正在飞行。
    # nous.anthropic_wire=auto: a wire switch decided from the previous response lands here,
    # before this iteration's request is built and with nothing in flight.
    if getattr(agent, "_nous_wire_pending", None):
        from agent.nous_wire import apply_pending_wire_switch
        apply_pending_wire_switch(agent)

    # 【触发网关单步执行钩子（agent:step 事件广播）】
    # Fire step_callback for gateway hooks (agent:step event).
    if agent.step_callback is not None:
        try:
            agent.step_callback(api_call_count, _previous_tool_round(messages))
        except Exception as _step_err:
            logger.debug("step_callback error (iteration %s): %s", api_call_count, _step_err)

    # 【技能沉淀（Skill Nudge）迭代计数器】
    # 统计两次使用技能管理工具之间经历的工具调用轮数；一旦调用 skill_manage 则重置。
    # Tool-calling iterations for the skill nudge; resets whenever skill_manage is used.
    if agent._skill_nudge_interval > 0 and "skill_manage" in agent.valid_tool_names:
        agent._iters_since_skill += 1

    # 【Nous Agent 临时凭据临期预刷新】
    # Nous Agent 密钥有效期约为 1 小时，而长任务单轮执行可能跨越数小时。
    # 在当前密钥过期前提前采纳保活的新密钥（本地读取 JWT exp，无网络请求损耗），彻底消除 401 凭证风暴。
    # Nous agent keys live ~1 h and a single turn can run for hours: adopt the keepalive's fresh
    # key before the one in hand expires (local JWT exp read; no network unless inside the skew)
    # instead of letting this iteration's request 401. With many agents sharing the hour that
    # 401 was a storm, and the pool benched the sole credential for all of them.
    try:
        agent._adopt_nous_key_before_expiry()
    except Exception:
        logger.debug("Nous key pre-expiry adoption failed", exc_info=True)

    # 【排空上一轮模型生成期间积攒的 /steer 用户指引】
    # 作为独立的 user 消息追加在最新 tool 结果之后（绝不涂抹污染已落盘的 tool 结果，保障前缀缓存绝对稳定）。
    # Drain a /steer sent during the last API call so it lands THIS iteration. Delivered as a
    # standalone user row after the newest tool result (never smeared onto the tool row: that
    # row is already persisted append-only, so replay would diverge from the live request and
    # break the prompt cache — same contract as apply_pending_steer_to_tool_results).
    _pre_api_steer = agent._drain_pending_steer()
    if _pre_api_steer:
        _inject_steer_after_newest_tool_result(agent, messages, _pre_api_steer)

    # 【单次任务运行时间预算（run_budget）超 80% 紧急收尾提醒注入】
    # One-shot run-budget wrap-up notice at 80% of agent.run_budget_seconds, appended to the
    # newest tool result; off with no budget.
    if getattr(agent, "run_budget_seconds", None):
        _maybe_inject_run_budget_wrapup(agent, messages)

    # 【迭代轮数预算（iteration_budget）预警注入】
    # Appended to the newest tool result; never a synthetic user/system row.
    _maybe_inject_iteration_budget_warning(agent, messages)

    request_logger = getattr(agent, "logger", None) or logger  # same name as the origin module
    # 【校验游标记忆：跳过已验证通过的工具调用参数】
    # Per-agent validation cursor skips re-parsing tool_call args already validated.
    # Identity-keyed; a rewritten list breaks the prefix match and forces a re-scan.
    _sanitize_cursor = getattr(agent, "_sanitize_args_cursor", None)
    if _sanitize_cursor is None:
        _sanitize_cursor = {}
        with suppress(Exception):
            agent._sanitize_args_cursor = _sanitize_cursor
    repaired_tool_calls = agent._sanitize_tool_call_arguments(
        messages, logger=request_logger, session_id=agent.session_id, cursor=_sanitize_cursor
    )
    if repaired_tool_calls > 0:
        # 【就地修复参数后强制全量刷盘扫描】
        # 参数清洗可能导致已落盘的字典丢失持久化标记，强制重置前缀扫描位以重写持久化行。
        # In-place arg repair may have popped _DB_PERSISTED_MARKER off stamped live dicts;
        # force a full flush scan so the repaired rows are rewritten.
        agent._db_flush_scan_prefix = None
        request_logger.info(
            "Sanitized %s corrupted tool_call arguments before request (session=%s)",
            repaired_tool_calls,
            agent.session_id or "-",
        )

    # 【剔除携带打断脚手架的废弃隐藏 assistant 幽灵行】
    # 防止大模型在回放该占位符时产生鹦鹉学舌般的复读回响。
    # Drop legacy hidden assistant placeholders carrying the raw interrupt scaffold
    # before repair: replayed, the model echoes/self-replicates.
    def _is_scaffold_ghost(msg: Dict[str, Any]) -> bool:
        return (
            msg.get("display_kind") == "hidden"
            and msg.get("role") == "assistant"
            and any(
                isinstance(msg.get(k), str) and msg[k].strip() == _INTERRUPT_SCAFFOLD_MARKER
                for k in ("content", "api_content")
            )
        )

    messages = [msg for msg in messages if not _is_scaffold_ghost(msg)]

    # 【修复破损的消息角色交替（如 tool→user 或连续 user）】
    # 严格角色交替守护：大模型接口遇到非法序列会返回空内容导致空转重试。
    # repair_message_sequence_with_cursor 同时在压缩后重新计算 SessionDB 刷盘游标。
    # Repair malformed role alternation (tool→user / user→user tails): providers
    # return empty content on them and the empty-retry loop spins. The _with_cursor
    # variant also recomputes the SessionDB flush cursor after compaction.
    from agent.agent_runtime_helpers import repair_message_sequence_with_cursor
    repaired_seq = repair_message_sequence_with_cursor(agent, messages)
    if repaired_seq > 0:
        request_logger.info(
            "Repaired %s message-alternation violations before request (session=%s)",
            repaired_seq,
            agent.session_id or "-",
        )
        # 【序列折叠后当前用户索引重锚定】
        # 序列合并缩减了列表长度，导致原先记录的索引可能越界；在此通过 _reanchor 重新锚定当前用户请求。
        # The merge shrank the list, so the index recorded at turn start can point past this
        # turn's user row: prefetch would inject into a historical row and index-settling hosts
        # (hermes-webui) would write the current turn to the FRONT of the context. Re-anchor as
        # the compression-restart path does (last verbatim row wins, never a historical copy);
        # without the text the index cannot be re-derived and is left detectably stale.
        if user_message is not None:
            _reanchored_idx = _reanchor(agent, messages, user_message)
            if _reanchored_idx != current_turn_user_idx:
                request_logger.info(
                    "Re-anchored current_turn_user_idx %s -> %s after alternation repair (session=%s)",
                    current_turn_user_idx, _reanchored_idx, agent.session_id or "-",
                )
                current_turn_user_idx = _reanchored_idx
    # 【轮次中途压缩后的当前用户索引重锚定】
    # 若经历过后置微压缩或溢出恢复，历史被重写但未更新索引，导致回放前缀在工具行内发生割裂；
    # 此处确保 current_turn_user_idx 始终精准锚定在当前轮次的用户行上。
    # Mid-turn compaction (post-tool gate, overflow restart, recovery) rebuilds ``messages`` without
    # handing back a new index. A stale index splits the request's replay prefix inside this turn's
    # tool rows: prefix canonicalization then drops the assistant tool_call whose result fell past the
    # split, the orphaned result is sanitized away, and the model silently loses tool output that
    # state.db still holds. A valid index always lands on this turn's user row; re-anchor otherwise.
    if user_message is not None and not _anchors_current_turn(messages, current_turn_user_idx, user_message):
        _reanchored_idx = _reanchor(agent, messages, user_message)
        request_logger.info(
            "Re-anchored stale current_turn_user_idx %s -> %s (session=%s)",
            current_turn_user_idx, _reanchored_idx, agent.session_id or "-",
        )
        current_turn_user_idx = _reanchored_idx
    return IterationPrep(
        action="fallthrough", messages=messages, request_logger=request_logger,
        current_turn_user_idx=current_turn_user_idx,
    )


def _previous_tool_round(messages: Any) -> list:
    """【提取最新一批 Assistant 工具调用及其对应执行结果】
    从历史逆序扫描最近一次 role=="assistant" 的 tool_calls 批次，并关联匹配后续的 tool 结果，供 agent:step 网关广播使用。

    The newest assistant tool_calls batch with each call's result, for ``agent:step``."""
    for _idx, _m in enumerate(reversed(messages)):
        if _m.get("role") == "assistant" and _m.get("tool_calls"):
            _results_by_id = {}
            for _tm in messages[len(messages) - _idx:]:
                if _tm.get("role") != "tool":
                    break
                _tcid = _tm.get("tool_call_id")
                if _tcid:
                    _results_by_id[_tcid] = _tm.get("content", "")
            return [
                {
                    "name": tc["function"]["name"],
                    "result": _results_by_id.get(tc.get("id")),
                    "arguments": tc["function"].get("arguments"),
                }
                for tc in _m["tool_calls"]
                if isinstance(tc, dict)
            ]
    return []


def _inject_steer_after_newest_tool_result(agent: Any, messages: Any, steer_text: str) -> None:
    """【在最新工具返回消息后追加 /steer 用户引导行】
    若找到 role=="tool" 消息，在其后插入一条由 steer_user_row 生成的独立 user 消息；
    若历史中尚无工具消息，则将指引文本重新放回待处理队列，留待工具执行完毕后再次追加。

    Append the steer marker as a standalone user row after the newest tool message; with no
    tool message, put the text back so the post-tool-execution drain delivers it later."""
    for _si in range(len(messages) - 1, -1, -1):
        _sm = messages[_si]
        if isinstance(_sm, dict) and _sm.get("role") == "tool":
            from agent.prompt_builder import steer_user_row
            messages.insert(_si + 1, steer_user_row(steer_text))
            logger.debug("Pre-API-call steer drain: appended user row after tool msg at index %d", _si)
            return
    from agent.agent_runtime_helpers import _requeue_pending_steer
    _requeue_pending_steer(agent, steer_text)


@dataclass
class ApiCallAnnouncement:
    """Always ``action == "fallthrough"``; ``thinking_spinner`` is the started raw spinner or
    None (TUI widget / streaming consumers / verbose mode)."""

    action: str
    thinking_spinner: Any


def announce_api_call(
    agent: Any, *, messages: Any, api_messages: Any, api_call_count: Any, approx_tokens: Any,
    total_chars: Any,
) -> ApiCallAnnouncement:
    """【发起模型 API 网络请求前的 UI 宣发通知】
    根据 quiet_mode 状态切换展示行为：
    - Verbose 模式：打印本次调用的轮次序号、消息数、Token 估算及工具数量；
    - Quiet 模式：启动 KawaiiSpinner 思考动画或向 TUI 回调推送随机表情包动效。

    Print the request summary (verbose) or start the quiet-mode thinking indicator."""
    thinking_spinner = None
    if not agent.quiet_mode:
        agent._vprint(f"\n{agent.log_prefix}🔄 Making API call #{api_call_count}/{agent.max_iterations}...")
        agent._vprint(f"{agent.log_prefix}   📊 Request size: {len(api_messages)} messages, ~{approx_tokens:,} tokens (~{total_chars:,} chars)")
        agent._vprint(f"{agent.log_prefix}   🔧 Available tools: {len(agent.tools) if agent.tools else 0}")
    else:
        # Animated thinking spinner in quiet mode
        face = random.choice(KawaiiSpinner.get_thinking_faces())
        verb = random.choice(KawaiiSpinner.get_thinking_verbs())
        if agent.thinking_callback:
            # CLI TUI mode: use prompt_toolkit widget instead of raw spinner
            # (works in both streaming and non-streaming modes)
            agent.thinking_callback(f"{face} {verb}...")
        elif not agent._has_stream_consumers() and agent._should_start_quiet_spinner():
            # Raw KawaiiSpinner only when no streaming consumers and the
            # spinner output has a safe sink.
            spinner_type = random.choice(['brain', 'sparkle', 'pulse', 'moon', 'star'])
            thinking_spinner = KawaiiSpinner(f"{face} {verb}...", spinner_type=spinner_type, print_fn=agent._print_fn)
            thinking_spinner.start()

    # Log request details if verbose
    if agent.verbose_logging:
        logging.debug(f"API Request - Model: {agent.model}, Messages: {len(messages)}, Tools: {len(agent.tools) if agent.tools else 0}")
        logging.debug(f"Last message role: {messages[-1]['role'] if messages else 'none'}")
        logging.debug(f"Total message size: ~{approx_tokens:,} tokens")
    return ApiCallAnnouncement(action="fallthrough", thinking_spinner=thinking_spinner)


@dataclass
class IterationStart:
    """``action``: ``"fallthrough"`` (run the iteration) or ``"break"`` (turn ends: interrupt,
    review input budget or iteration budget exhausted — ``_turn_exit_reason`` set)."""

    action: str
    original_user_message: Any
    api_call_count: Any
    interrupted: Any
    _turn_exit_reason: Any


def begin_iteration(
    agent: Any, *, messages: Any, conversation_history: Any, original_user_message: Any,
    api_call_count: Any, interrupted: Any, _turn_exit_reason: Any,
) -> IterationStart:
    """【开启单次迭代主流程（Phase 1 门禁预检）】
    严格按序执行迭代准入逻辑：
    1. 排空并应用用户纠偏重定向（/redirect），追加至当前消息历史并持久化；
    2. 重置单轮次检查点去重状态，允许本轮迭代创建新的文件快照；
    3. 检查用户打断（_interrupt_requested），若被中断则标记退出；
    4. 检查后台复盘输入预算（_review_input_budget_exhausted），超限则安全截断；
    5. 递增 API 调用计数，检查剩余迭代预算（IterationBudget）或消费宽限调用标记（_budget_grace_call）。

    Iteration entry in the original order: apply a pending redirect, reset the checkpoint
    dedup, then the interrupt / review-budget / iteration-budget exits. ``api_call_count`` is
    incremented here (the grace call consumes its flag instead of the budget)."""
    from agent.conversation_loop import (
        _apply_active_turn_redirect, _review_input_budget_exhausted
    )

    def _verdict(action: str) -> IterationStart:
        return IterationStart(
            action=action, original_user_message=original_user_message,
            api_call_count=api_call_count, interrupted=interrupted,
            _turn_exit_reason=_turn_exit_reason,
        )

    _redirect_text = agent._drain_pending_redirect()
    if _redirect_text:
        _apply_active_turn_redirect(agent, messages, _redirect_text)
        if isinstance(original_user_message, str):
            original_user_message = (
                f"{original_user_message}\n\n" f"User correction during the turn: {_redirect_text}"
            )
        agent._persist_session(messages, conversation_history)

    # Reset per-turn checkpoint dedup so each iteration can take one snapshot.
    agent._checkpoint_mgr.new_turn()

    if agent._interrupt_requested:
        interrupted = True
        _issuer = interrupt_issuer(agent)
        _turn_exit_reason = f"interrupted_by_system({_issuer})" if _issuer else "interrupted_by_user"
        if not agent.quiet_mode:
            agent._safe_print("\n⚡ Breaking out of tool loop due to interrupt...")
        return _verdict("break")

    # Aggregate input budget for detached auxiliary forks bounds the whole review, not
    # each request; checked between iterations so the crossing request's writes landed.
    if _review_input_budget_exhausted(agent):
        _turn_exit_reason = "review_input_budget_exhausted"
        if not agent.quiet_mode:
            agent._safe_print(
                f"\n⏹️  Review input budget exhausted "
                f"({int(agent.session_input_tokens):,} tokens) — stopping "
                f"the review tool loop before the next provider call.", diagnostic=True,
            )
        return _verdict("break")

    api_call_count += 1
    agent._api_call_count = api_call_count
    agent._touch_activity(f"starting API call #{api_call_count}")

    # Grace call: budget exhausted but the model gets one more call. Consume the
    # flag so the loop exits after this iteration regardless of outcome.
    if agent._budget_grace_call:
        # Exhaustion retains one toolless grace call regardless of whether an opt-in
        # checkpoint was emitted; the warning never extends the hard budget.
        agent._budget_grace_call = False
    elif not agent.iteration_budget.consume():
        _turn_exit_reason = "budget_exhausted"
        if not agent.quiet_mode:
            agent._safe_print(f"\n⚠️  Iteration budget exhausted ({agent.iteration_budget.used}/{agent.iteration_budget.max_total} iterations used)", diagnostic=True)
        return _verdict("break")
    return _verdict("fallthrough")


@dataclass
class RetryRestartVerdict:
    """``action``: ``"fallthrough"`` (a response is ready — process it), ``"continue"``
    (a restart flag re-issues the iteration: redirect / compressed / rebuilt-for-fallback /
    length continuation) or ``"break"`` (turn ends: interrupted, non-actionable compaction
    handoff, or every retry exhausted without a response)."""

    action: str
    current_turn_user_idx: Any
    final_response: Any
    retry_count: Any
    restart_count: Any
    api_call_count: Any
    _preflight_compression_blocked: Any
    _turn_exit_reason: Any


def apply_retry_restarts(
    agent: Any, *, _retry: Any, response: Any, interrupted: Any, messages: Any,
    conversation_history: Any, user_message: Any, api_kwargs: Any, current_turn_user_idx: Any,
    final_response: Any, retry_count: Any, max_retries: Any, api_call_count: Any,
    restart_count: Any, length_continue_retries: Any,
    _preflight_compression_blocked: Any, _turn_exit_reason: Any,
) -> RetryRestartVerdict:
    """【消费重试重启标志并调度控制流（Phase 7 重启核心决策）】
    在 API 重试循环结束后，按严格优先级消费 TurnRetryState 中的四大重启标志：
    1. restart_with_redirected_messages（重定向重启）：因用户插话打断，返还迭代预算并重走迭代；受 restart_count 硬顶防死循环；
    2. restart_with_compressed_messages（压缩后重启）：因 413 溢出触发上下文压缩后重走迭代；重新锚定当前用户索引，
       若压缩后仅剩交接摘要则触发 _should_skip_model_call_for_reference_handoff 优雅退出；
    3. restart_with_rebuilt_messages（故障转移模型重建重启）：主模型崩溃切换到备用模型后，返还预算并清除前置压缩阻塞位，
       以全新的模型上下文窗口重新运行起跑门禁；
    4. restart_with_length_continuation（超长截断续写重启）：因输出达到 max_tokens 截断时触发指数级输出预算上调（2x, 4x, 8x... 封顶 32k）。
    5. 所有重试耗尽且 response 为 None：记录诊断状态并退出循环。

    Consume the ``TurnRetryState`` restart flags after the retry loop, in the original
    priority order. Refunds the iteration budget/count for restarts that produced no valid
    assistant item; ``restart_with_rebuilt_messages`` is the single consumer that clears
    ``_preflight_compression_blocked`` so the fallback gets a fresh preflight (#84733).

    The two refunding restart paths (redirect and rebuilt-for-fallback) are bounded by
    ``max_retries`` via ``restart_count`` (a per-turn accumulator) so a runaway
    interrupt/redirect that keeps re-arming a restart flag cannot refund the budget
    forever and hold the turn lease indefinitely."""

    from agent.conversation_loop import (
        _HANDOFF_SKIP_FINAL_RESPONSE, _should_skip_model_call_for_reference_handoff
    )

    def _verdict(action: str) -> RetryRestartVerdict:
        return RetryRestartVerdict(
            action=action, current_turn_user_idx=current_turn_user_idx,
            final_response=final_response, retry_count=retry_count, restart_count=restart_count,
            api_call_count=api_call_count,
            _preflight_compression_blocked=_preflight_compression_blocked,
            _turn_exit_reason=_turn_exit_reason,
        )

    if _retry.restart_with_redirected_messages:
        restart_count += 1
        if restart_count > max_retries:
            # 【重定向防无限返还死循环熔断】
            # 若用户频繁打断重定向，restart_count 超过 max_retries 则停止返还预算，终结轮次；
            # 未生效的纠偏文本通过 steer() 返还给下一个轮次，防止在 clear_interrupt() 中静默丢失。
            # A redirect/interrupt keeps re-arming this flag: stop refunding the iteration
            # budget and re-issuing the same logical iteration, or a runaway turn holds the
            # turn lease indefinitely (redirect restarts previously had no bound).
            _turn_exit_reason = "redirect_restart_limit_exceeded"
            logger.warning(
                "Redirected-message restart limit (%s) exceeded; ending turn instead of "
                "refunding the iteration budget indefinitely.",
                max_retries,
            )
            # The correction that tripped the cap was never applied; hand it back as the
            # next user turn (result["pending_steer"]) instead of losing it to clear_interrupt().
            _unapplied = agent._drain_pending_redirect()
            if _unapplied:
                agent.steer(_unapplied)
            return _verdict("break")
        # Cancelled request produced no valid assistant item: reuse the same logical
        # iteration after the outer loop appends partial context + correction.
        api_call_count -= 1
        agent.iteration_budget.refund()
        _retry.restart_with_redirected_messages = False
        return _verdict("continue")

    if interrupted:
        _issuer = interrupt_issuer(agent)
        _turn_exit_reason = (
            f"interrupted_during_api_call({_issuer})" if _issuer else "interrupted_during_api_call"
        )
        return _verdict("break")

    if _retry.restart_with_compressed_messages:
        api_call_count -= 1
        agent.iteration_budget.refund()
        # 【压缩后重启：防无限无效压缩】
        # 压缩重启计入 retry_count，防止压缩缩减比例不足引发死循环；
        # 压缩重建了列表（尾部消息是崭新的压缩副本），重新锚定 current_turn_user_idx，
        # 避免外挂预取将上下文注入到历史轮次或破坏 SessionDB 刷盘游标（参见 issue #48677, #80622）。
        # Compression restarts count toward the retry limit so a compression that
        # shrinks messages but not enough can't loop forever.
        retry_count += 1
        _retry.restart_with_compressed_messages = False
        if _should_skip_model_call_for_reference_handoff(
            # Compression rebuilt the list (tail messages are fresh compaction copies), so the
            # pre-compression index of this turn's user message is stale. Re-anchor both index trackers: the
            # api_content stamp below, the loop's injection site, and the flush's persist-override row
            # (#48677) must all target the surviving dict, not a stale position. Exact-content match first
            # so a todo-snapshot user message appended after the tail can't steal the anchor.
            messages, user_message
        ):
            logger.info(
                "Skipping compressed-restart model call: reference-only "
                "handoff would be the sole active user turn (#80622)"
            )
            if not final_response:
                final_response = _HANDOFF_SKIP_FINAL_RESPONSE
            _turn_exit_reason = "compaction_handoff_not_actionable"
            return _verdict("break")
        # In-loop compression rebuilt `messages`; re-anchor the current-turn index
        # like the prologue, AFTER the handoff guard (it may re-append this turn's
        # ask). A stale anchor injects prefetch into a historical row.
        current_turn_user_idx = _reanchor(agent, messages, user_message)
        return _verdict("continue")

    if _retry.restart_with_rebuilt_messages:
        restart_count += 1
        if restart_count > max_retries:
            # A stall/failure keeps re-escalating to the fallback chain: stop refunding the
            # iteration budget and re-issuing, or a runaway turn holds the turn lease
            # indefinitely (rebuilt restarts previously had no bound).
            _turn_exit_reason = "rebuilt_restart_limit_exceeded"
            logger.warning(
                "Rebuilt-message restart limit (%s) exceeded; ending turn instead of "
                "refunding the iteration budget indefinitely.",
                max_retries,
            )
            return _verdict("break")
        # 【故障转移重建重启（Rebuilt for Fallback）】
        # 模型级级联回退激活：清除前置压缩阻塞标记（_preflight_compression_blocked=False），
        # 因为备用模型可能拥有不同的上下文窗口，确保首次回退调用前能够以新模型的真实窗口重新运行门禁。
        # A stall/failure escalated to the fallback chain: re-issue against the
        # active fallback provider, refunding budget/count for the stalled attempt.
        api_call_count -= 1
        agent.iteration_budget.refund()
        _retry.restart_with_rebuilt_messages = False
        # Failover shrank the compressor window: clear the preflight block so
        # preflight re-runs before the first fallback call (single consumer).
        _preflight_compression_blocked = False
        return _verdict("continue")

    if _retry.restart_with_length_continuation:
        # 【输出截断续写预算指数提升（Length Continuation Boost）】
        # 每次超长截断重试时将输出上限指数级提升：2x, 4x, 8x, 16x 基础值，最高至 32,768 tokens。
        # Boost output budget per retry: 2×, 4×, 8×, 16× base, capped at 32 768, via
        # _ephemeral_max_output_tokens. Keep a larger original provider/model
        # default as the floor so retries never downshift.
        _boost = (agent.max_tokens or 4096) * (2 ** length_continue_retries)
        _requested_cap = agent._requested_output_cap_from_api_kwargs(api_kwargs)
        if _requested_cap is not None:
            _boost = max(_boost, _requested_cap)
        _boost_cap = max(32768, _requested_cap or 0)
        agent._ephemeral_max_output_tokens = min(_boost, _boost_cap)
        return _verdict("continue")

    # All retries may exhaust with `response` still None; break out cleanly.
    if response is None:
        _turn_exit_reason = "all_retries_exhausted_no_response"
        agent._emit_diagnostic_status("❌ The model provider didn't answer after all retries. Send /retry, or switch models with /model.")
        agent._persist_session(messages, conversation_history)
        return _verdict("break")
    return _verdict("fallthrough")
