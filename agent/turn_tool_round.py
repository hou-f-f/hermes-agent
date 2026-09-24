"""【单轮工具调用执行中枢 / Tool Calling Round Execution Hub】
对话轮次循环中负责工具校验、持久化与执行分发的关键模块：
1. 校验、配额限制与参数去重（Validate/Cap/Dedupe）：清洗大模型生成的工具参数；
2. 执行前持久化铁律（Persist-Before-Execute Durability Invariant）：
   在产生任何真实副作用之前，必须先将模型输出的 tool_calls 落盘持久化到会话数据库（SessionDB）中！
   确保即使破坏性工具中途导致机器重启或进程崩溃，重启恢复（Resume）时也能识别已执行区块；
3. 工具并发与分段调度：结合 SegmentPlanner 区分只读安全工具与串行屏障工具；
4. 审批与安全拦截（Guardrail Halts）：响应 tools/approval 的确认请求；
5. 工具执行后的微压缩（Micro-compaction）：若工具返回超大日志/输出，立即就地压缩。

本模块严禁在模块顶层导入 agent.conversation_loop（防止循环依赖）。

One tool-calling round of the conversation turn loop: validate/cap/dedupe the model's
tool calls, persist the tool-call turn BEFORE any side effect, execute the tools, honour
guardrail halts / persistence failures, then compress after tool results. Nothing here
imports ``agent.conversation_loop`` at module level (cycle) — loop-internal helpers resolve
lazily.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional, Tuple

from agent.message_metadata import append_message
from agent.message_sanitization import coalesce_tool_call_id
from agent.turn_preflight import compress_after_tool_results
from agent.turn_tool_validation import validate_tool_calls

logger = logging.getLogger("agent.conversation_loop")

# Post-response housekeeping tools: a round made only of these mutes tool progress.
_HOUSEKEEPING_TOOLS = frozenset({"memory", "todo_list", "skill_manage", "session_search"})


@dataclass
class ToolRoundVerdict:
    """【工具轮次执行裁决 / Tool Round Verdict】
    ``action`` 取值说明：
    - ``"continue"``：工具执行完毕并将结果写回上下文，驱动主循环进入下一轮模型思考（API 调用）；
    - ``"break"``：轮次终止（如触发持久化失败、安全守卫硬拦截、工具结果超大触发压缩并收工）；
    - ``"return"``：直接产出本轮对话的最终结果字典。

    ``action``: ``"continue"`` (tools ran, next API call), ``"break"`` (turn ends:
    persistence failure, guardrail halt, post-tool compression end) or ``"return"``
    (``result`` is the turn's result dict). The other fields are the loop locals the round
    rebinds."""

    action: str
    messages: Any
    conversation_history: Any
    active_system_prompt: Any
    compression_attempts: Any
    final_response: Any
    failed: Any
    _turn_exit_reason: Any
    truncated_tool_call_retries: Any
    current_turn_user_idx: Any
    result: Optional[Dict[str, Any]] = None


def run_tool_round(
    agent: Any, *, assistant_message: Any, finish_reason: Any, messages: Any,
    conversation_history: Any, api_call_count: Any, effective_task_id: Any, user_message: Any,
    system_message: Any, active_system_prompt: Any, compression_attempts: Any,
    max_compression_attempts: Any, final_response: Any, failed: Any, _turn_exit_reason: Any,
    truncated_tool_call_retries: Any, current_turn_user_idx: Any,
) -> ToolRoundVerdict:
    """【执行单轮工具调用链 / Execute One Tool Round】
    在生命周期 Phase 9 执行，严格维持系统设计不变量：
    1. 执行前持久化（Persist-before-execute）：这是系统的耐久性底线（Durability Invariant）。
       如果落盘追加失败，必须立即中止轮次，绝不允许仅凭内存临时状态擅自触发带副作用的工具；
    2. 工具分派与错误包装：每个工具的调用结果严格配对为一条 `role: tool` 消息并附带匹配的 `tool_call_id`。

    Execute one tool round in the exact original order. Persist-before-execute is a
    durability invariant: resume must see the executed block if a destructive tool restarts
    Hermes; a failed canonical append ends the turn rather than running tools from
    process-only state."""
    from agent.conversation_loop import _invalid_tool_name_error_content

    def _verdict(action: str, result: Optional[Dict[str, Any]] = None) -> ToolRoundVerdict:
        return ToolRoundVerdict(
            action=action, messages=messages, conversation_history=conversation_history,
            active_system_prompt=active_system_prompt, compression_attempts=compression_attempts,
            final_response=final_response, failed=failed, _turn_exit_reason=_turn_exit_reason,
            truncated_tool_call_retries=truncated_tool_call_retries,
            current_turn_user_idx=current_turn_user_idx, result=result,
        )

    if not agent.quiet_mode:
        agent._vprint(f"{agent.log_prefix}🔧 Processing {len(assistant_message.tool_calls)} tool call(s)...")

    if agent.verbose_logging:
        for tc in assistant_message.tool_calls:
            raw_args = tc.function.arguments
            args_preview = raw_args[:200] if isinstance(raw_args, str) else repr(raw_args)[:200]
            logging.debug("Tool call: %s with args: %s...", tc.function.name, args_preview)

    _tvv = validate_tool_calls(
        agent, assistant_message, finish_reason, messages=messages,
        conversation_history=conversation_history, api_call_count=api_call_count,
        effective_task_id=effective_task_id,
    )
    if _tvv.action == "return":
        return _verdict("return", _tvv.result)
    if _tvv.action == "continue":
        return _verdict("continue")

    # Post-call guardrails.
    assistant_message.tool_calls = agent._deduplicate_tool_calls(
        agent._cap_delegate_task_calls(assistant_message.tool_calls)
    )

    # Mixed batch: the assistant message keeps EVERY emitted call (each tool_call needs a
    # matching result) while only valid ones dispatch.
    _invalid_batch_calls = [
        tc for tc in assistant_message.tool_calls if tc.function.name not in agent.valid_tool_names
    ] if _tvv.mixed_invalid_batch else []

    assistant_msg, duplicate_previous_interim = stage_tool_call_message(
        agent, assistant_message=assistant_message, finish_reason=finish_reason, messages=messages
    )
    append_message(messages, assistant_msg)

    # 【架构不变量 5 实践：工具调用与返回结果严格配对（Tool Pairing Invariant）】
    # 痛点剖析：大模型单轮次吐出多个工具调用（Mixed Batch），其中某个工具名非法（例如模型幻觉了一个不存在的工具）。
    # 为什么不能直接从列表里删掉它？
    # 因为主流大模型 API（Anthropic/OpenAI/DeepSeek）强制要求：assistant 消息中声明的所有 tool_call_id，
    # 在下一轮消息序列中必须有严格一一对应的 `role: "tool"` 响应！缺一个就会被 API 直接报 400 格式错误拒接！
    # 解决机制：为非法的工具调用就地构造合法的错误响应帧（Synthesized Error Result），确保消息骨架结构合法。
    if _invalid_batch_calls:
        for tc in _invalid_batch_calls:
            append_message(messages, {
                "role": "tool",
                "name": tc.function.name,
                "tool_call_id": coalesce_tool_call_id(tc),
                "content": _invalid_tool_name_error_content(
                    tc.function.name, agent.valid_tool_names
                ),
            })
        assistant_message.tool_calls = [
            tc for tc in assistant_message.tool_calls if tc.function.name in agent.valid_tool_names
        ]

    # 【架构不变量 2 实践：执行前强制持久化铁律（Persist-Before-Execute Durability Invariant）】
    # 核心原理：
    # 在真正调用工具（可能会写磁盘、运行脚本、删文件等产生副作用）之前，必须先将模型要执行的意图消息
    # 刷入底层的 SQLite SessionDB 数据库！
    # 痛点防范：
    # 如果一个带有破坏性的工具运行到一半导致操作系统断电或 Hermes 崩溃，
    # 系统在重启恢复会话（Resume）时，能够精准知道崩溃前系统下发了哪些命令，避免状态不一致。
    # 关键防线：如果落盘持久化失败（如磁盘写满或数据库被锁），宁可立即终止本轮对话（_verdict("break")），
    # 也绝不基于内存纯临时状态去冒失执行带副作用的工具！
    # Persist the tool-call turn before any tool side effects so resume sees the executed
    # block if a destructive tool restarts Hermes.
    try:
        _tool_turn_persisted = agent._flush_messages_to_session_db(messages, conversation_history)
    except Exception as exc:
        _tool_turn_persisted = False
        from hermes_state import classify_persistence_error
        agent._last_persistence_error_cause = classify_persistence_error(exc)
        logger.warning(
            "Incremental tool-call persistence failed before execution "
            "(session=%s): %s",
            agent.session_id or "none",
            exc,
        )

    if _tool_turn_persisted is False:
        # Canonical append failed: never project the row or run tools from process-only
        # state; break rather than retry. No recorded cause means genuinely unknown.
        if getattr(agent, "_last_persistence_error_cause", None) is None:
            agent._last_persistence_error_cause = "unknown"
        _turn_exit_reason = "session_persistence_failed"
        final_response = ""
        failed = True
        return _verdict("break")

    # 前端 UI 严禁观测到仅存在于内存中的助理/工具调用行：
    # 必须在成功落盘追加到数据库之后，才向外部广播临时阶段性解说（interim commentary）。
    # A UI must never observe an assistant/tool-call row that is only an in-memory
    # projection: emit interim commentary after the DB append.
    if not duplicate_previous_interim:
        agent._emit_interim_assistant_message(assistant_msg)

    # 在执行工具前冲刷掉当前未闭合的流式文本框，防止早先吐出的文本与工具执行日志缠绕混杂。
    # 仅针对前端显示回调生效 —— 文本转语音 TTS（_stream_callback）绝对不能接收 None（表示 EOS 流结束）。
    # Flush open streaming boxes before tools so early content doesn't wrap tool feed
    # lines. Display callback only — TTS (_stream_callback) must NOT receive None (EOS).
    if agent.stream_delta_callback:
        with suppress(Exception):
            agent.stream_delta_callback(None)

    agent._execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count)

    if getattr(agent, "_incremental_persistence_failed", False):
        # 工具执行结果无法持久化落盘：绝对不要将纯内存临时结果发送给模型，也不要在本轮投射后续事件，直接终止。
        # Tool result could not be made canonical: never send the in-memory result to
        # the model or project later events from this turn.
        _turn_exit_reason = "session_persistence_failed"
        final_response = ""
        failed = True
        return _verdict("break")

    if agent._tool_guardrail_halt_decision is not None:
        decision = agent._tool_guardrail_halt_decision
        _turn_exit_reason = "guardrail_halt"
        final_response = agent._toolguard_controlled_halt_response(decision)
        agent._emit_diagnostic_status(f"⚠️ Tool guardrail halted {decision.tool_name}: {decision.code}")
        append_message(messages, {"role": "assistant", "content": final_response})
        # 显式广播安全拦截原因，避免用户误以为程序崩溃；此时流式回调依然保活，SSE/TUI 客户端能完整展示拦截说明。
        # Emit the halt so it isn't mistaken for a crash; the stream callback is still
        # alive, so SSE/TUI clients see the explanation.
        if final_response:
            agent._safe_print(f"\n{final_response}\n")
            if agent.stream_delta_callback:
                with suppress(Exception):
                    agent.stream_delta_callback(final_response)
                    agent.stream_delta_callback(None)
        return _verdict("break")

    # 重置单轮重试计数器，防止某一次截断污染整轮会话
    # Reset per-turn retry counters so one truncation can't poison the turn.
    truncated_tool_call_retries = 0
    # 延迟分段换行：当真正的正文到来时，_fire_stream_delta() 会在前端补充一个 "\n\n"，
    # 从而避免多轮工具执行之间堆积大量无意义的空行。
    # Defer the paragraph break: _fire_stream_delta() prepends one "\n\n" when real
    # text arrives, so tool iterations don't stack blank lines.
    agent._stream_needs_break = True
    # 预算退还机制（Iteration Budget Refund）：
    # 当本次轮次调用的唯一工具是 `execute_code`（通过代码解释器进行编程式工具调用）时，
    # 这种开销极低的 RPC 风格快速调用不应白白扣减宝贵的模型交互迭代预算，在此将扣除的配额退还。
    # Refund the iteration when the ONLY tool was execute_code (programmatic tool
    # calling) — cheap RPC-style calls shouldn't eat the budget.
    if {tc.function.name for tc in assistant_message.tool_calls} == {"execute_code"}:
        agent.iteration_budget.refund()

    _ptc = compress_after_tool_results(
        agent, messages=messages, system_message=system_message, user_message=user_message,
        active_system_prompt=active_system_prompt, conversation_history=conversation_history,
        compression_attempts=compression_attempts,
        max_compression_attempts=max_compression_attempts, effective_task_id=effective_task_id,
        final_response=final_response, turn_exit_reason=_turn_exit_reason,
        current_turn_user_idx=current_turn_user_idx,
    )
    messages = _ptc.messages
    active_system_prompt = _ptc.active_system_prompt
    conversation_history = _ptc.conversation_history
    compression_attempts = _ptc.compression_attempts
    final_response = _ptc.final_response
    _turn_exit_reason = _ptc.turn_exit_reason
    current_turn_user_idx = _ptc.current_turn_user_idx
    if _ptc.end_turn:
        return _verdict("break")

    # 增量保存会话日志（确保即使后续被用户打断，之前的执行进度依然可见）
    # Save session log incrementally (so progress is visible even if interrupted)
    agent._session_messages = messages
    # 【刷新活跃度时间戳以防网关超时被杀 / Gateway Inactivity Timeout Prevention】
    # 关键机制：在继续下一轮之前触碰活跃状态（touch activity），
    # 避免工具执行完成到下一次 API 调用开始之间的耗时（如耗时的上下文压缩、数据库落盘以及较慢的后续 API）
    # 导致网关的空闲监控检测到陈旧时间戳而误将 Agent 杀掉（HERMES_AGENT_TIMEOUT 默认 1800 秒，参见 issue #69559, #69131）。
    # Touch activity so slow post-tool work plus a slow follow-up API call can't exceed
    # the gateway inactivity timeout (HERMES_AGENT_TIMEOUT).
    # Touch activity before continuing so the gateway's inactivity monitor never sees a stale timestamp
    # between tool completion and the start of the next API call. Without this, a tool-call result (which
    # takes ~0s to process) followed by slow post-tool processing (compression, persist) and a slow
    # follow-up API call can exceed the gateway inactivity timeout (HERMES_AGENT_TIMEOUT, default 1800s) and
    # the gateway kills the session before the next activity touch fires (#69559, #69131).
    agent._touch_activity(f"tool results posted, continuing iteration #{api_call_count}")
    return _verdict("continue")


def stage_tool_call_message(
    agent: Any, *, assistant_message: Any, finish_reason: Any, messages: Any
) -> Tuple[Dict[str, Any], bool]:
    """【暂存工具调用消息并更新单轮静音/兜底状态 / Stage Tool Call Message】
    构建 assistant 的 tool_calls 消息字典，并处理轮次级边界状态：
    1. 丢弃工具调用旁裸露的中括号协议脚手架标记（如 `[memory]`，防止重试死循环，issue #78148）；
    2. 分类内务/管家工具（Housekeeping Tools）：若全部为内存/待办/搜索等静默工具，则开启流式静音；若包含实际操作工具，清除静音；
    3. 保留伴随工具调用的可见文本作为兜底最终回复（fallback final response），防止后续轮次返回空内容；
    4. 弹出前置思考注入预填消息（`_thinking_prefill`），重置重试计数；
    5. 重置空结果催促与遗漏调用计数器；
    6. 检测是否与前序 incomplete 消息重复，返回 `(assistant_msg, duplicate_previous_interim)`。

    Build the assistant tool-call row and update the per-turn fallback/mute state.

    Drops a bare bracketed marker beside a call (#78148), classifies housekeeping-only
    rounds, keeps visible content as the empty-follow-up fallback, pops thinking-only
    prefills (resetting their counters), re-arms the post-tool nudge and the
    dropped-tool-call stall budget. Returns ``(assistant_msg, duplicate_previous_interim)``;
    the flag suppresses re-emitting interim commentary the previous ``incomplete`` row
    already showed."""
    from agent.conversation_loop import _STALE_MARKER_RE

    assistant_msg = agent._build_assistant_message(assistant_message, finish_reason)

    turn_content = assistant_message.content or ""

    # 工具调用旁孤立的中括号标记（例如 ``[memory]``）属于协议脚手架残留；
    # 若将其持久化，后续工具执行完毕后的兜底重放机制会永远重放该标记（参见 issue #78148）。
    # A bare bracketed token (e.g. ``[memory]``) beside a function call is protocol
    # scaffolding; persisting it lets the post-tool fallback replay it forever (#78148).
    if assistant_message.tool_calls and _STALE_MARKER_RE.fullmatch(turn_content.strip()):
        logger.warning(
            "Discarding bare tool-call marker from assistant content: %s", turn_content
        )
        turn_content = ""
        assistant_msg["content"] = ""

    # 无论可见内容如何，对工具进行分类：包含实体工具（非管家内务工具）的轮次必须
    # 使旧的管家工具兜底失效（避免两轮前的管家解说被错误归因于当前工具轮次），
    # 并清除先前管家轮次设置的静音标记，否则 _vprint 会压制当前工具的进度输出。
    # Classify tools regardless of visible content: a substantive tool-only turn must
    # invalidate any older housekeeping fallback (so a two-turn-old housekeeping
    # narration isn't attributed to the preceding tool turn), and clear the mute flag a
    # prior housekeeping turn set, else _vprint suppresses this turn's tool progress.
    _all_housekeeping = all(
        tc.function.name in _HOUSEKEEPING_TOOLS for tc in assistant_message.tool_calls
    )
    if assistant_message.tool_calls and not _all_housekeeping:
        agent._last_content_with_tools = None
        agent._last_content_tools_all_housekeeping = False
        agent._mute_post_response = False

    # 单轮内同时包含文本 content 和 tool_calls：保留该文本作为兜底最终回复（fallback final response），
    # 以防工具执行后的后续轮次返回空内容。
    # 仅当所有工具均为事后管家内务工具时静音；包含实体操作工具时保持输出开启。
    # Content + tool_calls in one turn: keep the content as a fallback final response in
    # case the follow-up turn after tools is empty. Mute only when EVERY tool call is
    # post-response housekeeping; substantive tools keep output on.
    if turn_content and agent._has_content_after_think_block(turn_content):
        agent._last_content_with_tools = turn_content
        agent._last_content_tools_all_housekeeping = _all_housekeeping
        if _all_housekeeping and agent._has_stream_consumers():
            agent._mute_post_response = True
        elif agent._should_emit_quiet_tool_messages():
            clean = agent._strip_think_blocks(turn_content).strip()
            if clean:
                agent._vprint(f"  ┊ 💬 {clean}")

    # 在追加新消息之前，弹出仅包含思考内容的预填消息（与 final-response 路径逻辑相同）。
    # 在预填恢复后成功触发工具调用会重置预填计数器，因此每次工具调用的成功都是全新的起点，而不是累积消耗。
    # Pop thinking-only prefill message(s) before appending (same rationale as the
    # final-response path). Tool calls after a prefill recovery reset the prefill
    # counter, so each tool-call success is a fresh start, not a cumulative burn.
    _had_prefill = False
    while messages and isinstance(messages[-1], dict) and messages[-1].get("_thinking_prefill"):
        messages.pop()
        _had_prefill = True
    if _had_prefill:
        agent._thinking_prefill_retries = 0
        agent._empty_content_retries = 0
    # 重新装载工具执行后的空响应催促（post-tool nudge），使其在后续的工具轮次中仍能触发；
    # 工具调用的成功落地说明已经从工具调用丢失停滞中恢复，因此每个停滞状态刷新该预算。
    # Re-arm the post-tool nudge so it can fire on a LATER tool round; a landed tool call
    # recovers any dropped-tool-call stall, so refresh that budget per stall.
    agent._post_tool_empty_retried = False
    agent._dropped_toolcall_retries = 0

    previous_msg = messages[-1] if messages else None
    current_interim_visible = agent._interim_assistant_visible_text(assistant_msg)
    duplicate_previous_interim = (
        bool(current_interim_visible)
        and isinstance(previous_msg, dict)
        and previous_msg.get("role") == "assistant"
        and previous_msg.get("finish_reason") == "incomplete"
        and agent._interim_assistant_visible_text(previous_msg) == current_interim_visible
    )
    return assistant_msg, duplicate_previous_interim
