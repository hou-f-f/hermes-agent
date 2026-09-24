"""【API 预调用守卫网关 / Pre-API Pressure Gate】
对话轮次循环中调用大模型 API 之前的安全预检与上下文压力闸门：
1. Ollama 本地模型运行时上下文下限守卫（防止本地小上下文模型因为加载了庞大工具定义而直接崩盘）；
2. 供应商上下文溢出（Context Overflow）重试恢复布防；
3. 压缩收益不足阻断器（Insufficient-progress Blocker）：对比完整请求 Token 压力，防止压缩后 Token 几乎没降却陷入无限重试；
4. 调度 turn_preflight.run_preflight_compression 执行前置微压缩或全量压缩。

本模块严禁在模块顶层导入 agent.conversation_loop（防止循环引用）。

Pre-API pressure gate for the conversation turn loop: the Ollama runtime-context floor,
the provider-overflow re-check arming, the insufficient-progress blocker (compares fully
assembled requests, not raw ``messages``) and the call into
``turn_preflight.run_preflight_compression``. Nothing here imports
``agent.conversation_loop`` at module level (cycle)."""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any

from agent.message_metadata import append_message
from agent.turn_context import _compression_warrants_another_preflight_pass
from agent.turn_preflight import PreflightGateVerdict, run_preflight_compression

logger = logging.getLogger("agent.conversation_loop")


def run_preflight_gate(
    agent: Any, *, request_pressure_tokens: Any, _moa_prepared_request: Any,
    pending_moa_prepared_request: Any, messages: Any, system_message: Any, user_message: Any,
    active_system_prompt: Any, conversation_history: Any, api_call_count: Any,
    compression_attempts: Any, max_compression_attempts: Any, effective_task_id: Any,
    final_response: Any, failed: Any, _turn_exit_reason: Any, _compression_timeout_exhausted: Any,
    _preflight_compression_blocked: Any, _provider_overflow_recovery_pending: Any,
    _last_preflight_pressure: Any,
) -> PreflightGateVerdict:
    """【执行 API 调用前置守卫链 / Run Pre-API Guard Chain】
    在生命周期 Phase 2 执行，按严格顺序执行三道防线：
    1. Ollama 上下文门槛检查：若本地模型上下文甚至放不下 Hermes 工具定义，立即阻断并退还迭代预算；
    2. 上下文压力（Request Pressure）评估：对比阈值决定是否启动前置压缩（Preflight Compression）；
    3. 压缩收益防死锁检测：如果上一次压缩后 Token 数量没有实质性下降（insufficient progress），
       则阻断本轮继续压缩，防止 Agent 在同一个轮次内死循环消耗压缩配额。
    
    参数说明：
    - `request_pressure_tokens`: 包含全量 System Prompt、历史消息和工具声明的精确 Token 预估值。
    - `_last_preflight_pressure`: 在此函数中被消费（置为 None），仅在发生实质性压缩后重新布防。

    Run the pre-API guard chain in the original order. ``_last_preflight_pressure`` is
    consumed here (set to None) and re-armed only by a compression pass, so a blocked
    preflight never compares against a stale figure."""
    from agent.conversation_loop import _ollama_context_limit_error

    v = PreflightGateVerdict(
        action="fallthrough", pending_moa_prepared_request=pending_moa_prepared_request,
        messages=messages, active_system_prompt=active_system_prompt,
        conversation_history=conversation_history, api_call_count=api_call_count,
        compression_attempts=compression_attempts, final_response=final_response, failed=failed,
        _turn_exit_reason=_turn_exit_reason,
        _compression_timeout_exhausted=_compression_timeout_exhausted,
        _preflight_compression_blocked=_preflight_compression_blocked,
        _provider_overflow_recovery_pending=_provider_overflow_recovery_pending,
        _last_preflight_pressure=None,
    )

    _runtime_context_error = _ollama_context_limit_error(agent, request_pressure_tokens)
    if _runtime_context_error:
        v.final_response = _runtime_context_error
        v.failed = True
        v._turn_exit_reason = "ollama_runtime_context_too_small"
        append_message(messages, {"role": "assistant", "content": v.final_response})
        agent._emit_diagnostic_status("❌ Ollama runtime context is too small for Hermes tool use")
        v.api_call_count -= 1
        agent._api_call_count = v.api_call_count
        with suppress(Exception):
            agent.iteration_budget.refund()
        v.action = "break"
        return v

    # Pre-API pressure check: tool results grow a turn and last_prompt_tokens lags
    # them. Mirror the turn-prologue guard chain: defer on noisy estimate, skip in
    # failure cooldown, then should_compress().
    _compressor = agent.context_compressor
    _preflight_threshold = int(getattr(_compressor, "threshold_tokens", 0) or 0)
    _provider_overflow_preflight = _provider_overflow_recovery_pending and (
        _preflight_threshold <= 0 or request_pressure_tokens >= _preflight_threshold
    )
    if _provider_overflow_recovery_pending and not _provider_overflow_preflight:
        # The outer-loop rebuild includes system prompt, request-only injections and
        # tool schemas; only that full request with output runway may be sent.
        v._provider_overflow_recovery_pending = False
    # Compare fully assembled requests, not raw ``messages`` (which omit
    # api_content, plugin injections, prefills, MoA context, ephemeral system text).
    if (
        _last_preflight_pressure is not None
        and request_pressure_tokens >= _preflight_threshold
        and not _compression_warrants_another_preflight_pass(
            _last_preflight_pressure, request_pressure_tokens, _preflight_threshold
        )
    ):
        # Stop proactive retries this turn without consuming the shared overflow-
        # recovery budget; the provider's error handler may still compact.
        v._preflight_compression_blocked = True
        logger.warning(
            "Pre-API compression made insufficient progress: ~%s -> "
            "~%s request tokens; skipping additional preflight passes",
            f"{_last_preflight_pressure:,}",
            f"{request_pressure_tokens:,}",
        )
    return run_preflight_compression(
        agent, v, compressor=_compressor, request_pressure_tokens=request_pressure_tokens,
        provider_overflow_preflight=_provider_overflow_preflight,
        # An anchored figure is real usage + delta: never deferred. Only a whole-context rough
        # estimate waits for the provider's count.
        defer_preflight=(
            (lambda _t: False) if getattr(agent, "_request_pressure_anchored", False)
            else getattr(_compressor, "should_defer_preflight_to_real_usage", lambda _t: False)
        ),
        moa_prepared_request=_moa_prepared_request, system_message=system_message,
        user_message=user_message, max_compression_attempts=max_compression_attempts,
        effective_task_id=effective_task_id,
    )
