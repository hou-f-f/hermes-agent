"""API-call exception handler for the conversation turn's retry loop: pre-/post-classification
recovery, interpreter-shutdown abandon, classified-error routing, overflow recovery, the
non-retryable client-error exit, max-retries exhaustion (primary transport recovery →
fallback → terminal result) and the interruptible backoff. Nothing here imports
``agent.conversation_loop`` at module level (cycle) — loop-internal helpers resolve lazily so
``patch("agent.conversation_loop.X")`` sites keep intercepting.

【阶段 7：模型 API 异常诊断与重试恢复门面 / Phase 7: API Error Handling & Recovery】
本模块负责对话轮次内部重试循环（Turn Retry Loop）中捕获的 API 调用异常的全生命周期处理：
1. 分类前/分类后自愈（Pre-/Post-classification recovery）：如畸变 JSON 截断修复、凭证池轮换等；
2. 解释器关闭熔断（Interpreter-shutdown abandon）：当主进程正在退出时立即放弃，防止日志刷屏；
3. 分类错误路由（Classified-error routing）：基于统一分类器对 429 速率限制、401 鉴权失效等做针对性处理；
4. 上下文溢出急救（Overflow recovery）：自动触发 Micro-compaction 上下文压缩或减小 max_tokens；
5. 不可重试客户端错误安全退出（Non-retryable client-error exit）：处理 Copilot 凭据自愈或触发模型级降级回退；
6. 最大重试耗尽与回退（Max-retries exhaustion）：主通道恢复 -> 级联回退（Fallback Model）-> 终态失败封装；
7. 可中断的指数退避等待（Interruptible backoff sleep）：退避期间随时响应用户打断或转向（/steer）。
架构铁律：任何回退模型的激活都必须带着 ``restart_with_rebuilt_messages = True`` 退出重试循环（返回 action="break"），
确保 Preflight 门禁在新回退模型的上下文窗口与配置下重新执行全量静态检查（issue #84733）。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import ssl
import time
from typing import Any, Dict, Optional

from agent.error_classifier import RETRYABLE_CLIENT_REASONS, FailoverReason, classify_api_error
from agent.turn_overflow import recover_from_overflow
from agent.turn_recovery import (
    _NONRETRYABLE_LABELS, abort_turn_on_interrupt, compute_error_backoff, interruptible_backoff_sleep,
    log_api_error_attempt,
    max_retries_exhausted_result, nonretryable_client_error_result, recover_after_classification,
    recover_before_classification, route_classified_error,
)

logger = logging.getLogger("agent.conversation_loop")


@dataclass
class ApiErrorVerdict:
    """【API 异常处理决策流结果 / API Error Verdict】
    封装异常恢复处理后的下一步动作：
    - ``action == "continue"``：已自愈（如换 Key 或修复格式成功），在当前循环内重试 API 调用；
    - ``action == "break"``：退出当前 API 重试循环（例如激活了备用模型 Fallback 或用户发起了 /steer 转向），将轮次控制权交还给外层 iteration_prep 进行上下文重建；
    - ``action == "return"``：已彻底失败或被打断，终止当前 Turn，``result`` 包含终态字典；
    - 绝不存在 ``"fallthrough"``（异常分支必须明确收敛到某种动作）；
    - 其余字段为重试循环中需要重新绑定的局部变量；``_provider_overflow_recovery_pending`` 仅在触发时合并置为 True。

    ``action``: ``"continue"`` (retry the API call), ``"break"`` (leave the retry loop:
    fallback armed / redirect pending) or ``"return"`` (``result`` is the turn's result dict);
    ``"fallthrough"`` never happens — the handler always ends in an exit. The other fields
    are the retry-loop locals the handler rebinds; ``_provider_overflow_recovery_pending`` is
    merge-only (caller sets True when set)."""

    action: str
    thinking_spinner: Any
    messages: Any
    active_system_prompt: Any
    conversation_history: Any
    approx_tokens: Any
    retry_count: Any
    max_retries: Any
    compression_attempts: Any
    _provider_overflow_recovery_pending: Any
    result: Optional[Dict[str, Any]] = None


def handle_api_error(
    agent: Any, *, api_error: Any, _retry: Any, thinking_spinner: Any, messages: Any,
    api_messages: Any, api_kwargs: Any, system_message: Any, active_system_prompt: Any,
    conversation_history: Any, approx_tokens: Any, retry_count: Any, max_retries: Any,
    compression_attempts: Any, max_compression_attempts: Any, api_call_count: Any,
    api_request_id: Any, api_start_time: Any, effective_task_id: Any, turn_id: Any,
) -> ApiErrorVerdict:
    """【处理单次 API 调用异常的核心入口 / Core API Error Handler】
    严格按既定防御梯队对 ``api_error`` 执行级联恢复：
    1. 停止思考动画 Spinner（静默暂存状态，只有当所有重试和回退都耗尽时才整体刷新输出）；
    2. 分类前预处理（recover_before_classification）；
    3. 进程退出快速熔断检测（interpreter_shutting_down）；
    4. 统一异常分类（classify_api_error）；
    5. 分类后恢复尝试（recover_after_classification）；
    6. 用户打断检测（abort_turn_on_interrupt，但注意保留转向 pending redirect）；
    7. 分类错误路由（route_classified_error）；
    8. 上下文超限急救与微压缩（recover_from_overflow）；
    9. 未恢复错误的最终决断（settle_unrecovered_error）。
    【关键架构约束】：任何激活备用模型（Fallback）的操作都必须以 ``"break"`` 形式离开重试循环，
    使得前置 Preflight 门禁能够在新模型的 Context Window 与特性限制下重新计算请求（参见 issue #84733）。

    Recover from ``api_error`` in the original order. Every fallback activation must leave
    the retry loop with ``restart_with_rebuilt_messages`` armed (``"break"``) so the pre-API
    preflight re-runs against the fallback's context window (#84733)."""
    _provider_overflow_recovery_pending = False

    def _verdict(action: str, result: Optional[Dict[str, Any]] = None) -> ApiErrorVerdict:
        return ApiErrorVerdict(
            action=action, thinking_spinner=thinking_spinner, messages=messages,
            active_system_prompt=active_system_prompt, conversation_history=conversation_history,
            approx_tokens=approx_tokens, retry_count=retry_count, max_retries=max_retries,
            compression_attempts=compression_attempts,
            _provider_overflow_recovery_pending=_provider_overflow_recovery_pending, result=result,
        )

    # 静默停止思考动画 Spinner —— 重试状态会在内部缓冲暂存，
    # 只有当所有的重试与备用模型（fallback）均告耗尽时，才会向终端统一输出诊断状态。
    # Stop spinner silently — retry status is buffered and only flushed when every
    # retry+fallback is exhausted.
    if thinking_spinner:
        thinking_spinner.stop("")
        thinking_spinner = None
    if agent.thinking_callback:
        agent.thinking_callback("")

    _recovered, active_system_prompt = recover_before_classification(
        agent, api_error, messages=messages, api_messages=api_messages, api_kwargs=api_kwargs,
        active_system_prompt=active_system_prompt,
    )
    if _recovered:
        return _verdict("continue")

    status_code = getattr(api_error, "status_code", None)
    error_context = agent._extract_api_error_context(api_error)

    # 进程正在退出（如收到 SIGINT / 解释器销毁）：此时继续重试、轮换凭据或模型回退毫无意义，
    # 且会向终端输出大量的无用回溯堆栈，因此打印单行日志并直接放弃本轮。
    # Process is exiting mid-flight: retries/rotation/fallbacks are futile and the
    # retry trace spams the shell. One log line.
    from tools.interpreter_shutdown import interpreter_shutting_down

    if interpreter_shutting_down(api_error):
        logger.warning(
            "%sInterpreter is shutting down — abandoning turn "
            "during API call #%d (%s)",
            agent.log_prefix, api_call_count, api_error,
        )
        _shutdown_summary = "Turn abandoned: the process was shutting down before the model call could complete."
        return _verdict("return", {
            "final_response": _shutdown_summary, "messages": messages, "api_calls": api_call_count,
            "completed": False, "failed": True, "error": _shutdown_summary,
            "failure_reason": "interpreter_shutdown", "failure_retryable": False,
        })

    _compressor = getattr(agent, "context_compressor", None)
    _ctx_len = getattr(_compressor, "context_length", 200000) if _compressor else 200000
    classified = classify_api_error(
        api_error, provider=getattr(agent, "provider", "") or "",
        model=getattr(agent, "model", "") or "", approx_tokens=approx_tokens,
        context_length=_ctx_len, num_messages=len(api_messages) if api_messages else 0,
        base_url=str(getattr(agent, "base_url", "") or ""),
        api_key=getattr(agent, "api_key", None),
    )
    logger.debug(
        "Error classified: reason=%s status=%s retryable=%s compress=%s rotate=%s fallback=%s",
        classified.reason.value, classified.status_code,
        classified.retryable, classified.should_compress,
        classified.should_rotate_credential, classified.should_fallback,
    )
    agent._invoke_api_request_error_hook(
        task_id=effective_task_id, turn_id=turn_id, api_request_id=api_request_id,
        api_call_count=api_call_count, api_start_time=api_start_time, api_kwargs=api_kwargs,
        error_type=type(api_error).__name__, error_message=str(api_error), status_code=status_code,
        retry_count=retry_count, max_retries=max_retries, retryable=classified.retryable,
        reason=classified.reason.value,
    )

    _recovered, recovered_with_pool = recover_after_classification(
        agent, api_error, classified, _retry, status_code=status_code, error_context=error_context,
        messages=messages, api_messages=api_messages,
    )
    if _recovered:
        return _verdict("continue")

    retry_count += 1
    elapsed_time = time.time() - api_start_time
    # 仅更新看门狗与活跃度标签（Liveness/Watchdog，不会显示在前端对话界面中），
    # 异常分类器的“不可重试”判断会在下方单独的日志行中详细记录。
    # Liveness/watchdog label only (never shown in chat), so the classifier's
    # "not retryable" verdict is named on the logged attempt line below instead.
    agent._touch_activity(f"API error recovery (attempt {retry_count}/{max_retries})")

    error_type, error_msg, _provider, _base, _model = log_api_error_attempt(
        agent, api_error, retry_count=retry_count, max_retries=max_retries, status_code=status_code,
        elapsed_time=elapsed_time, api_messages=api_messages, approx_tokens=approx_tokens,
        retryable=bool(classified.retryable),
    )

    if agent._interrupt_requested:
        # 保留待处理的用户转向（Redirect/Steer）：用户是在修正引导方向而非彻底放弃任务 ——
        # 此时应根据修正指示重新构建当前轮次，而不是直接中止退出。
        # Preserve a pending redirect: the user is steering, not stopping — rebuild the
        # turn from the correction instead of aborting.
        if agent.clear_interrupt(preserve_redirect=True):
            _retry.restart_with_redirected_messages = True
            return _verdict("break")
        return _verdict("return", abort_turn_on_interrupt(
            agent, messages, conversation_history, api_call_count,
            abort_message="Interrupt detected during error handling, aborting retries.",
            interrupt_text=f"Operation interrupted: handling API error ({error_type}: {agent._clean_error_message(str(api_error))}).",
        ))

    _ce = route_classified_error(
        agent, api_error, classified, _retry, error_msg=error_msg, error_context=error_context,
        recovered_with_pool=recovered_with_pool, base_url=_base, model=_model, messages=messages,
        api_messages=api_messages, system_message=system_message,
        active_system_prompt=active_system_prompt, conversation_history=conversation_history,
        retry_count=retry_count, max_retries=max_retries, compression_attempts=compression_attempts,
        max_compression_attempts=max_compression_attempts, api_call_count=api_call_count,
        effective_task_id=effective_task_id,
    )
    status_code = _ce.status_code
    messages = _ce.messages
    active_system_prompt = _ce.active_system_prompt
    conversation_history = _ce.conversation_history
    retry_count = _ce.retry_count
    max_retries = _ce.max_retries
    compression_attempts = _ce.compression_attempts
    is_rate_limited = _ce.is_rate_limited
    _wrapped_output_cap_budget = _ce.wrapped_output_cap_budget
    _is_zai_coding_overload = _ce.is_zai_coding_overload
    if _ce.provider_overflow_recovery_pending:
        _provider_overflow_recovery_pending = True
    if _ce.action != "fallthrough":
        return _verdict(_ce.action, _ce.result)

    _ov = recover_from_overflow(
        agent, api_error, classified, _retry, status_code=status_code, error_msg=error_msg,
        wrapped_output_cap_budget=_wrapped_output_cap_budget, messages=messages,
        api_messages=api_messages, system_message=system_message,
        active_system_prompt=active_system_prompt, conversation_history=conversation_history,
        approx_tokens=approx_tokens, compression_attempts=compression_attempts,
        max_compression_attempts=max_compression_attempts, api_call_count=api_call_count,
        effective_task_id=effective_task_id,
    )
    messages = _ov.messages
    active_system_prompt = _ov.active_system_prompt
    conversation_history = _ov.conversation_history
    approx_tokens = _ov.approx_tokens
    compression_attempts = _ov.compression_attempts
    is_context_length_error = _ov.is_context_length_error
    if _ov.provider_overflow_recovery_pending:
        _provider_overflow_recovery_pending = True
    if _ov.action != "fallthrough":
        return _verdict(_ov.action, _ov.result)

    _ue = settle_unrecovered_error(
        agent, api_error=api_error, classified=classified, _retry=_retry, status_code=status_code,
        error_msg=error_msg, error_context=error_context,
        is_context_length_error=is_context_length_error,
        is_rate_limited=is_rate_limited, _is_zai_coding_overload=_is_zai_coding_overload,
        _provider=_provider, _base=_base, _model=_model, messages=messages,
        api_messages=api_messages, api_kwargs=api_kwargs, active_system_prompt=active_system_prompt,
        conversation_history=conversation_history, approx_tokens=approx_tokens,
        retry_count=retry_count, max_retries=max_retries, compression_attempts=compression_attempts,
        api_call_count=api_call_count,
    )
    active_system_prompt = _ue.active_system_prompt
    retry_count = _ue.retry_count
    compression_attempts = _ue.compression_attempts
    return _verdict(_ue.action, _ue.result)


def _is_local_validation_error(api_error: Any) -> bool:
    """【判断是否为真正的本地代码/参数校验错误】
    通常 ValueError / TypeError 属于本地代码 bug（无需重试），但以下特例属于网络传输或服务端返回的畸变：
    1. UnicodeEncodeError：属于代理对（surrogate）字符转义问题，系统具备自动清洗恢复路径；
    2. json.JSONDecodeError：上游返回了截断的畸变 JSON 数据，属于偶发网络/Provider 故障，必须重试；
    3. ssl.SSLError：在 Python 中继承自 OSError 与 ValueError，TLS 握手中断属于传输层故障而非本地 bug；
    4. "NoneType is not iterable" 的 TypeError：上游响应结构不符合预期（如 Codex 输出为 null），允许重试或触发备用模型。

    ValueError/TypeError are local bugs, except: UnicodeEncodeError (surrogate recovery
    path), json.JSONDecodeError (transient provider/network failure, must retry),
    ssl.SSLError (inherits OSError *and* ValueError — a TLS failure is not a local bug)
    and "NoneType is not iterable" TypeErrors (upstream shape mismatches, e.g. Codex
    response.completed.output=null — retryable so the fallback path runs)."""
    if not isinstance(api_error, (ValueError, TypeError)):
        return False
    if isinstance(api_error, (UnicodeEncodeError, json.JSONDecodeError, ssl.SSLError)):
        return False
    _text = str(api_error).lower()
    return not (isinstance(api_error, TypeError) and "nonetype" in _text and "not iterable" in _text)


@dataclass
class UnrecoveredErrorVerdict:
    """【未自愈异常的决断结果 / Unrecovered Error Verdict】
    封装对所有恢复链路均未拦截的终态决断：
    - ``action``: ``"continue"``（重试当前 API）、``"break"``（已装载 Fallback 备用模型或排队重定向）、``"return"``（返回终态失败字典）；
    - 重新绑定 ``active_system_prompt``、``retry_count`` 和 ``compression_attempts``。

    ``action``: ``"continue"`` (retry), ``"break"`` (fallback armed / redirect pending) or
    ``"return"`` (``result`` is the terminal result dict). Rebinds ``active_system_prompt``,
    ``retry_count`` and ``compression_attempts``."""

    action: str
    active_system_prompt: Any
    retry_count: Any
    compression_attempts: Any
    result: Optional[Dict[str, Any]] = None


def settle_unrecovered_error(
    agent: Any, *, api_error: Any, classified: Any, _retry: Any, status_code: Any, error_msg: Any,
    is_context_length_error: Any, is_rate_limited: Any, _is_zai_coding_overload: Any,
    _provider: Any, _base: Any, _model: Any, messages: Any, api_messages: Any, api_kwargs: Any,
    active_system_prompt: Any, conversation_history: Any, approx_tokens: Any, retry_count: Any,
    max_retries: Any, compression_attempts: Any, api_call_count: Any, error_context: Any = None,
) -> UnrecoveredErrorVerdict:
    """【未恢复异常的终极裁决 / Settle Unrecovered Error】
    当常规恢复链路均无法消化异常时的决策梯队：
    1. 本地校验错误 / 不可重试客户端错误：优先尝试 Copilot 过期凭证自愈（400 转换为换 Token 重试），失败则尝试回退到 Fallback 模型，若无可用模型则返回终端失败结果；
    2. 最大重试次数耗尽（Max-retries exhaustion）：首先尝试重建底层 HTTP 连接池（primary transport recovery）消除 TCP Reset 伪故障；若仍失败则激活 Fallback 模型；最后走自动恢复阶梯（auto_recover_after_exhaustion）；
    3. 否则进入可中断的指数退避等待（interruptible_backoff_sleep），等待期过后重试当前 API。
    注意：HTTP 402（FailoverReason.billing 欠费）被刻意视为不可重试，防止在空账户上白白消耗重试次数（issue #31273）。

    Decide the fate of an API error that every recovery chain declined: local validation /
    non-retryable client errors (Copilot stale-credential self-heal first, then fallback, then a
    terminal result), max-retries exhaustion (primary transport recovery -> fallback -> terminal
    result), else the interruptible error backoff. ``FailoverReason.billing`` (402) is deliberately
    treated as non-retryable (#31273)."""
    from agent.conversation_loop import (
        _arm_fallback_restart, _is_copilot_provider, _is_stale_copilot_credential_error
    )

    def _verdict(action: str, result: Optional[Dict[str, Any]] = None) -> UnrecoveredErrorVerdict:
        return UnrecoveredErrorVerdict(
            action=action, active_system_prompt=active_system_prompt, retry_count=retry_count,
            compression_attempts=compression_attempts, result=result,
        )

    # 欠费错误 ``FailoverReason.billing`` (402) 被刻意排除在重试之外：
    # 凭证池轮换与降级回退均已放弃，继续重试只会无谓消耗空账户。与 401/403 的处理保持一致。
    # ``FailoverReason.billing`` (402) is deliberately NOT excluded: pool rotation and
    # eager fallback already gave up, so retrying only burns paid requests on a depleted
    # balance. Mirrors 401/403.
    is_local_validation_error = _is_local_validation_error(api_error)
    # ``recover_after_classification`` 会在执行图片压缩前设置 ``image_shrink_retry_attempted = True``。
    # 因此，如果带着该标记再次遇到 image-size 超限拒绝，说明已经没有任何可进一步压缩的图片了
    # （超出的其实是文本，或者图片本身无法再缩放）。继续重发完全相同的请求 ``max_retries`` 次毫无意义：
    # 此时应将其视为客户端错误并立即走 Fallback 备用模型降级链（参见 issue #112473）。
    # ``recover_after_classification`` sets ``image_shrink_retry_attempted`` BEFORE it runs the shrink,
    # so an image-size rejection reaching this point with the flag set had nothing left to shrink (the
    # excess is text on a host whose cap is payload-scoped, or an unshrinkable image). Re-sending the
    # byte-identical body ``max_retries`` times changes nothing: treat it as a client error and try
    # the fallback chain now, as the format_error verdict these 400s carried before did (#112473).
    shrink_spent = classified.reason == FailoverReason.image_too_large and bool(
        getattr(_retry, "image_shrink_retry_attempted", False)
    )
    # 对强制推理字段（reasoning_mandatory）被拒也采取同样的防死循环策略：
    # 上一次重试已经移除了该控制参数，若再次因推理字段被拒，说明该路由拒绝了配置的推理控制本身 ——
    # 此时已经没有任何多余参数可以剔除，必须立即进入备用模型 Fallback 降级链，而不是无休止地重试（参见 issue #114460）。
    # Same shape for the reasoning-disable rung: the retry already went out without the disable,
    # so a second reasoning-field rejection means the route refuses the configured reasoning
    # controls themselves — nothing left to drop, so take the fallback chain now instead of
    # replaying the identical request ``max_retries`` times (#114460).
    reasoning_spent = classified.reason == FailoverReason.reasoning_mandatory and bool(
        getattr(_retry, "reasoning_mandatory_retry_attempted", False)
    )
    is_client_error = (
        is_local_validation_error
        or shrink_spent
        or reasoning_spent
        or (
            not classified.retryable
            and not classified.should_compress
            and classified.reason not in RETRYABLE_CLIENT_REASONS
        )
    ) and not is_context_length_error

    if is_client_error:
        # Codex ChatGPT 账户的 Entitlement 400 错误指明了模型名称：在无凭据可轮换的情况下，
        # 该模型代号对于此账户已被判定永久无效，在开始回退前将其标记为不可用（参见 issue #106475）。
        # A Codex ChatGPT-account entitlement 400 names the model: with nothing to rotate the
        # slug is dead for this account, so record it before the fallback walk runs (#106475).
        from agent.fallback_cooldown import _mark_entitlement_rejected_model
        _mark_entitlement_rejected_model(agent, api_error)
        # Copilot 凭据自愈必须在 Fallback 之前执行：Copilot 凭证过期时返回的是 HTTP 400
        # ``model_not_available_for_integrator`` 或 ``model_not_supported``，而非 401。
        # 此时重新换取全新 Token 并重建 Client，在同一个 Provider 上进行一次重试即可自愈。
        # Copilot self-heal BEFORE fallback: a stale credential yields a 400
        # ``model_not_available_for_integrator`` / ``model_not_supported``, not a 401.
        # Fresh token + client rebuild, one retry, SAME provider.
        if (
            _is_copilot_provider(agent)
            and not _retry.copilot_stale_cred_retry_attempted
            and _is_stale_copilot_credential_error(
                status_code, str(getattr(api_error, "message", "") or api_error)
            )
        ):
            _retry.copilot_stale_cred_retry_attempted = True
            if agent._try_recover_stale_copilot_credential():
                agent._buffer_vprint(
                    "🔐 Copilot credential re-exchanged after "
                    "model_not_available 400. Retrying request..."
                )
                retry_count = 0
                return _verdict("continue")
        # ``should_fallback=False`` 标记了确定性的失败，任何备用 Provider 都无法解决它
        # （例如模型本身输出了格式损坏的工具调用 JSON，issue #12770；或 MoA 预设/适配器故障，issue #55933）：
        # 此时跳过 Fallback 级联。未分类的本地 ValueError/TypeError 保留历史上的回退行为；
        # 但被分类器明确判定为不应回退的错误拥有最高优先级。
        # ``should_fallback=False`` marks a deterministic failure no other provider can fix (the
        # model's own malformed tool-call JSON, #12770; MoA preset/adapter faults, #55933): skip
        # the cascade. An UNCLASSIFIED local ValueError/TypeError keeps its historical fallback;
        # a recognised verdict that opts out wins even when the exception is a ValueError subclass.
        _unclassified_local = is_local_validation_error and classified.reason == FailoverReason.unknown
        if classified.should_fallback or _unclassified_local or shrink_spent or reasoning_spent:
            # 仅在真正配置了备用模型链路时才向用户广播提示，避免在没有备用链的情况下假提示“正在尝试备用模型...”后直接崩溃退出。
            # Announce the fallback only when a chain exists, else "trying fallback..." lies
            # before a silent abort.
            if agent._has_pending_fallback():
                _label = _NONRETRYABLE_LABELS.get(classified.reason, f"Non-retryable error (HTTP {status_code})")
                agent._buffer_diagnostic_status(f"⚠️ {_label} — trying fallback...")
            reset_at = error_context.get("reset_at") if isinstance(error_context, dict) else None
            if agent._try_activate_fallback(reason=classified.reason, reset_at=reset_at):
                # 直接返回 ``return _verdict("break")`` 具有关键架构意义：
                # 必须跳出当前 API 重试循环，促使外层 Iteration Prep 针对 Fallback 模型的上下文窗口重新执行前置预检。
                # Direct ``return _verdict("break")`` is load-bearing: the restart handler
                # re-runs the pre-API preflight against the fallback's context window.
                active_system_prompt = _arm_fallback_restart(agent, api_messages, active_system_prompt, _retry)
                retry_count = compression_attempts = 0
                return _verdict("break")
        return _verdict("return", nonretryable_client_error_result(
            agent, api_error, classified, status_code=status_code, api_kwargs=api_kwargs,
            api_messages=api_messages, messages=messages, conversation_history=conversation_history,
            api_call_count=api_call_count, approx_tokens=approx_tokens, provider=_provider,
            base_url=_base, model=_model,
        ))

    if retry_count >= max_retries:
        # 在触发备用模型之前，先尝试为当前 API 调用块重建一次主传输客户端连接池，
        # 用于修复因连接池陈旧或 TCP Reset 导致的暂时性传输故障。
        # Before fallback, rebuild the primary client once per API call block for
        # transient transport errors (stale pool, TCP reset).
        if not _retry.primary_recovery_attempted and agent._try_recover_primary_transport(
            api_error, retry_count=retry_count, max_retries=max_retries,
        ):
            _retry.primary_recovery_attempted = True
            retry_count = 0
            # 开启全新尝试周期：重置 fallback 状态，使后续可能发生的 429 仍能正常触发 fallback_providers
            # Fresh attempt cycle: re-open fallback state so a follow-on 429 can still
            # activate fallback_providers.
            _retry.has_retried_429 = False
            agent._fallback_index = 0
            agent._fallback_activated = False
            return _verdict("continue")
        if agent._has_pending_fallback():
            agent._buffer_diagnostic_status(f"⚠️ Max retries ({max_retries}) exhausted — trying fallback...")
        reset_at = error_context.get("reset_at") if isinstance(error_context, dict) else None
        if agent._try_activate_fallback(reason=classified.reason, reset_at=reset_at):
            # 直接返回 ``return _verdict("break")`` 具有关键架构意义：
            # 必须跳出当前 API 重试循环，促使外层 Iteration Prep 针对 Fallback 模型的上下文窗口重新执行前置预检。
            # Direct ``return _verdict("break")`` is load-bearing: the restart handler
            # re-runs the pre-API preflight against the fallback's context window.
            active_system_prompt = _arm_fallback_restart(agent, api_messages, active_system_prompt, _retry)
            retry_count = compression_attempts = 0
            return _verdict("break")
        # 备用模型回退优先触发（见上方）；只有在没有任何备用模型可用的情况下，
        # 才会进入受控自动恢复阶梯（auto_recover_after_exhaustion），在突发故障期间挂起轮次等待服务恢复，而非直接判死（issue #85426, #107307）。
        # Fallback first (above); only with nothing left to move to does the bounded auto-recovery
        # ladder park the turn on a transient outage instead of ending it (#85426, #107307).
        from agent.turn_recovery_autorecover import auto_recover_after_exhaustion
        _ladder = auto_recover_after_exhaustion(
            agent, api_error, classified, _retry, messages=messages,
            conversation_history=conversation_history, api_call_count=api_call_count,
        )
        if _ladder is not None:
            if _ladder["action"] == "continue":
                retry_count = 0
            return _verdict(_ladder["action"], _ladder.get("result"))
        return _verdict("return", max_retries_exhausted_result(
            agent, api_error, classified, max_retries=max_retries, is_rate_limited=is_rate_limited,
            error_msg=error_msg, api_kwargs=api_kwargs, api_messages=api_messages,
            messages=messages, conversation_history=conversation_history,
            api_call_count=api_call_count, approx_tokens=approx_tokens, provider=_provider,
            base_url=_base, model=_model,
        ))

    wait_time = compute_error_backoff(
        agent, api_error, retry_count=retry_count, max_retries=max_retries,
        is_rate_limited=is_rate_limited, is_zai_coding_overload=_is_zai_coding_overload,
        base_url=_base, model=_model,
    )
    # 与非预期响应等待保持相同的“保留转向”（preserve-redirect）规则：
    # 用户发出的 /steer 转向修正必须在退避等待中得以保留，而绝不能被当做“操作被打断”而丢弃。
    # Same preserve-redirect rule as the invalid-response wait: a steering correction
    # must survive backoff, not die as "Operation interrupted".
    _interrupted = interruptible_backoff_sleep(
        agent, wait_time, _retry, messages=messages, conversation_history=conversation_history,
        api_call_count=api_call_count,
        abort_message="Interrupt detected during retry wait, aborting.",
        interrupt_text=f"Operation interrupted: retrying API call after error (retry {retry_count}/{max_retries}).",
        activity_label=f"error retry backoff ({retry_count}/{max_retries})",
    )
    if _interrupted is not None:
        return _verdict("return", _interrupted)
    if _retry.restart_with_redirected_messages:
        # 跳出 API 重试循环 —— 调用方将根据用户的修正指示重新构建当前迭代请求，而不是重发已过时的请求。
        # Leave the retry loop — the caller rebuilds this iteration from the correction
        # instead of re-firing the stale request.
        return _verdict("break")
    return _verdict("fallthrough")
