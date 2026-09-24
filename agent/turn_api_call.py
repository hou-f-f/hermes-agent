"""【模型提供商 API 调用执行器 / Model Provider API Call Executor】
对话轮次重试循环中与底层大模型网关直接通信的执行中枢：
1. nous_rate_limit_guard：速率限制守卫，检测到跨会话 Nous Portal 限流时自动挂起规避；
2. perform_api_call：核心调用分发，负责判断是否启用流式传输（Streaming）、
   MoA 预备请求握手、包裹 LLM 执行中间件（Middleware Pipeline）、
   管理 _model_request_active 活跃标志锁以支持双线程实时中断打断（_interruptible_api_call），
   以及裁决“模型返回”与“用户中途纠偏重定向（Redirect）”并发相撞时的冲突解决；
3. handle_api_interrupt：捕获请求中途触发的 InterruptedError 并安全清理临时现场。

本模块严禁在模块顶层导入 agent.conversation_loop（防止循环依赖）。

The provider call for the conversation turn's retry loop: ``nous_rate_limit_guard`` (skip
the attempt while another session's Nous Portal rate limit is active), ``perform_api_call``
(streaming decision, MoA prepared-request handshake, LLM execution middleware wrapper, the
redirect ``_model_request_active`` bracket and the response-vs-redirect crossing check) and
``handle_api_interrupt`` (``InterruptedError`` mid-call). Nothing here imports
``agent.conversation_loop`` at module level (cycle).
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import logging
import time
from typing import Any, Dict, Optional

from agent.error_classifier import FailoverReason
from agent.agent_runtime_helpers import _INTERRUPTED_PLACEHOLDER
from agent.message_metadata import append_message
from agent.repetition_guard import REPETITION_LOOP_INTERRUPTED, is_runaway_repetition
from agent.turn_failure_copy import site_copy, stamp_failure

logger = logging.getLogger("agent.conversation_loop")


def stop_thinking_spinner(agent: Any, thinking_spinner: Any) -> None:
    """【静默停止思考动画并清空回调】
    静默停止思考动画 Spinner 并清空 thinking 回调；返回 None 便于调用方以
    `thinking_spinner = stop_thinking_spinner(agent, thinking_spinner)` 形式就地重新绑定。

    Stop the spinner silently and clear the thinking callback; returns ``None`` so
    callers can rebind ``thinking_spinner = stop_thinking_spinner(agent, thinking_spinner)``."""
    if thinking_spinner:
        thinking_spinner.stop("")
    if agent.thinking_callback:
        agent.thinking_callback("")
    return None


@dataclass
class ApiCallVerdict:
    """【API 调用裁决结果 / API Call Verdict】
    ``action`` 取值说明：
    - ``"fallthrough"``：调用成功，响应数据进入后续校验（check_api_response）与工具执行；
    - ``"break"``：中途被用户纠偏（Redirect）击穿，丢弃过期响应，立即重建请求并重新发起。

    ``action``: ``"fallthrough"`` (``response`` is ready for verification) or ``"break"``
    (a redirect crossed the response — rebuild armed on ``_retry`` or ``interrupted``)."""

    action: str
    response: Any
    thinking_spinner: Any
    interrupted: Any


def _should_stream(agent: Any) -> bool:
    """【流式传输决策判定 / Streaming Decision】
    即使没有监听消费者也优先开启流式（用于陈旧连接检测与读取超时健康保活检查）；
    禁用流式的特例场景：
    1. 显式配置了 `_disable_streaming=True`；
    2. 使用 ACP 协议（`acp://`）或外部进程独立 Provider 配置文件；
    3. 无界面显示监听者的 MoA（混合模型架构）；
    4. 单元测试中的 Mock 客户端（非真实流迭代器）。

    Streaming is preferred even without consumers (stale-stream / read-timeout health
    checks); disabled on provider signal, ACP providers (``acp://`` scheme or an
    external-process provider profile), MoA without a display consumer, or Mock clients in
    tests (SimpleNamespace, not stream iterators)."""
    if getattr(agent, "_disable_streaming", False):
        return False
    _base = str(agent.base_url or "").lower()
    from hermes_cli.runtime_provider_backends import _is_external_process_provider

    if _base.startswith(("acp://", "acp+tcp://")) or _is_external_process_provider(agent.provider):
        return False
    if not agent._has_stream_consumers():
        if agent.provider == "moa":
            return False
        from unittest.mock import Mock
        if isinstance(getattr(agent, "client", None), Mock):
            return False
    return True


def perform_api_call(
    agent: Any, *, api_kwargs: Any, _original_api_kwargs: Any, _llm_middleware_trace: Any,
    _moa_prepared_request: Any, _retry: Any, thinking_spinner: Any, retry_count: Any,
    api_call_count: Any, api_request_id: Any, effective_task_id: Any, turn_id: Any,
    interrupted: Any,
) -> ApiCallVerdict:
    """Issue the request (see ``_should_stream`` for the streaming decision)."""
    response = None

    def _verdict(action: str) -> ApiCallVerdict:
        return ApiCallVerdict(
            action=action, response=response, thinking_spinner=thinking_spinner,
            interrupted=interrupted,
        )

    def _stop_spinner():
        nonlocal thinking_spinner
        thinking_spinner = stop_thinking_spinner(agent, thinking_spinner)

    _use_streaming = _should_stream(agent)

    def _perform_api_call(next_api_kwargs):
        if agent.api_mode == "codex_responses":
            next_api_kwargs = agent._get_transport().preflight_kwargs(
                next_api_kwargs, allow_stream=False, is_github_responses=agent._is_copilot_url(),
                sanitize_harmony_tokens=agent._is_codex_backend(),
            )
        if _use_streaming:
            return agent._interruptible_streaming_api_call(
                next_api_kwargs, on_first_delta=_stop_spinner
            )
        from agent import relay_llm

        return relay_llm.execute(
            next_api_kwargs,
            agent._interruptible_api_call,
            session_id=str(agent.session_id or ""),
            name=str(agent.provider or "provider"),
            model_name=str(agent.model or ""),
            metadata={
                "api_mode": agent.api_mode,
                "api_request_id": api_request_id,
                "call_role": (
                    "delegated"
                    if getattr(agent, "is_subagent", False)
                    else "fallback"
                    if int(getattr(agent, "_fallback_index", 0) or 0) > 0
                    else "primary"
                ),
                "retry_count": retry_count,
            },
            defer_logical_completion=True,
        )

    from hermes_cli.middleware import run_llm_execution_middleware

    # 【并发安全防护机制：nullcontext 与重定向锁】
    # 语法点：Python 的 contextlib.nullcontext() 是一个空上下文管理器（No-op Context Manager）。
    # 当 _redirect_lock 为 None（非并发环境）时，利用 nullcontext() 避免编写冗长混乱的 if/else 上下文判断，
    # 统一使用 `with _bracket:` 安全加锁。
    # 架构意图：_model_request_active 标记当前正在网络 I/O 阻塞中，必须在持锁状态下设置，
    # 严防多线程下外部传入的 redirect() 读到“半切换”状态的脏标记。
    # The ``_model_request_active`` bracket is taken under the redirect lock when one exists,
    # so redirect() can't observe a half-toggled flag.
    _model_request_active = getattr(agent, "_model_request_active", None)
    _redirect_lock = getattr(agent, "_pending_redirect_lock", None)
    _bracket = nullcontext() if _redirect_lock is None else _redirect_lock
    with _bracket:
        if _model_request_active is not None:
            _model_request_active.set()
    try:
        response = run_llm_execution_middleware(
            api_kwargs, _perform_api_call, original_request=_original_api_kwargs,
            task_id=effective_task_id, turn_id=turn_id, api_request_id=api_request_id,
            session_id=agent.session_id or "", platform=agent.platform or "", model=agent.model,
            provider=agent.provider, base_url=agent.base_url, api_mode=agent.api_mode,
            api_call_count=api_call_count, middleware_trace=list(_llm_middleware_trace),
        )
    finally:
        with _bracket:
            if _model_request_active is not None:
                _model_request_active.clear()
            _redirect_crossed_response = (
                bool(agent._pending_redirect) if _redirect_lock is not None
                else agent._has_pending_redirect()
            )
    # 【经典跨线程竞争条件：响应与重定向交叉碰撞（Race Condition）】
    # 痛点剖析：大模型 API 返回完整响应（线程 A）的瞬间，用户可能恰好在聊天端输入了新的纠偏指令（线程 B）。
    # 此时如果直接采纳大模型的旧响应，用户的纠偏就会被遗漏丢失。
    # 解决机制：检测到相撞后，果断丢弃已经过时的旧模型响应（stop_thinking_spinner），
    # 触发 `restart_with_redirected_messages`，将用户纠偏作为最新输入无缝重建并重新发起请求！
    if _redirect_crossed_response:
        # Response and redirect can cross threads: discard the now-stale
        # response and rebuild from the correction rather than lose it.
        thinking_spinner = stop_thinking_spinner(agent, thinking_spinner)
        if agent.clear_interrupt(preserve_redirect=True):
            _retry.restart_with_redirected_messages = True
        else:
            interrupted = True
        return _verdict("break")
    return _verdict("fallthrough")


@dataclass
class ApiInterruptVerdict:
    """【API 中断裁决结果 / API Interrupt Verdict】
    必定为 ``action == "break"``（跳出当前重试循环）：
    要么在 ``_retry`` 上装载了重定向重启（redirect restart），
    要么本轮被设为 ``interrupted`` 并设置了 ``final_response``。

    Always ``action == "break"`` (leave the retry loop): either a redirect restart was
    armed on ``_retry`` or the turn is ``interrupted`` with ``final_response`` set."""

    action: str
    thinking_spinner: Any
    interrupted: Any
    final_response: Any


def handle_api_interrupt(
    agent: Any, *, _retry: Any, thinking_spinner: Any, messages: Any, conversation_history: Any,
    api_start_time: Any, interrupted: Any, final_response: Any,
) -> ApiInterruptVerdict:
    """【处理 API 调用期间触发的中断异常 / Handle InterruptedError】
    调用模型中途捕获到 `InterruptedError` 时的恢复动作：
    - 若存在待处理的用户重定向（redirect），保持纠偏指令在队列中供外层重构使用；
    - 否则保留已接收到的流式局部文本，以便下一轮对话能够记录下该未完成的半截回复（避免彻底失忆）。

    ``InterruptedError`` during the provider call: a pending redirect keeps its correction
    queued for the outer-loop rebuild; otherwise keep any streamed partial text so the next
    turn has a record of the half-finished reply."""
    from agent.conversation_loop import INTERRUPT_WAITING_FOR_MODEL_PREFIX

    thinking_spinner = stop_thinking_spinner(agent, thinking_spinner)
    # redirect() 仅取消本次 HTTP 请求：保持纠偏指示在队列中，清除取消标记，
    # 交由外层循环重建请求。绝不具体化不完整的签名/加密推理项。
    # redirect() cancelled only this request: keep the correction queued, clear the
    # cancellation bit, let the outer loop rebuild. Never materialize incomplete
    # signed/encrypted reasoning items.
    if agent._has_pending_redirect() and agent.clear_interrupt(preserve_redirect=True):
        _retry.restart_with_redirected_messages = True
        return ApiInterruptVerdict("break", thinking_spinner, interrupted, final_response)
    api_elapsed = time.time() - api_start_time
    agent._vprint(f"{agent.log_prefix}⚡ Interrupted during API call.", force=True)
    interrupted = True
    _partial = agent._strip_think_blocks(
        getattr(agent, "_current_streamed_assistant_text", "") or ""
    ).strip()
    if _partial and is_runaway_repetition(_partial):
        # 被打断的消息行会在下一轮作为历史重放；若存在死循环重复的字节，重放会重新引发死循环（issue #112764）。
        # 与重定向占位符采用相同的隐藏形态（display_kind="hidden"）：在对话记录中对人类不可见，
        # 附带中性的 api_content，确保调用前的清洗器不会误对其二次自愈。
        # The interrupted row is replayed next turn; looped bytes there re-seed the loop
        # (#112764). Same hidden shape as the redirect placeholder: nothing visible in the
        # transcript, a neutral api_content so the pre-call sanitizer does not re-heal it.
        append_message(messages, {
            "role": "assistant", "content": "", "display_kind": "hidden",
            "api_content": _INTERRUPTED_PLACEHOLDER,
        })
        final_response = REPETITION_LOOP_INTERRUPTED
    elif _partial:
        append_message(messages, {"role": "assistant", "content": _partial})
        final_response = _partial
    else:
        final_response = f"{INTERRUPT_WAITING_FOR_MODEL_PREFIX}{api_elapsed:.1f}s elapsed)."
    agent._persist_session(messages, conversation_history)
    return ApiInterruptVerdict("break", thinking_spinner, interrupted, final_response)


@dataclass
class NousRateGuardVerdict:
    """【Nous 速率守卫裁决 / Nous Rate Guard Verdict】
    ``action``: ``"fallthrough"``（无活跃限制，正常发起调用）、
    ``"break"``（在 ``_retry`` 上装载备用模型 Fallback 重启）、
    ``"return"``（无可用备用模型，返回失败结果字典）。

    ``action``: ``"fallthrough"`` (no active limit — make the call), ``"break"``
    (fallback armed on ``_retry``) or ``"return"`` (``result``: no fallback available)."""

    action: str
    active_system_prompt: Any
    retry_count: Any
    compression_attempts: Any
    result: Optional[Dict[str, Any]] = None


def nous_rate_limit_guard(
    agent: Any, *, _retry: Any, api_messages: Any, messages: Any, conversation_history: Any,
    active_system_prompt: Any, retry_count: Any, compression_attempts: Any, api_call_count: Any,
) -> NousRateGuardVerdict:
    """【Nous Portal 速率限制防雪崩守卫】
    若另一个会话记录了 Nous Portal 速率限制（429），跳过当前调用尝试：
    因为每一次网络握手（包括 SDK 内部重试）都会扣减每小时请求配额（RPH）。
    守卫内部使用安全保护，绝不让速率守卫本身的异常击穿智能体主循环。

    Skip the call if another session recorded a Nous Portal rate limit: every attempt (incl.
    SDK retries) counts against RPH. Never lets the guard itself break the agent loop."""
    from agent.conversation_loop import _arm_fallback_restart

    def _verdict(action: str, result: Optional[Dict[str, Any]] = None) -> NousRateGuardVerdict:
        return NousRateGuardVerdict(
            action=action, active_system_prompt=active_system_prompt, retry_count=retry_count,
            compression_attempts=compression_attempts, result=result,
        )

    if agent.provider == "nous":
        # A gateway ``x-nous-model-switch`` recorded on the previous response moves this session
        # (and the config default, when it still names the free tier's model) before the next call.
        try:
            from hermes_cli.anon_auth import apply_model_switch
            apply_model_switch(agent)
        except Exception:
            pass
        try:
            from agent.nous_rate_guard import (
                nous_rate_limit_remaining, format_remaining as _fmt_nous_remaining
            )
            from hermes_cli import anon_auth
            _anonymous = anon_auth.is_anonymous_agent(agent)
            _nous_remaining = nous_rate_limit_remaining(anonymous=_anonymous)
            if _nous_remaining is not None and _nous_remaining > 0:
                reset = _fmt_nous_remaining(_nous_remaining)
                if _anonymous:
                    _nous_msg = anon_auth.FREE_TIER_RATE_LIMIT_CHAT.format(
                        reset=anon_auth.friendly_wait(_nous_remaining))
                else:
                    _nous_msg = f"Your Nous account has hit its rate limit; it resets in {reset}."
                agent._buffer_vprint(f"⏳ {_nous_msg} Trying fallback...")
                agent._buffer_diagnostic_status(f"⏳ {_nous_msg}")
                if agent._try_activate_fallback():
                    active_system_prompt = _arm_fallback_restart(
                        agent, api_messages, active_system_prompt, _retry)
                    retry_count = 0
                    compression_attempts = 0
                    return _verdict("break")
                # No fallback — surface the buffered rate-limit context that led here.
                agent._flush_status_buffer()
                agent._persist_session(messages, conversation_history)
                # The free tier's sentence already says what to do (wait, or sign in); the
                # fallback-provider advice is for an install that runs its own providers.
                return _verdict("return", stamp_failure({
                    "final_response": (f"⏳ {_nous_msg}" if _anonymous
                                       else f"⏳ {_nous_msg}\n\n{site_copy('nous_rate_limit')}"),
                    "messages": messages,
                    "api_calls": api_call_count,
                    "completed": False,
                    "failed": True,
                    "error": _nous_msg,
                    # The free tier's card body and its sign-in door (agent/error_surface.py).
                    **({"free_tier": {"kind": "rate_limited", "message": anon_auth.FREE_TIER_RATE_LIMIT_CARD.format(
                        reset=anon_auth.friendly_wait(_nous_remaining))}} if _anonymous else {}),
                }, FailoverReason.rate_limit.value, True))
        except Exception:
            pass  # Never let rate guard break the agent loop
    return _verdict("fallthrough")
