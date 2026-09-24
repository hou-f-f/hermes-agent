"""Response intake for the conversation turn loop: normalize the raw provider response into
the assistant message, splice agent-as-provider projections, fire ``post_api_request``, relay
reasoning to the progress callback, and apply the incomplete-scratchpad / Codex-incomplete
continuation guards. Nothing here imports ``agent.conversation_loop`` at module level (cycle).

【阶段 5：模型响应摄入与规范化 / Phase 5: Response Intake & Normalization】
本模块负责单轮循环中模型响应返回后的首道处理屏障：
1. 规范化响应：将各种 Provider 原始响应对象统一转换为 ``assistant_message``（规整 text 内容为字符串，处理 dict/list 畸形返回）；
2. Agent-as-Provider 映射拼接（``splice_provider_projection``）：若当前 Provider 底层由另一个 Agent 代理，则将其执行的工具过程拼接为前置 call/result 记录；
3. 触发生命周期钩子：触发 ``post_api_request``，上报网络耗时、TTFB、Token 用量与 MoA 指标；
4. 深度思考推理进度中继（``_relay_thinking``）：剥离 ``<think>`` 标签，向父 Agent 或 TUI 推送 ``reasoning.available`` 事件；
5. 不完整 Scratchpad 与 Codex 截断续问守卫：检测未闭合的 ``<REASONING_SCRATCHPAD>``（Token 耗尽导致半截思考）进行上限 2 次的重试；处理 Codex Responses API 下的 ``finish_reason == "incomplete"`` 并触发 fallback 回退。
注意：本模块禁止在模块级导入 ``agent.conversation_loop``，避免循环导入。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from typing import Any, Dict, Optional

from agent.provider_projection import splice_provider_projection
from agent.trajectory import has_incomplete_scratchpad
from agent.turn_truncation import (
    CODEX_FALLBACK_ACTIVATED, continue_codex_incomplete, normalize_response_for_agent, partial_result,
)

logger = logging.getLogger("agent.conversation_loop")

_REASONING_TAG_RE = re.compile(r'</?(?:REASONING_SCRATCHPAD|think|reasoning)>')


@dataclass
class ResponseIntakeVerdict:
    """【响应摄入决策结果 / Response Intake Verdict】
    封装响应摄入处理后的分支决策流向：
    - ``action == "fallthrough"``：响应完全正常，顺延流转至后续工具调用解析或直接结束；
    - ``action == "continue"``：需要立即重试当前轮次（如未闭合的推理草稿或 Codex 续问），不保存损坏消息；
    - ``action == "return"``：轮次提前终止并返回终态字典 ``result``（如重试 2 次仍截断时保存 partial 结果）；
    - ``assistant_message`` / ``finish_reason``：经统一规范化后的输出对象；
    - ``active_system_prompt``：在 Codex reasoning-only 降级重写后重新绑定的系统提示词（issue #67321）。

    ``action``: ``"fallthrough"`` (process ``assistant_message``), ``"continue"`` (retry the
    iteration: incomplete scratchpad / Codex continuation) or ``"return"`` (``result`` is the
    turn's result dict). ``assistant_message``/``finish_reason`` are the normalized outputs;
    ``active_system_prompt`` is rebound after a Codex reasoning-only fallover (#67321)."""

    action: str
    assistant_message: Any
    finish_reason: Any
    result: Optional[Dict[str, Any]] = None
    active_system_prompt: Any = None


def _coerce_content_text(raw: Any) -> str:
    """【强制将多态响应内容规整为纯文本字符串】
    部分兼容 OpenAI 接口的第三方服务（如 llama-server、某些本地推理引擎）可能将 content 字段
    序列化为 dict 或多模态 list，这会导致下游字符串处理（如 ``.strip()``）直接触发 AttributeError。
    本函数将多模态 list 递归提取 text 块并换行拼接，将 dict 转为 JSON 或提取 text/content，确保返回纯 str。

    Some OpenAI-compatible servers (llama-server) return content as dict/list, which
    crashes downstream ``.strip()``; normalize to str (multimodal lists → text parts)."""
    if isinstance(raw, dict):
        return raw.get("text", "") or raw.get("content", "") or json.dumps(raw)
    if isinstance(raw, list):
        parts = []
        for part in raw:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, dict) and "text" in part:
                parts.append(str(part["text"]))
        return "\n".join(parts)
    return str(raw)


def _fire_post_api_request_hook(
    agent: Any, response: Any, assistant_message: Any, finish_reason: Any, *, api_messages: Any,
    api_call_count: Any, api_duration: Any, api_start_time: Any, api_request_id: Any,
    effective_task_id: Any, turn_id: Any,
) -> None:
    from agent.conversation_loop import _moa_reference_metrics_for_hook

    try:
        from hermes_cli.lifecycle import has_hook, invoke_hook as _invoke_hook
        if has_hook("post_api_request"):
            _invoke_hook(
                "post_api_request",
                task_id=effective_task_id,
                turn_id=turn_id,
                api_request_id=api_request_id,
                session_id=agent.session_id or "",
                platform=agent.platform or "",
                model=agent.model,
                provider=agent.provider,
                base_url=agent.base_url,
                api_mode=agent.api_mode,
                api_call_count=api_call_count,
                api_duration=api_duration,
                started_at=api_start_time,
                ended_at=api_start_time + api_duration,
                # 首字到达时间戳（Epoch 秒）；若未开启流式或尚未收到首包则为 None。
                # 首字耗时 TTFB (Time To First Byte) = first_chunk_at - started_at。
                # First stream chunk time (epoch s); None if not streamed / no chunk.
                # TTFB = first_chunk_at - started_at.
                first_chunk_at=getattr(agent, "_last_api_first_chunk_at", None),
                finish_reason=finish_reason,
                message_count=len(api_messages),
                response_model=getattr(response, "model", None),
                response=agent._api_response_payload_for_hook(
                    response, assistant_message, finish_reason=finish_reason
                ),
                usage=agent._usage_summary_for_api_request_hook(response),
                assistant_message=assistant_message,
                assistant_content_chars=len(assistant_message.content or ""),
                assistant_tool_call_count=len(getattr(assistant_message, "tool_calls", None) or []),
                moa_references=_moa_reference_metrics_for_hook(agent),
            )
    except Exception:
        pass


def _relay_thinking(agent: Any, content: str) -> None:
    """【中继模型的深度思考与推理过程】
    将模型响应中的思考内容提取后广播给进度回调：
    - 若当前处于子 Agent 代理执行环境中（`_delegate_depth > 0`），仅提取思考首行（前 80 字符）显示在父界面；
    - 若配置了结构化回调，则发送 ``reasoning.available`` 事件，截取前 500 字符展示，保障实时交互体验。

    Relay the model's text to the progress callback: subagents send the first line to
    the parent display; any agent with a structured callback gets ``reasoning.available``."""
    _think_text = _REASONING_TAG_RE.sub('', content.strip()).strip()
    first_line = _think_text.split('\n')[0][:80] if _think_text else ""
    if first_line and getattr(agent, '_delegate_depth', 0) > 0:
        try:
            agent.tool_progress_callback("_thinking", first_line)
        except Exception:
            pass
    elif _think_text:
        try:
            agent.tool_progress_callback("reasoning.available", "_thinking", _think_text[:500], None)
        except Exception:
            pass


def normalize_model_response(
    agent: Any, *, response: Any, messages: Any, api_messages: Any, conversation_history: Any,
    api_call_count: Any, api_duration: Any, api_start_time: Any, api_request_id: Any,
    effective_task_id: Any, turn_id: Any, active_system_prompt: Any = None,
) -> ResponseIntakeVerdict:
    """【规范化模型响应并执行摄入守卫 / Normalize Response & Execute Guards】
    将底层 API 返回的原始 ``response`` 转换为结构化的 ``assistant_message``（确保 content 必为 str），
    并严格按照原逻辑顺序执行：
    1. 规整 content 格式；
    2. 拼接代理 Agent 工具调用行（Agent-as-provider projection）；
    3. 触发 post_api_request 插件钩子；
    4. 终端打日志与中继思考进度；
    5. 检测并处理未闭合的推理草稿 `<REASONING_SCRATCHPAD>`；
    6. 处理 Codex 截断续问及降级重试。

    Normalize ``response`` into ``assistant_message`` (str content, never dict/list) and run
    the post-response hooks and continuation guards, in the original order."""
    assistant_message = normalize_response_for_agent(agent, response)
    finish_reason = assistant_message.finish_reason

    def _verdict(action: str, result: Optional[Dict[str, Any]] = None) -> ResponseIntakeVerdict:
        return ResponseIntakeVerdict(
            action=action, assistant_message=assistant_message, finish_reason=finish_reason,
            result=result, active_system_prompt=active_system_prompt,
        )

    if assistant_message.content is not None and not isinstance(assistant_message.content, str):
        assistant_message.content = _coerce_content_text(assistant_message.content)

    # 【Agent-as-provider 投影机制】
    # 若当前模型服务底层其实由另一个 Agent 托管执行，将其执行过程中的工具调用与结果
    # 作为前置 call/result 插入本轮 assistant_message 之前；普通 Provider 为空操作。
    # Agent-as-provider projection: splice the provider-agent's own tool work in as
    # call/result rows before this turn's assistant message; no-op for ordinary providers.
    splice_provider_projection(agent, response, messages)

    _fire_post_api_request_hook(
        agent, response, assistant_message, finish_reason, api_messages=api_messages,
        api_call_count=api_call_count, api_duration=api_duration, api_start_time=api_start_time,
        api_request_id=api_request_id, effective_task_id=effective_task_id, turn_id=turn_id,
    )

    content = assistant_message.content
    if content and not agent.quiet_mode:
        if agent.verbose_logging:
            agent._vprint(f"{agent.log_prefix}🤖 Assistant: {content}")
        else:
            agent._vprint(f"{agent.log_prefix}🤖 Assistant: {content[:100]}{'...' if len(content) > 100 else ''}")
    if content and agent.tool_progress_callback:
        _relay_thinking(agent, content)

    # 【未闭合的推理草稿标签守卫 / Incomplete Scratchpad Guard】
    # 检测到 `<REASONING_SCRATCHPAD>`（已开启但未闭合）：说明模型在输出思考过程时耗尽了 max_tokens。
    # 策略：最多重试 2 次；若连续 2 次仍截断，则放弃并作为局部不完整结果（partial）保存，防止死循环。
    # Incomplete <REASONING_SCRATCHPAD> (opened, never closed): the model ran out of
    # output tokens mid-reasoning — retry up to 2 times, then save as partial.
    if has_incomplete_scratchpad(content or ""):
        agent._incomplete_scratchpad_retries += 1
        agent._buffer_vprint("⚠️  Incomplete <REASONING_SCRATCHPAD> detected (opened but never closed)")
        if agent._incomplete_scratchpad_retries <= 2:
            agent._buffer_vprint(f"🔄 Retrying API call ({agent._incomplete_scratchpad_retries}/2)...")
            return _verdict("continue")  # don't add the broken message
        agent._flush_status_buffer()
        agent._vprint(f"{agent.log_prefix}❌ Max retries (2) for incomplete scratchpad. Saving as partial.", force=True, diagnostic=True)
        agent._incomplete_scratchpad_retries = 0
        rolled_back_messages = agent._get_messages_up_to_last_assistant(messages)
        agent._cleanup_task_resources(effective_task_id)
        agent._persist_session(messages, conversation_history)
        return _verdict("return", partial_result(
            rolled_back_messages, api_call_count, "Incomplete REASONING_SCRATCHPAD after 2 retries"
        ))
    agent._incomplete_scratchpad_retries = 0

    if agent.api_mode == "codex_responses" and finish_reason == "incomplete":
        _codex_result = continue_codex_incomplete(
            agent, assistant_message, finish_reason, messages=messages,
            conversation_history=conversation_history, api_call_count=api_call_count,
            response=response,
        )
        if _codex_result is CODEX_FALLBACK_ACTIVATED:
            # 【Codex 故障转移提示词重构同步】
            # 故障转移（Failover）重写了缓存系统提示词中的 Model:/Provider: 身份标识；
            # 必须在此重新绑定 active_system_prompt，确保下一轮迭代根据新身份构建请求。
            # The failover rewrote the Model:/Provider: identity on the cached system prompt;
            # rebind it so the next iteration's request is rebuilt with the new identity.
            from agent.conversation_loop import _sync_failover_system_message
            active_system_prompt = _sync_failover_system_message(agent, api_messages, active_system_prompt)
            return _verdict("continue")
        if _codex_result is not None:
            return _verdict("return", _codex_result)
        return _verdict("continue")
    if hasattr(agent, "_codex_incomplete_retries"):
        agent._codex_incomplete_retries = 0
        agent._codex_reasoning_only_streak = 0
    return _verdict("fallthrough")
