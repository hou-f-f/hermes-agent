"""【单次迭代 API 请求装配器 / Per-Iteration API Request Assembly】
对话轮次循环中，每一轮模型交互（Iteration）发起前的请求报文组装核心：
1. 从对话轨迹（Transcript）构建临时的 `api_messages` 副本（绝不污染原始权威历史 `messages`）；
2. 聚合 MoA（Mixture-of-Agents 混合智能体）参考模型的上下文并追加至用户消息末尾；
3. 注入仅在发送时生效的临时预填充消息（Prefills）；
4. 触发 Context Engine（上下文引擎）选择钩子与字符串清洗消毒（Surrogate Sanitization）；
5. 规范化工具调用（Canonicalization），确保字节级绝对确定性以保活前缀缓存；
6. 在所有消息变换完成后，在最后一步构建针对当前请求的 Prompt Cache Plan（提示词缓存断点策略）；
7. 最终评估当前请求的 Token 压力（Request Pressure）。

Per-iteration API request assembly for the conversation turn loop: build ``api_messages``
from the transcript, append MoA context, inject prefills, run the context-engine selection
hook and the send-time sanitizers, canonicalize for bit-perfect cache prefixes, build the
request-local prompt-cache plan LAST (after every transcript mutation), prepare the
persistent-MoA request, then measure request pressure. Nothing here imports
``agent.conversation_loop`` at module level (cycle) — loop-internal helpers resolve lazily
so ``patch("agent.conversation_loop.X")`` sites keep intercepting.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from agent.message_sanitization import _sanitize_messages_surrogates
from agent.usage_anchor import anchored_context_tokens
from agent.prompt_caching import build_prompt_cache_plan, effective_cache_ttl
from agent.turn_context import build_api_messages

logger = logging.getLogger("agent.conversation_loop")


@dataclass
class AssembledRequest:
    """【装配完成的请求数据包 / Assembled Request Payload】
    包含本次迭代发送给 LLM 所需的全部物料副本（例如带缓存断点的 api_messages、格式化后的 tools_for_api）。
    权威原始数据（agent.messages 与 agent.tools）保持纯净不受修饰。

    Always ``action == "fallthrough"``; the fields are the iteration locals the assembly
    produces (``api_messages``/``tools_for_api`` are the decorated request copies — the
    canonical ``messages``/``agent.tools`` stay undecorated)."""

    action: str
    api_messages: Any
    tools_for_api: Any
    _moa_prepared_request: Any
    pending_moa_prepared_request: Any
    approx_tokens: Any
    request_pressure_tokens: Any
    total_chars: Any


def _append_moa_context(agent: Any, api_messages: Any, moa_config: Any, original_user_message: Any) -> None:
    """【并行聚合 MoA 参考模型上下文 / Append MoA Context】
    运行 MoA（Mixture-of-Agents）参考模型集合，并将它们聚合后的多视角推理上下文
    追加至最新一条用户消息末尾（在多模态轮次中作为尾部文本分块追加）。
    设计哲学：故障开放（Fail-open），任何参考模型故障均记录日志并放行，绝不阻断主模型调用。

    Run the MoA reference models and append their aggregated context to the last user
    message (as a trailing text part on multimodal turns). Fail-open."""
    try:
        from agent.message_content import flatten_message_text as _flatten_mt
        from agent.moa_loop import _preset_temperature, aggregate_moa_context

        _moa_context = aggregate_moa_context(
            user_prompt=(
                original_user_message
                if isinstance(original_user_message, str)
                # Multimodal content list: extract visible text rather than
                # str()-ing parts, which would leak base64 image payloads.
                else _flatten_mt(original_user_message)
            ),
            api_messages=api_messages,
            reference_models=moa_config.get("reference_models") or [],
            aggregator=moa_config.get("aggregator") or {},
            temperature=_preset_temperature(moa_config, "reference_temperature"),
            aggregator_temperature=_preset_temperature(moa_config, "aggregator_temperature"),

            # None = no per-preset override; inherit auxiliary.moa_reference.timeout.
            reference_timeout=(
                float(moa_config["reference_timeout"])
                if moa_config.get("reference_timeout")
                else None
            ),
            degraded_reference_policy=str(
                moa_config.get("degraded_reference_policy") or "loud"
            ),
            agent=agent,
        )
        if not _moa_context:
            return
        for _msg in reversed(api_messages):
            if _msg.get("role") == "user":
                _base = _msg.get("content", "")
                if isinstance(_base, str):
                    _msg["content"] = _base + "\n\n" + _moa_context
                elif isinstance(_base, list):
                    _msg["content"] = [*_base, {"type": "text", "text": "\n\n" + _moa_context}]
                break
    except Exception as _moa_exc:
        logger.warning("MoA context aggregation failed: %s", _moa_exc)


def _prepare_moa_request(agent: Any, api_messages: Any, pending_moa_prepared_request: Any) -> tuple:
    """【持久化 MoA 请求重定基底 / Prepare MoA Request】
    若客户端支持，将挂起的已准备 MoA 请求重定基底（rebase）到新消息上；
    否则重新准备全新请求。返回 ``(prepared_request, api_messages, pending_moa_prepared_request)``。

    Persistent-MoA request: rebase the pending prepared request onto the new messages
    when the client supports it, else prepare a fresh one. Returns
    ``(prepared_request, api_messages, pending_moa_prepared_request)``."""
    _moa_completions = getattr(getattr(agent.client, "chat", None), "completions", None)
    prepared: Any = None
    if pending_moa_prepared_request is not None:
        _rebase = getattr(_moa_completions, "rebase_prepared_request", None)
        if callable(_rebase):
            prepared = _rebase(pending_moa_prepared_request, api_messages)
        pending_moa_prepared_request = None
    if prepared is None:
        _prepare = getattr(_moa_completions, "prepare", None)
        if callable(_prepare):
            prepared = _prepare(api_messages)
    if prepared is not None:
        api_messages = prepared["messages"]
    return prepared, api_messages, pending_moa_prepared_request


def assemble_api_request(
    agent: Any, *, messages: Any, current_turn_user_idx: Any, _ext_prefetch_cache: Any,
    _plugin_user_context: Any, moa_config: Any, active_system_prompt: Any,
    original_user_message: Any, pending_moa_prepared_request: Any, request_logger: Any,
) -> AssembledRequest:
    """【装配单次 API 请求完整载荷 / Assemble Full API Request Payload】
    在生命周期 Phase 4 执行，严格按顺序执行清洗与组装。
    ⚠️ 顺序具有绝对结构承重性（ORDER IS LOAD-BEARING）：
    缓存断点（Cache Breakpoints）必须在所有空白归一化（Whitespace Normalization）、
    孤儿工具结果扫描、纯思考过程剔除/用户消息合并、以及乱码字符剥离之后才能注入！
    这样才能保证同一行消息在跨轮次传输中字节级恒定，绝不破坏底层前缀缓存（Prompt Caching）。

    Assemble the request in the original order. ORDER IS LOAD-BEARING: cache breakpoints
    are injected only after whitespace normalization, the orphan sweep, thinking-only drop /
    user merge and surrogate stripping, so the same row's bytes never vary across turns."""
    from agent.conversation_loop import (
        _CODEX_INCOMPLETE_NUDGE, _apply_context_engine_selection, _canonicalize_api_tool_calls,
        _clone_message_for_send, _midturn_request_pressure_tokens, _pressure_with_real_floor,
    )
    from agent.model_metadata import estimate_messages_tokens_rough

    api_messages, effective_system = build_api_messages(
        agent, messages, current_turn_user_idx=current_turn_user_idx,
        ext_prefetch_cache=_ext_prefetch_cache, plugin_user_context=_plugin_user_context,
        moa_config=moa_config, active_system_prompt=active_system_prompt,
    )

    if moa_config:
        _append_moa_context(agent, api_messages, moa_config, original_user_message)

    # 【请求级临时预填消息注入 / Ephemeral Prefills】
    # 仅在发起 API 调用时生效，紧随 System Prompt 之后注入。
    # 必须执行结构深拷贝（Structural Clone），防止后续就地清洗消毒操作穿透破坏原始预填容器。
    # Ephemeral prefill messages go right after the system prompt, API-call-time only.
    if agent.prefill_messages:
        sys_offset = 1 if (api_messages and api_messages[0].get("role") == "system") else 0
        for idx, pfm in enumerate(agent.prefill_messages):
            # Structural clone: the in-place sanitizers below must not write
            # through into agent.prefill_messages' nested containers.
            api_messages.insert(sys_offset + idx, _clone_message_for_send(pfm))

    # 【单轮上下文动态筛选钩子 / Context Engine Selection Hook】
    # 允许外部上下文引擎仅针对本次调用动态挑选或替换上下文（单次请求生效、故障开放，且与 should_compress 相互独立）。
    # Per-turn context selection hook: an engine may select/replace context for THIS
    # call only — request-only, fail-open, and independent of should_compress().
    _sel_incoming = (
        messages[current_turn_user_idx] if 0 <= current_turn_user_idx < len(messages) else None
    )
    api_messages = _apply_context_engine_selection(
        agent, api_messages, messages, _sel_incoming, logger=request_logger
    )

    # 【无条件执行 API 消息清洗消毒】
    # 不依赖 context_compressor 状态，确保在恢复会话或用户手动编辑历史消息后，孤立的工具结果（无对应 tool_call）必然被兜底捕获与清洗。
    # Runs unconditionally (not gated on context_compressor) so orphaned tool
    # results from session loading or manual message edits are always caught.
    api_messages = agent._sanitize_api_messages(api_messages)
    # 【发送路径多模态视觉过期驱逐（Vision Eviction）】
    # 压缩策略仅在剪枝时剔除陈旧截图，且 Anthropic 适配器的保留窗口无法识别 OpenAI 风格的 tool-result image_url。
    # 在此仅对当前调用的请求副本原地驱逐旧图，持久化在磁盘上的真实历史毫发无损（参见 issue #89296）。
    # Send-path vision eviction (#89296): compression only strips stale screenshots
    # when prune fires, and the Anthropic adapter's keep-window never sees
    # OpenAI-style tool-result image_url parts. The per-call clone is rewritten in
    # place; persisted history is untouched.
    from agent.context_compressor import evict_stale_outbound_tool_images

    evict_stale_outbound_tool_images(api_messages)

    # 【自愈提示广播铁律 / Sanitizer Heal Notice】
    # 重复清洗自愈通知仅通过 status/warning 回调向外广播，绝对严禁追加到消息列表中，
    # 确保长会话的前缀缓存（Prompt Caching）字节级绝对一致。
    # One-time repeated-heal notice goes out via the status/warning callback, NEVER
    # appended to messages: the cached prompt prefix stays byte-identical.
    try:
        from agent.agent_runtime_helpers import consume_pending_sanitizer_heal_notice

        _heal_notice = consume_pending_sanitizer_heal_notice()
        if _heal_notice:
            agent._emit_warning(_heal_notice)
    except Exception:
        logger.debug("sanitizer heal notice delivery failed", exc_info=True)

    # 【剔除纯思考回合与相邻用户消息合并 / Drop Thinking & Merge Users】
    # 仅修饰 API 发送副本：Anthropic 等模型服务对以 `thinking` 结尾的请求直接报 400 错误；
    # 同时在非 Codex 协议下剥离 Codex 专用的控制催促文本（issue #67321）。
    # Drop thinking-only assistant turns + merge adjacent users, API copy only:
    # Anthropic-style backends 400 on a trailing `thinking` block; history keeps it.
    # Off the Codex wire (e.g. after a reasoning-only stall fell over to a Chat Completions
    # provider, #67321) the synthetic continuation nudge is Codex-only control text: drop it
    # alongside the opaque replay state.
    _cross_protocol = agent.api_mode != "codex_responses"
    api_messages = agent._drop_thinking_only_and_merge_users(
        api_messages, drop_codex_reasoning_items=_cross_protocol,
        drop_nudge_marker=_CODEX_INCOMPLETE_NUDGE if _cross_protocol else None,
    )

    # 【空白符与工具 JSON 规范化 / Bit-Perfect Canonicalization】
    # 在跨轮次交互中消除首尾多余空白，规整工具参数 JSON，确保前缀缓存字节级命中（本地模型复用 KV 缓存，云端大幅降低 Token 计费）。
    # Normalize whitespace and tool-call JSON for bit-perfect prefixes across turns
    # (KV-cache reuse on local servers, better cloud cache hits); API copy only.
    for am in api_messages:
        if isinstance(am.get("content"), str):
            am["content"] = am["content"].strip()
    _canonicalize_api_tool_calls(api_messages)

    # 【清洗孤立代理对字符 / Strip Lone Surrogates】
    # 某些本地 Ollama 服务模型吐出的孤立代理字符（U+D800~U+DFFF）会导致 OpenAI SDK 内部的 json.dumps() 崩溃并触发 3 次重试，在此提前消毒。
    # Strip lone surrogates (U+D800-U+DFFF) that some Ollama-served models emit;
    # they crash json.dumps() inside the OpenAI SDK and trigger the 3-retry cycle.
    _sanitize_messages_surrogates(api_messages)

    # No send-time pad loop here: ``repair_empty_non_final_messages`` (inside
    # ``_sanitize_api_messages``) is the single owner of empty-turn repair.

    # 【在所有变换的最后构建 Prompt Cache 断点】
    # 缓存标记必须在所有消息原地变换之后再行构建；标准工具注册表保持纯净无装饰。
    # Build the request-local cache sections LAST, after every transcript mutation;
    # the canonical tool registry stays undecorated. Marked ``content`` becomes text
    # blocks the whitespace pass skips, so the same row's bytes vary across turns.
    tools_for_api = agent.tools
    if agent._use_prompt_caching and agent.provider != "moa":
        from agent.prompt_caching import envelope_tool_part_cache_markers_supported

        _static_system_prefix = getattr(agent, "_cached_system_prompt_static", None)
        _initial_cache_plan = build_prompt_cache_plan(
            api_messages,
            tools_for_api,
            # Clamp per-destination: a configured 1h regresses to 5m on
            # Qwen/Alibaba routes, whose context cache is 5m-only.
            cache_ttl=effective_cache_ttl(
                agent._cache_ttl, provider=agent.provider, model=agent.model
            ),
            native_anthropic=agent._use_native_cache_layout,
            static_system_prefix=(
                _static_system_prefix if isinstance(_static_system_prefix, str) else None
            ),
            direct_native_tool_cache=agent._direct_native_anthropic_tool_cache_capability(),
            # LiteLLM-style envelope routes forward part-level markers into
            # tool_result.content[] → non-retryable 400.
            tool_part_markers=envelope_tool_part_cache_markers_supported(
                getattr(agent, "provider", ""), getattr(agent, "base_url", "")
            ),
        )
        api_messages = _initial_cache_plan.messages
        tools_for_api = _initial_cache_plan.tools

    # 【持久化 MoA 请求重定基底】
    # 在计算压缩压力前准备 MoA 请求；顾问模型的临时输出不在 messages 中，create() 复用该请求避免重复运行顾问。
    # Prepare the persistent-MoA request before measuring compression pressure: the
    # ephemeral advisor output is absent from ``messages``; ``create()`` reuses the
    # prepared request instead of running the advisors again.
    _moa_prepared_request = None
    if agent.provider == "moa":
        _moa_prepared_request, api_messages, pending_moa_prepared_request = _prepare_moa_request(
            agent, api_messages, pending_moa_prepared_request
        )

    # 【双重上下文 Token 估算】
    # 剥离图片后的估算喂给两个指标；工具 Schema 独立计数（50+ 工具约占 2~3 万 Token）。
    # One image-stripped estimate feeds both figures; tools counted separately (50+
    # tools ≈ 20-30K tokens); total_chars is a rough proxy for logs/hooks only.
    # Charge stale thinking only when the active route replays it.
    from agent.turn_context import _agent_stale_thinking_on_wire

    if _agent_stale_thinking_on_wire(agent):
        approx_tokens = estimate_messages_tokens_rough(api_messages)
    else:
        approx_tokens = estimate_messages_tokens_rough(api_messages, charge_stale_thinking=False)
    # 【感知路由的上下文压力评估 / Route-Aware Context Pressure】
    # 当请求支持 Native Responses 原生压缩时，传输层在发送前会自动进行检查点剪枝。
    # 若用通用的持久化历史计算会严重高估 Token 数，导致误触发长达 600 秒的不必要本地压缩（参见 issue #96995）。
    # Route-aware: native Responses compaction prunes the wire payload, so the raw
    # history figure overstates it and fires needless local compression.
    # Route-aware pressure: when the upcoming request is eligible for native Responses compaction the
    # transport will checkpoint-prune the payload before sending — the generic durable-history figure
    # overstates the wire by orders of magnitude on a compacted session and fires a 600s local compression
    # the main request never needed (#96995, mirroring the turn-prologue preflight #96644/#96155).
    request_pressure_tokens = _midturn_request_pressure_tokens(
        agent, api_messages, effective_system or "", approx_tokens
    )
    # 【用量锚点优先覆盖 / Usage-Anchored Override】
    # 用真实的 API prompt_tokens（包含 system 和工具定义）+ 当前轮次增量估算，替代全量历史经验启发式估算；
    # 当锚点新鲜时提供最精确的 Token 读数。
    # Usage-anchored override: real prompt_tokens (incl. system + tool schemas) +
    # delta estimate replaces the whole-history heuristic when the anchor is fresh.
    _anchored_pressure = anchored_context_tokens(messages, getattr(agent, "_usage_anchor", None))
    agent._request_pressure_anchored = _anchored_pressure is not None
    if _anchored_pressure is not None:
        request_pressure_tokens = _anchored_pressure
    else:
        # Rough fallback only: floor at the provider's last REAL prompt size (an anchored
        # figure is provider-exact and is never floored — on MoA turns that would re-add
        # the fan-out tokens the anchor excludes).
        request_pressure_tokens = _pressure_with_real_floor(
            agent.context_compressor, request_pressure_tokens
        )
    return AssembledRequest(
        "fallthrough", api_messages, tools_for_api, _moa_prepared_request,
        pending_moa_prepared_request, approx_tokens, request_pressure_tokens, approx_tokens * 4,
    )
