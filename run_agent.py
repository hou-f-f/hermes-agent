#!/usr/bin/env python3
"""AIAgent: the tool-calling agent runner (conversation loop, tool execution, session lifecycle).

【模块概述 / Architecture Overview】
AIAgent 是整个 Hermes Agent 的核心执行引擎门面（Facade）。
在 2026 年 9 月的代码重构中，原先 1.5 万行以上的单体“上帝文件”被拆分为：
1. 门面类（Facade）：即本文件 `run_agent.py`，对外暴露统一的 `AIAgent` 类和 CLI 入口；
2. 会话循环实现：`agent/conversation_loop.py`，负责单轮/多轮 Turn 的核心驱动；
3. Turn 阶段子模块：`agent/turn_*.py`（包括 prep 准备、API 调用、错误处理、上下文压缩、异常恢复等）；
4. 业务 Mixin：拆分在 `agent/` 下的十多个 Mixin 类（如网络生命周期、流式输出、会话持久化等）；
5. 初始化逻辑：`agent/agent_init.py`。

本文件保留所有公共 API、CLI 入口、以及会话生命周期与系统级清理动作。

    from run_agent import AIAgent
    agent = AIAgent(base_url="http://localhost:30000/v1", model="claude-opus-4-20250514")
    response = agent.run_conversation("Tell me about the latest Python updates")
"""

# hermes_bootstrap 必须是首个导入项（在 Windows 上配置 UTF-8 标准输入输出；在 POSIX 系统上为无操作 no-op）。
# hermes_bootstrap must be the very first import (UTF-8 stdio on Windows; no-op on POSIX).
try:
    import hermes_bootstrap  # noqa: F401
except ModuleNotFoundError:
    pass  # partial `hermes update` — only skips the Windows UTF-8 stdio setup

import json
import logging
logger = logging.getLogger(__name__)
import os
import re
import sys
import time
import threading
import uuid
import warnings
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from pathlib import Path

from hermes_constants import get_hermes_home


def _launch_cwd_for_session(source: str) -> Optional[str]:
    """cwd to stamp on a new session row (``hermes -c`` / ``--resume``), or None.

    【设计原理 / Design Intent】
    为新会话记录初始工作目录（cwd），供后续 `hermes -c` 或 `--resume` 恢复会话时还原上下文环境。
    注意：仅本地 CLI 会话（source == 'cli' 且 TERMINAL_ENV 为 local）会记录 cwd。
    网关（Gateway）、定时任务（Cron）或 Docker/SSH/Modal 等远程执行环境（non-local TERMINAL_ENV）
    对宿主机没有稳定的本地工作目录，因此返回 None。
    """
    if source not in CLI_FAMILY_SOURCES or (os.environ.get("TERMINAL_ENV") or "local").strip().lower() not in ("", "local"):
        return None
    try:
        return os.getcwd()
    except OSError:  # cwd was unlinked out from under us (当前工作目录可能被外部进程删除)
        return None


# 【交互式 UI 会话源标记 / Interactive UI Transport Sources】
# 标记由交互式 UI 传输层托管的人类会话来源（TUI 终端界面或 Electron 桌面端）。
# 当在这些会话中启动一次性单次执行命令（如 `hermes chat -q`）或派生子进程时，
# 子进程会通过 terminal 工具继承父会话的环境变量 HERMES_SESSION_SOURCE，
# 但这些子会话并不是主对话本身：如果不加以区分并打上 `tui`/`desktop` 标签，
# 它们就会错误地出现在 TUI/WebUI 的可恢复会话列表（Resumable Chat Pickers）中，
# 导致用户误点并继续该一次性会话（参见 issue #112550）。
# 相反，自动化来源（kanban 看板、tool 工具调用、cron 定时任务、a2a 等）是故意允许继承的。
_UI_TRANSPORT_SOURCES = frozenset({"tui", "desktop"})

# Finite non-interactive CLI runs (``hermes chat -q``/``--oneshot``, ``hermes -z``) get their own source so human
# pickers hide them without title/cwd heuristics; ``hermes -c`` still treats them as CLI history.
ONESHOT_SOURCE = "oneshot"
CLI_FAMILY_SOURCES = frozenset({"cli", ONESHOT_SOURCE})


def _session_source_for_agent(platform: Optional[str]) -> str:
    """获取当前 Agent 的有效会话来源（Session Source）。
    
    综合读取环境变量 `HERMES_SESSION_SOURCE`、传入的 `platform`，兜底为 'cli'。
    如果检测到是 UI 传输来源但属于单次查询（HERMES_SINGLE_QUERY_SESSION=1）且未显式指定来源，
    则清空 source 以免污染会话选择列表。
    """
    try:
        from gateway.session_context import get_session_env
    except Exception:
        get_session_env = os.environ.get
    source = str(get_session_env("HERMES_SESSION_SOURCE", "") or "").strip()
    single_query = get_session_env("HERMES_SINGLE_QUERY_SESSION", "") == "1"
    explicit = get_session_env("HERMES_SESSION_SOURCE_EXPLICIT", "") == "1"
    if single_query and not explicit and source in _UI_TRANSPORT_SOURCES:
        source = ""
    if single_query and not source and (platform or "cli") == "cli":
        return ONESHOT_SOURCE
    return source or platform or "cli"


def _gateway_origin_json(agent: "AIAgent") -> Optional[str]:
    """Gateway routing ``origin_json`` for a session row; None when the agent carries no gateway identity.

    Mirrors ``SessionSource.to_dict()`` so state.db consumers see the same fields ``record_gateway_session_peer`` writes.

    【设计原理 / Design Intent】
    将网关路由身份信息打包为 JSON 字符串，写入 SQLite state.db 会话表中的 `origin_json` 字段。
    记录的信息包括：接入平台（Telegram、Discord、Slack 等）、chat_id、chat_name、
    chat_type（dm/群组）、user_id、user_name、thread_id 以及 Profile 配置名称。
    这样无论是通过 Web 仪表盘查看、还是恢复已中断的网关连接，都能精准回溯该会话的真实来源。
    """
    chat_id = getattr(agent, "_chat_id", None)
    session_key = getattr(agent, "_gateway_session_key", None)
    user_id = getattr(agent, "_user_id", None)
    if not (chat_id or session_key or user_id):
        return None
    origin: Dict[str, Any] = {
        "platform": getattr(agent, "platform", None) or "", "chat_id": chat_id,
        "chat_name": getattr(agent, "_chat_name", None), "chat_type": getattr(agent, "_chat_type", None) or "dm",
        "user_id": user_id, "user_name": getattr(agent, "_user_name", None), "thread_id": getattr(agent, "_thread_id", None),
    }
    if getattr(agent, "_user_id_alt", None):
        origin["user_id_alt"] = agent._user_id_alt
    profile = getattr(agent, "_profile_name", None)
    if not profile:
        try:
            from hermes_cli.profiles import get_active_profile_name
            profile = get_active_profile_name()
        except Exception:
            profile = None
        if profile == "default":
            profile = None
    if profile:
        origin["profile"] = profile
    try:
        return json.dumps(origin)
    except Exception:
        return None


from agent.iteration_budget import IterationBudget
from hermes_cli.env_loader import load_hermes_dotenv
from hermes_cli.timeouts import get_provider_request_timeout, get_provider_stale_timeout

_hermes_home = get_hermes_home()  # read by agent_init via _ra()._hermes_home
_loaded_env_paths = load_hermes_dotenv(hermes_home=_hermes_home, project_env=Path(__file__).parent / '.env')
for _env_path in _loaded_env_paths:
    logger.info("Loaded environment variables from %s", _env_path)
if not _loaded_env_paths:
    logger.info("No .env file found. Using system environment variables.")


from model_tools import get_toolset_for_tool
from tools.terminal_tool_lifecycle import cleanup_vm, get_active_env
from tools.interrupt import set_interrupt as _set_interrupt
from tools.browser_tool_lifecycle import cleanup_browser
from tools.connectors.turn import agent_connection_surface, scoped_connection_surface

# 【AIAgent 的能力拼装：14 个核心功能 Mixin / Core Functionality Mixins】
# Hermes Agent 采用 Facade + Mixin 架构，将复杂职能模块化解耦：
from agent.memory_provider import is_trivial_prompt
from agent.client_lifecycle import ClientLifecycleMixin       # HTTP 客户端生命周期管理（OpenAI/Anthropic 客户端连接池、复用与关闭）
from agent.stream_delivery import StreamDeliveryMixin           # 响应流式分发（解析 SSE 数据流，回调推送思考过程与文本增量）
from agent.status_output import StatusOutputMixin               # 状态输出与终端打印（支持 verbose 详细模式与静默模式）
from agent.api_request_hooks import ApiRequestHooksMixin        # API 请求插件拦截钩子（请求前后触发自定义 Hook）
from agent.api_error_summary import PROVIDER_STREAM_PARSE_MARKERS, ApiErrorSummaryMixin # API 异常诊断与错误分类（统一各大 Provider 错误格式）
from agent.interrupt_control import InterruptControlMixin       # 中断控制（支持用户 /stop、新消息插入或 SIGINT 时安全取消当前 turn）
from agent.turn_explainers import TurnExplainersMixin           # 交互解释器（生成向用户展示的重试、回退、压缩提示）
from agent.activity_tracking import ActivityTrackingMixin       # 会话活跃度跟踪（记录最新动作来源与时间戳，供网关检测僵死）
from agent.rate_limit_credits import RateLimitCreditsMixin      # 速率限制与配额账户追踪（429 熔断、Token 扣费监控）
from agent.session_persistence import SessionPersistenceMixin   # 会话持久化混合类（消息增量落盘与 SQLite 同步）
from agent.compression_facade import CompressionFacadeMixin     # 上下文压缩门面（触发 Micro-compaction、Native Compaction 等）
from agent.turn_facade import TurnFacadeMixin                   # Turn 调度门面（管理跨进程会话排他租约 Turn Lease 和看门狗）
from agent.vision_message_prep import VisionMessagePrepMixin    # 多模态视觉预处理（图片格式化、Base64 转换与尺寸缩放）
from agent.reasoning_params import ReasoningParamsMixin         # 深度思考参数管理（配置 Extended Thinking / Reasoning Effort）
from agent.lazy_forward import forward as _forward, forward_static as _forward_static
from agent.session_activity import ActivityProvenance
from agent.model_metadata import is_local_endpoint
from agent.message_sanitization import (
    coalesce_tool_call_id as _sanitize_coalesce_tool_call_id,
    deterministic_call_id as _codex_deterministic_call_id,
    uniquify_tool_call_ids as _sanitize_uniquify_tool_call_ids,
)
from agent.codex_responses_adapter import (
    _derive_responses_function_call_id as _codex_derive_responses_function_call_id,
    _split_responses_tool_id as _codex_split_responses_tool_id,
    _summarize_user_message_for_log,
)
from agent.tool_guardrails import ToolGuardrailDecision, append_toolguard_guidance, toolguard_synthetic_result
from utils import base_url_host_matches, base_url_hostname, env_float, model_forces_max_completion_tokens


_MAX_TOOL_WORKERS = 8  # 并发执行独立工具调用时的线程池最大并发数


# 每个进程仅启动一次 OpenRouter 端点预热线程，避免多 AIAgent 实例时在网关中发生线程泄漏
# Spawn the OpenRouter pre-warm thread once per process, not per AIAgent (gateway thread leak).
_openrouter_prewarm_done = threading.Event()


def _quietly(fn: Callable, *args, **kwargs) -> None:
    """Run one teardown step, swallowing any exception so sibling steps still run.
    
    【静默清理助手 / Teardown Helper】
    执行一个清理步骤并捕获所有异常。在 Agent 退出或资源释放流程中，确保单个子系统
    （如释放连接池或通知 DB）的崩溃绝不会阻断其他同级资源（如浏览器、Docker 虚拟机）的清理。
    """
    try:
        fn(*args, **kwargs)
    except Exception:
        pass


def _call_engine_hook(engine: Any, hook: str, *args, **kwargs) -> None:
    """Invoke an optional context-engine lifecycle hook; failures are logged, never raised.
    
    【上下文引擎钩子安全调用】
    调用 context_engine（上下文压缩引擎或插件引擎）的可选生命周期方法。失败只记录调试日志，不向上抛出。
    """
    if not hasattr(engine, hook):
        return
    try:
        getattr(engine, hook)(*args, **kwargs)
    except Exception as exc:
        logger.debug("context engine %s during transition: %s", hook, exc)


def _positive_int(value: Any) -> Optional[int]:
    """若 ``value`` 是真正的正整数（排除布尔值），则返回该整数，否则返回 None。"""
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _review_should_defer(agent: Any, task_cfg: Optional[Dict[str, Any]]) -> bool:
    """True when an automatic background review targets the managed local runtime under ``defer: auto``.
    
    【后台反思推迟判断 / Background Review Deferral】
    当 Hermes 在本地运行（使用托管的本地 llama-server）且配置了 `defer: auto` 时，
    自动后台审查（Memory/Skill Review）应当推迟到系统空闲时进行，
    避免在用户正进行交互式对话时强行占用本地 GPU 显存和推理算力导致卡顿。
    """
    from agent.review_idle_queue import defer_mode, review_targets_managed_local
    return defer_mode(task_cfg) == "auto" and review_targets_managed_local(agent, task_cfg)


def _review_queue_key(agent: Any) -> str:
    """获取审查空闲队列的键（优先使用 session_id，回退到 agent 内存 id）。"""
    return str(getattr(agent, "session_id", None) or id(agent))


def _notify_context_engine_session_end(agent: Any, messages: Optional[list]) -> None:
    """Tell the context engine the session ended (flush DAG, close DBs) at the same lifecycle moment as the
    memory manager, so per-session engine state never leaks into the next session.
    
    【通知上下文引擎会话结束】
    在会话结束或重置时通知压缩引擎（如 ContextCompressor），刷新有向无环图（DAG）并关闭底层 DB，
    防止当前会话的上下文记忆泄漏到下一个独立的会话中。
    """
    engine = getattr(agent, "context_compressor", None)
    if engine:
        _quietly(lambda: engine.on_session_end(agent.session_id or "", messages or []))


def _pool_may_recover_from_rate_limit(pool) -> bool:
    """Wait for credential-pool rotation (True) or fall back to ``fallback_model`` (False) after a 429.

    Rotation only helps when the pool has somewhere to go; a single-credential pool would retry the same quota.

    See issues #11314 and #13636.

    【429 速率限制恢复策略判断】
    当收到 429 错误时，判断是否可以通过 API Key 轮换恢复：
    - 如果配置了 CredentialPool（凭证池）且池中有多个可用凭据，返回 True，触发换 Key 重试；
    - 如果只有一个凭据，换 Key 毫无意义（只会继续打到同一配额），此时返回 False，直接触发回退到备用模型（fallback_model）。
    """
    return pool is not None and pool.has_available() and len(pool.entries()) > 1


class _StreamErrorEvent(Exception):
    """Provider error synthesized from a standalone Responses ``type=error`` SSE frame (Codex-style backends).

    Gives ``_summarize_api_error`` / the entitlement detector the familiar ``.body`` / ``.status_code`` shape.

    【Codex 流式错误事件封装】
    当使用 Codex Responses API 时，服务端在流式连接中返回单独的 `type=error` SSE 帧。
    本异常类将其包装为具备标准 `.body` 和 `.status_code` 属性的异常对象，
    使得下层的通用错误分类器与配额检测器能以统一的 OpenAI SDK 格式解析处理。
    """

    def __init__(self, message: str, *, code: Optional[str] = None, param: Optional[str] = None,
                 status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.message, self.code, self.param, self.status_code = message, code, param, status_code
        # 构造符合 OpenAI SDK 规范的错误 body，供 _summarize_api_error 和 classify_api_error 识别
        self.body: Dict[str, Any] = {"error": {"message": message, "code": code, "param": param, "type": "error"}}


class AIAgent(
    ClientLifecycleMixin, StreamDeliveryMixin, StatusOutputMixin, ApiRequestHooksMixin, ApiErrorSummaryMixin,
    InterruptControlMixin, TurnExplainersMixin, ActivityTrackingMixin, RateLimitCreditsMixin,
    SessionPersistenceMixin, CompressionFacadeMixin, TurnFacadeMixin, VisionMessagePrepMixin, ReasoningParamsMixin,
):
    """AI Agent with tool calling capabilities.
    
    【AIAgent 核心类 / Central Orchestration Facade】
    Hermes Agent 的核心智能体门面类。汇聚了以下职能：
    1. 提示词（Prompt）与工具 Schema 动态构建（通过 prompt_builder）；
    2. 多 Provider / API 模式选择（Chat Completions、Codex Responses、Anthropic Messages）；
    3. 支持异步中断的可取消模型调用与工具执行（并发 ThreadPoolExecutor 或串行）；
    4. 对话历史状态机维护、自动上下文微压缩（Micro-compaction）与跨模型回退（Fallback）；
    5. 跨父子 Agent 的迭代预算追踪（Iteration Budget）；
    6. 会话持久化与状态恢复（SQLite SessionDB 与外部向量记忆管理）。
    """

    # 工具调用参数损坏时的安全兜底标记（当流式传输损坏导致 JSON 无法解析时，丢弃该参数以保持对话不崩溃，见 issue #15236）
    _TOOL_CALL_ARGUMENTS_CORRUPTION_MARKER = (
        "[hermes-agent: tool call arguments were corrupted in this session and "
        "have been dropped to keep the conversation alive. See issue #15236.]"
    )

    @property
    def base_url(self) -> str:
        return self._base_url

    @base_url.setter
    def base_url(self, value: str) -> None:
        """设置 API base_url，并自动同步计算小写 URL 及 host 域名，便于快速模式匹配。"""
        self._base_url = value
        self._base_url_lower = value.lower() if value else ""
        self._base_url_hostname = base_url_hostname(value)

    def __init__(
        self,
        base_url: str = None, api_key: str = None, provider: str = None, api_mode: str = None,
        acp_command: str = None, acp_args: list[str] | None = None, command: str = None, args: list[str] | None = None,
        model: str = "",
        max_iterations: int = sys.maxsize,  # unlimited tool-calling iterations by default (shared with subagents)
        tool_delay: float = None,  # deprecated: accepted for compatibility, ignored
        enabled_toolsets: List[str] = None, disabled_toolsets: List[str] = None,
        save_trajectories: bool = False, verbose_logging: bool = False, quiet_mode: bool = False,
        tool_progress_mode: str = "all", ephemeral_system_prompt: str = None,
        log_prefix_chars: int = 100, log_prefix: str = "",
        providers_allowed: List[str] = None, providers_ignored: List[str] = None, providers_order: List[str] = None,
        provider_sort: str = None, provider_require_parameters: bool = False, provider_data_collection: str = None,
        openrouter_min_coding_score: Optional[float] = None,
        session_id: str = None,
        tool_progress_callback: callable = None, tool_start_callback: callable = None,
        tool_complete_callback: callable = None, thinking_callback: callable = None,
        reasoning_callback: callable = None, clarify_callback: callable = None,
        read_terminal_callback: callable = None, read_preview_callback: callable = None,
        drive_preview_callback: callable = None, read_window_below_callback: callable = None,
        connection_callback: callable = None, tour_callback: callable = None, step_callback: callable = None,
        stream_delta_callback: callable = None, interim_assistant_callback: callable = None,
        tool_gen_callback: callable = None, status_callback: callable = None,
        notice_callback: callable = None, notice_clear_callback: callable = None,
        event_callback: Optional[Callable[[str, dict], None]] = None,
        reaction_callback: Optional[Callable[[str], None]] = None,
        max_tokens: int = None, reasoning_config: Dict[str, Any] = None, service_tier: str = None,
        request_overrides: Dict[str, Any] = None, prefill_messages: List[Dict[str, Any]] = None,
        platform: str = None, user_id: str = None, user_id_alt: str = None, user_name: str = None,
        chat_id: str = None, chat_name: str = None, chat_type: str = None, thread_id: str = None,
        gateway_session_key: str = None,
        skip_context_files: bool = False, load_soul_identity: bool = False,
        skip_memory: bool = False, skip_background_review: bool = False,
        session_db=None, parent_session_id: str = None,
        iteration_budget: "IterationBudget" = None, run_budget_seconds: Optional[float] = None,
        fallback_model: Dict[str, Any] = None, credential_pool=None,
        checkpoints_enabled: bool = False, checkpoint_max_snapshots: int = 20,
        checkpoint_max_total_size_mb: int = 500, checkpoint_max_file_size_mb: int = 10,
        pass_session_id: bool = False, requested_provider: str = None,
        capabilities: Dict[str, bool] | None = None, cwd: str | None = None,
        side_agent: bool = False, memory_manager=None,
    ):
        """Forwarder — see ``agent.agent_init.init_agent`` (same keyword parameters, minus ``tool_delay``).
        
        【构造函数转发 / Forwarder】
        `AIAgent` 的初始化参数非常丰富，涵盖基础模型信息、工具集白黑名单、各类观察者与流式回调、
        网关身份、会话数据库、预算限制、快照检查点等。
        所有实际初始化逻辑被委托给 `agent.agent_init.init_agent` 执行。
        """
        init_kwargs = {k: v for k, v in locals().items() if k not in ("self", "tool_delay")}
        if tool_delay is not None:
            warnings.warn("tool_delay is deprecated and ignored; sequential tool calls "
                          "no longer sleep between executions.", DeprecationWarning, stacklevel=2)
        from agent.agent_init import init_agent
        init_agent(self, **init_kwargs)

    def _get_session_db_for_recall(self):
        """SessionDB for recall, opening the default state DB when no ``session_db`` was passed so the
        advertised ``session_search`` tool stays usable.

        【设计原理 / Design Intent】
        用于记忆检索（Recall）的会话数据库句柄。若初始化时未显式传入 `session_db`，则惰性打开默认的
        SQLite state.db，确保内置的 `session_search` 工具始终可用。
        【关键安全边界】：
        若设置了 `_persist_disabled=True`（例如后台 Memory Review 分支或临时测试 Harness），
        严禁惰性打开规范的 state DB，否则会将审查过程中的临时伪造轮次误写到用户的真实会话中。
        """
        # Persistence-isolated forks (background review) must not lazily open the canonical state DB —
        # that would re-arm the flush to write the fork's harness turn into the user's real session.
        if getattr(self, "_persist_disabled", False):
            return None
        if self._session_db is not None:
            return self._session_db
        try:
            from hermes_state_registry import acquire

            self._session_db = acquire()
            self._owns_session_db = True  # 我们自己打开的数据库，必须在 close() 时负责释放引用
            return self._session_db
        except Exception:
            logger.debug("SessionDB unavailable for recall", exc_info=True)
            return None

    def _session_row_model_config(self) -> Any:
        """``model_config`` for the session row: the init config plus the live YOLO bypass.

        The row is created lazily on the first turn, so this is the only chance to record a pre-first-turn
        /yolo toggle for ``hermes --resume``.

        【会话元数据配置捕获】
        计算写入数据库会话行的 model_config。由于会话行是在首个 turn 时惰性创建的，
        这里是捕获用户在对话开始前通过 `/yolo`（危险操作免确认模式）开关设置的唯一时机，
        确保后续 `hermes --resume` 时能正确恢复 YOLO 权限模式。
        """
        model_config = self._session_init_model_config
        try:
            from tools.approval import is_session_yolo_enabled
            if is_session_yolo_enabled(self.session_id):
                model_config = dict(model_config or {})
                model_config["yolo_mode"] = True
        except Exception:
            pass
        return model_config

    def _ensure_db_session(self) -> None:
        """Create the session DB row on first use; a transient failure leaves it to retry next turn.

        【首轮对话惰性建会话行 / Lazy Session Creation】
        在对话开始前确保 SQLite 会话行已创建。关键细节：
        1. 显式记录 `profile_name`（包括 "default"）：多 Profile 隔离架构下，NULL 会被识别为孤立无主记录导致侧边栏丢失（issue #99222）；
        2. 携带完整的网关路由信息 `_gateway_origin_json`：即使网关临时出现数据库锁降级为 JSONL，这里也是唯一的持久化锚点；
        3. 遇到 SQLite 锁冲突等临时异常时保持 `_session_db_created = False`，静默等待下轮自动重试。
        """
        if getattr(self, "_persist_disabled", False) or self._session_db_created or not self._session_db:
            return
        source = _session_source_for_agent(self.platform)
        try:
            # 显式持久化 Profile 名称（包括 "default"）
            try:
                from hermes_cli.profiles import get_active_profile_name
                profile_for_session = get_active_profile_name()
            except Exception:
                profile_for_session = None
            self._session_db.create_session(
                session_id=self.session_id, source=source, model=self.model,
                model_config=self._session_row_model_config(), system_prompt=self._cached_system_prompt,
                user_id=getattr(self, "_user_id", None), session_key=getattr(self, "_gateway_session_key", None),
                chat_id=getattr(self, "_chat_id", None), chat_type=getattr(self, "_chat_type", None),
                thread_id=getattr(self, "_thread_id", None),
                display_name=getattr(self, "_chat_name", None) or getattr(self, "_user_name", None),
                origin_json=_gateway_origin_json(self), parent_session_id=self._parent_session_id,
                cwd=_launch_cwd_for_session(source), profile_name=profile_for_session,
            )
            self._session_db_created = True
        except Exception as e:
            # 临时错误（如 SQLite 锁死）：保持 _session_db_created 为 False，下一轮继续重试
            logger.warning("Session DB creation failed (will retry next turn): %s", e)

    def _transition_context_engine_session(
        self, *, old_session_id: Optional[str] = None, new_session_id: Optional[str] = None,
        previous_messages: Optional[list] = None, carry_over_context: bool = False, reset_engine: bool = True,
        **extra_context,
    ) -> None:
        """Drive the context engine's session transition: on_session_end → on_session_reset → on_session_start
        → carry_over_new_session_context. Each hook is optional (the built-in compressor only resets).

        【上下文引擎会话生命周期流转】
        当发生会话切换或重置时，按标准协议依次通知上下文压缩引擎：
        旧会话结束(on_session_end) -> 引擎重置(on_session_reset) -> 新会话启动(on_session_start) -> 上下文延续(carry_over)。
        """
        engine = getattr(self, "context_compressor", None)
        if not engine:
            return
        if old_session_id and previous_messages is not None:
            _call_engine_hook(engine, "on_session_end", old_session_id, previous_messages)
        if reset_engine:
            _call_engine_hook(engine, "on_session_reset")

        should_start = bool(old_session_id or previous_messages is not None or carry_over_context or extra_context)
        target_session_id = new_session_id or getattr(self, "session_id", "") or ""
        if should_start and target_session_id and hasattr(engine, "on_session_start"):
            start_context = {
                "old_session_id": old_session_id, "carry_over_context": carry_over_context,
                "platform": _session_source_for_agent(getattr(self, "platform", None)),
                "model": getattr(self, "model", ""), "context_length": getattr(engine, "context_length", None),
                "conversation_id": getattr(self, "_gateway_session_key", None), **extra_context,
            }
            start_context = {k: v for k, v in start_context.items() if v not in (None, "")}
            _call_engine_hook(engine, "on_session_start", target_session_id, **start_context)
        if carry_over_context and old_session_id and target_session_id:
            _call_engine_hook(engine, "carry_over_new_session_context", old_session_id, target_session_id)

    def reset_session_state(self, previous_messages: Optional[list] = None, old_session_id: Optional[str] = None,
                            carry_over_context: bool = False):
        """Reset session-scoped token/cost counters and compressor state for a fresh session.

        With ``previous_messages`` / ``old_session_id`` / ``carry_over_context`` the context engine gets the
        full transition lifecycle instead of a bare reset.

        【重置会话内部状态 / Fresh Session Reset】
        在用户执行 `/new`（新建会话）、`/resume`（恢复会话）或 `/branch`（分支对话）时调用：
        1. 归零本会话累计的 Token 消耗、推理 Token、API 调用次数与预估成本（USD）；
        2. 清空用量锚点 `_usage_anchor` 与工作区快照 `_frozen_workspace_snapshot`（确保重新生成系统提示词）；
        3. 重置用户轮次计数器 `_user_turn_count`；
        4. 重新挂载上下文压缩器（ContextCompressor）的数据库绑定状态。
        """
        for counter in (
            "session_total_tokens", "session_input_tokens", "session_output_tokens", "session_prompt_tokens",
            "session_completion_tokens", "session_cache_read_tokens", "session_cache_write_tokens",
            "session_reasoning_tokens", "session_api_calls",
        ):
            setattr(self, counter, 0)
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "unknown"
        self.session_cost_source = "none"

        # Session boundary: the usage anchor describes the OLD transcript; fall back to full estimation.
        self._usage_anchor = None
        self._turn_base_usage_anchor = None
        # The workspace snapshot is pinned per session (agent/system_prompt.py::_coding_parts); a
        # /new, /resume or /branch on the same agent must re-snapshot at its own session start.
        self._frozen_workspace_snapshot = None

        # Turn counter (added after reset_session_state was first written — #2635)
        self._user_turn_count = 0
        # The drifted-prompt compaction INFO is once per session, so a /new or /resume re-arms it.
        self._compaction_prompt_drift_logged = False
        # Who wrote the current turn. build_turn_context() sets it at the start of every turn.
        self._turn_author = None
        # Copilot x-initiator: True for the first API call of a user turn, False for tool-loop follow-ups.
        self._is_user_initiated_turn = False

        self._transition_context_engine_session(
            old_session_id=old_session_id, new_session_id=getattr(self, "session_id", None),
            previous_messages=previous_messages, carry_over_context=carry_over_context, reset_engine=True,
        )

        # Reset-only switches (/new, /resume, /branch) change session_id before this call; rebind the
        # built-in compressor's session-keyed cooldown state when no full start hook ran.
        engine = getattr(self, "context_compressor", None)
        target_session_id = getattr(self, "session_id", "") or ""
        if (engine is not None and hasattr(engine, "bind_session_state") and target_session_id
                and target_session_id != getattr(engine, "_session_id", "")):
            try:
                engine.bind_session_state(getattr(self, "_session_db", None), target_session_id)
            except Exception as exc:
                logger.debug("context engine bind_session_state during reset: %s", exc)

    @staticmethod
    def _effective_lmstudio_context_length(config_context_length: Optional[int], runtime_context_length: Any) -> Optional[int]:
        """Return a safe context budget from explicit intent and verified runtime.
        
        【LM Studio 上下文长度协商】
        结合用户在配置文件中显式指定的 context_length 与 LM Studio 运行时报告的实际显存上下文窗口，
        取二者的最小值（min），确保不会向本地模型发送超出显存分配能力的超长上下文。
        """
        explicit = _positive_int(config_context_length)
        runtime = _positive_int(getattr(runtime_context_length, "context_length", runtime_context_length))
        if bool(getattr(runtime_context_length, "rejected", False)) or (
            bool(getattr(runtime_context_length, "load_attempted", False)) and runtime is None
        ):
            return None
        if runtime is not None and explicit is not None:
            return min(runtime, explicit)
        return runtime if runtime is not None else explicit

    @staticmethod
    def _lmstudio_load_was_unverified(load_result: Any) -> bool:
        """Return true when a management load was rejected or unverifiable."""
        return bool(getattr(load_result, "rejected", False)) or (
            bool(getattr(load_result, "load_attempted", False)) and getattr(load_result, "context_length", None) is None
        )

    def _ensure_lmstudio_runtime_loaded(self, config_context_length: Optional[int] = None) -> Any:
        """Preload LM Studio unless configured to rely on JIT loading.
        
        【LM Studio 模型预加载】
        对于 LM Studio 本地部署，除非用户配置了 `lmstudio_load_mode=jit`（即时按需加载），
        否则在 Agent 启动时显式调用 LM Studio API 预先将模型装载进 GPU 显存，
        避免首次推理时由于漫长的大模型载入导致网关请求超时。
        """
        if (self.provider or "").strip().lower() != "lmstudio":
            return None
        if (getattr(self, "lmstudio_load_mode", "explicit") or "explicit").strip().lower() == "jit":
            logger.debug("LM Studio explicit preload skipped: lmstudio_load_mode=jit")
            return None
        from hermes_cli.models_local import ensure_lmstudio_model_loaded

        if config_context_length is None:
            config_context_length = getattr(self, "_config_context_length", None)
        return ensure_lmstudio_model_loaded(
            self.model, self.base_url, getattr(self, "api_key", ""), config_context_length, return_load_result=True,
        )

    switch_model = _forward("agent.agent_runtime_helpers", "switch_model")

    def _disable_codex_reasoning_replay(self, messages: Optional[List[Dict[str, Any]]] = None) -> Dict[str, int]:
        """On HTTP 400 ``invalid_encrypted_content``: disable Responses reasoning replay and pop
        ``codex_reasoning_items`` from every assistant message. Returns ``{"messages", "items"}`` counts.

        【Codex 加密思考内容失效恢复 / Reasoning Replay Recovery】
        ChatGPT OAuth Codex Responses API 会返回带有加密签名的思考项（codex_reasoning_items）。
        当会话持续一段时间或密钥轮换后，重发旧思考项可能导致服务端返回 HTTP 400 invalid_encrypted_content。
        本方法遍历历史消息，将所有 assistant 消息中的加密思考项全部剥离并禁用回放，使得会话能够自愈并继续进行。
        """
        stripped_messages = stripped_items = 0
        for msg in (messages if isinstance(messages, list) else []):
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            items = msg.pop("codex_reasoning_items", None)
            if isinstance(items, list) and items:
                stripped_messages += 1
                stripped_items += len(items)
        self._codex_reasoning_replay_enabled = False
        return {"messages": stripped_messages, "items": stripped_items}

    _stream_diag_init = _forward_static("agent.stream_diag", "stream_diag_init")
    _stream_diag_capture_response = _forward("agent.stream_diag", "stream_diag_capture_response")
    _flatten_exception_chain = _forward_static("agent.stream_diag", "flatten_exception_chain")

    def _is_provider_stream_parse_error(self, error: BaseException) -> bool:
        """True for a malformed Anthropic event-stream frame (surfaced by the SDK as a plain ``ValueError``);
        that is wire trouble, not local validation, so it follows the truncated-JSON retry path.

        【Anthropic 流式解析异常判定】
        Anthropic SDK 在遇到网络中断导致的畸变 SSE 帧时，会直接抛出 Python 原生的 `ValueError`。
        如果不做上下文特征匹配，容易被当成本地参数校验错误而放弃。
        本方法通过匹配特定的标志符（PROVIDER_STREAM_PARSE_MARKERS），确认这是底层网络残损，
        从而自动触发截断 JSON 重试流程（truncated-JSON retry path）。
        """
        return (getattr(self, "api_mode", None) == "anthropic_messages" and isinstance(error, ValueError)
                and not isinstance(error, (UnicodeEncodeError, json.JSONDecodeError))
                and any(marker in str(error).strip().lower() for marker in PROVIDER_STREAM_PARSE_MARKERS))

    _log_stream_retry = _forward("agent.stream_diag", "log_stream_retry")
    _emit_stream_drop = _forward("agent.stream_diag", "emit_stream_drop")

    def _emit_auxiliary_failure(self, task: str, exc: BaseException) -> None:
        """Surface a compact warning for failed auxiliary work.
        
        【辅助任务异常提示】
        向用户输出格式化警告。辅助任务（如生成标题、自动总结记忆等）即使失败也不应中断主流程，
        将错误信息修剪在 220 字符以内，以精简的警告形式展示。
        """
        try:
            detail = self._summarize_api_error(exc)
        except Exception:
            detail = str(exc)
        detail = (detail or exc.__class__.__name__).strip()
        if len(detail) > 220:
            detail = detail[:217].rstrip() + "..."
        self._emit_warning(f"⚠ Auxiliary {task} failed: {detail}")

    def _current_main_runtime(self) -> Dict[str, str]:
        """Return the live main runtime for session-scoped auxiliary routing.
        
        【获取当前运行环境参数】
        导出当前会话的模型、provider、base_url、api_key、api_mode 等核心元数据，
        供辅助任务路由到正确的模型服务。
        """
        return {
            key: getattr(self, key, "") or ""
            for key in ("model", "provider", "base_url", "api_key", "api_mode", "auth_mode", "session_id")
        }

    _check_compression_model_feasibility = _forward("agent.conversation_compression", "check_compression_model_feasibility")
    _replay_compression_warning = _forward("agent.conversation_compression", "replay_compression_warning")

    def _hostname_for(self, base_url: Optional[str]) -> str:
        """获取 base_url 的主机名，若未提供则使用 agent 自身的 base_url 主机名。"""
        if base_url is not None:
            return base_url_hostname(base_url)
        return getattr(self, "_base_url_hostname", "") or base_url_hostname(getattr(self, "_base_url_lower", ""))

    def _is_direct_openai_url(self, base_url: str = None) -> bool:
        """判断 base_url 是否直接指向 OpenAI 官方原生 API (api.openai.com)。"""
        return self._hostname_for(base_url) == "api.openai.com"

    def _is_azure_openai_url(self, base_url: str = None) -> bool:
        """判断 base_url 是否指向 Azure OpenAI 端点（注意：Azure 使用标准客户端，但不支持 Responses API）。"""
        url = str(base_url).lower() if base_url is not None else (getattr(self, "_base_url_lower", "") or "")
        return base_url_host_matches(url, "openai.azure.com")

    def _is_github_copilot_url(self, base_url: str = None) -> bool:
        """判断 base_url 是否指向 GitHub Copilot 的 OpenAI 兼容接口。"""
        hostname = self._hostname_for(base_url)
        return bool(hostname) and (hostname == "api.githubcopilot.com" or hostname.endswith(".githubcopilot.com"))

    def _resolved_api_call_timeout(self) -> float:
        """Per-call request timeout: per-model ``timeout_seconds`` > provider ``request_timeout_seconds`` >
        ``HERMES_API_TIMEOUT`` > 1800s.

        【API 单次调用硬超时解析优先级】
        严格遵循以下优先级层级解析：
        1. 针对具体模型的配置项（timeout_seconds）
        2. Provider 全局配置项（request_timeout_seconds）
        3. 环境变量 `HERMES_API_TIMEOUT`
        4. 兜底默认值：1800 秒（30 分钟）
        """
        cfg = get_provider_request_timeout(self.provider, self.model)
        return cfg if cfg is not None else env_float("HERMES_API_TIMEOUT", 1800.0)

    def _resolved_api_call_stale_timeout_base(self) -> tuple[float, bool]:
        """Base non-stream stale timeout: per-model ``stale_timeout_seconds`` > provider-wide >
        ``HERMES_API_CALL_STALE_TIMEOUT`` > reasoning floor > 90s.

        Returns ``(seconds, uses_implicit_default)``; the implicit flag lets callers auto-disable the detector
        for local endpoints only when the user configured nothing.

        【非流式请求静默卡死检测基准 / Stale Timeout Base】
        用于检测非流式调用是否因为服务端断连但未关闭 Socket 而产生僵死。
        解析层级：模型级别 > Provider 级别 > 环境变量 `HERMES_API_CALL_STALE_TIMEOUT` >
        深度思考模型保护下限（Reasoning floor，防止 o1/o3/r1 长考时被误判为卡死） > 兜底 90 秒。
        返回 (seconds, uses_implicit_default) 元组，如果是隐式默认值，对于本地模型（Ollama/vLLM）会自动关闭该检测。
        """
        cfg = get_provider_stale_timeout(self.provider, self.model)
        if cfg is not None:
            return cfg, False
        env_timeout = os.getenv("HERMES_API_CALL_STALE_TIMEOUT")
        if env_timeout is not None:
            return float(env_timeout), False
        # 深度思考模型保护下限（云端网关常常在模型深思时切断连接，这里设定特殊保底时间）
        from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor
        reasoning_floor = get_reasoning_stale_timeout_floor(self.model)
        if reasoning_floor is not None:
            return reasoning_floor, False
        return 90.0, True

    def _compute_non_stream_stale_timeout(self, api_payload: Any) -> float:
        """Effective non-stream stale timeout for ``api_payload`` (an ``api_kwargs`` dict or legacy ``messages``
        list), scaled by estimated context size and capped by the run budget.

        【动态计算非流式请求假死超时时间 / Dynamic Stale Timeout】
        根据请求规模和整体任务预算动态调整超时门限：
        1. 针对超长上下文自适应放大：预估 Token > 100k 时至少 240 秒，> 50k 时至少 150 秒；
        2. 针对高思考强度的 Codex 深度推理（High-effort Reasoning），设定专门的静默保底时间；
        3. 受总运行预算（run_budget_seconds）上限保护：隐式超时时间不得超过剩余预算的一半（且 >= 60s），
           确保单次挂起的网络请求不会吃光整个 Agent 的任务运行时间。
        """
        stale_base, uses_implicit_default = self._resolved_api_call_stale_timeout_base()
        base_url = getattr(self, "_base_url", None) or self.base_url or ""
        if uses_implicit_default and base_url and is_local_endpoint(base_url):
            return float("inf")

        from agent.chat_completion_helpers import _high_effort_silence_floor, estimate_request_context_tokens
        est_tokens = estimate_request_context_tokens(api_payload)
        timeout = max(stale_base, 240.0) if est_tokens > 100_000 else max(stale_base, 150.0) if est_tokens > 50_000 else stale_base
        explicit = self._stale_timeout_is_explicit()
        # High-effort Codex reasoning (#112909) floors the IMPLICIT stale timeout before the run-budget
        # cap below, so the floor can never outlive the run budget.
        if self.api_mode == "codex_responses" and not explicit:
            timeout = max(timeout, _high_effort_silence_floor(self))
        # 运行预算上限约束：隐式超时受限于剩余预算的一半，避免单次挂死消耗整个会话
        run_budget = getattr(self, "run_budget_seconds", None)
        started = getattr(self, "_run_budget_started_at", None)
        if run_budget and started and not explicit:
            remaining = float(run_budget) - (time.time() - started)
            timeout = min(timeout, max(60.0, remaining * 0.5))
        return timeout

    def _stale_timeout_is_explicit(self) -> bool:
        """判断用户是否显式配置了假死超时（显式配置的超时不向 run_budget 妥协截断）。"""
        return (get_provider_stale_timeout(self.provider, self.model) is not None
                or os.getenv("HERMES_API_CALL_STALE_TIMEOUT") is not None)

    def _codex_silent_hang_hint(self, model: Optional[str] = None) -> Optional[str]:
        """Actionable hint when the request matches a known Codex silent-reject shape (currently the ``gpt-5.5``
        family: connection accepted, no events, no error), else None. Makes the stale timeout actionable.

        【Codex 后端静默断连诊断 / Actionable Diagnostic Hint】
        针对 ChatGPT Codex 后端偶尔对 gpt-5.5 系列模型出现的“假死静默断连”（TCP 连接成功但无事件无报错）
        自动提供排查诊断指南，指引用户无缝降级到 gpt-5.4 或其它后备模型（issue #21444）。
        """
        if self.api_mode != "codex_responses":
            return None
        from agent.codex_responses_adapter import classify_responses_route

        if not classify_responses_route(self).is_codex_backend:
            return None
        eff_model = (model if model is not None else self.model) or ""
        # 匹配 gpt-5.5 系列（包括裸名、-codex 或带厂商前缀），排除 gpt-5.50
        if not re.search(r"(?:^|[/\-_])gpt-5\.5(?:$|[\-_])", eff_model.lower()):
            return None
        return (
            f"Codex backend appears to be silently rejecting {eff_model!r} "
            "on chatgpt.com/backend-api/codex (no stream events, no error). "
            "This is a known backend-side pattern that has affected ChatGPT "
            "Plus accounts intermittently. "
            "Workaround: try `gpt-5.4` on the same OAuth profile, "
            "or switch to a different model/provider in your fallback chain. "
            "Some ChatGPT Codex accounts do not support `gpt-5.4-codex`. "
            "See hermes-agent#21444 for symptom history."
        )

    def _is_openrouter_url(self) -> bool:
        """判断 base_url 是否指向 OpenRouter。"""
        return base_url_host_matches(self._base_url_lower, "openrouter.ai")

    def _is_copilot_url(self) -> bool:
        """判断 base_url 是否指向 GitHub Copilot 或 GitHub Models。"""
        return any(base_url_host_matches(self._base_url_lower, h) for h in ("api.githubcopilot.com", "models.github.ai"))

    def _is_copilot_provider(self) -> bool:
        """判断当前 provider 是否属于 GitHub Copilot 系列（支持 copilot/github-copilot/github 等别名）。"""
        return (self.provider or "").strip().lower() in {"copilot", "github-copilot", "github"} or self._is_copilot_url()

    def _is_codex_backend(self) -> bool:
        """判断是否为 ChatGPT OAuth Codex Responses 后端。"""
        return (getattr(self, "api_mode", None) == "codex_responses"
                and getattr(self, "_base_url_hostname", "") == "chatgpt.com"
                and "/backend-api/codex" in (getattr(self, "_base_url_lower", "") or ""))

    _anthropic_prompt_cache_policy = _forward("agent.agent_runtime_helpers", "anthropic_prompt_cache_policy")
    _direct_native_anthropic_tool_cache_capability = _forward("agent.agent_runtime_helpers", "_direct_native_anthropic_tool_cache_capability")

    @staticmethod
    def _model_requires_responses_api(model: str) -> bool:
        """True for GPT-5.x, which OpenAI and OpenRouter reject on /v1/chat/completions
        (``unsupported_api_for_model``).
        
        【Responses API 强校验】
        GPT-5.x 系列在 OpenAI 官方和 OpenRouter 上均已弃用传统的 /v1/chat/completions 接口，
        强行调用会报 unsupported_api_for_model 错误，必须切换为 Responses API。
        """
        return model.lower().rsplit("/", 1)[-1].startswith("gpt-5")  # 剥离厂商前缀如 "openai/gpt-5.4"

    @staticmethod
    def _provider_model_requires_responses_api(model: str, *, provider: Optional[str] = None) -> bool:
        """判断特定的 provider / model 组合是否需要使用 Responses API。"""
        from hermes_cli.providers import is_actual_route
        normalized_provider = (provider or "").strip().lower()
        # Nous 自有端点通过 chat completions 提供 GPT-5.x 服务；通用自定义网关可能不支持 Responses 语义
        if normalized_provider in ("nous", "custom") or is_actual_route(provider):
            return False
        # ACP facades expose the OpenAI-compatible chat.completions shape regardless of model
        # family and have no ``responses`` attribute, so neither primary routing nor GPT-5
        # fallback activation may upgrade them. Keyed on the profile's auth_type: every
        # external-process provider, not one vendor's names.
        from hermes_cli.runtime_provider_backends import _is_external_process_provider
        if _is_external_process_provider(normalized_provider):
            return False
        if normalized_provider == "copilot":
            try:
                from hermes_cli.models import _should_use_copilot_responses_api
                return _should_use_copilot_responses_api(model)
            except Exception:
                pass  # 回退到通用的 GPT-5 判断规则
        return AIAgent._model_requires_responses_api(model)

    def _max_tokens_param(self, value: int) -> dict:
        """``max_completion_tokens`` for newer OpenAI families (and Azure / Copilot serving them), else
        ``max_tokens``. URL-first, then model-name fallback for third-party endpoints fronting those models.

        【Token 参数适配】
        OpenAI 新模型族（包括 Azure/Copilot）改用 `max_completion_tokens`，而旧版或第三方模型使用 `max_tokens`。
        优先根据 URL 端点识别，次选根据模型名称判断。
        """
        if (self._is_direct_openai_url() or self._is_azure_openai_url() or self._is_github_copilot_url()
                or model_forces_max_completion_tokens(self.model)):
            return {"max_completion_tokens": value}
        return {"max_tokens": value}

    @staticmethod
    def _requested_output_cap_from_api_kwargs(api_kwargs: Any) -> Optional[int]:
        """从已构造的请求参数中提取响应输出的最大 Token 上限。"""
        if not isinstance(api_kwargs, dict):
            return None
        for key in ("max_output_tokens", "max_completion_tokens", "max_tokens"):
            try:
                value = int(api_kwargs.get(key))
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
        return None

    def _has_content_after_think_block(self, content: str) -> bool:
        """剥离 `<think>` 思考块后是否仍然存在可见内容。若仅有思考而无实际输出，系统会触发重试。"""
        return bool(content) and bool(self._strip_think_blocks(content).strip())

    _strip_think_blocks = _forward("agent.agent_runtime_helpers", "strip_think_blocks")

    @staticmethod
    def _has_natural_response_ending(content: str) -> bool:
        """Heuristic: does visible assistant text look intentionally finished?
        
        【响应自然结尾启发式判断】
        检查可见文本末尾是否具有合乎语法的结束符（句号、问号、感叹号、反引号代码块闭合标记或 Emoji），
        用于快速鉴别模型是否因被服务端静默截断而遗留半句话。
        """
        stripped = (content or "").rstrip()
        if not stripped:
            return False
        last = stripped[-1]
        # 结尾标点、代码块闭合标记 ``` 或常见 Emoji 符号
        return stripped.endswith("```") or last in '.!?:)"\']}。！？：）】」』》^' or ord(last) >= 0x1F300

    def _is_ollama_glm_backend(self) -> bool:
        """Ollama-hosted GLM models misreport finish_reason='stop'. Matches only explicit Ollama signatures
        (port 11434, "ollama" in URL, provider ollama), never arbitrary local proxies; excludes Ollama Cloud
        (``ollama.com`` / ``:cloud``), which reports faithfully — rewriting it would manufacture truncations.

        【Ollama GLM 特殊后端识别 / Ollama GLM Bug Mitigation】
        本地 Ollama 部署的 GLM 系列模型存在一个历史已知缺陷：即使输出因达到上下文上限被截断，
        依然会错误地返回 finish_reason='stop'。
        本方法精准识别真正的本地 Ollama 实例（端口 11434 或 URL 带有 ollama），同时排除上报正确的 Ollama Cloud，
        避免对正规代理造成误伤。
        """
        model_lower = (self.model or "").lower()
        provider_lower = (self.provider or "").lower()
        if "glm" not in model_lower and provider_lower != "zai":
            return False
        base = self._base_url_lower
        # Ollama Cloud 能够如实转发 finish_reason，不进行重写
        if "ollama.com" in base or ":cloud" in model_lower:
            return False
        if "ollama" in base or ":11434" in base:
            return True
        return provider_lower == "ollama"

    def _should_treat_stop_as_truncated(self, finish_reason: str, assistant_message, messages: Optional[list] = None) -> bool:
        """Detect conservative stop->length misreports for Ollama-hosted GLM models.
        
        【将错误的 stop 判定为截断截流 / Stop->Length Correction】
        结合当前后端类型、是否有 tool 消息交互、文本长度以及末尾是否缺少自然标点，
        判定该 stop 是否为误报的假完成。如果是，则将其修正为截断，触发续写机制。
        """
        if finish_reason != "stop" or self.api_mode != "chat_completions" or not self._is_ollama_glm_backend():
            return False
        if not any(isinstance(msg, dict) and msg.get("role") == "tool" for msg in (messages or [])):
            return False
        if assistant_message is None or getattr(assistant_message, "tool_calls", None):
            return False
        content = getattr(assistant_message, "content", None)
        if not isinstance(content, str):
            return False
        visible_text = self._strip_think_blocks(content).strip()
        if len(visible_text) < 20 or not re.search(r"\s", visible_text):
            return False
        return not self._has_natural_response_ending(visible_text)

    _looks_like_codex_intermediate_ack = _forward("agent.agent_runtime_helpers", "looks_like_codex_intermediate_ack")
    _extract_reasoning = _forward("agent.agent_runtime_helpers", "extract_reasoning")
    _cleanup_task_resources = _forward("agent.chat_completion_helpers", "cleanup_task_resources")

    # 后台记忆与技能提炼提示词（Prompts 维护在 agent.background_review 中）
    from agent.background_review import _MEMORY_REVIEW_PROMPT, _SKILL_REVIEW_PROMPT, _COMBINED_REVIEW_PROMPT
    _summarize_background_review_actions = _forward_static("agent.background_review", "summarize_background_review_actions")

    def _spawn_background_review(self, messages_snapshot: List[Dict], review_memory: bool = False,
                                 review_skills: bool = False, focus: Optional[str] = None, explicit: bool = False) -> None:
        """Post-turn review entry point: decide WHEN, then spawn.

        A review whose runtime is the MANAGED LOCAL llama-server is queued for machine idle (``defer: auto``)
        instead of hitting the user's GPU mid-session; everything else spawns immediately. ``explicit``
        (/refine) is never deferred but does not touch the ``focus``-keyed delegate/enabled gates.

        【后台记忆/技能审查入口 / Post-Turn Background Review】
        每轮 Turn 结束后的自我沉淀机制：
        1. 检查门控条件（子 Agent 深度 > 0 时默认跳过，避免重复提炼）；
        2. 若在本地部署且处于 defer: auto 模式，将审查任务放入 idle_queue 队列，等系统空闲时再执行；
        3. 进行深层结构克隆（_clone_background_review_messages），防止后台审查过程的脱敏或就地修改污染前台活跃会话历史；
        4. 其余情况直接通过 _spawn_background_review_now 启动后台守护线程。
        """
        # Gates run at enqueue/spawn time; the idle dispatcher re-checks `enabled` at dispatch time.
        if focus is None and getattr(self, "_delegate_depth", 0) > 0:
            return
        task_cfg = None
        if focus is None:
            from agent.background_review import load_background_review_settings
            enabled, task_cfg = load_background_review_settings()
            if not enabled:
                return

        # Structural clone at the single chokepoint: the fork sanitizes in place, and a shallow copy would
        # alias the live history's nested tool_calls/content.
        # Structural clone at the single chokepoint every review path (automatic, /refine, idle-queue
        # deferral) goes through. See #100795.
        from agent.turn_finalizer import _clone_background_review_messages
        kwargs = dict(messages_snapshot=_clone_background_review_messages(messages_snapshot),
                      review_memory=review_memory, review_skills=review_skills, focus=focus, task_cfg=task_cfg,
                      explicit=explicit)
        if focus is None and not explicit and _review_should_defer(self, task_cfg):
            from agent.review_idle_queue import QUEUE
            QUEUE.enqueue(self, _review_queue_key(self), kwargs)
            return
        self._spawn_background_review_now(**kwargs)

    def _spawn_background_review_now(self, messages_snapshot: List[Dict], review_memory: bool = False,
                                     review_skills: bool = False, focus: Optional[str] = None,
                                     task_cfg: Optional[Dict[str, Any]] = None, _requeue_attempts: int = 0,
                                     explicit: bool = False) -> None:
        """Spawn the background memory/skill review thread.

        ``threading.Thread`` is constructed here so tests patching ``run_agent.threading.Thread`` keep working.
        ``focus`` is /refine steering text; ``task_cfg`` is the pre-loaded config block (None on direct calls).
        ``explicit`` (/refine) forks under the ``refine_review`` write origin, keeping the full
        memory operation set. A deferred review preempted by a live turn is requeued (bounded)
        rather than lost.

        【启动后台审查守护线程 / Spawn Background Review Thread】
        执行记忆提取与技能提炼的线程启动逻辑：
        1. 显式通过 `propagate_context_to_thread` 传递当前的 Profile 运行时上下文，
           确保后台写入的 `MEMORY.md` 和技能文件正确落入当前 Profile 目录下；
        2. 设置为 daemon 守护线程，不阻塞主程序退出；
        3. 若该审查在后续被用户新发起的 live turn 抢占中断，会触发重新入队逻辑。
        """
        from agent.background_review import (
            finish_background_review_run, prepare_background_review_run, spawn_background_review_thread,
        )
        from tools.thread_context import propagate_context_to_thread

        review_run = prepare_background_review_run(self)
        if review_run is None:
            return
        try:
            target, _prompt = spawn_background_review_thread(
                self, messages_snapshot, review_memory=review_memory, review_skills=review_skills,
                focus=focus, task_cfg=task_cfg, review_run=review_run, explicit=explicit,
            )

            def _target_with_requeue() -> None:
                target()
                self._maybe_requeue_preempted_review(review_run, dict(
                    messages_snapshot=messages_snapshot, review_memory=review_memory, review_skills=review_skills,
                    focus=focus, task_cfg=task_cfg, _requeue_attempts=_requeue_attempts + 1,
                    explicit=explicit))

            # 跨线程传递当前活跃 Profile 上下文，防止文件落入默认 profile
            threading.Thread(target=propagate_context_to_thread(_target_with_requeue), daemon=True, name="bg-review").start()
        except Exception:
            finish_background_review_run(self, review_run)
            raise

    _REVIEW_REQUEUE_MAX_ATTEMPTS = 3  # 被用户新交互打断时的最大重新入队重试次数

    def _maybe_requeue_preempted_review(self, review_run, kwargs) -> None:
        """Requeue a deferred-mode review that a live turn cancelled.

        Only for automatic reviews on the managed local runtime; bounded attempts stop a busy box cycling
        forever.

        【被打断的审查任务重新入队 / Preempted Review Requeue】
        当本地后台审查因为用户突然发送新消息而被取消时（抢占优先级），
        若重试次数未达到上限（3 次），自动重新塞入 idle 队列，等待下一轮空闲时再次执行。
        """
        try:
            # 只有确实被 cancel 打断、且不是显式 /refine 时才重新入队
            if not review_run.cancel_requested.is_set() or kwargs.get("focus") is not None:
                return
            if kwargs.get("_requeue_attempts", 0) > self._REVIEW_REQUEUE_MAX_ATTEMPTS:
                logger.info("Preempted background review dropped after %d requeues", self._REVIEW_REQUEUE_MAX_ATTEMPTS)
                return
            if not _review_should_defer(self, kwargs.get("task_cfg")):
                return
            from agent.review_idle_queue import QUEUE
            QUEUE.enqueue(self, _review_queue_key(self), dict(kwargs))
        except Exception:  # noqa: BLE001 — requeue is best-effort
            logger.debug("Preempted-review requeue failed", exc_info=True)

    _build_memory_write_metadata = _forward("agent.background_review", "build_memory_write_metadata")
    _apply_pending_steer_to_tool_results = _forward("agent.agent_runtime_helpers", "apply_pending_steer_to_tool_results")

    def get_activity_summary(self) -> dict:
        """Diagnostic snapshot: ``last_activity_*`` plus the short aliases gateway and delegate readers use.
        
        【会话活跃度诊断快照 / Activity Summary】
        采集当前 Agent 的实时健康与执行状态快照，供 Gateway、Subagent 协调器及 WebUI 观察：
        包含最后活动时间戳、当前正在执行的工具、API 调用计数、迭代预算消耗等。
        """
        from agent.session_activity import build_activity_snapshot

        provenance = getattr(self, "_last_activity_provenance", None)
        return build_activity_snapshot(
            last_activity_at=getattr(self, "_last_activity_ts", None),
            last_activity_description=getattr(self, "_last_activity_desc", None) or "",
            last_activity_provenance=provenance if provenance is not None else ActivityProvenance.UNKNOWN,
            extra={
                "current_tool": self._current_tool, "api_call_count": self._api_call_count,
                "max_iterations": self.max_iterations, "budget_used": self.iteration_budget.used,
                "budget_max": self.iteration_budget.max_total,
            },
        )

    def shutdown_memory_provider(self, messages: list = None) -> None:
        """Shut down the memory provider and context engine at session end (idempotent: gateway cleanup and
        ``close()`` may both call it).
        
        【幂等关闭记忆提供者与上下文引擎】
        会话结束时触发 memory_manager 和 context_engine 的收尾动作，幂等设计（多次调用安全）。
        """
        if getattr(self, "_memory_provider_shutdown", False):
            return
        self._memory_provider_shutdown = True
        if self._memory_manager:
            try:
                self._memory_manager.on_session_end(messages or [])
            except Exception as e:
                logger.warning("Memory provider on_session_end failed during shutdown: %s", e, exc_info=True)
            _quietly(lambda: self._memory_manager.shutdown_all())
        _notify_context_engine_session_end(self, messages)

    def commit_memory_session(self, messages: list = None) -> None:
        """Flush end-of-session extraction on session_id rotation (/new, compression) without tearing providers
        down.
        
        【提交当前会话记忆】
        在会话轮转（如用户执行 `/new` 或发生上下文压缩）时触发记忆提取，但不关闭 provider 连接。
        """
        if self._memory_manager:
            _quietly(lambda: self._memory_manager.on_session_end(messages or []))
        _notify_context_engine_session_end(self, messages)

    def _sync_external_memory_for_turn(self, *, original_user_message: Any, final_response: Any, interrupted: bool,
                                       messages: list | None = None) -> None:
        """Mirror a completed turn into external memory providers (``sync_all`` + ``queue_prefetch_all``).

        Uses ``original_user_message`` (``user_message`` may carry injected skill content). Interrupted turns
        are skipped: partial output is not durable truth. Best-effort — an offline backend never blocks.

        A partial assistant output, an aborted tool chain, or a mid-stream reset is not durable
        conversational truth — mirroring it into an external memory backend pollutes future recall with
        state the user never saw completed. The prefetch is gated on the same flag: the user's next message
        is almost certainly a retry of the same intent, and a prefetch keyed on the interrupted turn would
        fire against stale context. See #15218.

        【单轮对话同步至外部记忆 / External Memory Sync Invariants】
        将本轮对话的内容同步至外部记忆后端（如 Mem0 / 向量存储）：
        【核心不变量 / Core Invariant】：
        若轮次被中断（interrupted=True），绝对不进行同步！
        因为被用户掐断的半截响应、未完成的工具链不是持久的真实对话事实，将其写入外部存储会严重污染
        未来的检索与召回（参见 issue #15218）。同时跳过无实际有效信号的低信息量提示词。
        """
        if interrupted or not (self._memory_manager and final_response and original_user_message):
            return
        # 扁平化多模态部分为纯文本（换行连接供记忆提取）
        user_text = _summarize_user_message_for_log(original_user_message, sep="\n")
        response_text = _summarize_user_message_for_log(final_response, sep="\n")
        if not (user_text and response_text):
            return
        try:
            sync_kwargs = {"session_id": self.session_id or "", **({"messages": messages} if messages is not None else {})}
            # 本轮发言者身份
            turn_author = getattr(self, "_turn_author", None)
            if turn_author is not None:
                sync_kwargs["turn_author"] = turn_author
            self._memory_manager.sync_all(user_text, response_text, **sync_kwargs)
            # 无有效信号的短语不触发预取
            if not is_trivial_prompt(user_text):
                self._memory_manager.queue_prefetch_all(user_text, session_id=self.session_id or "")
        except Exception:
            pass

    def release_clients(self) -> None:
        """Release LLM clients and child agents WITHOUT tearing down session tool state (gateway cache
        eviction: the session may resume on the same task_id, so processes, sandbox, browser, computer-use and
        memory provider are kept). Idempotent; distinct from ``close()``。

        【网关缓存淘汰时的软释放 / Gateway Cache Eviction】
        不同于彻底销毁的 `close()`，网关内存管理器淘汰空闲会话时调用此方法：
        - 仅释放 HTTP/API 客户端连接池（关闭 Socket、释放 TLS 文件描述符）；
        - 【保留】终端环境、Docker 虚拟机、沙箱、浏览器与记忆提供者；
        - 这样当用户在同一会话中再次发消息时，可以秒级无缝唤醒，无需重建沉重的沙箱环境。
        """
        self._close_active_children(soft=True)
        # 退休（而不是硬关闭）共享客户端：淘汰运行在网关内存管理线程上，跨线程硬关闭可能释放正在析构的工作线程的 TLS FD
        _quietly(self._drop_shared_client, lambda c: self._retire_shared_openai_client(c, reason="cache_evict"))
        self._close_request_clients("cache_evict")
        # The Codex app-server child is an LLM client, not session tool state: the evicted instance is popped
        # from the cache and a rebuilt agent spawns its own child, so an unclosed one leaks for the gateway's life.
        _quietly(self._close_codex_session)

    def close(self) -> None:
        """Release every resource this agent holds (idempotent); each phase is guarded so one failure never
        blocks the rest.

        【彻底销毁 Agent / Full Teardown】
        释放 Agent 持有的所有资源（幂等安全执行）：
        1. 触发外部记忆关闭与刷盘；
        2. 彻底关闭任务沙箱与终端工具资源；
        3. 关闭所有派生的子 Agent；
        4. 物理关闭 HTTP 客户端连接与 Codex 会话；
        5. 主动清空对话历史列表，断开对大内存的循环引用；
        6. 调用 `_trim_process_memory`（在 Linux glibc 下调用 malloc_trim 将堆内存物理归还操作系统）；
        7. 结算并关闭 SQLite 会话行。
        """
        session_messages = getattr(self, "_session_messages", None)
        _quietly(self.shutdown_memory_provider, session_messages if isinstance(session_messages, list) else None)
        self._close_task_resources(getattr(self, "session_id", None) or "")
        self._close_active_children(soft=False)
        _quietly(self._drop_shared_client, lambda c: self._close_openai_client(c, reason="agent_close", shared=True))
        self._close_request_clients("agent_close")
        _quietly(self._close_codex_session)
        # 主动释放对话历史，避免闭包在父堆内存中持有已关闭的 Agent 历史
        self._session_messages = []
        self._db_flush_scan_prefix = None
        self._streamed_assistant_text_parts = []
        _quietly(self._trim_process_memory)
        _quietly(self._finalize_owned_session_row)

    # -- close()/release_clients() 细分阶段 / Teardown Phases -----------------------------------------

    def _close_active_children(self, *, soft: bool) -> None:
        """Detach and close per-turn child agents; ``soft`` releases their clients first, falling back to close().
        
        【分流并清理子 Agent / Child Agent Teardown】
        从当前 Agent 中分离活跃的子 Agent。
        soft=True 时仅执行 release_clients() 释放其网络客户端；soft=False 时执行物理销毁 close()。
        """
        try:
            with self._active_children_lock:
                children = list(self._active_children)
                self._active_children.clear()
        except Exception:
            return
        for child in children:
            if soft:
                try:
                    child.release_clients()
                    continue
                except Exception:
                    pass
            _quietly(lambda: child.close())

    def _drop_shared_client(self, close_fn: Callable[[Any], None]) -> None:
        """Hand the shared OpenAI/httpx client to ``close_fn`` and clear the attribute.
        
        【释放共享客户端连接 / Shared Client Socket Retirement】
        安全释放底层 OpenAI / httpx 客户端连接。
        【关键并发安全 (issue #70773)】：淘汰机制运行在网关内存管理器线程上，
        如果直接跨线程硬关闭共享客户端，会导致底层 TLS 文件描述符（FD）被操作系统立即释放，
        此时其他尚未完全退出的工作线程若正好复用了被关闭的 FD，会引发极难定位的 SQLite 数据库写损坏。
        因此通过 Retirement（退役连接池）优雅关闭连接，等待各线程引用归零后由 GC 安全回收。
        """
        client = getattr(self, "client", None)
        if client is not None:
            close_fn(client)
            self.client = None

    def _close_request_clients(self, reason: str) -> None:
        """释放按请求级别缓存的底层 OpenAI / Anthropic 网络客户端。"""
        _quietly(self._close_cached_request_openai_client, reason=reason)
        _quietly(self._close_cached_request_anthropic_client, reason=reason)

    def _close_codex_session(self) -> None:
        """关闭 Codex app-server 子进程会话（在 close() 之前清空引用，防止并发线程读取半关闭的脏会话）。"""
        codex_session = getattr(self, "_codex_session", None)
        if codex_session is not None:
            self._codex_session = None
            codex_session.close()

    @staticmethod
    def _trim_process_memory() -> None:
        """Return freed heap pages to the OS on glibc; safe no-op elsewhere.
        
        【物理内存归还内核 / malloc_trim】
        在 Linux glibc 环境下，Python 内部虽然释放了对象，但 glibc 内存分配器通常不会立即把空闲堆内存
        还给宿主机操作系统。这里调用 trim_memory 触发 `malloc_trim(0)`，真正缩减进程 RSS 占用。
        """
        from hermes_cli.mem_trim import trim_memory
        trim_memory(force=True, reason="agent close")

    def _finalize_owned_session_row(self) -> None:
        """结束本 Agent 独占的会话行，并关闭 SQLite 句柄（仅当本 Agent 是句柄的所有者时关闭，共享实例只扣减引用计数）。"""
        session_db = getattr(self, "_session_db", None)
        session_id = getattr(self, "session_id", None)
        if getattr(self, "_end_session_on_close", True) and session_db and session_id:
            _quietly(lambda: session_db.end_session(session_id, "agent_close"))
        if getattr(self, "_owns_session_db", False) and session_db is not None:
            self._owns_session_db = False
            from hermes_state_registry import release_or_close
            release_or_close(session_db)

    def _hydrate_todo_store(self, history: List[Dict[str, Any]]) -> None:
        """Replay the most recent todo tool response (the gateway builds a fresh AIAgent per message). Only
        results paired with an earlier assistant ``todo`` call count — a forged bare ``role: tool`` message
        must not seed the store (GHSA-5g4g-6jrg-mw3g).

        【历史 Todo 待办事项状态恢复 / Todo Hydration】
        由于网关架构在每条新消息到来时可能会重新构建一个 AIAgent 实例，因此需要从对话历史中回溯恢复 todo 状态。
        【关键安全边界 / Security Notice (GHSA-5g4g-6jrg-mw3g)】：
        必须严格校验：只有与前一条 assistant 消息发出的 `todo` tool_call 严格配对的 tool 响应才会被恢复！
        任何单独伪造的 `role: tool` 消息都坚决丢弃，防止外部恶意注入篡改系统待办事项存储。
        """
        found = self._latest_todo_response(history)
        if found is not None:
            last_todo_response, last_todo_revision = found
            try:
                history_revision = max(0, int(last_todo_revision or 0))
            except (TypeError, ValueError):
                history_revision = 1
            if history_revision > int(self._todo_store.snapshot().get("revision", 0) or 0):
                self._todo_store.restore(last_todo_response, revision=history_revision)
                if not self.quiet_mode:
                    self._vprint(f"{self.log_prefix}📋 Restored {len(last_todo_response)} todo item(s) from history")
        _set_interrupt(False)

    def _latest_todo_response(self, history: List[Dict[str, Any]]) -> Optional[tuple]:
        """倒序遍历对话历史，寻找最新配对且长度合规的 todo 工具调用响应 -> (todos, revision)。"""
        from tools.todo_tool import MAX_TODO_RESULT_CHARS

        for idx in range(len(history) - 1, -1, -1):
            msg = history[idx]
            content = msg.get("content", "")
            if msg.get("role") != "tool" or not isinstance(content, str) or not self._tool_response_matches_todo_call(history, idx):
                continue
            if len(content) > MAX_TODO_RESULT_CHARS:
                logger.warning("Skipping oversized todo tool response during hydration: "
                               "session=%s chars=%d", self.session_id or "none", len(content))
                continue
            if '"todos"' not in content:  # json.loads 解析前的轻量级前置过滤
                continue
            try:
                data = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                continue
            if "todos" in data and isinstance(data["todos"], list):
                return data["todos"], data.get("revision", 1)
        return None

    @classmethod
    def _tool_response_matches_todo_call(cls, history: List[Dict[str, Any]], tool_index: int) -> bool:
        """校验：紧邻的前一条 assistant 消息是否确实发出了拥有该 tool_call_id 的 todo 工具调用。"""
        tool_call_id = history[tool_index].get("tool_call_id") if 0 <= tool_index < len(history) else None
        if not tool_call_id:
            return False
        for prior in reversed(history[:tool_index]):
            role = prior.get("role")
            if role == "assistant":
                return cls._assistant_has_todo_tool_call(prior, tool_call_id)
            if role in {"user", "system"}:
                return False
        return False

    @classmethod
    def _assistant_has_todo_tool_call(cls, assistant_msg: Dict[str, Any], tool_call_id: str) -> bool:
        """检查 assistant 消息中的 tool_calls 是否包含该 id 的 todo 调用。"""
        tool_calls = assistant_msg.get("tool_calls")
        return isinstance(tool_calls, list) and any(
            cls._get_tool_call_id_static(tc) == tool_call_id and cls._get_tool_call_name_static(tc) == "todo"
            for tc in tool_calls
        )

    @property
    def is_interrupted(self) -> bool:
        """检查当前 Agent 是否已被请求中断。"""
        return self._interrupt_requested

    _build_system_prompt = _forward("agent.system_prompt", "build_system_prompt")

    # 提取 tool_call 的 Call ID（统一由 message_sanitization.coalesce_tool_call_id 管理）
    _get_tool_call_id_static = staticmethod(_sanitize_coalesce_tool_call_id)

    @staticmethod
    def _get_tool_call_name_static(tc) -> str:
        """提取 tool_call 的函数名称（Gemini 模型要求在每条 role: tool 消息上必须标明函数名）。"""
        if isinstance(tc, dict):
            fn = tc.get("function")
            return (fn.get("name", "") or "") if isinstance(fn, dict) else ""
        return getattr(getattr(tc, "function", None), "name", "") or ""

    _VALID_API_ROLES = frozenset({"system", "user", "assistant", "tool", "function", "developer"})
    _sanitize_api_messages = _forward_static("agent.agent_runtime_helpers", "sanitize_api_messages")

    @staticmethod
    def _is_thinking_only_assistant(msg: Dict[str, Any], *, drop_codex_reasoning_items: bool = True) -> bool:
        """True if ``msg`` is an assistant turn whose only payload is reasoning (no text, no tool_calls).

        Providers converting reasoning to thinking blocks reject it (400 "final block cannot be thinking"), so
        the turn is dropped from the API copy; the transcript keeps the reasoning block.

        【纯思考轮次识别 / Thinking-Only Turn Pruning】
        某些 API Provider 在将 reasoning 转换为 thinking 块时，若尾部消息只有思考过程而无任何文本或工具调用，
        会直接抛出 HTTP 400（"final block cannot be thinking"）。
        本方法识别出此类纯思考轮次，以便在向 API 发送的消息副本中将其剔除，而本地 transcript 中完整保留。
        """
        if not isinstance(msg, dict) or msg.get("role") != "assistant" or msg.get("tool_calls"):
            return False
        if msg.get("_thinking_prefill"):
            return True
        if AIAgent._content_has_real_payload(msg.get("content")):
            return False
        # 原生压缩检查点（Native Compaction Checkpoint）使 carrier 消息永远不属于纯思考
        from agent.native_compaction import has_compaction_checkpoint

        if has_compaction_checkpoint(msg.get("codex_reasoning_items")):
            return False
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        rd = msg.get("reasoning_details")
        if (isinstance(reasoning, str) and reasoning.strip()) or (isinstance(rd, list) and rd):
            return True
        codex_items = msg.get("codex_reasoning_items")
        if drop_codex_reasoning_items and isinstance(codex_items, list):
            return any(isinstance(item, dict) and item.get("type") == "reasoning" for item in codex_items)
        return False

    @staticmethod
    def _content_has_real_payload(content: Any) -> bool:
        """判断 assistant 的 content 中除了脱敏思考块和空白字符外，是否具有实际可见内容。"""
        if isinstance(content, str):
            return bool(content.strip())
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    if block:
                        return True
                    continue
                btype = block.get("type")
                if btype == "text":
                    text = block.get("text", "")
                    if isinstance(text, str) and text.strip():
                        return True
                elif btype not in {"thinking", "redacted_thinking"}:
                    return True  # tool_use, image, document 等真正有效载荷
            return False
        return content is not None and content != ""

    _drop_thinking_only_and_merge_users = _forward_static("agent.agent_runtime_helpers", "drop_thinking_only_and_merge_users")

    @staticmethod
    def _cap_delegate_task_calls(tool_calls: list) -> list:
        """Cap delegate_task calls in one turn at max_concurrent_children (non-delegate calls all kept);
        returns the original list when nothing was truncated.
        
        【限制单轮并发派发子 Agent 数量】
        防止大模型在单轮中一次性并发调起几十个 delegate_task 子任务拖垮系统。
        将单轮派发的子任务限制在 max_concurrent_children 之内，其余的保留其它常规工具调用。
        """
        from tools.delegate_tool import _get_max_concurrent_children
        max_children = _get_max_concurrent_children()
        delegate_count = sum(1 for tc in tool_calls if tc.function.name == "delegate_task")
        if delegate_count <= max_children:
            return tool_calls
        kept_delegates, truncated = 0, []
        for tc in tool_calls:
            if tc.function.name == "delegate_task":
                if kept_delegates >= max_children:
                    continue
                kept_delegates += 1
            truncated.append(tc)
        logger.warning("Truncated %d excess delegate_task call(s) to enforce "
                       "max_concurrent_children=%d limit", delegate_count - max_children, max_children)
        return truncated

    @staticmethod
    def _deduplicate_tool_calls(tool_calls: list) -> list:
        """Drop duplicate (tool_name, arguments) pairs in one turn (first wins). Valid JSON arguments are
        canonicalized so key order/whitespace can't evade dedup; returns the original list when nothing was removed.

        【单轮重复工具调用去重 / Tool Call Deduplication】
        当大模型在单轮中输出了多个具有相同工具名和完全相同参数的重复工具调用时，剔除多余项（保留首个）。
        参数通过规范化格式化（JSON key 排序且去除冗余空白），防止因字典键顺序不同或空格差异绕过去重逻辑。
        """
        seen, unique = set(), []
        for tc in tool_calls:
            arguments = tc.function.arguments
            try:
                arguments = json.dumps(json.loads(arguments), separators=(",", ":"), sort_keys=True)
            except (TypeError, ValueError):
                pass
            key = (tc.function.name, arguments)
            if key in seen:
                logger.warning("Removed duplicate tool call: %s", tc.function.name)
                continue
            seen.add(key)
            unique.append(tc)
        return unique if len(unique) < len(tool_calls) else tool_calls

    # 为每轮 assistant 工具调用分配确定性唯一 ID（策略归属于 message_sanitization）
    # 【核心不变量 / Prompt Cache Invariant】：冲突时追加确定性后缀 `_d<n>`，绝对不能生成随机 uuid4，
    # 确保相同历史生成的 Prompt 前缀字节级完全一致，维持大模型供应商的前缀缓存（Prompt Caching）命中率。
    _uniquify_tool_call_ids = staticmethod(_sanitize_uniquify_tool_call_ids)

    _repair_tool_call = _forward("agent.agent_runtime_helpers", "repair_tool_call")
    _invalidate_system_prompt = _forward("agent.system_prompt", "invalidate_system_prompt")

    # Codex Responses API ID 策略（agent.codex_responses_adapter）：API 省略 ID 时生成确定性 ID
    _deterministic_call_id = staticmethod(_codex_deterministic_call_id)
    _split_responses_tool_id = staticmethod(_codex_split_responses_tool_id)
    _derive_responses_function_call_id = staticmethod(_codex_derive_responses_function_call_id)

    _interruptible_api_call = _forward("agent.chat_completion_helpers", "interruptible_api_call")
    _interruptible_streaming_api_call = _forward("agent.chat_completion_helpers", "interruptible_streaming_api_call")
    _try_activate_fallback = _forward("agent.chat_completion_helpers", "try_activate_fallback")

    def _has_pending_fallback(self) -> bool:
        """检查后备模型链（fallback_chain）中是否还有未尝试的后备 Provider，防止虚报状态。"""
        return getattr(self, "_fallback_index", 0) < len(getattr(self, "_fallback_chain", None) or [])

    _restore_primary_runtime = _forward("agent.agent_runtime_helpers", "restore_primary_runtime")
    _try_recover_primary_transport = _forward("agent.agent_runtime_helpers", "try_recover_primary_transport")
    _build_api_kwargs = _forward("agent.chat_completion_helpers", "build_api_kwargs")

    def _set_tool_guardrail_halt(self, decision: ToolGuardrailDecision) -> None:
        """记录首个触发当前 turn 强制终止的工具防护栏决策。"""
        if decision.should_halt and self._tool_guardrail_halt_decision is None:
            self._tool_guardrail_halt_decision = decision

    def _toolguard_controlled_halt_response(self, decision: ToolGuardrailDecision) -> str:
        """生成展示给用户的友好停机说明，告知具体哪个工具因为持续无进展而被主动停止。"""
        return (
            f"I stopped retrying because I kept running {decision.tool_name or 'the same tool'} "
            f"{decision.count} times without making progress. The last result above shows what "
            "blocked it. Tell me how you'd like to proceed, or send `continue` and I'll try a "
            "different approach."
        )

    def _append_guardrail_observation(self, tool_name: str, function_args: dict, function_result: str, *,
                                      failed: bool, tool_call_id: str = "") -> str:
        """【工具防死循环与停滞防护栏 / Anti-Stall Guardrails】
        在每次工具执行完毕后进行观测分析：
        1. 跟踪工具失败频率；
        2. 相同调用停滞防护：若检测到相同参数且返回完全相同文本的工具重复执行，
           使用存根（Result-reference stubbing）替换结果，防止模型陷入复读死循环并节省大量 Token；
        3. 连续失败超出阈值时，自动注入指导性 Guidance 或触发 halt 停机。
        """
        decision = self._tool_guardrails.after_call(tool_name, function_args, function_result, failed=failed)
        stall_notice = result_stub = None
        if self._stall_guards_enabled():
            try:
                observation = self._tool_guardrails.observe_call(
                    tool_name, function_args, function_result if isinstance(function_result, str) else None,
                    tool_call_id=tool_call_id, failed=failed,
                )
                stall_notice, result_stub = observation.notice, observation.stub
            except Exception as exc:
                logger.debug("stall-guard identical-call observation failed: %s", exc)
        # 结果存根化替换：多次相同调用的相同结果直接替换为存根
        if result_stub and isinstance(function_result, str):
            function_result = result_stub
        if decision.action in {"warn", "halt"}:
            function_result = append_toolguard_guidance(function_result, decision)
        if decision.should_halt:
            self._set_tool_guardrail_halt(decision)
        else:
            streak_halt = self._tool_guardrails.halt_decision
            if streak_halt is not None and streak_halt.code in ("identical_call_streak_halt", "identical_cycle_halt"):
                function_result = append_toolguard_guidance(function_result, streak_halt)
                self._set_tool_guardrail_halt(streak_halt)
        if stall_notice:
            function_result = (function_result or "") + "\n\n" + stall_notice
        return function_result

    def _stall_guards_enabled(self) -> bool:
        """检查是否开启了运行时反停滞防护栏开关。"""
        return bool(getattr(self, "_stall_guards", True))

    def _guardrail_block_result(self, decision: ToolGuardrailDecision) -> str:
        self._set_tool_guardrail_halt(decision)
        return toolguard_synthetic_result(decision)

    def _execute_tool_calls(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """Execute the assistant's tool calls and append results to ``messages``.

        The segment planner splits the batch into runs of parallel-safe calls (read-only, non-overlapping file
        targets, opted-in MCP) separated by sequential barriers, run in emission order.

        【分段并发工具调度器 / Segment Planner & Tool Dispatcher】
        Hermes 工具执行的核心调度逻辑：
        1. 单个工具调用直接在当前线程串行执行；
        2. 多个工具调用时，调用 `_plan_tool_batch_segments` 分段规划器：
           - 自动分析工具只读性（如只读文件搜索、网页抓取）；
           - 自动分析文件修改路径（无冲突的文件写操作可并发，同路径写操作强制串行）；
           - 区分 MCP 工具并发支持；
        3. 将工具执行切分为“并发执行段”与“串行同步栅栏”，兼顾吞吐效率与副作用安全。
        """
        tool_calls = assistant_message.tool_calls
        args = (assistant_message, messages, effective_task_id, api_call_count)
        self._executing_tools = True
        try:
            with scoped_connection_surface(agent_connection_surface(self)):
                if len(tool_calls) <= 1:
                    self._execute_tool_calls_sequential(*args)
                else:
                    from agent.tool_dispatch_helpers import _plan_tool_batch_segments
                    active_env = get_active_env(effective_task_id)
                    exec_cwd = Path(active_env.cwd) if active_env is not None and active_env.cwd else None
                    segments = _plan_tool_batch_segments(tool_calls, execution_cwd=exec_cwd)
                    if len(segments) == 1:
                        run = self._execute_tool_calls_concurrent if segments[0][0] == "parallel" else self._execute_tool_calls_sequential
                        run(*args)
                    else:
                        from agent.tool_executor import execute_tool_calls_segmented
                        execute_tool_calls_segmented(self, *args, segments=segments)
        finally:
            self._executing_tools = False
        # getattr: test stubs built without _set_defaults drive this method too
        if getattr(self, "_trim_after_tool_batch", False):
            # Only on normal completion: every executor frame that held a >=1 MB raw result has
            # unwound and just the spilled preview lives in ``messages``. An in-flight exception
            # would pin those frames via its traceback, so that path leaves the flag for the
            # next completed batch (agent/tool_executor.py, #70684).
            self._trim_after_tool_batch = False
            from hermes_cli.mem_trim import trim_memory
            trim_memory(reason="large tool result")

    def _dispatch_delegate_task(self, function_args: dict) -> str:
        """Single call site for delegate_task dispatch; new DELEGATE_TASK_SCHEMA fields are added only here.
        
        【子 Agent 任务派发总入口 / Subagent Task Delegation】
        顶层模型发起的委托始终在后台异步运行（立即返回 handle，后续通过消息管道获取结果）；
        编排型子 Agent（深度 > 0）保持同步执行，以便在本轮中拿到结果汇总结算。
        """
        from tools.delegate_tool import _strip_model_hidden_task_fields, delegate_task as _delegate_task
        return _delegate_task(
            goal=function_args.get("goal"), context=function_args.get("context"),
            tasks=_strip_model_hidden_task_fields(function_args.get("tasks")),
            max_iterations=function_args.get("max_iterations"), role=function_args.get("role"),
            background=not (getattr(self, "_delegate_depth", 0) > 0), images=function_args.get("images"),
            action=function_args.get("action"),
            subagent_id=function_args.get("subagent_id"), message=function_args.get("message"), parent_agent=self,
        )

    _invoke_tool = _forward("agent.agent_runtime_helpers", "invoke_tool")

    @staticmethod
    def _wrap_verbose(label: str, text: str, indent: str = "     ") -> str:
        """将详细工具输出按终端宽度自动换行（带缩进对齐）。"""
        import shutil, textwrap
        wrap_width = max(40, shutil.get_terminal_size((120, 24)).columns - len(indent))
        out_lines: list[str] = []
        for raw_line in text.split("\n"):
            if len(raw_line) <= wrap_width:
                out_lines.append(raw_line)
            else:
                out_lines.extend(textwrap.wrap(raw_line, width=wrap_width, break_long_words=True, break_on_hyphens=False) or [raw_line])
        return f"{indent}{label}" + ("\n" + indent).join(out_lines)

    _execute_tool_calls_concurrent = _forward("agent.tool_executor", "execute_tool_calls_concurrent")
    _execute_tool_calls_sequential = _forward("agent.tool_executor", "execute_tool_calls_sequential")
    _handle_max_iterations = _forward("agent.chat_completion_helpers", "handle_max_iterations")

    def _conversation_root_id(self) -> Optional[str]:
        """Session-lineage ROOT id for Portal usage attribution, so one conversation keeps a single
        ``conversation=`` tag across compression rotation; subagents resolve via ``_parent_session_id``.

        【会话根 ID 血统追踪 / Conversation Lineage Tracking】
        用于 Portal 平台计费与用量聚合。使一个长期长生命周期的对话在经历多次上下文压缩、重置后，
        始终维持同一个统一的 conversation 根标签；子 Agent 则自动通过 parent_session_id 向上追溯到根。
        """
        cached = getattr(self, "_cached_conversation_root", None)
        if cached:
            return str(cached)
        sid = getattr(self, "session_id", None)
        if not sid:
            return None
        start = getattr(self, "_parent_session_id", None) or sid
        db = getattr(self, "_session_db", None)
        if db is None:
            return start
        try:
            return db.get_conversation_root(start) or start
        except Exception:
            logger.debug("Conversation root lineage walk failed", exc_info=True)
            return start


# 【预定义基础工具集与复合工具集分类 / Builtin Toolsets】
_BASIC_TOOLSETS = {"web", "terminal", "vision", "creative", "reasoning"}
_COMPOSITE_TOOLSETS = {"research", "development", "analysis", "content_creation", "full_stack"}
_LIST_TOOLS_USAGE = """
💡 Usage Examples:
  # Use predefined toolsets
  python run_agent.py --enabled_toolsets=research --query='search for Python news'
  python run_agent.py --enabled_toolsets=development --query='debug this code'
  python run_agent.py --enabled_toolsets=safe --query='analyze without terminal'

  # Combine multiple toolsets
  python run_agent.py --enabled_toolsets=web,vision --query='analyze website'

  # Disable toolsets
  python run_agent.py --disabled_toolsets=terminal --query='no command execution'

  # Run with trajectory saving enabled
  python run_agent.py --save_trajectories --query='your question here'"""


def _print_tool_listing() -> None:
    """``--list_tools``: 格式化打印当前系统已安装的所有基础工具集、复合工具集、场景工具集及具体独立工具列表。"""
    from model_tools import get_all_tool_names, get_available_toolsets
    from toolsets import get_all_toolsets, get_toolset_info

    print("📋 Available Tools & Toolsets:")
    print("-" * 50)
    print("\n🎯 Predefined Toolsets (New System):")
    print("-" * 40)
    basic_toolsets, composite_toolsets, scenario_toolsets = [], [], []
    for name in get_all_toolsets():
        info = get_toolset_info(name)
        if info:
            bucket = basic_toolsets if name in _BASIC_TOOLSETS else composite_toolsets if name in _COMPOSITE_TOOLSETS else scenario_toolsets
            bucket.append((name, info))
    print("\n📌 Basic Toolsets:")
    for name, info in basic_toolsets:
        print(f"  • {name:15} - {info['description']}")
        print(f"    Tools: {', '.join(info['resolved_tools']) if info['resolved_tools'] else 'none'}")
    print("\n📂 Composite Toolsets (built from other toolsets):")
    for name, info in composite_toolsets:
        print(f"  • {name:15} - {info['description']}")
        print(f"    Includes: {', '.join(info['includes']) if info['includes'] else 'none'}")
        print(f"    Total tools: {info['tool_count']}")
    print("\n🎭 Scenario-Specific Toolsets:")
    for name, info in scenario_toolsets:
        print(f"  • {name:20} - {info['description']}")
        print(f"    Total tools: {info['tool_count']}")
    print("\n📦 Legacy Toolsets (for backward compatibility):")
    for name, info in get_available_toolsets().items():
        print(f"  {'✅' if info['available'] else '❌'} {name}: {info['description']}")
        if not info["available"]:
            print(f"    Requirements: {', '.join(info['requirements'])}")
    all_tools = get_all_tool_names()
    print(f"\n🔧 Individual Tools ({len(all_tools)} available):")
    for tool_name in sorted(all_tools):
        print(f"  📌 {tool_name} (from {get_toolset_for_tool(tool_name)})")
    print(_LIST_TOOLS_USAGE)


def _parse_toolset_arg(raw: Optional[str], label: str) -> Optional[List[str]]:
    """解析以逗号分隔的工具集命令行参数 -> 列表。"""
    if not raw:
        return None
    names = [t.strip() for t in raw.split(",")]
    print(f"{label}: {names}")
    return names


def _save_sample_trajectory(agent: "AIAgent", result: dict, user_query: str, model: str) -> None:
    """``--save_sample``: 将单次对话轨迹按照 benchmark / batch_runner 格式保存为 UUID 命名的 JSON 样本文件。"""
    sample_filename = f"sample_{str(uuid.uuid4())[:8]}.json"
    entry = {
        "conversations": agent._convert_to_trajectory_format(result['messages'], user_query, result['completed']),
        "timestamp": datetime.now().isoformat(), "model": model, "completed": result['completed'], "query": user_query,
    }
    try:
        with open(sample_filename, "w", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, indent=2))
        print(f"\n💾 Sample trajectory saved to: {sample_filename}")
    except Exception as e:
        print(f"\n⚠️ Failed to save sample: {e}")


def main(
    query: str = None, model: str = "", api_key: str = None, base_url: str = "", max_turns: int = 10,
    enabled_toolsets: str = None, disabled_toolsets: str = None, list_tools: bool = False,
    save_trajectories: bool = False, save_sample: bool = False, verbose: bool = False, log_prefix_chars: int = 20,
):
    """
    Main function for running the agent directly via CLI.
    
    【直接运行 Agent 的 CLI 主函数】
    直接通过命令行启动 AIAgent 调试或执行单任务查询。

    Args:
        query (str): 自然语言提示词（默认展示 Python 3.13 特性查询示例）。
        model (str): 模型名称（如 openrouter 格式 provider/model 或 Claude/GPT-5）。
        api_key (str): 认证 API Key（默认优先从环境变量读取）。
        base_url (str): API 端点 base_url。
        max_turns (int): 最大 API 迭代轮数限制（默认 10 轮）。
        enabled_toolsets (str): 启用的工具集（如 'research', 'development', 'web,vision'）。
        disabled_toolsets (str): 禁用的工具集（如 'terminal'）。
        list_tools (bool): 仅列出可用工具并退出。
        save_trajectories (bool): 追加保存轨迹到 trajectory_samples.jsonl。
        save_sample (bool): 将本次会话保存为独立的 sample_xxxx.json 文件。
        verbose (bool): 启用详细调试日志。
        log_prefix_chars (int): 工具日志预览截断字数。
    """
    print("🤖 AI Agent with Tool Calling")
    print("=" * 50)
    if list_tools:
        return _print_tool_listing()

    enabled_toolsets_list = _parse_toolset_arg(enabled_toolsets, "🎯 Enabled toolsets")
    disabled_toolsets_list = _parse_toolset_arg(disabled_toolsets, "🚫 Disabled toolsets")
    if save_trajectories:
        print("💾 Trajectory saving: ENABLED")
        print("   - Successful conversations → trajectory_samples.jsonl")
        print("   - Failed conversations → failed_trajectories.jsonl")

    try:
        agent = AIAgent(
            base_url=base_url, model=model, api_key=api_key, max_iterations=max_turns,
            enabled_toolsets=enabled_toolsets_list, disabled_toolsets=disabled_toolsets_list,
            save_trajectories=save_trajectories, verbose_logging=verbose, log_prefix_chars=log_prefix_chars,
        )
    except RuntimeError as e:
        print(f"❌ Failed to initialize agent: {e}")
        return

    user_query = query if query is not None else ("Tell me about the latest developments in Python 3.13 and what new features "
                                                  "developers should know about. Please search for current information and try it out.")
    print(f"\n📝 User Query: {user_query}")
    print("\n" + "=" * 50)

    result = agent.run_conversation(user_query)

    print("\n" + "=" * 50 + "\n📋 CONVERSATION SUMMARY\n" + "=" * 50)
    print(f"✅ Completed: {result['completed']}\n📞 API Calls: {result['api_calls']}\n💬 Messages: {len(result['messages'])}")
    if result['final_response']:
        print("\n🎯 FINAL RESPONSE:\n" + "-" * 30 + "\n" + result['final_response'])
    if save_sample:
        _save_sample_trajectory(agent, result, user_query, model)
    print("\n👋 Agent execution completed!")


if __name__ == "__main__":
    import fire
    fire.Fire(main)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# 【外部插件向后兼容层 / PEP 562 Backward Compatibility Layer】
# 在 2026 年 9 月的代码大重构中，原先直接位于 `run_agent.py` 的许多内部函数与类被拆分到了各个专业模块中。
# 为了避免外部生态已发布的第三方插件在更新时因 ImportError 崩溃，
# 此处通过 Python PEP 562 的 `__getattr__` 机制维护了一个惰性转发字典：
# 当外部插件从 run_agent 导入旧名称时，动态加载并返回新模块的对应属性，同时发出一次性弃用警告（warn_once）。
# 内部代码严禁使用这些旧导出（CI 会自动运行 check_compat_pointers.py 进行静态检查）。
from types import SimpleNamespace  # noqa: F401,E402
import asyncio  # noqa: F401,E402
import base64  # noqa: F401,E402
import copy  # noqa: F401,E402
import hashlib  # noqa: F401,E402
import tempfile  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'COMPRESSED_SUMMARY_METADATA_KEY': ('agent.context_compressor', 'COMPRESSED_SUMMARY_METADATA_KEY'),
    'ContextCompressor': ('agent.context_compressor', 'ContextCompressor'),
    'DEFAULT_AGENT_IDENTITY': ('agent.prompt_builder', 'DEFAULT_AGENT_IDENTITY'),
    'FailoverReason': ('agent.error_classifier', 'FailoverReason'),
    'OpenAI': ('agent.process_bootstrap', 'OpenAI'),
    'atomic_json_write': ('utils', 'atomic_json_write'),
    'build_context_files_prompt': ('agent.prompt_builder', 'build_context_files_prompt'),
    'build_environment_hints': ('agent.prompt_builder', 'build_environment_hints'),
    'build_skills_system_prompt': ('agent.prompt_builder', 'build_skills_system_prompt'),
    'check_toolset_requirements': ('model_tools', 'check_toolset_requirements'),
    'convert_scratchpad_to_think': ('agent.trajectory', 'convert_scratchpad_to_think'),
    'estimate_request_tokens_rough': ('agent.model_metadata', 'estimate_request_tokens_rough'),
    'file_mutation_result_landed': ('agent.tool_result_classification', 'file_mutation_result_landed'),
    'flatten_message_text': ('agent.message_content', 'flatten_message_text'),
    'get_tool_definitions': ('model_tools', 'get_tool_definitions'),
    'handle_function_call': ('model_tools', 'handle_function_call'),
    'is_truthy_value': ('utils', 'is_truthy_value'),
    'jittered_backoff': ('agent.retry_utils', 'jittered_backoff'),
    'load_soul_md': ('agent.prompt_builder', 'load_soul_md'),
    'normalize_usage': ('agent.usage_pricing', 'normalize_usage'),
    'redact_sensitive_text': ('agent.redact', 'redact_sensitive_text'),
    'request_hard_interrupt': ('agent.interrupt_compat', 'request_hard_interrupt'),
    'sanitize_context': ('agent.memory_manager', 'sanitize_context'),
    'user_originated_turn_view': ('agent.context_compressor', 'user_originated_turn_view'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
