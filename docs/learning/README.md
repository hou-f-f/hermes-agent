# Hermes Agent 核心架构与原理学习路线

欢迎阅读 Hermes Agent 系统化进阶学习知识库。本项目由团队维护，旨在系统化剖析 Hermes 核心控制循环、提示词工程、工具分段调度、多进程租约管理及长期记忆体系。

> 📌 **项目协作必读**：所有参与本项目代码变更与学习文档编写的成员及 AI 助手，请首先阅读 [《项目学习与代码协作规范手册》](./项目学习与代码协作规范手册.md) 与项目根目录的 [`GEMINI.md`](../../GEMINI.md)。

---

## 从这里开始学习

先读[第 00 章：项目架构与源码导航](<00_项目架构与源码导航/architecture.md>)，建立系统地图，区分界面、后端、Agent 核心与扩展机制。

然后，有 Java 基础、还不熟悉 Python 的读者，请打开[第一章学习步骤与完成标准](<01_Hermes Agent Loop 内部机制深度学习手册/agent_loop_internals_manual.md#learning-route>)。按“门面 → 准入 → 主循环 → 状态传递 → 一次工具问答”的顺序读，语法解释就在章节对应位置。无需一开始通读三个大文件。

每一步都列出具体函数、阅读范围、暂时跳过的内容和检查点；完成四个练习并通过章末自查后，再进入下一专题。下表状态描述文档编写进度，不代表个人已经学会。

## 📚 学习专题目录

| 序号 | 专题模块 | 核心源码位置 | 学习文档与资源 | 状态 |
| :--- | :--- | :--- | :--- | :--- |
| **00** | **项目架构与源码导航** | 多入口、`agent/`、`tools/`、会话与插件 | [当前架构、数据流与学习检查点](<00_项目架构与源码导航/architecture.md>)<br/>[系统全景交互式可视化组件](00_项目架构与源码导航/architecture_visualizer.html) | 📝 文档已编写 |
| **01** | **Agent Loop 核心机制** | `run_agent.py`<br/>`agent/turn_facade.py`<br/>`agent/conversation_loop.py`<br/>`agent/turn_*.py` | [Agent Loop 内部机制深度学习手册](01_Hermes Agent Loop 内部机制深度学习手册/agent_loop_internals_manual.md)<br/>[交互式架构可视化组件](01_Hermes Agent Loop 内部机制深度学习手册/agent_loop_visualizer.html) | 📝 文档已编写 |
| **02** | **提示词组装与前缀缓存保活** | `agent/prompt_builder.py`<br/>`agent/system_prompt.py`<br/>`agent/prompt_caching.py` | *Prompt Caching is Sacred* 专题 | ⏳ 规划中 |
| **03** | **工具分段规划与安全调度** | `agent/turn_tool_round.py`<br/>`model_tools.py`<br/>`tools/registry.py` | *SegmentPlanner* 并发与串行屏障机制 | ⏳ 规划中 |
| **04** | **上下文压缩与微压缩体系** | `agent/context_engine.py`<br/>`agent/context_compressor.py`<br/>`agent/conversation_compression.py` | 三级自适应压缩架构 | ⏳ 规划中 |
| **05** | **状态持久化与长期记忆系统** | `hermes_state.py`<br/>`agent/turn_finalizer.py`<br/>`plugins/memory/` | SessionDB 增量入库与 Memory Sync Guard | ⏳ 规划中 |
| **06** | **多平台接入与网关通信** | `gateway/`<br/>`tui_gateway/`<br/>`apps/desktop/` | Gateway 会话管理与轻量断开机制 | ⏳ 规划中 |

---

## 🛠️ 本地学习与调试指南

1. **查看流程图与交互组件**：
   可以直接在支持 HTML 渲染的浏览器中打开 [`01_Hermes Agent Loop 内部机制深度学习手册/agent_loop_visualizer.html`](01_Hermes Agent Loop 内部机制深度学习手册/agent_loop_visualizer.html) 进行沉浸式交互学习。
2. **源码对应阅读**：
   - 门面与准入：[`run_agent.py`](../../run_agent.py) 和 [`agent/turn_facade.py`](../../agent/turn_facade.py)
   - 驱动与状态机：[`agent/conversation_loop.py`](../../agent/conversation_loop.py)
   - 执行阶段实现：[`agent/turn_*.py`](../../agent/)
