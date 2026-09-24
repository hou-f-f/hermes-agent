# Hermes Agent 项目学习与代码协作规范 (Antigravity Persistent Rules)

本文档是当前工作区（`hou-f-f/hermes-agent`）的最高优先级持久化规则文件。AI Assistant 在处理本项目的任何任务、代码修改或文档撰写时，**必须无条件严格遵循**以下规则：

---

## 1. 源码修改与中英文双语注释准则

1. **中英文双语对照，中文在前，原英文在后**：
   - 对已有英文 Docstring 或关键注释进行翻译注解时，**必须将中文讲解放置在前面，原版英文完整保留在后面**。
   - 严禁随意删除或覆盖原始的英文 Docstring 与技术注释。
2. **拒绝机械翻译，必须深度结合架构上下文**：
   - 严禁字面生硬的直译与无意义的机器翻译；
   - 翻译与讲解必须深度结合 Hermes Agent 系统的底层架构意图与核心概念（如 `Prompt Caching is Sacred`、`Durable Turn Lease`、`SegmentPlanner`、`Micro-compaction`、`Native Responses Compaction`、`Failover Restart` 等）。
3. **补充核心原理与机制剖析**：
   - 在翻译的基础上，必须追加对该函数/类在单轮生命周期（Turn Lifecycle）中所处位置、解决的问题痛点（如关联 issue 编号背景）、参数作用域及副作用的中文深入讲解。
4. **编译与语法校验铁律**：
   - 每次代码变更后，必须在终端执行 `python3 -m py_compile <modified_file>` 进行语法严格校验，确保退出码为 0，零语法损坏。

---

## 2. 章节学习手册与目录归档规范

1. **统一归档在 `docs/learning/` 目录下**：
   - 每一个学习主题/模块，必须在 `docs/learning/` 下建立独立的章节子目录，统一产出结构化的深入学习手册。
2. **章节文件夹命名必须使用中文**：
   - 章节文件夹命名格式示例：`docs/learning/01_Hermes Agent Loop 内部机制深度学习手册/`、`docs/learning/02_提示词工程与前缀缓存保活/` 等。
3. **手册内容结构标准**：
   - 必须包含：**架构演进背景**（指出官方旧文档滞后点）、**现代模块拓扑**、**高清彩色流程图**（Mermaid 高对比度现代配色）、**生命周期核心阶段拆解**、**设计不变量（Invariants）** 及 **重点源码阅读路径**。
   - 章节中的重要流程图可配套生成独立的交互式可视化组件（HTML Widget）。
4. **同步维护索引**：
   - 每完成一个学习章节，必须同步更新 [`docs/learning/README.md`](./docs/learning/README.md) 中的专题目录与学习路线状态。

---

## 3. Git 远程仓库与提交推送纪律

1. **远程仓库约定**：
   - `origin` 必须始终指向个人 Fork 仓库：`https://github.com/hou-f-f/hermes-agent.git`；
   - `upstream` 必须始终指向 NousResearch 官方主仓库：`https://github.com/NousResearch/hermes-agent.git`。
2. **代理环境维护**：
   - 必须保持针对 GitHub 的局部代理规则（`http.https://github.com.proxy http://127.0.0.1:6789`），杜绝因国内直连导致断流。
3. **提交与推送**：
   - 日常代码注释、重构、学习手册及文档均提交并推送到 `origin`（用户的 Fork 仓库）。
   - 拉取官方新特性使用 `git fetch upstream && git merge upstream/main`。

---

## 4. 系统级四大架构不变量（严禁破坏）

在修改或扩展任何核心代码时，绝不能违反 Hermes Agent 的四大钢铁设计戒条：
1. **Prompt Caching is Sacred（提示词缓存神圣不可侵犯）**：保持长会话 System Prompt 字节稳定；冻结工具声明顺序；多界面切换在请求末尾注入注记；工具 ID 碰撞消解必须使用确定性 `_d<n>` 后缀，**严禁引入随机 UUID**。
2. **Durable Multi-Process Turn Lease（跨进程持久化轮次租约）**：严禁单会话并发写入，依赖租约和心跳看门狗避免多端争抢。
3. **Memory Sync Invariant（记忆同步安全守卫）**：被打断（`interrupted=True`）的轮次**绝对禁止**同步至外部向量数据库（Mem0/Zep），严防脏上下文污染知识库。
4. **Segment Planner Concurrency（工具分段安全调度）**：只读安全工具方可并发，有写副作用或交互式工具（如 clarify）必须作为串行屏障同步执行。
