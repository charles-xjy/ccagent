# Nano Claude Code

一个基于 **LangGraph** 实现的轻量级 AI 编程助手，采用 Manager/Executor 多智能体架构，具备完整的三层记忆系统（Redis 热缓存 + 语义记忆提炼 + MySQL 冷归档）、人机交互确认机制和上下文自动压缩能力。

---

## 目录

- [项目概述](#项目概述)
- [架构设计](#架构设计)
  - [主图：Manager](#主图manager)
  - [子 Agent 专家团队](#子-agent-专家团队)
- [核心功能](#核心功能)
  - [意图分析与需求文档生成](#1-意图分析与需求文档生成)
  - [三层记忆系统](#2-三层记忆系统)
  - [上下文自动压缩](#3-上下文自动压缩)
  - [人机交互确认机制](#4-人机交互确认机制)
  - [会话管理与恢复](#5-会话管理与恢复)
- [记忆管理系统](#记忆管理系统)
  - [启动方式](#启动方式)
  - [菜单结构](#菜单结构)
  - [检查点浏览](#检查点浏览)
  - [检查点删除](#检查点删除)
  - [长期记忆管理](#长期记忆管理)
  - [新旧格式兼容](#新旧格式兼容)
- [项目优点](#项目优点)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [技术栈](#技术栈)

---

## 项目概述

Nano Claude Code 是对 Anthropic Claude Code 核心机制的轻量复现与创新延伸。它不是简单的 LLM 聊天封装，而是一个**可执行复杂多步骤编程任务**的 AI Agent 系统：

- Manager 负责**规划与协调**，不直接写代码
- 专家子 Agent 各司其职，专注于自己的领域
- 每次任务结束后自动**提炼长期记忆**，下次会话直接继承
- 对话退出时**自动归档到 MySQL**，Redis 过期后仍可恢复
- 上下文接近溢出时**自动压缩**，可连续工作超长任务

---

## 架构设计

### 主图：Manager

```
START
  │
  ▼
intent_analysis  ──── 需求确认 Q&A + 生成需求文档（仅新会话）
  │
  ▼
agent (Manager)  ──── 规划任务、协调专家
  │
  ├─ task_tool(coder)       → coder 子图
  ├─ task_tool(researcher)  → researcher 子图
  ├─ task_tool(reviewer)    → reviewer 子图
  ├─ todo_manager           → tools → plan_confirm（首次规划）
  │                                 → token 路由
  │
  └─ (无 tool_call)         → END
       ↑
       └─ [warn] ──── 用户选择 ────┐
       └─ [compress] ─────────────┘
```

Manager 通过 `task_tool` 虚拟工具向专家发出指令，条件边在运行时拦截并路由到对应子图节点，Manager 本身**不执行任何工具**（除任务面板管理）。

### 子 Agent 专家团队

| Agent | 职责 | 工具 |
|-------|------|------|
| **Intent-Analyst** | 需求确认问答、生成结构化需求文档 | 无（纯 LLM 对话 + `interrupt()` 交互） |
| **Coder** | 编写、修改代码文件 | `read_file`、`write_file`、`edit_file`、`bash`（只读） |
| **Tech-Researcher** | 技术调研，查文档、搜索 | `web_search`、`langchain_docs`、`GitHub` 系列、`fetch`、`read_file` |
| **Reviewer** | 代码审查、测试运行 | `read_file`、`run_in_sandbox`、`run_python_test`、`check_code_style`、`run_bash_command` |

Coder、Tech-Researcher、Reviewer 三个子 Agent 各是独立的 LangGraph 子图，拥有**独立的 Redis checkpoint**，互不干扰。Intent-Analyst 是 Manager 主图中的内联节点，与 Manager 共享同一 thread_id，不单独占用 checkpoint。

---

## 核心功能

### 1. 意图分析与需求文档生成

任务开始前，`intent_analysis` 节点与用户进行多轮对话，确认需求细节：

1. AI 提出 2-4 个关键问题（功能、架构、技术选型等）
2. 用户逐一回答，直到 AI 判断信息充足（输出 `[READY]` 标记）
3. AI 自动生成结构化 Markdown 需求文档
4. 用户确认文档或提供修改意见（可多轮修订）
5. 文档保存到项目目录，作为 Manager 的执行依据

> 恢复旧会话时自动跳过此步骤，直接进入 Manager。

### 2. 三层记忆系统

仿照 Claude Code 的记忆架构，三层分工明确：

```
第一层：热缓存 (Redis Checkpoint)
  └── LangGraph AsyncRedisSaver → Redis (TTL 7天)
      每次节点执行后自动快照完整 AgentState
      7天内回访直接毫秒级恢复，无需查询 MySQL

第二层：语义记忆 (LongTermMemory)
  └── Redis (ltm:mem:* 键空间)
      从对话中提炼结构化知识，跨会话注入背景信息
      四种类型: user / feedback / project / reference

第三层：冷归档 (MySQL)
  └── MemoryStore → MySQL conversations 表
      对话退出时自动归档完整消息历史
      Redis checkpoint 过期后，从这里恢复对话上下文
      恢复后自动重建 Redis checkpoint，TTL 重新计时
```

**数据流**：

```
对话中:   LLM → State → AsyncRedisSaver → Redis (每步快照)
退出时:   Redis checkpoint → 语义提炼 → Redis ltm:mem:*
          Redis checkpoint → 消息序列化 → MySQL UPSERT
          Redis checkpoint 保留不删 (7天 TTL 自然淘汰)
再进入:   先查 Redis → 命中 → LangGraph 自动恢复 (热路径)
                    → 未命中 → MySQL loads() → 注入 State → 重建 Redis (冷路径)
```

**长期记忆** 支持 4 种类型，完整映射用户画像：

| 类型 | 内容 | 示例 |
|------|------|------|
| `user` | 用户偏好、知识背景 | "用户偏好 Python，排斥冗余注释" |
| `feedback` | AI 行为被纠正的规则（含 Why） | "删文件前必须确认，曾因此丢失工作" |
| `project` | 当前项目决策、约束、目标 | "正在重构 Manager/Executor 架构" |
| `reference` | 外部资源地址 | "Redis 服务地址 10.x.x.x:6379" |

每次会话退出时，LLM 自动分析最后 40 条消息并将有价值的信息提炼为记忆条目持久化至 Redis；下次会话启动时自动注入 Manager 的 system prompt。

### 3. 上下文自动压缩

每个 Agent 的 `tools` 节点执行完毕后，触发三档 token 路由：

```
token 用量检测
  │
  ├── < 50%   → agent（正常继续）
  ├── 50~92%  → warn 节点（interrupt 暂停，用户选择是否压缩）
  │              ⚠️  50~75% 黄色警告
  │              🚨  75~92% 红色警告
  └── ≥ 92%   → compress 节点（自动压缩，无需确认）
```

压缩分两阶段执行：

- **Phase 1**：删除旧 `ToolMessage`（体积大、语义价值低）
- **Phase 2**：若仍超阈值，调用 LLM 生成 8 段式摘要替换全部旧消息

8 段式摘要涵盖：主要请求、关键概念、文件路径与行号、错误与修复、解决过程、用户消息摘录、待处理任务、当前状态与下一步计划。

### 4. 人机交互确认机制

系统在 5 个关键节点使用 LangGraph `interrupt()` 暂停，等待用户输入：

| 场景 | 触发标志 | 用户操作 |
|------|----------|----------|
| 需求确认问答 | `❓` | 回答问题 / 回车完成 / `skip` 跳过 |
| 需求文档确认 | `📄` | 确认 / 提供修改意见 / 跳过 |
| 任务规划确认 | `📋` | 确认开始 / 提供反馈重新规划 / 取消 |
| 危险工具确认 | `🔐` | 确认执行 / 跳过 / 中止任务 |
| 上下文压缩提醒 | token 警告 | 立即压缩 / 跳过继续 |

**危险工具** 在执行前强制请求用户授权：

- Coder Agent：`write_file`、`edit_file`
- Reviewer Agent：`run_in_sandbox`、`run_python_test`

### 5. 会话管理与恢复

启动时列出所有历史会话（从 Redis 动态扫描 + MySQL 归档），支持：

- **恢复 Redis 会话**：自动检测未完成任务（`pending` / `in_progress` 状态），提示用户选择是否自动继续。LangGraph 从 checkpoint 自动恢复完整 State。
- **从 MySQL 恢复**：Redis checkpoint 过期后，自动列出 MySQL 中已归档的会话，选择后从 MySQL 加载完整消息历史，注入 State 并重建 Redis checkpoint（TTL 重新计时 7 天）。
- **新建会话**：输入项目名称，生成带时间戳的唯一 session ID（格式 `session_{project}_{timestamp}`）。
- **会话隔离**：Manager 与每个子 Agent 使用独立 thread_id，互不污染。仅主 Manager 会话归档到 MySQL。

---

## 记忆管理系统

`memory/memory_manager.py` 是一个独立的交互式 CLI 工具，用于在**不启动主程序**的情况下直接浏览、检查和清理 Redis 中存储的所有记忆数据。

### 启动方式

```bash
python nano_claude_code/memory/memory_manager.py
```

### 菜单结构

```
主菜单（动态列出所有会话 + 长期记忆入口）
  │
  ├── <session_项目名_YYYYMMDD_HHMM>        ← 新格式会话
  │     └── 选择 Agent
  │           ├── 主 Agent (Manager)
  │           ├── 编程专家 (Coder)
  │           ├── 调研专家 (Researcher)
  │           ├── 代码审核专家 (Reviewer)
  │           └── [每个 Agent 下]
  │                 ├── 查看记忆（5 种浏览方式）
  │                 └── 删除记忆（3 种删除方式）
  │
  ├── [旧] <legacy_thread_id>               ← 旧格式会话（向后兼容）
  │     ├── 查看记忆
  │     └── 删除记忆
  │
  ├── 长期记忆管理
  │     ├── 查看全部记忆（按类型分组展示）
  │     ├── 删除指定记忆（方向键选择）
  │     └── 清空全部
  │
  └── 退出
```

> 所有菜单均使用方向键交互，用 `q` 返回上级，无需记忆数字编号。

### 检查点浏览

进入任意 Agent 的「查看记忆」后，提供 5 种查看方式：

| 选项 | 功能 |
|------|------|
| 查看全部消息 | 展示该 Agent 最新检查点的所有消息 |
| 查看最后 N 条 | 输入数字，显示最近 N 条（聚焦最新对话） |
| 查看前 N 条 | 显示最早的 N 条（了解任务起点） |
| 查看指定范围 (X~Y) | 精确定位消息区间，适合大上下文的局部检查 |
| 选择某个检查点查看 | 列出全部历史检查点（含消息数），可选择任意时间点的快照 |

消息展示格式：
```
────────────────────────────────────────────────────────────
#12 [AI] [工具调用: write_file]
────────────────────────────────────────────────────────────
<消息内容>
```

### 检查点删除

3 种删除策略，满足不同清理需求：

| 策略 | 适用场景 |
|------|----------|
| 删除指定检查点 | 精确删除某个时间点的快照，不影响其他检查点 |
| 保留最近 N 个 | 清理历史积累，释放 Redis 空间，同时保留近期上下文 |
| 删除全部 | 彻底清空某个 Agent 的所有记忆，从头开始 |

删除操作均有**二次确认**弹窗，防止误操作。删除完成后显示实际清理的 Redis 键数量。

### 长期记忆管理

长期记忆独立于检查点，存储于 `ltm:mem:*` 键空间，按 4 种类型分组展示：

```
【用户信息】
  ID=a1b2c3d4  name=user-prefers-python
  描述: 用户偏好 Python，排斥冗余注释
  内容: 用户偏好使用 Python 编写脚本，且明确要求不写多行注释或 docstring。
  创建: 2026-05-14T10:23:11

【行为反馈】
  ID=e5f6g7h8  name=confirm-before-delete
  描述: 删除文件前必须获得用户确认
  内容: 删除任何文件前须先展示文件路径并请求确认。Why: 曾因误删丢失工作...
  创建: 2026-05-14T15:07:42
```

支持按索引**精确删除单条**记忆（方向键选择），或一键**清空全部**。

### 新旧格式兼容

系统支持两种 session ID 格式，自动区分并分类展示：

| 格式 | 示例 | 说明 |
|------|------|------|
| 新格式 | `session_weather_agent_20260514_1023` | 按项目名+时间戳命名，支持子 Agent 分级查看 |
| 旧格式 | `manager_executor_v2` | 早期硬编码 thread_id，直接操作 |

启动时通过 `r.keys("*")` 全量扫描 Redis，自动提取存在的会话，不依赖外部索引，即使 Redis 中有孤立键也能正确展示。

---

## 项目优点

### 清晰的职责边界

Manager 只规划和协调，不写代码；每个子 Agent 专注于自己的领域，工具集严格隔离。这种分工使得每个节点的行为可预测、可调试，避免了单一 Agent 权限过大的安全风险。

### 安全优先的设计

- 危险工具（文件写入、代码执行）在执行前**强制人工确认**
- Reviewer 使用**隔离 venv 沙箱**执行代码，不污染宿主环境
- Coder 被明确禁止运行脚本和测试，只能操作文件系统
- 工具输出超过 10000 字符时自动截断，防止 prompt 污染

### 真正的跨会话记忆

不依赖外部向量数据库，使用 Redis + MySQL + LLM 提炼实现三层记忆。语义记忆在 session 结束时自动提炼，完整对话历史自动归档到 MySQL。Redis 过期后可从 MySQL 恢复对话上下文，下次会话启动时自动注入，用户无需重复说明项目背景和偏好。

### 智能的上下文管理

两阶段压缩策略（先删工具输出再摘要）参考 Claude Code 的设计，最大限度保留语义价值密度高的内容。三档路由（正常/警告/自动压缩）平衡了用户控制权与系统鲁棒性，支持超长任务不中断执行。

### 开放的工具扩展能力

Researcher Agent 通过 **MCP 协议**动态加载工具，启动时自动连接 MCP 服务器（LangChain 官方文档、GitHub 等），工具不可用时优雅降级，不影响核心功能。新增工具只需注册到对应 Agent，无需修改路由逻辑。

### 需求驱动的执行模式

意图分析节点在任务执行前捕获并澄清需求，将模糊指令转化为结构化需求文档，显著降低因需求理解偏差导致的返工成本。文档持久化保存，既是开发依据，也是项目文档。

### 可观测的执行过程

每个节点更新都会实时打印到终端，包括当前执行的子 Agent、调用的工具名称、AI 思考内容预览等。图结构在启动时自动导出为 `manager_graph.png`（或 `manager_graph.mmd`），便于理解和调试整体流程。

---

## 项目结构

```
nano_claude_code/
├── main.py                    # 主程序：Manager 图构建与交互主循环
├── core/
│   ├── state.py               # AgentState 定义（messages, current_todo 等）
│   ├── config.py              # 模型/Redis/MySQL 配置
│   ├── tools.py               # 共用工具：read_file，WORKDIR 定义
│   ├── memory_store.py        # MySQL 归档存储（archive/load/delete/list）
│   ├── confirm.py             # 危险工具确认节点工厂函数
│   └── prompt_ui.py           # 方向键交互 select 组件
├── intent_agent/
│   └── agent.py               # 意图分析节点：Q&A + 需求文档生成
├── coder_agent/
│   ├── agent.py               # Coder 子图：文件读写
│   └── tools.py               # bash, write_file, edit_file
├── researcher_agent/
│   ├── agent.py               # Researcher 子图：调研（含工具轮次限制）
│   └── tools.py               # web_search, MCP 工具加载
├── reviewer_agent/
│   ├── agent.py               # Reviewer 子图：代码审查
│   └── tools.py               # run_in_sandbox, run_python_test, check_code_style
├── memory/
│   ├── long_term_memory.py    # LongTermMemory：Redis 存取 + LLM 提炼
│   ├── compressor.py          # 两阶段上下文压缩 + 三档 token 路由
│   ├── memory_manager.py      # 记忆管理 CLI（查看/删除检查点和长期记忆）
│   └── __init__.py            # 统一导出压缩相关函数
├── MEMORY_DESIGN.md           # 记忆系统详细设计文档
├── README.md                  # 本文件
└── ../docs/
    └── memory_architecture.md # 三层记忆架构完整文档（含数据流、表结构、维护指南）
```

---

## 快速开始

**前置依赖：**

- Python 3.11+
- Redis 实例（用于 Checkpoint 和长期记忆）
- MySQL 8.0（用于对话归档持久化，可选，不可用时自动降级）
- 兼容 OpenAI API 的 LLM 服务（如 vLLM 部署的 Qwen 模型）

**安装与运行：**

```bash
# 安装依赖
pip install langchain langgraph langchain-core redis python-dotenv aiomysql

# 启动 Redis 和 MySQL（Docker 方式）
docker run -d --name redis-stack-server -p 6379:6379 redis/redis-stack-server:latest
docker run -d --name mysql-ccagent --restart always -p 3306:3306 \
  -e MYSQL_ALLOW_EMPTY_PASSWORD=yes -e MYSQL_DATABASE=ccagent mysql:8.0

# 修改 core/config.py 中的 Redis/MySQL 连接地址
# 启动
python nano_claude_code/main.py
```

**记忆管理工具：**

```bash
python nano_claude_code/memory/memory_manager.py
```

---

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| Agent 框架 | LangGraph (StateGraph + Checkpoint + interrupt) |
| LLM 接入 | LangChain + OpenAI 兼容接口（vLLM） |
| 热缓存 | Redis (AsyncRedisSaver, TTL 7天, checkpoint 自动快照) |
| 语义记忆 | Redis (LongTermMemory, ltm:mem:* 键空间, LLM 自动提炼) |
| 冷归档 | MySQL 8.0 (MemoryStore, aiomysql, 完整消息历史持久化) |
| 工具扩展 | MCP 协议（Model Context Protocol） |
| 异步运行时 | Python asyncio（Windows SelectorEventLoop） |
| 交互界面 | 终端 ANSI 颜色 + 自定义方向键 select 组件 |
| 容器化 | Docker (Redis Stack Server + MySQL 8.0) |
