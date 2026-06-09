# CCAgent

一个基于 LangGraph 构建的生产级多 Agent 编程助手，核心设计目标是**长任务不飘逸、上下文不爆炸、跨会话有记忆**。

---

## 架构概览

```
用户输入
   │
   ▼
Supervisor（任务规划 + 路由调度）
   │
   ├──[需求不确定时]──► analyze_intent（Q&A 澄清 + 需求文档生成）
   │                         │
   │                    确认后继续规划
   │
   └──► spawn_agents（并行启动子 Agent）
            ├──► skills=[coder]      （编写/修改代码，执行 Shell）
            ├──► skills=[researcher] （技术调研，MCP 工具调用）
            └──► skills=[reviewer]   （代码审查，测试执行）
```

子 Agent 不再是固定的独立模块，而是通过 **skill 注入** 动态组装——`skills/` 目录下的 Markdown 文件定义角色身份和工具权限，多个 skill 可叠加（如 `["coder", "react-patterns"]`）。

`analyze_intent` 是 Supervisor 的一个工具，仅在需求边界不清晰时主动调用（如新功能目标明确但实现方式未定），需求清晰的任务直接进入执行阶段。

---

## 亮点一：Supervisor 多 Agent 架构

### 上下文隔离防止任务飘逸

传统单 Agent 方案在复杂任务中容易出现"上下文污染"——前一个子任务的细节干扰后续决策，导致任务偏离。CCAgent 采用 **Supervisor 架构**，子 Agent 通过 `run_agent()` 独立运行，上下文彼此隔离：

- **Supervisor**：只负责任务规划与路由，不接触代码细节
- **Coder**：只处理编码任务，不感知调研内容
- **Researcher**：只做技术调研，结果以摘要形式回传
- **Reviewer**：只做代码审查与测试，独立于编写过程

Supervisor 通过结构化的任务指令（而非消息转发）与子 Agent 通信，天然隔断了上下文串扰。

### Skill 系统：动态角色组装

子 Agent 的身份和能力由 `skills/` 目录下的 Markdown 文件定义，Supervisor 在 `spawn_agents` 时按需注入：

```
skills/
├── coder.md       # 编码专家角色定义 + 允许使用的工具白名单
├── researcher.md  # 调研专家角色定义
├── reviewer.md    # 审查专家角色定义
└── *.md           # 可按技术领域自由扩展（react-patterns、fastapi 等）
```

多个 skill 可叠加，例如 `["coder", "react-patterns"]` 让 Coder 额外具备 React 领域知识。新增领域 skill 无需修改任何代码。

### 强制审查闭环

Coder 完成编码后，Supervisor 被要求**强制调用 Reviewer** 审查，只有 Reviewer 通过后任务才能标记为完成，避免未经验证的代码直接交付。

### MCP 工具扩展

Researcher 支持 MCP（Model Context Protocol）工具动态加载，可在运行时接入外部知识源、API 和数据库，无需修改代码。

---

## 亮点二：三层记忆系统

CCAgent 实现了一套完整的分层记忆架构，覆盖从毫秒级上下文到跨月跨项目的长期知识积累。

### 层级结构

```
┌─────────────────────────────────────────────────────┐
│  短期记忆（会话级）                                    │
│  Redis LangGraph Checkpoint                          │
│  · 存储完整对话状态（消息、工具调用、任务进度）          │
│  · 按 Agent 角色独立存储，7天 TTL 自动淘汰             │
├─────────────────────────────────────────────────────┤
│  长期记忆（跨会话语义知识）                             │
│  Redis HNSW 热存储  +  MySQL 冷备份                   │
│  · LLM 从对话中提炼结构化知识                          │
│  · 4 种类型：user / feedback / project / reference   │
│  · Redis 崩溃时自动从 MySQL 恢复                      │
├─────────────────────────────────────────────────────┤
│  归档层（完整对话冷存储）                              │
│  MySQL conversations 表                              │
│  · 保存原始消息流，用于历史回溯                         │
│  · 支持从归档恢复历史会话继续对话                       │
└─────────────────────────────────────────────────────┘
```

### 上下文压缩（两阶段策略）

短期记忆会随对话积累不断增长，当 token 使用率触及阈值时自动触发压缩：

```
token 使用率
  < 50%   →  正常运行
  50~75%  →  ⚠️  提醒用户，可选择是否压缩
  75~92%  →  🚨  强提醒，建议立即压缩
  ≥ 92%  →  自动压缩，不中断任务
```

**Phase 1**：优先删除旧的 ToolMessage（工具输出体积大但价值低），若压缩后使用率降到安全线则结束。

**Phase 2**：若仍超阈值，调用 LLM 将旧消息生成 **8 段式结构化摘要**（请求意图 / 技术概念 / 文件变更 / 错误修复 / 问题解决 / 用户输入 / 待处理任务 / 当前状态），以单条摘要消息替换旧消息流，保留近期原文。

### 长期记忆的混合检索

每次 Manager Agent 被调用时，用**当前用户消息**作为查询，只把相关记忆注入 System Prompt，而非把全部记忆塞进去（全量注入在记忆条目增多后会浪费大量 token，且引入无关噪声）。

检索分两步：

**第一步：HNSW 向量粗召回**

用 `bge-small-zh-v1.5` 将查询文本编码为 512 维 float32 向量，在 Redis Stack 的 HNSW 索引上做 KNN 搜索，取前 `top_k × 3` 条候选。HNSW（Hierarchical Navigable Small World）是一种近似最近邻图索引，查询复杂度为 O(log n)，相比暴力遍历在记忆条数增大后有显著优势。

```
用户消息 "我想用 pytest 给这个模块写测试"
   │
   └─► [0.032, -0.187, 0.094, ...]  (512维)
          │
          └─► HNSW KNN → 召回 top_k×3 条最相近的记忆
                  例如：
                  · "禁止 mock 数据库，必须连真实 DB"     距离 0.08
                  · "测试框架用 pytest，不用 unittest"    距离 0.11
                  · "用户偏好 Python"                    距离 0.21
                  · ...（共 top_k×3 条候选）
```

**第二步：BM25 关键词 Re-ranking**

HNSW 用语义相似度召回，但可能遗漏关键词精确匹配的记忆（例如查询里有专有名词但语义距离稍远）。对召回的候选集额外做 BM25 关键词打分，再与向量相似度加权合并，取最终 Top-K：

```
综合得分 = 0.6 × 向量相似度 + 0.4 × BM25得分（均归一化到 [0,1]）
```

最终得分最高的 K 条记忆被格式化后注入 System Prompt，Manager Agent 据此做出有历史上下文的决策。

Embedding 模型（`bge-small-zh-v1.5`）通过 ModelScope 下载，本地 CPU 推理，不依赖外部 API。

### 长期记忆的去重/合并 Pipeline

会话结束时，LLM 从对话中提炼记忆条目，写入前经过去重检查：

```
新提炼的记忆条目
   │
   ├─► HNSW 找最近邻（相似度 ≥ 0.78 的候选）
   │
   ├─► LLM 决策：ADD / UPDATE / NOOP
   │     ADD    → 全新知识，直接写入
   │     UPDATE → 与已有记忆合并，更新内容和向量
   │     NOOP   → 重复或已有记忆更准确，跳过
   │
   └─► 双写 Redis（热）+ MySQL（备份）
```

4 种记忆类型：
- **user**：用户偏好、技术背景、工作角色
- **feedback**：用户纠正过的 AI 行为（含 Why + How to apply）
- **project**：关键决策、技术约束、截止日期
- **reference**：文件路径、服务地址、外部 URL

---

## 亮点三：沙箱隔离执行

Coder Agent 的所有代码执行、命令运行均发生在 **e2b CubeSandbox KVM 沙箱**中，与宿主机完全隔离：

- `bash(command)`：在沙箱中运行 Shell 命令
- `run_python(code)`：在沙箱中直接执行 Python 代码片段
- `write_file / edit_file / read_file`：沙箱文件系统操作

沙箱在会话期间保持同一实例（环境变量、已安装包等状态持续保留），会话结束时自动销毁。即使 Agent 执行了危险命令，宿主机也不受影响。

---

## 技术栈

| 组件 | 选型 |
|------|------|
| Agent 框架 | LangGraph（StateGraph + Checkpoint） |
| LLM | Qwen（vLLM 自托管） |
| 短期记忆 | Redis（LangGraph AsyncRedisSaver） |
| 长期记忆向量检索 | Redis Stack HNSW |
| Embedding 模型 | bge-small-zh-v1.5（ModelScope） |
| 长期记忆冷备份 | MySQL |
| 完整对话归档 | MySQL |
| 沙箱隔离 | e2b CubeSandbox（KVM） |
| MCP 工具 | Model Context Protocol |

---

## 目录结构

```
nano_claude_code/
├── main.py                  # 主入口，图定义，会话管理
├── core/
│   ├── config.py            # 模型与存储配置
│   ├── sandbox.py           # e2b 沙箱会话管理（含 SSL/DNS patch）
│   ├── tools.py             # 工具集（沙箱执行、文件读写、搜索等）
│   ├── agent_runner.py      # 轻量子 Agent 运行器（无 LangGraph）
│   ├── agent_loader.py      # Skill 加载与工具白名单解析
│   ├── spawn_tool.py        # spawn_agents 工具，并行启动子 Agent
│   ├── memory_store.py      # MySQL 归档与长期记忆备份
│   └── mcp.py               # MCP 工具加载
├── intent_agent/
│   └── agent.py             # analyze_intent 工具（需求 Q&A + 文档生成）
├── memory/
│   ├── long_term_memory.py  # 长期记忆：HNSW检索、去重Pipeline
│   ├── compressor.py        # 上下文两阶段压缩
│   └── memory_manager.py    # 记忆管理 CLI（查看/删除）
├── skills/                  # Skill 定义文件（角色身份 + 工具权限）
│   ├── coder.md
│   ├── researcher.md
│   ├── reviewer.md
│   └── *.md                 # 可自由扩展领域 skill
└── tests/
    └── test_sandbox.py      # 沙箱单元测试（23 个）
```
