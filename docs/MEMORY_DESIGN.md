# 记忆系统设计说明

## 整体架构

仿照 Claude Code 的记忆架构，将记忆分为三层，分工明确：

```
短期记忆（LangGraph Checkpoint）
  └── 存储：每一步执行后的完整状态快照（消息列表）
      作用：保持单次会话内的上下文连贯性
      位置：Redis，键前缀由 LangGraph 自动管理

上下文压缩（Compressor）
  └── 存储：对旧消息的 LLM 生成摘要，替换原始消息
      作用：防止 context window 溢出，保持 Agent 执行效率
      触发：token 使用率超过阈值时自动触发

长期记忆（LongTermMemory）
  └── 存储：从对话中提炼的结构化知识条目
      作用：跨会话注入背景知识，无需用户重复说明
      位置：Redis，键前缀 ltm:
```

---

## 一、短期记忆（LangGraph Checkpoint）

### 原理

LangGraph 在每次节点执行后自动将完整 `AgentState` 快照写入 Redis。每个快照称为一个 **检查点（Checkpoint）**，按时间倒序排列，最新的排在最前面（`alist()` 返回顺序）。

### Redis 键结构

由 `AsyncRedisSaver` 自动管理，每个检查点包含多个 Redis 键（含 writes、metadata 等），所有键都包含 `thread_id` 和 `checkpoint_id`。

### 注意事项

- `alist()` 返回顺序：**新 → 旧**（index 0 = 最新）
- 每个检查点是完整快照，越旧的消息数越少，这是正常现象
- `AgentState.messages` 使用 `add_messages` reducer（而非 `operator.add`），支持 `RemoveMessage` 删除指定消息

---

## 二、上下文压缩（Compressor）

### 设计来源

仿照 Claude Code 的两阶段压缩策略：
1. **Phase 1**：优先删除旧 `ToolMessage`（工具输出体积大、语义价值低）
2. **Phase 2**：若仍超阈值，调用 LLM 生成 8 段式摘要替换旧消息

### 触发条件

```python
MAX_TOKENS = 163840     # Qwen3.5-27B，vLLM --max-model-len 163840
TOKEN_THRESHOLD = 0.92  # ≥ 92% 自动压缩（约 15.1 万 token）
KEEP_RECENT = 3         # 无论如何保留最近 3 条消息原文
```

token 计数优先读取 `response_metadata.token_usage.total_tokens`（精确值），无则按字符数估算（`字符数 / 3`）。

### 三档路由

```
工具节点执行完毕
  ↓
make_token_router() 检查 token 用量
  ├── < 50%  → agent（正常继续）
  ├── 50~92% → warn 节点（interrupt 暂停，等待用户选择）
  │              用户输入 compress → compress 节点
  │              直接回车跳过    → agent
  └── ≥ 92%  → compress 节点（自动压缩，无需确认）
```

warn 节点用 LangGraph `interrupt()` 暂停图执行；50~75% 显示 ⚠️ ，75~85% 显示 🚨 。  
主循环和子 Agent wrapper 均用 `while True + Command(resume=...)` 处理中断并恢复。

### 执行流程（自动压缩路径）

```
进入 compress 节点
        │
        ├── Phase 1: 找出 messages[:-3] 中的所有 ToolMessage
        │     └── 删除后 token < 85%？
        │           ├── 是 → 只返回 RemoveMessage，结束
        │           └── 否 → 进入 Phase 2
        │
        └── Phase 2: 将 messages[:-3] 格式化为对话文本
              ↓
              调用 LLM 生成 8 段式摘要
              ↓
              删除所有旧消息（RemoveMessage × N）
              重建顺序：[摘要 AIMessage] + [recent 3 条]
              ↓
              回到 agent
```

### 消息顺序重建原理

由于 `add_messages` reducer 按顺序处理：先执行所有 `RemoveMessage`（清空），再追加新消息，因此返回：

```python
[RemoveMessage(id) for all] + [summary_msg] + list(recent)
```

最终 state 消息顺序为：`[摘要, recent[-3], recent[-2], recent[-1]]` ✓

### 8 段式摘要内容

LLM 被要求按以下结构生成摘要：

1. 主要请求和意图
2. 关键技术概念
3. 文件和代码段（含路径、关键行号）
4. 错误和修复
5. 问题解决过程
6. 用户消息摘录（按时间顺序）
7. 待处理任务
8. 当前工作状态和下一步计划

### 图结构变更

所有 Agent（Manager + Coder + Researcher + Reviewer）的图均从：

```
tools → agent
```

改为：

```
tools → [should_compress] → compress → agent
                          → agent（直接）
```

### 涉及文件

| 文件 | 变更 |
|------|------|
| `compressor.py` | 新建，包含全部压缩逻辑 |
| `core/state.py` | `operator.add` → `add_messages`（支持 RemoveMessage）|
| `coder_agent/agent.py` | 新增 compress 节点和条件边 |
| `researcher_agent/agent.py` | 同上 |
| `reviewer_agent/agent.py` | 同上 |
| `main.py` | Manager 图新增 compress 节点，4 条回 agent 的边改为条件路由 |

---

## 三、长期记忆（LongTermMemory）

### 记忆类型（4 种，对应 Claude Code）

| 类型 | 存什么 | 举例 |
|------|--------|------|
| `user` | 用户偏好、知识背景、工作角色 | "用户偏好 Python，排斥冗余注释" |
| `feedback` | AI 行为被纠正，或非显而易见的做法被确认 | "删文件前必须确认，曾因此丢失工作" |
| `project` | 当前项目的关键决策、约束、目标 | "正在重构 Manager/Executor 架构" |
| `reference` | 外部资源位置（地址、路径、工具名） | "Redis 服务地址 10.129.107.145:6379" |

### Redis 存储结构

```
ltm:index               → Set，存所有记忆的 ID
ltm:mem:{id}            → Hash，单条记忆的完整字段
  ├── id          string   8位随机 hex
  ├── type        string   user / feedback / project / reference
  ├── name        string   kebab-case slug（如 user-prefers-python）
  ├── description string   一句话摘要（用于判断相关性）
  ├── content     string   具体内容（含 Why 和 How to apply）
  └── created_at  string   ISO 8601 时间戳
```

与 Checkpoint 的键空间完全隔离（Checkpoint 使用 LangGraph 自有前缀）。

### 数据流（每次会话）

```
启动
  ↓
ltm.load_all() → cached_memories（加载已有记忆）
  ↓
ltm.format_for_prompt() → 注入 Manager system_prompt 头部
  ↓
对话循环（LangGraph Checkpoint + 压缩 正常工作）
  ↓
用户退出
  ↓
app.aget_state() 取出本次全量消息
  ↓
ltm.extract_and_save(model, msgs)
  LLM 分析最后 40 条消息 → 返回 JSON → 写入 Redis
```

### 记忆提炼 Prompt 要求

- 只提炼 4 种类型之一
- 没有新信息则返回空列表 `[]`
- `feedback` 类型必须包含 Why（原因）和 How to apply（应用场景）
- 以 JSON 数组返回，每项含 `type / name / description / content`

### system_prompt 注入格式

```
## 长期记忆（跨会话积累）

### 用户信息
**user-prefers-python**: 用户偏好 Python，排斥冗余注释...

### 行为反馈
**confirm-before-delete**: 删除文件前必须确认。Why: ...

### 项目上下文
**current-arch**: 正在重构 Manager/Executor 架构...
```

### 涉及文件

| 文件 | 作用 |
|------|------|
| `long_term_memory.py` | `LongTermMemory` 类，所有读写/提炼逻辑 |
| `main.py` | 启动时加载记忆、注入 prompt；退出时提炼保存 |
| `memory_manager.py` | 菜单项 5：长期记忆的查看/删除/清空 |

---

## 四、memory_manager.py 导航说明

所有菜单统一使用 `q` 返回上级，不再使用 `0`。菜单层级：

```
主菜单（选择 Agent 或长期记忆）
  ├── 1-4. 选择 Agent
  │     ├── 1. 查看记忆（检查点浏览）
  │     │     ├── 1-5. 各种查看方式
  │     │     └── q. 返回
  │     ├── 2. 删除记忆
  │     │     ├── 1. 选择删除指定检查点（输入索引，q 取消）
  │     │     ├── 2. 保留最近 N 个（输入数字，q 取消）
  │     │     ├── 3. 删除全部
  │     │     └── q. 返回
  │     └── q. 返回
  ├── 5. 长期记忆管理
  │     ├── 1. 查看全部记忆
  │     ├── 2. 删除指定记忆（输入索引，q 取消）
  │     ├── 3. 清空全部
  │     └── q. 返回
  └── q. 退出
```
