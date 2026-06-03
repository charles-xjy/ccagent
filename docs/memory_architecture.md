# 两层记忆系统架构文档

## 概述

本系统为 LangGraph 多 Agent 对话实现了两层记忆架构：

| 层级 | 存储 | 生命周期 | 用途 |
|------|------|----------|------|
| 热存储 (Hot) | Redis | 对话执行期间 + 之后 7 天 | LangGraph checkpoint 自动快照，毫秒级恢复 |
| 冷存储 (Cold) | MySQL | 永久保存（直到手动删除） | 对话归档，Redis 过期后的恢复来源 |

---

## 数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                      对话执行期间                                 │
│                                                                  │
│  LLM 每次输出 ──► LangGraph State ──► AsyncRedisSaver            │
│                                       │                          │
│                                       ▼                          │
│                               Redis checkpoint                   │
│                               (key: thread_id)                   │
│                               (TTL: 每步执行后刷新为 7 天)         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       对话结束时 (/q 退出)                        │
│                                                                  │
│  Redis checkpoint ──► 提取 messages ──► dumps() 序列化           │
│                                               │                  │
│                                               ▼                  │
│                                       MySQL conversations 表     │
│                                       (UPSERT, 按 thread_id)     │
│                                                                  │
│  Redis checkpoint 保留不删，继续作为热缓存                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     下次用户进入时                                │
│                                                                  │
│  查 Redis ──► 命中 ──► LangGraph 自动恢复 state ──► 继续对话     │
│    │                                                             │
│    └──► 未命中 ──► 查 MySQL ──► loads() 反序列化                 │
│                      │                                           │
│                      ▼                                           │
│               注入 AgentState.messages                            │
│               LangGraph 重建 Redis checkpoint                    │
│               TTL 重新计时 7 天                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 存储的数据

### 1. Redis（热存储）

**存储引擎**：LangGraph `AsyncRedisSaver`

**存储内容**：每个 `thread_id` 下完整的 `AgentState` 快照

```python
# AgentState 结构 (core/state.py)
class AgentState(TypedDict):
    messages: List[BaseMessage]     # 完整消息历史
    current_todo: List[dict]        # 当前任务面板状态
```

**Key 格式**：LangGraph 内部格式，前缀 `checkpoint` / `checkpoint_write`，按 `thread_id` 分区

**存储的具体消息类型**：
- `HumanMessage` — 用户输入
- `AIMessage` — LLM 输出（含 `tool_calls` 子结构）
- `ToolMessage` — 工具执行结果（含 `tool_call_id` 关联）
- `SystemMessage` — Manager 的系统提示（每次 invoke 时动态注入，不持久化在 checkpoint 中）

**TTL 策略**：
- 每次 `astream()` 执行会触发 checkpoint 写入，刷新该 key 的 TTL
- 默认 TTL = 604800 秒（7 天），配置在 `core/config.py:17`
- 7 天内无任何对话活动 → key 自动过期删除
- 子 Agent（coder/researcher/reviewer）的 checkpoint 同样享有 7 天 TTL

**生命周期管理**：

| 操作 | 触发时机 | 效果 |
|------|----------|------|
| 写入 | 每个 LangGraph 节点执行后自动触发 | 保存完整 State 快照 |
| 刷新 | 每次 `astream()` 调用 | TTL 重置为 7 天 |
| 过期 | 7 天无活动 | Redis 自动删除 key |
| 手动删除 | 通过 `memory_manager.py` | 立即删除指定 thread 的所有 checkpoint |

### 2. MySQL（冷存储）

**存储引擎**：`aiomysql`，封装在 `core/memory_store.py` 的 `MemoryStore` 类

**表结构**：

```sql
CREATE TABLE conversations (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    thread_id       VARCHAR(255) NOT NULL UNIQUE,
    messages_json   LONGTEXT NOT NULL,           -- 完整消息序列化 JSON
    message_count   INT NOT NULL DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_thread_id (thread_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

**存储内容**：对话退出时从 Redis checkpoint 提取出的完整 `messages` 列表，通过 `langchain_core.load.dumps()` 序列化为 JSON

**序列化格式**（LangChain 官方 LC 格式）：
```json
[
  {
    "lc": 1,
    "type": "constructor",
    "id": ["langchain", "schema", "messages", "HumanMessage"],
    "kwargs": {"content": "你好", "type": "human"}
  },
  {
    "lc": 1,
    "type": "constructor",
    "id": ["langchain", "schema", "messages", "AIMessage"],
    "kwargs": {
      "content": "你好！有什么可以帮你的？",
      "tool_calls": [],
      "type": "ai"
    }
  }
]
```

**存储的会话**：仅主 Manager 的 `thread_id = "manager_executor_v2"`。子 Agent 的 checkpoint 不归档（任务级、临时性）。

**生命周期管理**：

| 操作 | 触发时机 | SQL |
|------|----------|-----|
| 写入 | 每次用户正常退出（q/exit） | `INSERT ... ON DUPLICATE KEY UPDATE` |
| 更新 | 同一 thread_id 再次退出 | 同上 UPSERT，覆盖旧数据，刷新 `updated_at` |
| 读取 | 下次进入时 Redis 未命中 | `SELECT ... WHERE thread_id = ?` |
| 手动删除 | 通过 `memory_manager.py` 选项 5 | `DELETE ... WHERE thread_id = ?` |

---

## 涉及的 Thread ID

| thread_id | 用途 | Redis | MySQL |
|-----------|------|-------|-------|
| `manager_executor_v2` | 主 Manager Agent | TTL 7天 | 归档 |
| `manager_executor_v2_coder` | Coder 子 Agent | TTL 7天 | 不归档 |
| `manager_executor_v2_tech-researcher` | Researcher 子 Agent | TTL 7天 | 不归档 |
| `manager_executor_v2_reviewer` | Reviewer 子 Agent | TTL 7天 | 不归档 |

---

## 配置参数

所有配置集中在 `nano_claude_code/core/config.py`：

```python
# Redis
REDIS_URI = "redis://10.129.107.145:6379"
REDIS_TTL = 604800          # checkpoint 过期时间，秒（7天）

# MySQL
MYSQL_HOST = "10.129.107.145"
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_PASSWORD = ""
MYSQL_DATABASE = "ccagent"
```

**调整建议**：
- 高频使用场景 → 调大 `REDIS_TTL`（如 2592000 = 30 天）
- Redis 内存紧张 → 调小 `REDIS_TTL`（如 259200 = 3 天）
- Redis 服务端建议配置 `maxmemory-policy allkeys-lru`，内存满时自动淘汰最久未用的 key

---

## 维护工具：memory_manager.py

```bash
conda run -n langgraph python3 nano_claude_code/memory_manager.py
```

交互式菜单：

```
================ 记忆管理器 ================
--- Redis 热存储 ---
  1. 主 Agent (Manager) (ID: manager_executor_v2)
  2. 编程专家 (Coder) (ID: manager_executor_v2_coder)
  3. 调研专家 (Researcher) (ID: manager_executor_v2_tech-researcher)
  4. 代码审核专家 (Reviewer) (ID: manager_executor_v2_reviewer)
--- MySQL 冷存储 ---
  5. 查看 MySQL 归档列表
q. 退出
```

**Redis 操作**（选项 1-4 → 选 1/2）：
- 查看记忆：浏览 checkpoint 快照、按范围查看消息、选择特定 checkpoint
- 删除记忆：删除指定 thread 的所有 Redis checkpoint（不可恢复）

**MySQL 操作**（选项 5）：
- 查看归档列表：列出所有归档的 thread_id、消息数、更新时间
- 查看消息内容：选中一个归档，完整浏览历史消息
- 删除归档：删除指定 thread 的 MySQL 归档（不可恢复）

---

## 关键源文件

| 文件 | 职责 |
|------|------|
| `nano_claude_code/core/config.py` | Redis/MySQL 连接配置和 TTL 参数 |
| `nano_claude_code/core/state.py` | `AgentState` 结构定义（messages + current_todo） |
| `nano_claude_code/core/memory_store.py` | `MemoryStore` 类，MySQL 建表/归档/读取/删除/列表 |
| `nano_claude_code/main.py` | `restore_session()` 恢复逻辑 + `archive_session()` 归档逻辑，主循环集成 |
| `nano_claude_code/memory_manager.py` | 交互式记忆管理工具（Redis + MySQL 两面） |

---

## 故障降级

- **MySQL 不可用**：启动时打印 `[!] MySQL 不可用，将仅使用 Redis 记忆`，归档步骤跳过，系统正常运行，只是没有长期持久化
- **Redis 不可用**：无法启动（LangGraph 依赖 checkpointer），需要恢复 Redis 服务
- **序列化异常**：`langchain_core.load.dumps/loads` 处理所有标准消息类型，如果遇到未知类型会抛出异常，由调用方的 try/except 捕获并打印警告
