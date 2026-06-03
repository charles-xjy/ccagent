# 记忆系统完整架构文档

## 概述

本系统为 LangGraph 多 Agent 对话实现了**三层记忆架构**，覆盖不同粒度和生命周期的数据：

| 层级 | 存储 | 生命周期 | 数据粒度 | 用途 |
|------|------|----------|----------|------|
| 实时 (Live) | Redis checkpoint | 对话中每步快照 + 7 天 TTL | 完整 AgentState | LangGraph 自动快照，跨进程状态恢复 |
| 语义 (Semantic) | Redis hash set | 跨会话持久（无 TTL） | LLM 提炼的知识片段 | 用户偏好、行为反馈、项目上下文 |
| 归档 (Archive) | MySQL | 永久保存 | 完整消息历史 | 对话结束归档，Redis 过期后的恢复来源 |

---

## 数据流全景

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          对话执行期间                                         │
│                                                                              │
│  用户输入 → intent_analysis  →  Manager Agent  →  SubAgents                  │
│                  │                  │                 │                       │
│                  ▼                  ▼                 ▼                       │
│           需求文档确认        plan_confirm      coder/researcher/reviewer     │
│           (interrupt)        (interrupt)       (独立 thread_id checkpoint)    │
│                                                                              │
│  每步执行后: LangGraph State → AsyncRedisSaver → Redis checkpoint            │
│  token 超限时: warn → compress → 压缩消息历史 → 继续执行                      │
│  TTL: 每步写入刷新 7 天过期时间                                               │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                          对话结束时 (/q 退出)                                  │
│                                                                              │
│  Step 1: ltm.extract_and_save(model, messages)                               │
│          LLM 从消息历史提炼语义记忆 → Redis hash set                          │
│          类型: user / feedback / project / reference                         │
│                                                                              │
│  Step 2: _archive_to_mysql(session_id, checkpointer, memory_store)           │
│          从 Redis checkpoint 提取完整 messages                                │
│          → dumps() 序列化 → MySQL conversations 表 (UPSERT)                   │
│                                                                              │
│  Redis checkpoint 保留不删，作为热缓存（7 天 TTL 自然淘汰）                    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                         下次用户进入时                                         │
│                                                                              │
│  prompt_ui 选择器:                                                            │
│  ┌─────────────────────────────────────────────────┐                         │
│  │ ▶  恢复: session_myproject_20260603_1200  (Redis)│                         │
│  │ ▶  恢复: session_demo_20260601_0930       (Redis)│                         │
│  │ 🗄  从 MySQL 恢复: session_old_20260515_1800     │                         │
│  │ ✨  新建会话                                      │                         │
│  └─────────────────────────────────────────────────┘                         │
│                                                                              │
│  resume:SID → Redis 命中 → LangGraph 自动恢复 state → 继续对话                │
│  mysql:SID → Redis 过期 → 从 MySQL loads() → aupdate_state() → 重建 Redis    │
│  新建     → session_{project}_{timestamp} → 全新对话                          │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 三层存储详解

### 第一层：Redis Checkpoint（实时状态快照）

**存储引擎**: LangGraph `AsyncRedisSaver`

**配置**:
```python
REDIS_URI = "redis://10.129.107.145:6379"
REDIS_TTL = 604800  # 7 天
```

**存储的数据** — 每个 thread_id 下的完整 `AgentState`:
```python
class AgentState(TypedDict):
    messages: List[BaseMessage]     # HumanMessage, AIMessage, ToolMessage
    current_todo: List[dict]        # 任务面板
    plan_needs_confirm: bool        # 是否需要计划确认
    compress_choice: str            # 压缩选项
```

**Thread ID 体系**（动态生成）:
| 会话 | thread_id 格式 | 示例 |
|------|---------------|------|
| 主 Manager | `session_{project}_{timestamp}` | `session_myapp_20260603_1200` |
| Coder 子 Agent | `{main_session_id}_coder` | `session_myapp_20260603_1200_coder` |
| Researcher | `{main_session_id}_tech-researcher` | `session_myapp_20260603_1200_tech-researcher` |
| Reviewer | `{main_session_id}_reviewer` | `session_myapp_20260603_1200_reviewer` |

**生命周期**:
- **写入**: 每个 LangGraph 节点执行后自动触发
- **刷新**: 每次 `astream()` 调用重置 TTL 为 7 天
- **过期**: 7 天无活动后 Redis 自动删除 key
- **压缩**: token 超限时 `compress` 节点对消息历史做摘要压缩

**Key 格式**: LangGraph 内部管理，前缀 `checkpoint` / `checkpoint_write`，按 thread_id 分区

---

### 第二层：Redis LongTermMemory（语义记忆提炼）

**存储引擎**: 自定义 `LongTermMemory` 类（`memory/long_term_memory.py`）

**存储结构**:
```
ltm:index              → Redis Set    (存储所有 memory ID)
ltm:mem:{mem_id}       → Redis Hash   (单条记忆的详细信息)
```

**Hash 字段**:
| 字段 | 说明 |
|------|------|
| `id` | 8 位 hex，uuid4 前 8 字符 |
| `type` | `user` / `feedback` / `project` / `reference` |
| `name` | kebab-case 短名 |
| `description` | 一句话摘要（用于相关性判断） |
| `content` | 具体内容 |
| `created_at` | ISO 时间戳 |

**四种记忆类型**:
| Type | 标签 | 提炼内容 | 示例 |
|------|------|----------|------|
| `user` | 用户信息 | 角色、偏好、知识背景 | "用户是数据科学家，关注日志系统" |
| `feedback` | 行为反馈 | AI 行为纠正或确认 | "不要 mock 数据库，上次出过生产事故" |
| `project` | 项目上下文 | 决策、约束、截止日期 | "auth 中间件重写由合规性驱动" |
| `reference` | 资源引用 | 外部 URL/路径/服务地址 | "pipeline bug 跟踪: Linear INGEST 项目" |

**提炼流程**:
```
对话结束 → ltm.extract_and_save(model, messages)
         → 截取最后 40 条消息
         → LLM 提炼 → JSON 数组
         → 逐条存入 Redis hash set
```

**提炼触发条件**:
- 对话消息 ≥ 6 条（`_MIN_MSGS_FOR_MEMORY = 6`）
- LLM 返回非空 JSON 数组
- 仅接受 `MEMORY_TYPES` 中定义的四种类型

**管理工具**: `memory/memory_manager.py` — 交互式查看/删除语义记忆

---

### 第三层：MySQL 对话归档（完整消息历史）

**存储引擎**: `MemoryStore` 类（`core/memory_store.py`），封装 `aiomysql`

**表结构**:
```sql
CREATE TABLE conversations (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    thread_id       VARCHAR(255) NOT NULL UNIQUE,
    messages_json   LONGTEXT NOT NULL,          -- langchain_core.load.dumps() 序列化
    message_count   INT NOT NULL DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_thread_id (thread_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

**配置**:
```python
MYSQL_HOST = "10.129.107.145"
MYSQL_PORT = 3306
MYSQL_USER = "ccagent"
MYSQL_PASSWORD = "ccagent123"
MYSQL_DATABASE = "ccagent"
```

**序列化格式**（LangChain 官方 LC 格式）:
```json
[
  {
    "lc": 1,
    "type": "constructor",
    "id": ["langchain", "schema", "messages", "HumanMessage"],
    "kwargs": {"content": "帮我写一个快排", "type": "human"}
  },
  {
    "lc": 1,
    "type": "constructor",
    "id": ["langchain", "schema", "messages", "AIMessage"],
    "kwargs": {
      "content": "好的，我来写快排",
      "tool_calls": [
        {"name": "task_tool", "args": {"subagent_type": "coder"}, "id": "call_1", "type": "tool_call"}
      ],
      "type": "ai"
    }
  },
  {
    "lc": 1,
    "type": "constructor",
    "id": ["langchain", "schema", "messages", "ToolMessage"],
    "kwargs": {"content": "快速排序已写入 quicksort.py", "tool_call_id": "call_1", "type": "tool"}
  }
]
```

**生命周期操作**:
| 操作 | 触发时机 | SQL / 方法 |
|------|----------|------------|
| 写入 | 正常退出（q/exit） | `archive()` → `INSERT ... ON DUPLICATE KEY UPDATE` |
| 更新 | 同一会话再次退出 | 同上 UPSERT，覆盖旧数据 |
| 读取 | 进入时 Redis 未命中 | `load()` → `SELECT ... WHERE thread_id = ?` |
| 扫描 | 启动时构建会话列表 | `list_threads()` → `SELECT ... ORDER BY updated_at DESC` |
| 注入 | 从 MySQL 恢复会话 | `aupdate_state()` 写入 LangGraph State |
| 删除 | memory_manager 工具 | `delete()` → `DELETE ... WHERE thread_id = ?` |

**归档范围**: 仅主 Manager 的 thread_id。子 Agent 的 checkpoint 不归档（临时任务级）。

---

## 会话生命周期完整时序

```
┌─ 启动 ─────────────────────────────────────────────────────────────┐
│ 1. init MemoryStore (可选, 失败降级)                                 │
│ 2. AsyncRedisSaver.from_conn_string(REDIS_URI, ttl=REDIS_TTL)      │
│ 3. LongTermMemory(REDIS_URI)                                        │
│ 4. 加载 cached_memories (语义记忆)                                   │
│ 5. _scan_base_sessions(REDIS_URI) → Redis 中的会话列表              │
│ 6. _scan_mysql_sessions(memory_store) → MySQL 中的归档列表           │
│ 7. prompt_ui 选择器 (Redis 会话 + MySQL 会话 + 新建)                 │
└────────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    resume:SID           mysql:SID            新建会话
    Redis 恢复           MySQL 恢复           session_{project}_{timestamp}
    LangGraph 自动       loads() →            全新 AgentState
    恢复 checkpoint      aupdate_state()
                         → 重建 Redis
                              │
┌─ 对话循环 ───────────────────┼──────────────────────────────────────┐
│ while True:                                                         │
│   user_input → app.astream({messages: [HumanMessage]})              │
│   → intent_analysis → agent → tools/subagents → ...                 │
│   → plan_confirm (interrupt, 首次规划确认)                           │
│   → warn/compress (token 超限时)                                    │
│   → 每步自动 Redis checkpoint                                       │
│   → "__interrupt__" → _handle_interrupt → Command(resume=...)       │
│   输入 q/exit → break                                               │
└────────────────────────────────────────────────────────────────────┘
                              │
┌─ 退出 ─────────────────────────────────────────────────────────────┐
│ 1. 消息 ≥ 6 条? → ltm.extract_and_save(model, messages)            │
│    LLM 提炼语义记忆 → Redis hash set                                 │
│                                                                     │
│ 2. _archive_to_mysql(session_id, checkpointer, memory_store)        │
│    Redis checkpoint → 提取 messages → dumps() → MySQL UPSERT       │
│    Redis checkpoint 保留不删 (7 天 TTL)                              │
│                                                                     │
│ 3. memory_store.close()                                             │
│    LongTermMemory / checkpointer 由 async with 自动关闭             │
└────────────────────────────────────────────────────────────────────┘
```

---

## 故障降级矩阵

| 故障场景 | 第一层 (Redis checkpoint) | 第二层 (语义记忆) | 第三层 (MySQL 归档) |
|----------|--------------------------|-------------------|---------------------|
| MySQL 不可用 | 正常 | 正常 | 跳过归档，打印警告 |
| Redis 不可用 | 无法启动 | 无法启动 | — |
| MySQL 连接丢失（运行中） | 正常 | 正常 | 归档失败，消息仅存 Redis |
| Redis key 过期 | 自动从 MySQL 恢复 | 语义记忆独立存在 | 提供恢复数据 |
| checkpoint 损坏 | 从 MySQL 恢复 | 语义记忆独立存在 | 提供恢复数据 |

---

## 关键源文件

| 文件 | 层级 | 职责 |
|------|------|------|
| `core/config.py` | 配置 | Redis/MySQL 连接参数 + TTL |
| `core/state.py` | 数据模型 | `AgentState` TypedDict |
| `core/memory_store.py` | 第三层 | `MemoryStore` — MySQL 归档/读取/删除/列表 |
| `memory/__init__.py` | 第二层 | 包入口，导出 LongTermMemory + compressor |
| `memory/long_term_memory.py` | 第二层 | `LongTermMemory` — LLM 提炼语义记忆存 Redis |
| `memory/compressor.py` | 第一层 | Token 压缩节点 (warn/compress/router) |
| `memory/memory_manager.py` | 工具 | 交互式记忆管理 (Redis 语义记忆 + checkpoint 管理) |
| `main.py` | 编排 | `_scan_mysql_sessions` / `_restore_from_mysql` / `_archive_to_mysql` |
| `intent_agent/agent.py` | 辅助 | 意图分析（需求澄清多轮对话） |

---

## 维护工具

### memory_manager.py（语义记忆 + Redis checkpoint）

```bash
conda run -n langgraph python3 nano_claude_code/memory/memory_manager.py
```

管理 `LongTermMemory` 中存储的语义记忆（增删查）。

### 旧的 memory_manager.py（我们写的，在根目录下）

```bash
conda run -n langgraph python3 nano_claude_code/memory_manager.py
```

管理 Redis checkpoint + MySQL 归档。由于远程代码已将 memory_manager 迁移到 `memory/` 包，这个文件可能已过时，推荐使用 `memory/memory_manager.py`。

### MySQL 直接查询

```bash
docker exec mysql-ccagent mysql -uccagent -pccagent123 ccagent \
  -e "SELECT thread_id, message_count, updated_at FROM conversations ORDER BY updated_at DESC;"
```
