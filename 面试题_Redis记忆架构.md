# Redis 记忆架构面试题

> 基于 ccagent 项目的三层记忆架构（Redis 热缓存 + 语义提炼 + MySQL 冷归档）

---

## Q1：Redis 的 checkpoint 为什么不用 List 存？

因为 checkpoint 的核心需求是**按 ID 精确查找和更新**，List 做不到这个。

List 只能从两端操作，或者按下标访问，比如"第3个"、"最后一个"。但你不知道某个 checkpoint_id 对应 List 里的第几个位置，只能从头扫，效率很低。

Hash 可以直接用 checkpoint_id 当 field 名，`HGET` 一次就拿到，O(1) 复杂度。

另外压缩时需要**精确删除中间某条消息**，Hash 支持按 field 删，List 做不到。

---

## Q2：checkpoint_writes 为什么不用 String 存？

因为一个节点可能有**多个输出**，String 只能存一个值。

比如 Agent 并行调用了 3 个工具，就会有 3 条 writes：

```
checkpoint_writes:...:task0:0   → 工具1的输出
checkpoint_writes:...:task0:1   → 工具2的输出
checkpoint_writes:...:task0:2   → 工具3的输出
```

用 Hash 可以把这些输出都挂在同一个 key 下，按 field 区分，读的时候一次 `HGETALL` 全部拿到。

如果用 String，就得建 3 个独立的 key，管理起来麻烦，而且没法原子性地读取"这一步所有的输出"。

---

## Q3：ltm:index 为什么用 Set 不用 List？

因为记忆 ID **不能重复**，Set 自动去重，List 不行。

比如同一条记忆被触发保存两次，用 List 就存了两个重复 ID，后面 `load_all()` 遍历时就会读两遍同一条记忆，注入到 prompt 里就乱了。

用 Set 的话，`SADD` 同一个 ID 两次，第二次直接忽略，天然保证唯一性，不需要自己写去重逻辑。

---

## Q4：ltm:mem 为什么用 Hash 不用 String？

因为一条记忆有**多个字段**，type、name、description、content、created_at，如果用 String 就得把这些字段打包成一个 JSON 字符串存进去。

这样有个问题：每次只想更新 `content` 一个字段，也得把整个 JSON 读出来，改完再整体写回去。

用 Hash 的话，每个字段独立存，想改哪个就 `HSET` 哪个，想读哪个就 `HGET` 哪个，不用动其他字段。

---

## Q5：messages 为什么用 Hash 存？

messages 其实**不适合用 Hash 存**，它是一个有序列表，天然应该用 List。

但这里存的不是单独的 messages，而是**整个 state 对象**，messages 只是 state 里的一个字段：

```
Hash 的 field: channel_values
Hash 的 value: {"messages": [...], "compress_choice": "..."}
                    ↑
              整个 JSON 字符串，messages 在里面
```

messages 本身是被序列化成 JSON 字符串，塞进 Hash 的一个 field 里的，不是直接用 Redis 数据类型存的。

选 Hash 是因为 state 有多个字段（channel_values、channel_versions、versions_seen），需要按字段读写，所以 state 这一层用 Hash。messages 列表在 JSON 内部，Redis 感知不到它的结构。

---

## Q6：channel_versions 和 versions_seen 分别存什么？

这两个是 LangGraph 内部的**并发控制**字段，不是业务数据。

**channel_versions** — 记录每个 channel 当前的版本号：

```json
{
  "messages": 10,
  "compress_choice": 3
}
```

意思是 messages 这个 channel 已经被更新了 10 次。

**versions_seen** — 记录这个 checkpoint 是基于哪个版本生成的：

```json
{
  "agent": {"messages": 9},
  "compress": {"messages": 7}
}
```

两个配合起来判断某个节点的输出是否过期：

```
versions_seen["agent"]["messages"] == channel_versions["messages"]
        ↑ 我看到的                          ↑ 当前实际的
相等 → 没人动过，我的结果有效
不等 → 中间有人改过，我的结果过期了
```

本质是**乐观锁**，不用真正加锁，靠版本号比对来保证一致性。

---

## Q7：短期记忆的完整字段有哪些？

一个完整的 checkpoint Hash 有以下字段：

```
Key: checkpoint:session_xxx_coder::01J1ABC003...
Type: Hash

Field                Value
─────────────────────────────────────────────────────
channel_values       {
                       "messages": [...所有消息...],
                       "compress_choice": ""
                     }

channel_versions     {
                       "messages": 3,
                       "compress_choice": 1
                     }

versions_seen        {
                       "agent": {"messages": 2},
                       "tools": {"messages": 1}
                     }

checkpoint_ns        ""

checkpoint_id        "01J1ABC003..."
```

三类字段：

-   **业务数据** — `channel_values`，真正的对话内容在这里
-   **并发控制** — `channel_versions` + `versions_seen`，乐观锁用
-   **元数据** — `checkpoint_ns` + `checkpoint_id`，标识这个 checkpoint 是谁

---

## Q8：记忆压缩具体是怎么做的？比如现在有 100 条消息

### 第一步：检测是否需要压缩

每次 Agent 节点执行完，token 路由器先算用量：

```
token / (163840 - 2000) = 使用率

< 50%   → 直接给 Agent，不管
50~92%  → 暂停，问用户要不要压缩
≥ 92%   → 直接触发压缩
```

### 第二步：Phase 1 轻量清理

把 100 条里的**旧 ToolMessage** 找出来删掉（保留最近 3 条不动）：

```
前97条里有 30 条 ToolMessage → 删掉
剩 70 条，重新算 token
  → 不超 92% → 完成，结束
  → 还超     → 进入 Phase 2
```

### 第三步：Phase 2 LLM 生成摘要

保留最近 3 条不动，前 97 条全部压缩成 1 条 8 段式摘要：

```
1. 主要请求和意图
2. 关键技术概念
3. 文件和代码段
4. 错误和修复
5. 问题解决
6. 用户消息摘录
7. 待处理任务
8. 当前工作状态
```

### 第四步：写回 Redis

```
压缩前：[msg1...msg97, msg98, msg99, msg100]
压缩后：[摘要, msg98, msg99, msg100]
```

压缩本身不删旧 checkpoint，只新增一个压缩后的 checkpoint，所以需要配合清理逻辑（保留最新 2 个，删除其余）。

---

## Q9：压缩操作是在 Redis 层面做的吗？

不是，是在 **Python 内存里**做的。Redis 对 messages 的内部结构完全无感知。

完整流程：

```
从 Redis 读出 checkpoint
  ↓
反序列化 channel_values 的 JSON 字符串
  ↓
得到 Python 列表：[msg1, msg2, ..., msg100]
  ↓
在内存里遍历，isinstance(m, ToolMessage) 判断类型
  ↓
生成 RemoveMessage 列表 / 调 LLM 生成摘要
  ↓
序列化成 JSON，写回 Redis 新 checkpoint
```

Redis 在这里只是个**持久化容器**，业务逻辑全在 Python 里。

---

## Q10：为什么 Redis 存了还需要 MySQL？

Redis 有两个致命问题：

**问题一：数据会丢**

Redis 默认把数据存在内存里，服务器重启、内存不够触发 LRU 淘汰、或者崩了，数据就没了。MySQL 写磁盘，天然持久化。

**问题二：内存贵，存不了太多**

```
100 个会话 × 平均 500 条消息 × 每条 1KB = 50MB    还好
10000 个会话                               = 5GB     开始心疼
100000 个会话                              = 50GB    放不下
```

两者分工：

```
Redis   → 热数据，当前正在跑的会话，要求极速读写
MySQL   → 冷数据，已结束的会话归档，要求持久可靠
```

---

## Q11：会话结束后是怎么触发归档到 MySQL 的？

**用户退出对话时触发，不是自动的。**

```
用户输入 exit / Ctrl+C
  ↓
主循环结束，走到 finally 块（main.py:759）
  ↓
_archive_to_mysql() 执行
  ↓
从 Redis 读最新 checkpoint 里的 messages
  ↓
整体序列化写入 MySQL conversations 表
  ↓
关闭 MySQL 连接
```

潜在问题：如果进程被强杀（`kill -9`）或服务器断电，`finally` 来不及执行，会话不会归档到 MySQL。更健壮的做法是每隔 N 轮对话就归档一次。

---

## Q12：Redis 的数据结构是如何与 MySQL 对齐的？

归档时只提取 `channel_values.messages`，其他字段全部丢弃：

| | Redis | MySQL |
|---|---|---|
| 存的内容 | 完整 state（messages + 控制字段）| 只存 messages |
| 格式 | 多个 Hash field | 一整条 LONGTEXT |
| 条数 | 每步一个 checkpoint，共 N 个 | 一个 thread_id 只有一行 |
| 用途 | 断点恢复、回滚 | 历史查询、跨会话持久化 |

恢复时反向对齐：

```
MySQL LONGTEXT
  ↓ loads() 反序列化
Python messages 列表
  ↓ aupdate_state()
Redis 新 checkpoint Hash
```

**一句话总结**：Redis 存运行时完整快照，MySQL 存业务数据的最终归档。归档时瘦身（只保留 messages），恢复时再重新膨胀回 Redis 格式。

---

## Q13 完整记忆过程示例

> 以用户输入 **"帮我用 Python 写一个快速排序"** 为例，走完整流程。

---

### 阶段一：对话执行中（短期记忆写入）

**Step 1：用户输入**

Agent 节点执行完，LangGraph 写入：

```
# 指针 key（新建）
Key:   checkpoint:session_proj_20260604_1430_coder::
Hash:  {checkpoint_id: "01J1ABC001"}

# 快照 key
Key:   checkpoint:session_proj_20260604_1430_coder::01J1ABC001
Hash:
  channel_values   → {"messages": [
                         {"type": "human", "content": "帮我用 Python 写一个快速排序", "id": "m001"}
                       ], "compress_choice": ""}
  channel_versions → {"messages": 1}
  versions_seen    → {"agent": {"messages": 0}}
  checkpoint_id    → "01J1ABC001"

# writes key
Key:   checkpoint_writes:session_proj_20260604_1430_coder::01J1ABC001:task0:0
Hash:  {channel: "messages", value: '[{"type":"human","content":"..."}]'}
```

---

**Step 2：Agent 调工具**

```
# 指针更新
Key:   checkpoint:session_proj_20260604_1430_coder::
Hash:  {checkpoint_id: "01J1ABC002"}    ← 更新

# 新快照
Key:   checkpoint:session_proj_20260604_1430_coder::01J1ABC002
Hash:
  channel_values   → {"messages": [
                         {"type": "human", "content": "帮我用 Python 写一个快速排序", "id": "m001"},
                         {"type": "ai",    "content": "", "tool_calls": [{"name": "code_exec", "args": {"code": "..."}}], "id": "m002"}
                       ], "compress_choice": ""}
  channel_versions → {"messages": 2}
  versions_seen    → {"agent": {"messages": 1}}
  checkpoint_id    → "01J1ABC002"

# writes key
Key:   checkpoint_writes:...:01J1ABC002:task0:0
Hash:  {channel: "messages", value: '[{"type":"ai","tool_calls":[...]}]'}
```

---

**Step 3：工具返回结果**

```
# 指针更新
Key:   checkpoint:session_proj_20260604_1430_coder::
Hash:  {checkpoint_id: "01J1ABC003"}

# 新快照
Key:   checkpoint:session_proj_20260604_1430_coder::01J1ABC003
Hash:
  channel_values   → {"messages": [
                         {"type": "human",  "content": "帮我用 Python 写一个快速排序", "id": "m001"},
                         {"type": "ai",     "content": "", "tool_calls": [...], "id": "m002"},
                         {"type": "tool",   "content": "代码执行成功，输出：[1,2,3]", "id": "m003"}
                       ], "compress_choice": ""}
  channel_versions → {"messages": 3}
  versions_seen    → {"tools": {"messages": 2}}
  checkpoint_id    → "01J1ABC003"
```

---

**Step 4：Agent 生成最终回复**

```
Key:   checkpoint:session_proj_20260604_1430_coder::01J1ABC004
Hash:
  channel_values   → {"messages": [
                         {"type": "human", "content": "帮我用 Python 写一个快速排序", "id": "m001"},
                         {"type": "ai",    "content": "", "tool_calls": [...], "id": "m002"},
                         {"type": "tool",  "content": "代码执行成功，输出：[1,2,3]", "id": "m003"},
                         {"type": "ai",    "content": "这是快速排序代码：n```pythonndef quicksort...```", "id": "m004"}
                       ], "compress_choice": ""}
  channel_versions → {"messages": 4}
```

**此时 Redis 全貌（4步完成后）：**

```
checkpoint:session_proj_20260604_1430_coder::                   ← 指针
checkpoint:session_proj_20260604_1430_coder::01J1ABC001         ← 步骤1快照
checkpoint:session_proj_20260604_1430_coder::01J1ABC002         ← 步骤2快照
checkpoint:session_proj_20260604_1430_coder::01J1ABC003         ← 步骤3快照
checkpoint:session_proj_20260604_1430_coder::01J1ABC004         ← 步骤4快照
checkpoint_writes:...:01J1ABC001:task0:0
checkpoint_writes:...:01J1ABC002:task0:0
checkpoint_writes:...:01J1ABC003:task0:0
checkpoint_writes:...:01J1ABC004:task0:0
                                                  共 9 个 key
```

---

### 阶段二：token 超阈值，触发压缩

假设执行到第 100 步，token 超 92%：

**压缩前（最新 checkpoint）：**

```
channel_values.messages = [msg1, msg2, ..., msg97, msg98, msg99, msg100]
```

**Phase 1：删旧 ToolMessage（在 Python 内存里操作）**

```
从 Redis 读出 → 反序列化 → Python 对象列表
找出前97条里的 ToolMessage → [m003, m007, m011, ...]
模拟删除后 token 还超？→ 超，进 Phase 2
```

**Phase 2：LLM 生成摘要，写入新 checkpoint**

```
Key:   checkpoint:session_proj_20260604_1430_coder::01J1ABCnew
Hash:
  channel_values → {"messages": [
                      {"type": "human", "content":
                        "📋 对话摘要（压缩于 2026-06-04 15:00:00，原97条→摘要1条）
                         1. 主要请求：用户要写 Python 快速排序
                         2. 关键技术：Python、递归、分治
                         3. 文件代码：quicksort.py ...
                         ...", "id": "summary001"},
                      {"type": "ai",   "content": "...", "id": "m098"},
                      {"type": "tool", "content": "...", "id": "m099"},
                      {"type": "ai",   "content": "...", "id": "m100"}
                    ]}
```

**压缩后触发清理，只保留最新 2 个 checkpoint：**

```
删除：checkpoint:...:01J1ABC001 ~ 01J1ABC099（共99个旧快照及对应 writes key）
保留：checkpoint:...:01J1ABCpre（压缩前最后一个，用于回滚）
      checkpoint:...:01J1ABCnew（压缩后新快照，继续跑）
```

---

### 阶段三：会话结束，写入长期记忆

LLM 从对话中提炼有价值的信息：

```
# 索引 Set（新增一个 ID）
Key:    ltm:index
Value:  {"a1b2c3d4"}       ← sadd 写入

# 记忆内容 Hash
Key:    ltm:mem:a1b2c3d4
Hash:
  id           → a1b2c3d4
  type         → feedback
  name         → python-quicksort-comment-preference
  description  → 用户写代码时偏好有注释
  content      → 用户要求写快速排序时明确说"要有注释"，
                 未来写代码默认加行内注释
  created_at   → 2026-06-04T15:00:15.123456
```

---

### 阶段四：会话退出，归档到 MySQL

```
从 Redis 读最新 checkpoint 的 messages
  ↓
LangChain dumps() 序列化成 JSON 字符串
  ↓
MySQL conversations 表写入：

  thread_id     → session_proj_20260604_1430_coder
  messages_json → '[{"type":"human","content":"帮我用 Python 写..."},
                    {"type":"human","content":"📋 对话摘要..."},
                    {"type":"ai","content":"..."},
                    {"type":"tool","content":"..."},
                    {"type":"ai","content":"这是快速排序代码..."}]'
  message_count → 4
  updated_at    → 2026-06-04 15:00:20
```

---

### 最终各层存储状态

```
Redis 短期（checkpoint）
  └── 1 个指针 key
  └── 2 个快照 key（压缩前最后一个 + 压缩后新快照）

Redis 长期（ltm）
  └── ltm:index        (Set：1个ID)
  └── ltm:mem:a1b2c3d4 (Hash：6个字段)

MySQL 冷归档
  └── conversations 表 1 行（完整消息 JSON）
```

---

## Q14：MySQL 的 conversations 表存了什么？

### 表结构

表名固定叫 `conversations`，有 6 列：

| 列名 | 数据类型 | 说明 |
|------|---------|------|
| id | BIGINT | 自增主键 |
| thread_id | VARCHAR(255) | 会话唯一 ID，对应 Redis 的 thread_id |
| messages_json | LONGTEXT | 全部消息序列化后的 JSON 字符串 |
| message_count | INT | 消息条数 |
| created_at | TIMESTAMP | 首次归档时间 |
| updated_at | TIMESTAMP | 最后更新时间 |

列是固定的，行是动态的——每个会话一行，有多少会话就有多少行。

---

### 一行具体内容

```
id            → 1
thread_id     → session_proj_20260604_1430_coder
message_count → 4
created_at    → 2026-06-04 14:30:00
updated_at    → 2026-06-04 15:00:20
```

messages_json 展开：

```json
[
  {
    "type": "constructor",
    "id": ["langchain", "schema", "messages", "HumanMessage"],
    "kwargs": {
      "content": "帮我用 Python 写一个快速排序",
      "id": "m001"
    }
  },
  {
    "type": "constructor",
    "id": ["langchain", "schema", "messages", "AIMessage"],
    "kwargs": {
      "content": "",
      "tool_calls": [{"name": "code_exec", "args": {"code": "def quicksort..."}, "id": "call001"}],
      "id": "m002"
    }
  },
  {
    "type": "constructor",
    "id": ["langchain", "schema", "messages", "ToolMessage"],
    "kwargs": {
      "content": "执行成功，输出：[1, 2, 3, 5, 8]",
      "tool_call_id": "call001",
      "id": "m003"
    }
  },
  {
    "type": "constructor",
    "id": ["langchain", "schema", "messages", "AIMessage"],
    "kwargs": {
      "content": "这是快速排序代码：\n```python\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    ...\n```",
      "id": "m004"
    }
  }
]
```

---

### 三个注意点

**1. 不是裸 JSON，是 LangChain 序列化格式**

每条消息外面包了 `type: constructor` + `id: [类路径]`，目的是反序列化时能还原回正确的 Python 对象（HumanMessage / AIMessage / ToolMessage）。

**2. messages_json 是一整个字符串**

MySQL 不知道里面有几条消息，结构全靠 Python 的 `loads()` 解析。

**3. 和 Redis 的区别**

```
Redis channel_values → {"messages": [...], "compress_choice": ""}  含 LangGraph 控制字段
MySQL messages_json  → [...]                                        只有消息列表，做了瘦身
```