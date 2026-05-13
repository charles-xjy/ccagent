"""
上下文压缩模块（仿 Claude Code 两阶段压缩策略）

Token 用量路由：
  < 50%            → agent（正常）
  50% ~ 75%        → warn 节点（⚠️  提醒，用户选择）
  75% ~ 85%        → warn 节点（🚨  强提醒，用户选择）
  ≥ 85%            → compress 节点（自动压缩）

压缩两阶段：
  Phase 1: 优先删除旧 ToolMessage（工具输出体积大、价值低）
  Phase 2: 若仍超阈值，调用 LLM 生成 8 段式摘要替换旧消息
"""
import json
from datetime import datetime
from typing import List

from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage,
    RemoveMessage, SystemMessage, ToolMessage,
)
from langgraph.types import interrupt

# ── 压缩触发参数 ──────────────────────────────────────────────
KEEP_RECENT = 3         # 无论如何保留最近 N 条消息原文
MAX_TOKENS = 163840     # Qwen3.5-27B，vLLM --max-model-len 163840
TOKEN_THRESHOLD = 0.92  # 使用率超过此比例自动压缩

# ── 8 段式压缩 Prompt（源自 Claude Code 设计）──────────────────
COMPRESSION_PROMPT = """你的任务是将以下 AI Agent 对话历史压缩为详细摘要。
摘要必须保留足够的信息，使另一个 AI 实例能够无缝继续工作。

请按以下 8 个部分组织摘要：

1. **主要请求和意图** — 用户的所有明确请求和核心目标
2. **关键技术概念** — 涉及的技术栈、框架、架构决策
3. **文件和代码段** — 检查/修改/创建的文件（含关键代码片段和路径）
4. **错误和修复** — 遇到的具体错误及解决方案
5. **问题解决** — 已解决的问题和进行中的故障排除
6. **用户消息摘录** — 按时间顺序列出所有用户输入（非工具结果）
7. **待处理任务** — 明确被要求但尚未完成的任务
8. **当前工作状态** — 压缩时刻正在进行的工作及下一步计划

对话历史：
"""


# ── Token 估算 ────────────────────────────────────────────────

def estimate_tokens(messages: List[BaseMessage]) -> int:
    total = 0
    for msg in messages:
        content = msg.content if hasattr(msg, "content") else str(msg)
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False)
        total += len(str(content))
    return int(total / 3)  # 中英文混合约 3 字符/token


def get_token_usage(messages: List[BaseMessage]) -> int:
    """优先从 response_metadata 读取精确值，否则估算"""
    for msg in reversed(messages):
        usage = getattr(msg, "response_metadata", {}).get("token_usage", {})
        if usage:
            total = usage.get("total_tokens", 0)
            if total > 0:
                return total
    return estimate_tokens(messages)


def needs_compression(
    messages: List[BaseMessage],
    max_tokens: int = MAX_TOKENS,
    threshold: float = TOKEN_THRESHOLD,
) -> bool:
    if len(messages) < KEEP_RECENT + 2:
        return False
    current = get_token_usage(messages)
    available = max_tokens - 2000  # 为输出预留
    return (current / available) > threshold


# ── 工厂函数 ──────────────────────────────────────────────────

def create_compression_node(model):
    """返回绑定了 model 的异步压缩节点函数。"""

    async def compression_node(state: dict) -> dict:
        messages: List[BaseMessage] = state.get("messages", [])

        if len(messages) <= KEEP_RECENT:
            return {}

        boundary = len(messages) - KEEP_RECENT
        old_msgs = messages[:boundary]
        recent = messages[boundary:]

        # ── Phase 1: 尝试只删除旧 ToolMessage ──────────────────
        old_tool_indices = [i for i, m in enumerate(old_msgs) if isinstance(m, ToolMessage)]
        if old_tool_indices:
            pruned = [m for i, m in enumerate(messages) if i not in set(old_tool_indices)]
            if not needs_compression(pruned):
                to_remove = [old_msgs[i] for i in old_tool_indices]
                removes = [RemoveMessage(id=m.id) for m in to_remove if hasattr(m, "id")]
                print(f"\033[33m[压缩-P1] 清理 {len(removes)} 条旧 ToolMessage\033[0m")
                return {"messages": removes}

        # ── Phase 2: 生成摘要替换旧消息 ────────────────────────
        # 若 recent[0] 是悬空 ToolMessage，纳入压缩范围
        if recent and isinstance(recent[0], ToolMessage):
            old_msgs = old_msgs + [recent[0]]
            recent = recent[1:]

        lines = []
        for msg in old_msgs:
            content = msg.content if hasattr(msg, "content") else ""
            if isinstance(content, list):
                content = json.dumps(content, ensure_ascii=False)
            role = getattr(msg, "type", "?").upper()
            lines.append(f"[{role}]: {str(content)[:600]}")

        try:
            tokens_before = get_token_usage(messages)
            resp = await model.ainvoke([
                SystemMessage(content=COMPRESSION_PROMPT),
                HumanMessage(content="\n".join(lines)),
            ])
            summary_msg = AIMessage(
                content=(
                    f"📋 **对话摘要** (压缩于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}，"
                    f"原 {len(old_msgs)} 条 → 摘要 1 条)\n\n{resp.content}"
                )
            )
            tokens_after = estimate_tokens([summary_msg] + list(recent))
            print(
                f"\033[32m[压缩-P2] {len(old_msgs)} 条 → 摘要 | "
                f"tokens: {tokens_before:,} → {tokens_after:,}\033[0m"
            )

            # 删除所有旧消息，按 [summary, recent...] 顺序重建
            # add_messages 处理顺序：先执行 Remove，再追加新消息
            removes = [RemoveMessage(id=m.id) for m in messages if hasattr(m, "id")]
            return {"messages": removes + [summary_msg] + list(recent)}

        except Exception as e:
            print(f"\033[31m[压缩] 生成摘要失败: {e}，跳过\033[0m")
            return {}

    return compression_node


def create_warn_node():
    """返回 warn 节点：用 interrupt() 暂停图，等待用户选择是否压缩。"""
    async def warn_node(state: dict) -> dict:
        messages = state.get("messages", [])
        current = get_token_usage(messages)
        available = MAX_TOKENS - 2000
        ratio = current / available
        level = "🚨" if ratio >= 0.75 else "⚠️ "
        choice = interrupt(
            f"{level} Token 使用率 {ratio:.0%}（{current:,} / {MAX_TOKENS:,} tokens）\n"
            "是否立即压缩上下文？输入 compress 压缩，直接回车跳过"
        )
        return {"compress_choice": str(choice).strip().lower()}
    return warn_node


def route_after_warn(state: dict) -> str:
    """warn 节点执行完后的路由：根据用户选择决定是否压缩。"""
    return "compress" if state.get("compress_choice") == "compress" else "agent"


def make_token_router(
    max_tokens: int = MAX_TOKENS,
    threshold: float = TOKEN_THRESHOLD,
    warn_high: float = 0.75,
    warn_low: float = 0.50,
):
    """返回路由函数：根据 token 用量决定下一步节点。"""
    def token_router(state: dict) -> str:
        messages = state.get("messages", [])
        if len(messages) < KEEP_RECENT + 2:
            return "agent"
        current = get_token_usage(messages)
        available = max_tokens - 2000
        ratio = current / available
        if ratio >= threshold:
            return "compress"
        if ratio >= warn_low:   # 50% ~ 85% 均走 warn
            return "warn"
        return "agent"
    return token_router
