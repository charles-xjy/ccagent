"""
工具调用确认节点工厂

在危险工具执行前暂停，等待用户确认：
  输入 y / yes / 是 / 确认 → 继续执行（返回 {}，tools 节点正常运行）
  其他 / 直接回车          → 取消本轮全部待执行工具，注入拒绝 ToolMessage，返回 agent
"""
import difflib
from pathlib import Path
from typing import Dict, Set
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import interrupt

from core.state import AgentState
from core.tools import WORKDIR

_CONFIRM_WORDS = {"y", "yes", "是", "确认"}
_DIFF_CONTEXT = 3   # unified diff 上下文行数
_DIFF_MAX = 60      # 最多展示的 diff 行数


def _unified_diff(old: str, new: str) -> str:
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    diff = list(difflib.unified_diff(old_lines, new_lines,
                                     lineterm="", n=_DIFF_CONTEXT))
    if not diff:
        return "  (内容无变化)"
    body = diff[2:]          # 跳过 --- +++ 头部
    shown = body[:_DIFF_MAX]
    out = []
    for ln in shown:
        if ln.startswith("+"):
            out.append(f"\033[32m{ln}\033[0m")   # 绿色：新增
        elif ln.startswith("-"):
            out.append(f"\033[31m{ln}\033[0m")   # 红色：删除
        else:
            out.append(ln)
    if len(body) > _DIFF_MAX:
        out.append(f"\033[33m  ... (共 {len(body)} 行变更，仅显示前 {_DIFF_MAX} 行)\033[0m")
    added   = sum(1 for l in body if l.startswith("+"))
    removed = sum(1 for l in body if l.startswith("-"))
    header  = f"\033[33m  +{added} 行  -{removed} 行\033[0m"
    return header + "\n" + "\n".join(out)


def _preview(name: str, args: dict) -> str:
    if name == "write_file":
        path = args.get("path", "?")
        new_content = args.get("content", "")
        full_path = (WORKDIR / path).resolve()
        if full_path.exists():
            try:
                old_content = full_path.read_text(encoding="utf-8", errors="replace")
                return f"路径: {path}\n{_unified_diff(old_content, new_content)}"
            except Exception:
                pass
        # 新建文件：显示前 30 行
        lines = new_content.splitlines()
        shown = lines[:30]
        body = "\n".join(f"\033[32m+  {l}\033[0m" for l in shown)
        suffix = f"\n\033[33m  ... (共 {len(lines)} 行)\033[0m" if len(lines) > 30 else ""
        return f"新建文件: {path}  (+{len(lines)} 行)\n{body}{suffix}"

    if name == "edit_file":
        path = args.get("path", "?")
        old_text = args.get("old_text", "")
        new_text = args.get("new_text", "")
        return f"路径: {path}\n{_unified_diff(old_text, new_text)}"

    if name in ("run_in_sandbox", "run_python_test"):
        return f"文件: {args.get('file_path', '?')}"

    return str(args)[:80]


def create_tool_confirm_node(dangerous_tools: Set[str], agent_label: str):
    """返回一个异步确认节点，仅当有危险工具调用时才触发 interrupt。"""

    async def tool_confirm(state: AgentState) -> Dict:
        last_msg = state["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", []) or []
        dangerous = [tc for tc in tool_calls if tc["name"] in dangerous_tools]
        if not dangerous:
            return {}

        lines = [f"  • [{tc['name']}] {_preview(tc['name'], tc['args'])}"
                 for tc in dangerous]
        choice = interrupt(
            f"🔐 [{agent_label}] 即将执行以下操作，请确认：\n\n"
            + "\n".join(lines)
        )

        val = str(choice).strip().lower()
        if val in _CONFIRM_WORDS:
            return {}

        abort_msg = "用户已中止本次任务，请停止执行并输出已完成内容的摘要。" if val == "abort" \
                    else "用户已取消此操作。请重新考虑方案，或向用户询问下一步。"

        return {"messages": [
            ToolMessage(content=abort_msg, tool_call_id=tc["id"])
            for tc in tool_calls
        ]}

    return tool_confirm


def make_agent_router(dangerous_tools: Set[str]):
    """替换 should_continue_sub：有危险调用 → tool_confirm，无危险 → tools，无调用 → __end__"""
    def route(state: AgentState) -> str:
        last_msg = state["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", []) or []
        if not tool_calls:
            return "__end__"
        if any(tc["name"] in dangerous_tools for tc in tool_calls):
            return "tool_confirm"
        return "tools"
    return route


def route_after_tool_confirm(state: AgentState) -> str:
    """确认节点出口：最后一条是 ToolMessage（用户拒绝）→ agent，否则 → tools"""
    if isinstance(state["messages"][-1], ToolMessage):
        return "agent"
    return "tools"
