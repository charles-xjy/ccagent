"""
轻量级 Agent 运行器。

run_agent(task, model, skills=[...]) 是一个独立 async 协程：
- 从 TOOL_REGISTRY 取全量工具
- 通过 skills 列表注入身份 + 领域知识到 system prompt
- 跑自己的 agent 循环（最多 MAX_TURNS 轮）
- 无 LangGraph、无 checkpoint，任务完即销毁
- 多个 run_agent() 可通过 asyncio.gather 并行执行
"""

from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool as lc_tool

from core.agent_loader import build_skill_index, load_skill, resolve_allowed_tools
from core.tools import TOOL_REGISTRY

MAX_TURNS = 30


async def run_agent(
    task: str,
    model,
    *,
    skills: Optional[List[str]] = None,
    extra_tools: Optional[list] = None,
) -> str:
    """
    启动一个一次性 agent，完成任务后返回最终文本输出。

    Args:
        task:        任务描述
        model:       LangChain chat model 实例
        skills:      注入的 skill 名称列表，第一项通常是身份 skill
                     （如 ["coder", "react-patterns"]）
        extra_tools: 额外工具（如 MCP tools），追加到全量工具集后
    """
    # ── 工具集：全量 TOOL_REGISTRY + extra_tools + load_skill ────────────────
    tools = list(TOOL_REGISTRY.values())

    if extra_tools:
        existing = {t.name for t in tools}
        for t in extra_tools:
            if t.name not in existing:
                tools.append(t)
                existing.add(t.name)

    @lc_tool
    def load_skill_tool(skill_name: str) -> str:
        """加载指定 skill 的详细文档。需要参考规范、API 用法或项目约定时调用。"""
        return load_skill(skill_name)

    tools.append(load_skill_tool)

    # ── allowed_tools 过滤：取所有已加载 skill 的并集 ────────────────────────
    # None = 无任何 skill 声明限制，保持全量（兼容纯知识 skill）
    allowed = resolve_allowed_tools(skills or [])
    if allowed is not None:
        tools = [t for t in tools if t.name in allowed]

    tools_by_name = {t.name: t for t in tools}
    model_with_tools = model.bind_tools(tools)

    # ── System Prompt：预加载 skills 正文 + skill 目录 ────────────────────────
    parts = []
    for s in (skills or []):
        content = load_skill(s)
        parts.append(content)

    idx = build_skill_index()
    if idx:
        parts.append(idx)

    system_content = "\n\n".join(parts) if parts else "你是一个通用助手。"

    # ── Agent 循环 ────────────────────────────────────────────────────────────
    messages = [SystemMessage(content=system_content), HumanMessage(content=task)]

    for _turn in range(MAX_TURNS):
        response = await model_with_tools.ainvoke(messages)
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            return response.content or ""

        tool_results = []
        for tc in tool_calls:
            name = tc["name"]
            tool_obj = tools_by_name.get(name)
            try:
                if tool_obj is None:
                    obs = f"Error: 工具 '{name}' 不存在。"
                else:
                    obs = await tool_obj.ainvoke(tc["args"])
                    if isinstance(obs, str) and len(obs) > 10000:
                        obs = obs[:10000] + "\n...(已截断)"
            except Exception as e:
                obs = f"Error executing {name}: {e}"

            tool_results.append(ToolMessage(content=str(obs), tool_call_id=tc["id"]))
            preview = str(obs)[:80].replace("\n", " ")
            print(f"    \033[34m[agent]\033[0m {name}: {preview}")

        messages.extend(tool_results)

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return f"[达到最大轮次 {MAX_TURNS}]\n{msg.content}"
    return f"达到最大轮次 ({MAX_TURNS})，无有效输出。"
