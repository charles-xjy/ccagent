"""
spawn_agents 工具：Supervisor 调用后并行启动多个子 Agent。

使用前必须调用 init_spawn_tool(model, mcp_tools) 完成初始化。
"""

import asyncio
from typing import List

from langchain_core.tools import tool as lc_tool

_model = None
_mcp_tools: list = []


def init_spawn_tool(model, mcp_tools: list = None) -> None:
    """在 main.py 启动时调用，注入 model 和 MCP tools。"""
    global _model, _mcp_tools
    _model = model
    _mcp_tools = mcp_tools or []


def create_spawn_tool():
    """
    构建 spawn_agents 工具函数。
    在 build_skill_index() 可用后调用，以便将 skill 目录写入 docstring。
    """
    from core.agent_loader import list_skills

    skills = list_skills()
    catalog = "\n".join(
        f"  - `{s['name']}`: {s['description']}" for s in skills
    ) or "  （skills/ 目录为空）"

    description = (
        "并行启动多个专家 Agent，全部完成后返回汇总报告。\n\n"
        "参数 agents 是一个列表，每项格式：\n"
        '  {"task": "<具体任务>", "skills": ["<身份skill>", "<领域skill>", ...], "name": "<可选别名>"}\n\n'
        "skills 列表说明：\n"
        "- 第一项通常是身份 skill（决定 agent 是谁、用什么工具）\n"
        "- 后续项是领域知识 skill（按需加载）\n\n"
        f"可用 skill：\n{catalog}\n\n"
        "示例：\n"
        '  [{"task": "实现 UserCard 组件", "skills": ["coder", "react-patterns"], "name": "前端"},\n'
        '   {"task": "搜索 React 18 并发特性", "skills": ["researcher"], "name": "调研"}]'
    )

    @lc_tool(description=description)
    async def spawn_agents(agents) -> str:
        import json as _json
        from core.agent_runner import run_agent

        if isinstance(agents, str):
            try:
                agents = _json.loads(agents)
            except Exception:
                return "Error: agents 参数解析失败，请传入合法的 JSON 列表。"
        if not isinstance(agents, list):
            return "Error: agents 必须是列表。"
        if not agents:
            return "Error: agents 列表为空。"
        if _model is None:
            return "Error: spawn_tool 未初始化，请先调用 init_spawn_tool()。"

        print(f"\n\033[35m[Supervisor] 并行启动 {len(agents)} 个子 Agent\033[0m")

        async def _run_one(spec: dict) -> dict:
            task   = spec.get("task", "")
            skills = spec.get("skills") or []
            label  = spec.get("name") or (skills[0] if skills else "agent")

            if not task:
                return {"name": label, "ok": False, "result": "Error: 缺少 task 字段"}

            print(f"  \033[36m▶ [{label}]\033[0m {task[:60]}")
            try:
                result = await run_agent(
                    task,
                    _model,
                    skills=skills,
                    extra_tools=_mcp_tools if _mcp_tools else None,
                )
                print(f"  \033[32m✓ [{label}]\033[0m 完成")
                return {"name": label, "ok": True, "result": result}
            except Exception as e:
                print(f"  \033[31m✗ [{label}]\033[0m 失败: {e}")
                return {"name": label, "ok": False, "result": f"Error: {e}"}

        results = await asyncio.gather(*[_run_one(s) for s in agents])

        lines = [f"=== 子 Agent 执行报告（共 {len(results)} 个）===\n"]
        for r in results:
            icon = "✅" if r["ok"] else "❌"
            lines.append(f"{icon} **{r['name']}**\n{r['result']}\n")

        return "\n".join(lines)

    return spawn_agents
