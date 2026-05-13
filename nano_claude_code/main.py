import asyncio
import json
from typing import Dict, List, Literal
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from core.state import AgentState
from core.config import get_model
from core.tools import WORKDIR
from coder_agent.agent import create_coder_agent
from researcher_agent.agent import create_researcher_agent
from reviewer_agent.agent import create_reviewer_agent


@tool
def todo_manager(items: List[dict]) -> str:
    """更新当前会话的任务列表

    ## 任务对象必须包含以下键：
    - 'id': 任务编号 (如 "1")
    - 'text': 任务描述内容 (注意是 'text' 而不是 'description')
    - 'status': 状态，只能是 "pending", "in_progress", "completed", "failed" 之一。
    """
    status_headers = {
        "in_progress": "🔄进行中:",
        "pending": "⏳待处理:",
        "completed": "✅已完成:",
        "failed": "❌失败:"
    }
    lines = []
    for item in items:
        raw_status = item.get("status", "❌error")
        header = status_headers.get(raw_status, "❌error")
        tid = item.get("id", "❌id?")
        text = item.get("text") or item.get("description", "❌无内容")
        lines.append(f"{header} #{tid} {text}")
    if not lines:
        return "任务列表为空。"
    return f"--- 当前任务面板 ---\n" + "\n".join(lines)


# ==========================================================================
# task_tool 仅作为"虚拟工具"存在 —— 其 tool schema 用于让 Manager LLM 理解
# 可用的子 agent 类型及其参数。实际执行在子图 wrapper 节点中完成。
# 此函数不会被 tools 节点调用（tools_by_name 中不包含它）。
# ==========================================================================
@tool
def task_tool(description: str, subagent_type: str) -> str:
    """
    委派复杂的专项任务给专家处理。
    subagent_type 必须是 'tech-researcher'、'coder' 或 'reviewer'。
    """
    return ""  # 虚拟工具，实际执行在子图节点中


# ==========================================================================
# 子图 wrapper 工厂函数 —— 为每个子 agent 生成原生子图挂载节点
# ==========================================================================
def create_subagent_wrapper(agent, subagent_type: str, thread_id: str):
    """
    创建一个异步节点函数，该函数：
    1. 从父图 state 中提取 Manager 发出的 task_tool 调用参数
    2. 构造子图初始 state（隔离的 messages）
    3. 以独立 thread_id 隔离记忆的方式调用子图
    4. 将子图输出作为 ToolMessage 返回，闭合 Manager 的 tool_call
    """

    async def run_subagent(state: AgentState) -> Dict:
        # 从消息历史中找到最近的 task_tool 调用
        task_desc = ""
        tool_call_id = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
                for tc in msg.tool_calls:
                    if (tc["name"] == "task_tool"
                            and tc["args"].get("subagent_type") == subagent_type):
                        task_desc = tc["args"]["description"]
                        tool_call_id = tc["id"]
                        break
            if task_desc:
                break

        if not task_desc:
            return {"messages": [ToolMessage(
                content=f"Error: 未找到 {subagent_type} 子图的任务描述",
                tool_call_id="unknown"
            )]}

        print(f"\n\033[35m[系统] >>> 子 Agent ({subagent_type}) 开始工作...\033[0m")

        sub_state = {
            "messages": [HumanMessage(content=task_desc)],
            "current_todo": []  # 保持 AgentState 结构完整性
        }
        sub_config = {
            "configurable": {
                "thread_id": thread_id,  # 独立 thread_id 隔离各子 agent 记忆
            }
        }

        full_output = ""
        async for chunk in agent.astream(sub_state, sub_config, stream_mode="updates"):
            for node, data in chunk.items():
                print(f"  \033[34m└─ [{subagent_type}.{node}]\033[0m 正在处理...")
                if "messages" in data:
                    for msg in data["messages"]:
                        if isinstance(msg, AIMessage) and msg.content:
                            preview = msg.content[:200].replace("\n", " ")
                            print(f"    \033[37m思考: {preview}...\033[0m")
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            for tc in msg.tool_calls:
                                print(f"    \033[32m工具调用: {tc['name']}\033[0m")
                        if isinstance(msg, AIMessage):
                            full_output = msg.content

        print(f"\033[35m[系统] <<< 子 Agent ({subagent_type}) 任务完成。\033[0m")

        return {"messages": [ToolMessage(
            content=f"--- SubAgent [{subagent_type}] 执行报告 ---\n\n{full_output}",
            tool_call_id=tool_call_id
        )]}

    return run_subagent


async def main():
    print("\033[34m[*] 系统正在启动...\033[0m")
    model = get_model()

    DB_URI = "redis://10.129.107.145:6379"
    async with AsyncRedisSaver.from_conn_string(DB_URI) as checkpointer:
        print("[*] 正在同步专家组 (Executor SubAgents) 配置...")

        coder_agent = create_coder_agent(model, checkpointer)
        researcher_agent = await create_researcher_agent(model, checkpointer)
        reviewer_agent = create_reviewer_agent(model, checkpointer)

        # =====================================================================
        # Manager 工具配置
        # task_tool 用于 LLM 路由选择（会出现在 tool schema 中），
        # 但不在 tools_by_name 中 —— 路由在条件边中拦截，不进入 tools 节点
        # =====================================================================
        manager_tools = [todo_manager, task_tool]
        model_with_tools = model.bind_tools(manager_tools)
        tools_by_name = {"todo_manager": todo_manager}  # 只有真正的工具

        # =====================================================================
        # 图节点定义
        # =====================================================================
        async def call_manager_model(state: AgentState) -> Dict:
            todo_status = json.dumps(state.get("current_todo", []), ensure_ascii=False)
            system_prompt = SystemMessage(content=(
                f"你是一个高级编程协调员 (Manager)。当前工作路径: {WORKDIR}\n"
                f"当前任务计划进度: {todo_status}\n\n"
                "核心操作守则：\n"
                "1. 规划优先：面对复杂任务，必须先调用 'todo_manager' 编排分步计划。\n"
                "2. 专家委派：你自己不直接写代码或查文档。请通过 'task_tool' 指挥专家：\n"
                "   - 查阅技术资料 -> 调用 'tech-researcher'\n"
                "   - 编写、修改代码或执行只读 Bash -> 调用 'coder'\n"
                "   - 审查代码质量、运行测试 -> 调用 'reviewer'\n"
                "3. 状态闭环：开始执行步骤前更新为 'in_progress'，完成后更新为 'completed'。\n"
                "4. 审查强制：'coder' 完成代码编写/修改后，你必须立即调用 'reviewer' 对该代码进行审查。\n"
                "   reviewer 会运行测试、检查代码质量。只有 reviewer 确认通过后，才能将任务标记为 completed。\n"
                "5. 严谨性：在委派 'coder' 修改文件前，必须确保自己或专家已调用过 'read_file'。"
            ))
            messages = state["messages"]
            response = await model_with_tools.ainvoke([system_prompt] + messages)
            return {"messages": [response]}

        async def execute_todo_only(state: AgentState) -> Dict:
            """仅执行 todo_manager 工具；task_tool 调用已被条件边拦截"""
            last_message = state["messages"][-1]
            updates = {"messages": []}
            if hasattr(last_message, "tool_calls"):
                for tool_call in last_message.tool_calls:
                    name = tool_call["name"]
                    if name == "task_tool":
                        continue  # 防御性跳过

                    print(f"\n\033[33m[Manager 正在执行工具: {name}]\033[0m")

                    if name == "todo_manager":
                        updates["current_todo"] = tool_call["args"]["items"]

                    tool_obj = tools_by_name.get(name)
                    if not tool_obj:
                        observation = f"Error: 工具 '{name}' 未在系统中注册。"
                    else:
                        try:
                            observation = await tool_obj.ainvoke(tool_call["args"])
                            if isinstance(observation, str) and len(observation) > 10000:
                                observation = observation[:10000] + "\n... (内容过长，已自动截断)"
                        except Exception as e:
                            observation = f"Error executing {name}: {e}"
                    updates["messages"].append(ToolMessage(
                        content=str(observation),
                        tool_call_id=tool_call["id"]
                    ))
            return updates

        # =====================================================================
        # 条件路由：解析 Manager 的 tool_calls 决定下一步
        # =====================================================================
        def route_after_agent(state: AgentState) -> Literal[
            "coder", "researcher", "reviewer", "tools", "__end__"
        ]:
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                for tc in last_message.tool_calls:
                    if tc["name"] == "task_tool":
                        target = tc["args"].get("subagent_type", "")
                        if target in ("coder", "tech-researcher", "reviewer"):
                            return "researcher" if target == "tech-researcher" else target
                # 其他 tool calls（如 todo_manager）→ tools 节点
                return "tools"
            return "__end__"

        # =====================================================================
        # 构建主图 (Manager + 原生挂载的子图节点)
        # =====================================================================
        builder = StateGraph(AgentState)

        # 主节点
        builder.add_node("agent", call_manager_model)
        builder.add_node("tools", execute_todo_only)

        # 子图 wrapper 节点 —— 子图作为父图的节点挂载
        builder.add_node("coder",
                         create_subagent_wrapper(coder_agent, "coder", "manager_executor_v2_coder"))
        builder.add_node("researcher",
                         create_subagent_wrapper(researcher_agent, "tech-researcher", "manager_executor_v2_tech-researcher"))
        builder.add_node("reviewer",
                         create_subagent_wrapper(reviewer_agent, "reviewer", "manager_executor_v2_reviewer"))

        # 边
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", route_after_agent, {
            "coder": "coder",
            "researcher": "researcher",
            "reviewer": "reviewer",
            "tools": "tools",
            "__end__": "__end__",
        })
        # 子图 / tools 执行完后回到 agent，形成 agent 循环
        builder.add_edge("coder", "agent")
        builder.add_edge("researcher", "agent")
        builder.add_edge("reviewer", "agent")
        builder.add_edge("tools", "agent")

        app = builder.compile(checkpointer=checkpointer)

        # --- 保存主图到本地 ---
        try:
            graph_image = app.get_graph().draw_mermaid_png()
            with open("manager_graph.png", "wb") as f:
                f.write(graph_image)
            print("\033[32m[OK] 主图 (Manager) 已保存至 manager_graph.png\033[0m")
        except Exception as e:
            print(f"\033[33m[!] 无法生成 PNG (可能缺少 graphviz): {e}\033[0m")
            try:
                mermaid_text = app.get_graph().draw_mermaid()
                with open("manager_graph.mmd", "w") as f:
                    f.write(mermaid_text)
                print("\033[32m[OK] 主图 (Manager) Mermaid 已保存至 manager_graph.mmd\033[0m")
            except Exception as e2:
                print(f"\033[31m[!] 保存 Mermaid 也失败: {e2}\033[0m")

        print("\033[32m[OK] Nano Claude Code (Manager/Executor 架构) 已就绪。\033[0m")
        session_id = "manager_executor_v2"
        session_config = {"configurable": {"thread_id": session_id}}

        # =====================================================================
        # 交互主循环
        # =====================================================================
        while True:
            try:
                print("=" * 60)
                print("请输入您的需求 (输入 q/exit 退出):")
                user_input = input("\n\033[36m >> \033[0m")
                if user_input.strip().lower() in ("q", "exit"):
                    break
                if not user_input.strip():
                    continue
            except (EOFError, KeyboardInterrupt):
                break

            initial_state = {"messages": [HumanMessage(content=user_input)]}
            async for chunk in app.astream(initial_state, session_config, stream_mode="updates"):
                for node_name, node_update in chunk.items():
                    if "messages" in node_update:
                        for msg in node_update["messages"]:
                            print(f"\n--- 节点 [{node_name}] 输出 ---")
                            msg.pretty_print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
