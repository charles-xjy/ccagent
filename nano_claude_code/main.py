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
        # 兼容处理，如果模型传了 description 则取 description，否则取 text
        text = item.get("text") or item.get("description", "❌无内容")
        lines.append(f"{header} #{tid} {text}")
    if not lines:
        return "任务列表为空。"
    return f"--- 当前任务面板 ---\n" + "\n".join(lines)


async def main():
    print("\033[34m[*] 系统正在启动...\033[0m")
    model = get_model()
    
    DB_URI = "redis://10.129.107.145:6379"
    async with AsyncRedisSaver.from_conn_string(DB_URI) as checkpointer:
        print("[*] 正在同步专家组 (Executor SubAgents) 配置...")
        
        # 将实例化的 checkpointer 传递给子 Agent 的工厂函数
        coder_agent = create_coder_agent(model, checkpointer)
        researcher_agent = await create_researcher_agent(model, checkpointer)

        agent_instances = {
            "coder": coder_agent,
            "tech-researcher": researcher_agent
        }

        @tool
        async def task_tool(description: str, subagent_type: str, session_id: str) -> str:
            """
            委派复杂的专项任务给专家处理。
            subagent_type 必须是 'tech-researcher' 或 'coder'。
            session_id 为当前主图的线程 ID，用于保持子 Agent 的记忆连贯性。
            """
            agent = agent_instances.get(subagent_type)
            if not agent:
                return f"Error: 找不到类型为 '{subagent_type}' 的专家。"

            print(f"\n\033[35m[系统] >>> 子 Agent ({subagent_type}) 开始工作...\033[0m")
            full_subagent_output = ""
            
            sub_config = {"configurable": {"thread_id": f"{session_id}_{subagent_type}"}}

            # 在 astream（相当于异步的 invoke/stream）时传入 config，激活记忆读写
            async for chunk in agent.astream({"messages": [HumanMessage(content=description)]}, sub_config, stream_mode="updates"):
                for node, data in chunk.items():
                    print(f"  \033[34m└─ [{subagent_type}.{node}]\033[0m 正在处理...")
                    if "messages" in data:
                        for msg in data["messages"]:
                            if isinstance(msg, AIMessage) and msg.content:
                                print(f"    \033[37m思考: {msg.content}...\033[0m")
                            if isinstance(msg, AIMessage) and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    print(f"    \033[32m工具调用: {tc['name']}\033[0m")
                            if isinstance(msg, AIMessage):
                                full_subagent_output = msg.content

            print(f"\033[35m[系统] <<< 子 Agent ({subagent_type}) 任务完成。\033[0m")
            return f"--- SubAgent [{subagent_type}] 执行报告 ---\n\n{full_subagent_output}"

        manager_tools = [todo_manager, task_tool]
        model_with_tools = model.bind_tools(manager_tools)
        tools_by_name = {t.name: t for t in manager_tools}

        async def call_manager_model(state: AgentState) -> Dict:
            todo_status = json.dumps(state.get("current_todo", []), ensure_ascii=False)
            system_prompt = SystemMessage(content=(
                f"你是一个高级编程协调员 (Manager)。当前工作路径: {WORKDIR}\n"
                f"当前任务计划进度: {todo_status}\n\n"
                "核心操作守则：\n"
                "1. 规划优先：面对复杂任务，必须先调用 'todo_manager' 编排分步计划。\n"
                "2. 专家委派：你自己不直接写代码或查文档。请通过 'task_tool' 指挥专家：\n"
                "   - 查阅 LangChain/LangGraph 技术资料 -> 调用 'tech-researcher'\n"
                "   - 编写、修改、测试代码或运行 Bash -> 调用 'coder'\n"
                "   - 调用时务必传递你自己的 thread_id（即 'manager_executor_v2'）作为 session_id。\n"
                "3. 状态闭环：开始执行步骤前更新为 'in_progress'，完成后更新为 'completed'。\n"
                "4. 严谨性：在委派 'coder' 修改文件前，必须确保自己或专家已调用过 'read_file'。"
            ))
            messages = state["messages"]
            response = await model_with_tools.ainvoke([system_prompt] + messages)
            return {"messages": [response]}

        async def execute_manager_tools(state: AgentState) -> Dict:
            last_message = state["messages"][-1]
            updates = {"messages": []}
            if hasattr(last_message, "tool_calls"):
                for tool_call in last_message.tool_calls:
                    name = tool_call["name"]
                    print(f"\n\033[33m[Manager 正在分派工具: {name}]\033[0m")
                    if name == "todo_manager":
                        updates["current_todo"] = tool_call["args"]["items"]
                    
                    tool_obj = tools_by_name.get(name)
                    if not tool_obj:
                        observation = f"Error: 工具 '{name}' 未在系统中注册。"
                    else:
                        try:
                            args = tool_call["args"].copy()
                            if name == "task_tool" and "session_id" not in args:
                                args["session_id"] = "manager_executor_v2"

                            observation = await tool_obj.ainvoke(args)
                            if isinstance(observation, str) and len(observation) > 10000:
                                observation = observation[:10000] + "\n... (内容过长，已自动截断)"
                        except Exception as e:
                            observation = f"Error executing {name}: {e}"
                    updates["messages"].append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
            return updates

        def should_continue_manager(state: AgentState) -> Literal["tools", "__end__"]:
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return "__end__"

        builder = StateGraph(AgentState)
        builder.add_node("agent", call_manager_model)
        builder.add_node("tools", execute_manager_tools)
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", should_continue_manager)
        builder.add_edge("tools", "agent")

        app = builder.compile(checkpointer=checkpointer)

        print("\033[32m[OK] Nano Claude Code (Manager/Executor 架构) 已就绪。\033[0m")
        session_id = "manager_executor_v2"
        session_config = {"configurable": {"thread_id": session_id}}

        while True:
            try:
                print("===============================请输入您的需求or输入q,exit退出=================================")
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