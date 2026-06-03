from typing import Dict, Literal
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START

from core.state import AgentState
from core.tools import read_file
from core.confirm import create_tool_confirm_node, make_agent_router, route_after_tool_confirm
from coder_agent.tools import bash, run_python, write_file, edit_file, read_sandbox_file
from memory import create_compression_node, create_warn_node, make_token_router, route_after_warn

_DANGEROUS = {"write_file", "edit_file", "bash", "run_python"}

def create_coder_agent(model, checkpointer):
    tools = [read_file, bash, run_python, write_file, edit_file, read_sandbox_file]
    system_prompt = (
        "你是一个编程专家，在 CubeSandbox KVM 沙箱中工作。每次会话共享同一个沙箱实例（状态持久）。\n"
        "你可以使用以下工具：\n"
        "1. read_file — 读取宿主机本地项目文件（只读，用于参考已有代码）\n"
        "2. read_sandbox_file — 读取沙箱内已创建的文件\n"
        "3. write_file — 在沙箱中创建新文件（路径相对于 /home/user/workspace/）\n"
        "4. edit_file — 精确替换沙箱中文件的片段\n"
        "5. bash — 在沙箱中执行 shell 命令（安装依赖、查看文件、运行脚本等）\n"
        "6. run_python — 在沙箱中直接执行 Python 代码片段\n\n"
        "沙箱说明：\n"
        "- 所有代码执行均在隔离的 KVM MicroVM 中进行，宿主机完全安全\n"
        "- 沙箱工作目录为 /home/user/workspace/\n"
        "- 可以自由安装包（pip install）、执行任意命令，无需顾虑宿主机安全\n"
        "- 完成编码后输出简短总结，交给 Manager 判断是否需要审查"
    )
    model_with_tools = model.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    async def call_sub_model(state: AgentState) -> Dict:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages
        response = await model_with_tools.ainvoke(messages)
        return {"messages": [response]}

    async def execute_sub_tools(state: AgentState) -> Dict:
        last_message = state["messages"][-1]
        updates = {"messages": []}
        if hasattr(last_message, "tool_calls"):
            for tool_call in last_message.tool_calls:
                name = tool_call["name"]
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
                updates["messages"].append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
        return updates

    _router = make_token_router()
    _agent_router = make_agent_router(_DANGEROUS)

    builder = StateGraph(AgentState)
    builder.add_node("agent", call_sub_model)
    builder.add_node("tool_confirm", create_tool_confirm_node(_DANGEROUS, "Coder"))
    builder.add_node("tools", execute_sub_tools)
    builder.add_node("warn", create_warn_node())
    builder.add_node("compress", create_compression_node(model))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", _agent_router,
                                  {"tool_confirm": "tool_confirm", "tools": "tools", "__end__": "__end__"})
    builder.add_conditional_edges("tool_confirm", route_after_tool_confirm,
                                  {"tools": "tools", "agent": "agent"})
    builder.add_conditional_edges("tools", _router, {"compress": "compress", "warn": "warn", "agent": "agent"})
    builder.add_conditional_edges("warn", route_after_warn, {"compress": "compress", "agent": "agent"})
    builder.add_edge("compress", "agent")

    return builder.compile(checkpointer=checkpointer)
