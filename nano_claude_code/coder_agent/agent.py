from typing import Dict, Literal
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START

from core.state import AgentState
from core.tools import read_file
from core.confirm import create_tool_confirm_node, make_agent_router, route_after_tool_confirm
from coder_agent.tools import bash, write_file, edit_file
from memory import create_compression_node, create_warn_node, make_token_router, route_after_warn

_DANGEROUS = {"write_file", "edit_file"}

def create_coder_agent(model, checkpointer):
    tools = [read_file, bash, write_file, edit_file]
    system_prompt = (
        "你是一个编程专家。你的唯一职责是编写和修改代码文件。\n"
        "你可以使用以下工具：\n"
        "1. read_file — 在修改前先阅读文件内容\n"
        "2. write_file — 创建新文件\n"
        "3. edit_file — 精确替换文件中的片段\n"
        "4. bash — 仅用于只读文件系统操作（如 ls、cat、grep、find），帮助定位文件或查看目录结构\n\n"
        "严格限制：\n"
        "- 绝对禁止运行任何 Python 脚本、测试或编译命令（如 python、pytest、npm test、make 等）\n"
        "- 绝对禁止调用 todo_manager 更新任务状态\n"
        "- 你的输出就是写完代码后的简短总结，交给 Manager 判断是否需要审查\n"
        "- 确保代码可读且高效"
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
