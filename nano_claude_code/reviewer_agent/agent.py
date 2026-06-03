from typing import Dict, Literal
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START

from core.state import AgentState
from core.tools import read_file
from core.confirm import create_tool_confirm_node, make_agent_router, route_after_tool_confirm
from reviewer_agent.tools import run_in_sandbox, run_python_test, run_bash_command, check_code_style
from memory import create_compression_node, create_warn_node, make_token_router, route_after_warn

_DANGEROUS = {"run_in_sandbox", "run_python_test"}


def create_reviewer_agent(model, checkpointer):
    tools = [read_file, run_in_sandbox, run_python_test, run_bash_command, check_code_style]
    system_prompt = (
        "你是一个代码审查专家。你的职责是审查代码的正确性、安全性和代码风格。\n"
        "你可以使用以下工具：\n"
        "1. read_file        — 读取需要审查的代码文件\n"
        "2. run_in_sandbox   — 【唯一合法的代码执行工具】在隔离 venv 中运行代码或测试\n"
        "3. check_code_style — 检查代码语法和基础规范（无需执行）\n"
        "4. run_bash_command — 仅用于只读文件操作：ls、grep、cat、find 等\n"
        "5. run_python_test  — 仅在明确知道环境已安装依赖时才使用\n\n"
        "严格限制：\n"
        "- 【禁止】使用 run_bash_command 执行任何 Python 脚本（python xxx.py）\n"
        "- 【禁止】使用 run_bash_command 运行测试（pytest、unittest 等）\n"
        "- 所有代码执行、测试运行，必须且只能通过 run_in_sandbox 完成\n\n"
        "审查流程：\n"
        "1. read_file 阅读代码，理解逻辑\n"
        "2. check_code_style 做语法检查\n"
        "3. run_in_sandbox 运行测试验证正确性（这是执行代码的唯一方式）\n"
        "4. 输出结构化审查报告，包含问题清单和改进建议"
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
                updates["messages"].append(
                    ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
                )
        return updates

    _router = make_token_router()
    _agent_router = make_agent_router(_DANGEROUS)

    builder = StateGraph(AgentState)
    builder.add_node("agent", call_sub_model)
    builder.add_node("tool_confirm", create_tool_confirm_node(_DANGEROUS, "Reviewer"))
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
