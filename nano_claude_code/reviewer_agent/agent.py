from typing import Dict, Literal
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START

from core.state import AgentState
from core.tools import read_file
from reviewer_agent.tools import run_python_test, run_bash_command, check_code_style


def create_reviewer_agent(model, checkpointer):
    tools = [read_file, run_python_test, run_bash_command, check_code_style]
    system_prompt = (
        "你是一个代码审查专家。你的职责是审查代码的正确性、安全性和代码风格。\n"
        "你可以使用以下工具：\n"
        "1. read_file — 读取需要审查的代码文件\n"
        "2. run_python_test — 运行 Python 单元测试并查看结果\n"
        "3. check_code_style — 检查代码语法和基础规范\n"
        "4. run_bash_command — 执行只读辅助命令（如 grep、ls 等）\n\n"
        "审查流程：\n"
        "- 先阅读代码文件，理解逻辑\n"
        "- 必要时运行测试验证功能正确性\n"
        "- 检查潜在的安全风险、错误处理、边界情况\n"
        "- 输出结构化的审查报告，包含问题清单和改进建议"
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

    def should_continue_sub(state: AgentState) -> Literal["tools", "__end__"]:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "__end__"

    builder = StateGraph(AgentState)
    builder.add_node("agent", call_sub_model)
    builder.add_node("tools", execute_sub_tools)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue_sub)
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer)
