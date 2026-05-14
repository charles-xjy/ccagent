from typing import Dict, List, Literal
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START

from core.state import AgentState
from core.tools import read_file
from researcher_agent.tools import web_search
from memory import create_compression_node, create_warn_node, make_token_router, route_after_warn

_MAX_TOOL_ROUNDS = 3  # 最多允许的工具调用轮次

_SYSTEM_PROMPT = (
    "你是一个技术调研专家，擅长从互联网和 GitHub 获取最新技术资料。\n"
    "你可以使用以下工具：\n"
    "1. read_file        — 读取本地代码或文档文件\n"
    "2. web_search       — DuckDuckGo 关键词搜索，无需 API Key\n"
    "3. langchain_docs 系列工具 — 直接查询 LangChain/LangGraph/LangSmith 官方文档（最新、最准确）\n"
    "4. GitHub 系列工具  — 搜索仓库、查看源码、阅读 issue / PR（需 GITHUB_TOKEN）\n"
    "5. fetch            — 抓取任意网页内容（官方文档、博客、API 参考页面）\n\n"
    "调研流程（最多 3 轮工具调用，请合理规划每轮）：\n"
    "- 第 1 轮：用 web_search 或 langchain_docs 快速定位权威来源\n"
    "- 第 2 轮：用 fetch 或 GitHub 工具精读关键页面/源码\n"
    "- 第 3 轮（最后一轮）：补充遗漏的关键细节，然后直接输出报告\n"
    "- 每轮可并行调用多个工具，充分利用单轮配额\n"
    "- 综合后给出准确、带示例的调研报告"
)


def create_researcher_agent(model, checkpointer, mcp_tools: List = None):
    tools = [read_file, web_search] + (mcp_tools or [])
    model_with_tools = model.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    async def call_sub_model(state: AgentState) -> Dict:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=_SYSTEM_PROMPT)] + messages
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
        if not (hasattr(last_message, "tool_calls") and last_message.tool_calls):
            return "__end__"
        # 从最后一条 HumanMessage 开始计数，每次新任务自动清零
        messages = state["messages"]
        last_human_idx = max(
            (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)),
            default=0
        )
        rounds = sum(
            1 for m in messages[last_human_idx:]
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
        )
        if rounds >= _MAX_TOOL_ROUNDS:
            print(f"\033[33m[!] tech-researcher 已达最大工具调用轮次 ({_MAX_TOOL_ROUNDS})，强制结束\033[0m")
            return "__end__"
        return "tools"

    _router = make_token_router()

    builder = StateGraph(AgentState)
    builder.add_node("agent", call_sub_model)
    builder.add_node("tools", execute_sub_tools)
    builder.add_node("warn", create_warn_node())
    builder.add_node("compress", create_compression_node(model))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue_sub)
    builder.add_conditional_edges("tools", _router, {"compress": "compress", "warn": "warn", "agent": "agent"})
    builder.add_conditional_edges("warn", route_after_warn, {"compress": "compress", "agent": "agent"})
    builder.add_edge("compress", "agent")

    return builder.compile(checkpointer=checkpointer)
