"""
Calculator Agent - 四则运算 Agent 核心实现

本模块使用 LangGraph 框架创建一个能够执行四则运算的智能 Agent。
Agent 可以接收自然语言指令，自动选择并调用相应的计算工具。

作者：Charles
日期：2024
"""

from typing import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


# =============================================================================
# 工具定义 - 四则运算函数
# =============================================================================

@tool
def add(a: float, b: float) -> float:
    """
    执行两个数的加法运算。
    
    Args:
        a: 第一个加数（可以是整数或小数）
        b: 第二个加数（可以是整数或小数）
    
    Returns:
        两个数的和
    
    Examples:
        >>> add(5, 3)
        8
        >>> add(2.5, 3.5)
        6.0
        >>> add(-5, 10)
        5
    """
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """
    执行两个数的减法运算。
    
    Args:
        a: 被减数（可以是整数或小数）
        b: 减数（可以是整数或小数）
    
    Returns:
        两个数的差（a - b）
    
    Examples:
        >>> subtract(10, 3)
        7
        >>> subtract(5, 8)
        -3
        >>> subtract(7.5, 2.5)
        5.0
    """
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """
    执行两个数的乘法运算。
    
    Args:
        a: 第一个乘数（可以是整数或小数）
        b: 第二个乘数（可以是整数或小数）
    
    Returns:
        两个数的积
    
    Examples:
        >>> multiply(4, 5)
        20
        >>> multiply(2.5, 4)
        10.0
        >>> multiply(-3, 7)
        -21
    """
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """
    执行两个数的除法运算。
    
    Args:
        a: 被除数（可以是整数或小数）
        b: 除数（可以是整数或小数，不能为 0）
    
    Returns:
        两个数的商
    
    Raises:
        ValueError: 当除数为 0 时抛出异常
    
    Examples:
        >>> divide(20, 4)
        5.0
        >>> divide(7, 2)
        3.5
        >>> divide(-15, 3)
        -5.0
    """
    if b == 0:
        raise ValueError("除数不能为 0")
    return a / b


# =============================================================================
# Agent 状态定义
# =============================================================================

class AgentState(TypedDict):
    """
    Agent 的状态定义。
    
    使用 TypedDict 定义 Agent 在运行过程中需要维护的状态。
    messages: 存储对话历史中的所有消息（用户输入和 AI 回复）
    """
    messages: Annotated[list[BaseMessage], operator.add]


# =============================================================================
# Agent 节点定义
# =============================================================================

def create_calculator_agent(llm):
    """
    创建四则运算 Agent。
    
    使用 LangGraph 框架创建一个能够执行四则运算的智能 Agent。
    Agent 的工作流程：
    1. 接收用户消息
    2. LLM 分析消息，决定是否需要调用工具
    3. 如果需要，调用相应的计算工具
    4. 将工具结果返回给 LLM 生成最终回复
    
    Args:
        llm: 语言模型实例（如 ChatOpenAI, ChatAnthropic 等）
    
    Returns:
        编译后的 LangGraph Agent 实例
    """
    
    # 获取所有工具
    tools = [add, subtract, multiply, divide]
    
    # 创建工具节点
    # ToolNode 会自动处理工具调用和结果返回
    tool_node = ToolNode(tools)
    
    # 定义 LLM 调用节点
    def call_llm(state: AgentState) -> AgentState:
        """
        调用 LLM 处理当前状态。
        
        LLM 会根据消息历史：
        1. 理解用户意图
        2. 决定是否需要调用工具
        3. 如果需要，生成工具调用请求
        4. 如果不需要，直接生成回复
        
        Args:
            state: 当前 Agent 状态，包含消息历史
        
        Returns:
            更新后的状态，包含 LLM 的回复
        """
        # 绑定工具到 LLM，使其能够调用工具
        llm_with_tools = llm.bind_tools(tools)
        
        # 调用 LLM
        response = llm_with_tools.invoke(state["messages"])
        
        # 返回新消息
        return {"messages": [response]}
    
    # 定义条件路由函数
    def should_continue(state: AgentState) -> str:
        """
        判断是否继续执行工具调用。
        
        检查 LLM 的回复是否包含工具调用请求：
        - 如果有工具调用，返回 "tools" 继续执行工具
        - 如果没有，返回 END 结束流程
        
        Args:
            state: 当前 Agent 状态
        
        Returns:
            下一步要执行的节点名称（"tools" 或 END）
        """
        messages = state["messages"]
        last_message = messages[-1]
        
        # 检查是否有工具调用
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        
        return END
    
    # =============================================================================
    # 构建工作流图
    # =============================================================================
    
    # 创建状态图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("agent", call_llm)      # LLM 处理节点
    workflow.add_node("tools", tool_node)     # 工具执行节点
    
    # 设置入口点
    workflow.set_entry_point("agent")
    
    # 添加条件边：根据 should_continue 的结果决定下一步
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",   # 需要调用工具时
            END: END            # 不需要调用工具时直接结束
        }
    )
    
    # 添加边：工具执行完成后返回给 agent
    workflow.add_edge("tools", "agent")
    
    # 编译工作流
    app = workflow.compile()
    
    return app


# =============================================================================
# 便捷函数
# =============================================================================

def get_available_tools() -> list:
    """
    获取可用的工具列表。
    
    Returns:
        包含所有可用工具的列表
    """
    return [add, subtract, multiply, divide]


def print_tool_info():
    """
    打印所有可用工具的信息。
    
    用于调试和展示 Agent 的能力。
    """
    print("=" * 50)
    print("可用工具列表：")
    print("=" * 50)
    for tool in get_available_tools():
        print(f"\n工具名称：{tool.name}")
        print(f"描述：{tool.description}")
        print(f"参数：{tool.inputs}")
    print("=" * 50)
