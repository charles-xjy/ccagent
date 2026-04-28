import re
from typing import Annotated, TypedDict, Sequence
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

model = init_chat_model(
    base_url="http://localhost:8001/v1",
    api_key="vllm-no-key",
    model="Qwen_agent",
    model_provider="openai",
    temperature=0,
)


@tool
def search_web(query: str) -> str:
    """搜索网络（模拟）

    Args:
        query: 搜索查询
    """
    # 模拟搜索结果
    print("使用search_web工具")
    return f"关于'{query}'的搜索结果：这是一些模拟的搜索结果..."


@tool
def ask_human(question: str) -> str:
    """向用户请求帮助或澄清

    当你不确定用户意图、需要额外信息、或需要用户确认时使用此工具。

    Args:
        question: 要问用户的问题
    """
    # 这个工具内部会触发interrupt
    prompt = "请回答以下问题"
    while True:
        # 第一次或校验失败后，都会停在这里
        human_response = interrupt({"question": question, "prompt": prompt})

        # 简单的邮箱正则校验
        if re.match(r"[^@]+@[^@]+\.[^@]+", str(human_response)):
            return human_response  # 只有格式对了，才 return 给 LLM

        # 如果格式不对，修改 prompt，继续 while 循环，再次触发 interrupt
        prompt = f"'{human_response}' 不是有效邮箱格式，请重新输入："


smart_tools = [search_web, ask_human]
llm_with_smart_tools = model.bind_tools(smart_tools)

print("✅ 定义了智能工具")
from langchain.agents import create_agent

# 使用预构建的agent，简化代码
smart_agent = create_agent(
    model=llm_with_smart_tools,
    tools=smart_tools,
    system_prompt="""你是一个智能助手。
    
当你遇到以下情况时，应该使用ask_human工具：
- 用户的需求不明确，需要澄清
- 你需要用户提供额外信息
- 执行重要操作前需要确认
- 你不确定应该怎么做

主动询问用户，直到你获得足够信息为止
""",
    checkpointer=MemorySaver(),  # 必须有checkpointer
)

print("✅ 智能Agent创建完成")
# 可视化
png_data = smart_agent.get_graph(xray=True).draw_mermaid_png()

with open("my_agent_graph.png", "wb") as f:
    f.write(png_data)


config = {"configurable": {"thread_id": "smart-1"}}

result = smart_agent.invoke(
    {
        "messages": [
            HumanMessage(
                content="帮我把这段话搜索一下并根据结果发邮件给查尔斯，但我不知道他的邮箱，你帮我问一下"
            )
        ]
    },
    config,
)
while "__interrupt__" in result:
    print("\n🤔 Agent请求澄清！")
    interrupts = result["__interrupt__"]
    print(f"问题: {interrupts}")

    response = input(
        f"{interrupts[0].value['prompt']}:{interrupts[0].value['question']}"
    )

    result = smart_agent.invoke(
        Command(resume=response), config  # resume的值会传给ask_human中的human_response
    )

    print("\n最终回答:")
    result["messages"][-1].pretty_print()
