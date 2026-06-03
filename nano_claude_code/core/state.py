from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    current_todo: List[dict]
    compress_choice: str       # "compress" | "skip" | ""，由 warn 节点写入
    plan_needs_confirm: bool   # True 时暂停等待用户确认任务规划
