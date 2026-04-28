from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, MessagesState, START
from langchain_core.messages import message_to_dict

checkpointer = InMemorySaver()


def call_model(state: MessagesState):
    # state["messages"] 是一个列表，取出最后一条消息对象
    last_message = state["messages"][-1]

    # 打印该消息对象的内容
    print(last_message.content)
    return {"messages": [{"role": "ai", "content": last_message.content}]}


builder = StateGraph(MessagesState)
builder.add_node(call_model)
builder.add_edge(START, "call_model")
graph = builder.compile(checkpointer=checkpointer)

graph.invoke(
    {"messages": [{"role": "user", "content": "hi! i am Bob"}]},
    {"configurable": {"thread_id": "1"}},
)
config = {
    "configurable": {
        "thread_id": "1",
    }
}
snapshot = graph.get_state(config)
print(snapshot)
messages_dict = [message_to_dict(m) for m in snapshot.values["messages"]]
print(messages_dict)
