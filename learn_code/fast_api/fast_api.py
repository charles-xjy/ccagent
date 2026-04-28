"""
=============================================================================================
##########################         fast api封装       ####################################
=============================================================================================
"""
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from learn_code.test.test_prettyprint import agent

app = FastAPI()


# -------- 数据结构 --------
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]


# -------- 模型列表 --------
@app.get("/v1/models")
def models():
    return {
        "data": [{"id": "langgraph-agent", "object": "model"}]
    }


import json


def sse(content: str):
    chunk = {
        "choices": [
            {
                "delta": {
                    "content": content
                }
            }
        ]
    }
    return f"data: {json.dumps(chunk)}\n\n"


# -------- 核心：stream trace --------
def stream_agent(user_msg: str):
    from langchain.messages import HumanMessage

    messages = [HumanMessage(content=user_msg)]

    # 开头提示
    yield sse("🤔 思考中...\n")

    for event in agent.stream({"messages": messages}):

        for node_name, node_output in event.items():

            # ===== LLM 节点 =====
            if node_name == "llm_call":
                msg = node_output["messages"][-1]

                # 👉 如果有 tool_calls = 调工具
                if msg.tool_calls:
                    for call in msg.tool_calls:
                        yield sse(f"🔧 调用工具: {call['name']}({call['args']})\n")

                else:
                    # 👉 没有 tool_calls 才是最终答案
                    yield sse(f"💬 最终答案: {msg.content}\n")

            # ===== TOOL 节点 =====
            elif node_name == "tool_node":
                for m in node_output["messages"]:
                    yield sse(f"✅ 工具返回: {m.content}\n")

    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    user_msg = req.messages[-1].content

    return StreamingResponse(
        stream_agent(user_msg),
        media_type="text/event-stream"
    )
