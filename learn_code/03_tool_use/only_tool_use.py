import json
import operator
import subprocess
from pathlib import Path
from typing import Annotated, List, TypedDict, Union

from openai import OpenAI
from langgraph.graph import StateGraph, END

# --- 环境配置 ---
WORKDIR = Path.cwd()

# 修改为 vLLM 本地部署的配置
client = OpenAI(
    base_url="http://localhost:8001/v1",  # 请确保端口与 vLLM 启动端口一致
    api_key="vllm-no-key",  # vLLM 本地通常不需要 Key
)
# 必须与 vLLM 启动时的 --model 参数完全一致
MODEL = "Qwen_agent"


# --- 1. 定义状态 (State) ---
class AgentState(TypedDict):
    # 使用 Annotated[..., operator.add] 实现消息列表的自动追加
    messages: Annotated[List[dict], operator.add]


# --- 2. 工具逻辑实现 ---
class ToolExecutor:
    def __init__(self):
        self.todo_items = []

    def bash(self, command: str) -> str:
        dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
        if any(d in command for d in dangerous):
            return "Error: Dangerous command blocked"
        try:
            r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=120)
            return (r.stdout + r.stderr).strip()[:5000]
        except Exception as e:
            return f"Error: {e}"

    def read_file(self, path: str, limit: int = None) -> str:
        try:
            p = (WORKDIR / path).resolve()
            lines = p.read_text().splitlines()
            if limit: lines = lines[:limit]
            return "\n".join(lines)[:5000]
        except Exception as e:
            return f"Error: {e}"

    def write_file(self, path: str, content: str) -> str:
        try:
            p = (WORKDIR / path).resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return f"Successfully wrote to {path}"
        except Exception as e:
            return f"Error: {e}"

    def edit_file(self, path: str, old_text: str, new_text: str) -> str:
        try:
            p = (WORKDIR / path).resolve()
            content = p.read_text()
            if old_text not in content:
                return f"Error: old_text not found in {path}"
            p.write_text(content.replace(old_text, new_text, 1))
            return f"Successfully edited {path}"
        except Exception as e:
            return f"Error: {e}"

    def todo(self, items: list) -> str:
        self.todo_items = items
        lines = []
        for it in items:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}[it["status"]]
            lines.append(f"{marker} #{it['id']}: {it['text']}")
        return "\n".join(lines) + f"\n({sum(1 for t in items if t['status'] == 'completed')}/{len(items)} done)"


executor = ToolExecutor()


# --- 3. 定义节点 (Nodes) ---

def call_model(state: AgentState):
    """调用本地 vLLM 模型"""
    system_prompt = {
        "role": "system",
        "content": f"You are a coding agent at {WORKDIR}. Use todo tool to plan tasks."
    }

    # 打印调试信息，确认发送的消息格式
    # print(f"DEBUG: Sending {len(state['messages'])} messages to vLLM")

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[system_prompt] + state["messages"],
            tools=TOOLS_DEFINITION,
            temperature=0  # 建议设为 0 以保证工具调用的稳定性
        )
    except Exception as e:
        print(f"\033[31mAPI Call Failed: {e}\033[0m")
        raise e

    message = response.choices[0].message

    # vLLM/Qwen 有时会返回 content=None，这里强制转为空字符串防止后续节点崩溃
    res_content = message.content if message.content is not None else ""

    msg_dict = {"role": "assistant", "content": res_content}

    if message.tool_calls:
        msg_dict["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            } for tc in message.tool_calls
        ]

    return {"messages": [msg_dict]}


def execute_tools(state: AgentState):
    """执行工具调用"""
    last_message = state["messages"][-1]
    tool_outputs = []

    if "tool_calls" not in last_message:
        return {"messages": []}

    for tool_call in last_message["tool_calls"]:
        func_name = tool_call["function"]["name"]
        args = json.loads(tool_call["function"]["arguments"])

        print(f"\033[33m[vLLM 执行工具: {func_name}...]\033[0m")

        if func_name == "bash":
            result = executor.bash(args["command"])
        elif func_name == "read_file":
            result = executor.read_file(args["path"], args.get("limit"))
        elif func_name == "write_file":
            result = executor.write_file(args["path"], args["content"])
        elif func_name == "edit_file":
            result = executor.edit_file(args["path"], args["old_text"], args["new_text"])
        elif func_name == "todo":
            result = executor.todo(args["items"])
        else:
            result = f"Unknown tool: {func_name}"

        tool_outputs.append({
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "name": func_name,
            "content": str(result)
        })

    return {"messages": tool_outputs}


# --- 4. 流程控制 ---

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.get("tool_calls"):
        return "tools"
    return END


# --- 5. 构建图 ---
from langgraph.graph import START

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", execute_tools)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

app = workflow.compile()

# --- 工具定义 (严格遵循 OpenAI 标准，不含任何 input_schema 字样) ---
TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run shell commands.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read content from a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Find and replace text in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"}
                },
                "required": ["path", "old_text", "new_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "todo",
            "description": "Manage the task list and progress.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "text": {"type": "string"},
                                "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}
                            },
                            "required": ["id", "text", "status"]
                        }
                    }
                },
                "required": ["items"]
            }
        }
    }
]

if __name__ == "__main__":
    # 这里的 app 是你代码里编译后的工作流对象
    png_data = app.get_graph(xray=True).draw_mermaid_png()

    with open("my_agent_graph.png", "wb") as f:
        f.write(png_data)
    print("\033[32m--- vLLM LangGraph Agent Starting ---\033[0m")
    query = input("\033[36m请输入需求: \033[0m")
    # query = "我想要写一个hello word"
    inputs = {"messages": [{"role": "user", "content": query}]}
    # 使用 stream 模式运行，可以看到每个节点的执行过程
    try:

        final_state = app.invoke(inputs)
        # 直接从结果里取消息
        for msg in final_state["messages"]:
            msg.pretty_print()
        # for output in app.stream(inputs):
        #     for node_name, state in output.items():
        #         last_msg = state["messages"][-1]
        #         print(last_msg)
        #         if last_msg.get("content"):
        #             print(f"\n\033[32m[{node_name}]:\033[0m {last_msg['content']}")
    except Exception as e:
        print(f"\n\033[31m[Error]: {e}\033[0m")
