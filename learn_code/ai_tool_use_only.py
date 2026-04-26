import json
import operator
import subprocess
from pathlib import Path
from typing import Annotated, List, TypedDict, Union, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI

# --- 环境配置 ---
WORKDIR = Path.cwd()

model = ChatOpenAI(
    base_url="http://localhost:8001/v1",
    api_key="vllm-no-key",
    model="Qwen_agent",
    temperature=0
)


# --- 1. 定义工具 ---

@tool
def bash(command: str) -> str:
    """
    在当前工作目录运行 shell 命令。
    """
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "错误：危险命令已被拦截。"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=120)
        return (r.stdout + r.stderr).strip()[:5000]
    except Exception as e:
        return f"错误: {e}"


@tool
def read_file(path: str) -> str:
    """
    读取文件内容。在编辑文件前必须先读取，以获取正确的上下文。
    """
    try:
        p = (WORKDIR / path).resolve()
        return p.read_text(encoding="utf-8")[:5000]
    except Exception as e:
        return f"错误: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """
    创建新文件。
    """
    try:
        p = (WORKDIR / path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"成功写入文件 {path}"
    except Exception as e:
        return f"错误: {e}"


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """
    通过精准匹配替换文件内容。

    注意：为了避免误替换，old_text 必须在文件中是唯一的。
    如果有多处重复代码，请在 old_text 中包含前后几行代码作为‘锚点’以确保唯一性。

    Args:
        path: 文件路径。
        old_text: 要替换的唯一文本块。
        new_text: 新的文本块。
    """
    try:
        p = (WORKDIR / path).resolve()
        content = p.read_text(encoding="utf-8")

        # 核心逻辑：检查匹配次数
        occurrence = content.count(old_text)

        if occurrence == 0:
            return f"错误：找不到指定的文本。请检查拼写、空格或换行符是否完全一致。"

        if occurrence > 1:
            return (f"错误：在文件中找到了 {occurrence} 处匹配。替换请求具有歧义！\n"
                    f"请提供更多的上下文（包含目标行前后的代码）作为 old_text，确保它是唯一的。")

        # 只有在唯一匹配时才执行替换
        new_content = content.replace(old_text, new_text, 1)
        p.write_text(new_content, encoding="utf-8")
        return f"成功编辑文件 {path}"
    except Exception as e:
        return f"错误: {e}"


@tool
def todo_manager(items: List[dict]) -> str:
    """
    对于任何包含多个步骤的任务，你需要先调用任务规划工具来编排任务

        Args:
        items: 任务对象列表。每个对象必须严格包含以下键：
               - 'id': 任务编号 (如 "1")
               - 'text': 任务描述内容
               - 'status': 状态，只能是 "pending", "in_progress", "completed" 之一。
    """
    summary = "\n".join([f"[{i['status']}] #{i['id']}: {i['text']}" for i in items])
    return f"任务列表已更新：\n{summary}"


tools = [bash, read_file, write_file, edit_file, todo_manager]
tools_by_name = {t.name: t for t in tools}
model_with_tools = model.bind_tools(tools)


# --- 2. 定义状态 ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    current_todo: List[dict]


# --- 3. 定义节点 ---

def call_model(state: AgentState):
    """LLM 决策节点"""
    todo_status = json.dumps(state.get("current_todo", []), ensure_ascii=False)

    # 将 SystemMessage优化
    system_prompt = SystemMessage(content=(
        f"你是一个位于 {WORKDIR} 的编程助手。\n"
        f"当前任务计划: {todo_status}\n\n"
        "核心操作规则：\n"
        "1. 任务规划：对于任何包含多个步骤的任务（如先读取再编辑、创建多个文件等），你必须先创建详细计划。\n"
        "2. 进度更新：在开始执行某个步骤前，将该任务状态更新为 'in_progress'；完成后更新为 'completed'。\n"
        "3. 精确编辑：'edit_file' 使用的是完全字符串匹配。如果文件中存在多个相同的代码块，你必须在 'old_text' 中包含上下文‘锚点’以确保匹配唯一。\n"
        "4. 先读后改：在调用 'edit_file' 之前，必须先调用 'read_file' 确认文件内容，严禁凭空猜测代码内容。\n"
        "5. 错误处理：如果遇到‘多重匹配’错误，请重新读取文件并提供更长、更唯一的代码片段进行替换。"
    ))
    response = model_with_tools.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


def execute_tools(state: AgentState):
    """工具执行节点转换逻辑"""
    last_message = state["messages"][-1]
    updates = {"messages": []}

    if hasattr(last_message, "tool_calls"):
        for tool_call in last_message.tool_calls:
            print(f"\033[33m[正在执行工具: {tool_call['name']}]\033[0m")

            if tool_call["name"] == "todo_manager":
                updates["current_todo"] = tool_call["args"]["items"]

            tool_func = tools_by_name[tool_call["name"]]
            observation = tool_func.invoke(tool_call["args"])

            updates["messages"].append(ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"]
            ))
    return updates


def should_continue(state: AgentState) -> Literal["tools", END]:
    if state["messages"][-1].tool_calls:
        return "tools"
    return END


# --- 5. 构建图 ---

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", execute_tools)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# --- 6. 运行 ---

if __name__ == "__main__":
    print("\033[32m--- 鲁棒编辑智能体（中文指令版）已就绪 ---\033[0m")
    # 模拟一个需求
    # query = input("\033[36m请输入需求: \033[0m")
    query = "把 inplace_quick_sort.py里改为原地的快速排序算法"
    inputs = {"messages": [HumanMessage(content=query)], "current_todo": []}

    for chunk in app.stream(inputs, stream_mode="updates", version="v2"):
        for node_name, state_update in chunk.items():
            if isinstance(state_update, dict):
                for msg in state_update.get("messages", []):
                    # 使用 LangChain 的漂亮打印输出消息
                    msg.pretty_print()
