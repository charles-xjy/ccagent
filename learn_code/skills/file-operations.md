---
name: file-operations
description: Perform file system operations including reading, writing, editing files and running shell commands. Use when user asks to read/write/edit files or run terminal commands.
---

# File Operations Skill (LangGraph Compatible)

You now have expertise in file system operations. This skill is designed to work with LangGraph framework.

## Tool Definitions

### 1. bash - Run shell commands

```python
from langchain.tools import tool
import subprocess
from pathlib import Path

WORKDIR = Path.cwd()

@tool
def bash(command: str) -> str:
    """
    Run a shell command in the current working directory and return its output.
    Use this to create files, list directories, or run scripts.
    """
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=120)
        return (r.stdout + r.stderr).strip()[:5000]
    except Exception as e:
        return f"Error: {e}"
bash.is_tool = True  # 标记为工具
```

**Safety**: Blocks dangerous commands like `rm -rf /`, `sudo`, `shutdown`, `reboot`
**Timeout**: 120 seconds max

### 2. read_file - Read file contents

```python
@tool
def read_file(path: str) -> str:
    """读取文件内容。在编辑文件前必须先读取，以获取正确的上下文。"""
    try:
        p = (WORKDIR / path).resolve()
        return p.read_text(encoding="utf-8")[:5000]
    except Exception as e:
        return f"Error: {e}"
read_file.is_tool = True  # 标记为工具
```

### 3. write_file - Write content to file

```python
@tool
def write_file(path: str, content: str) -> str:
    """
    在指定路径创建一个文件并写入内容。
    自动创建不存在的中间目录。
    """
    try:
        p = (WORKDIR / path).resolve()
        if not str(p).startswith(str(WORKDIR.resolve())):
            return "Error: 越界访问！你不准去 WORKDIR 以外的地方。"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error: {e}"
write_file.is_tool = True  # 标记为工具
```

### 4. edit_file - Replace exact text in file

```python
@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """
    通过查找并替换的方式修改现有文件的一小部分内容。
    """
    try:
        p = (WORKDIR / path).resolve()
        content = p.read_text(encoding="utf-8")
        occurrence = content.count(old_text)
        if occurrence == 0:
            return f"Error: 找不到指定的文本。请检查拼写、空格或换行符是否完全一致。"
        if occurrence > 1:
            return f"Error: 在文件中找到了 {occurrence} 处匹配。替换请求具有歧义！请提供更多的上下文。"
        new_content = content.replace(old_text, new_text, 1)
        p.write_text(new_content, encoding="utf-8")
        return f"Successfully edited {path}"
    except Exception as e:
        return f"Error: {e}"
edit_file.is_tool = True  # 标记为工具
```

### 5. todo_manager - Task planning

```python
from typing import List

@tool
def todo_manager(items: List[dict]) -> str:
    """
    对于任何包含多个步骤的任务，你需要先调用任务规划工具来编排任务
    """
    summary = "\n".join([f"[{i['status']}] #{i['id']}: {i['text']}" for i in items])
    return f"任务列表已更新：\n{summary}"
todo_manager.is_tool = True  # 标记为工具
```

## LangGraph Integration

### State Definition

```python
from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    current_todo: List[dict]
```

### Node Functions

```python
from langchain_core.messages import SystemMessage, ToolMessage
from typing import Literal
from langgraph.graph import StateGraph, START, END
import json

# System Prompt (from skill)
SYSTEM_PROMPT = """你是一个位于 {WORKDIR} 的编程助手。

核心操作规则：
1. 任务规划：对于任何包含多个步骤的任务（如先读取再编辑、创建多个文件等），你必须先创建详细计划。
2. 进度更新：在开始执行某个步骤前，将该任务状态更新为 'in_progress'；完成后更新为 'completed'。
3. 精确编辑：'edit_file' 使用的是完全字符串匹配。如果文件中存在多个相同的代码块，你必须在 'old_text' 中包含上下文'锚点'以确保匹配唯一。
4. 先读后改：在调用 'edit_file' 之前，必须先调用 'read_file' 确认文件内容，严禁凭空猜测代码内容。
5. 错误处理：如果遇到'多重匹配'错误，请重新读取文件并提供更长、更唯一的代码片段进行替换。"""

def call_model(state: AgentState) -> dict[str, list[BaseMessage]]:
    """LLM 决策节点"""
    todo_status = json.dumps(state.get("current_todo", []), ensure_ascii=False)
    
    system_prompt = SystemMessage(content=SYSTEM_PROMPT.format(WORKDIR=WORKDIR))
    task_prompt = SystemMessage(content=f"\n当前任务计划: {todo_status}")
    
    response = model_with_tools.invoke([system_prompt, task_prompt] + state["messages"])
    return {"messages": [response]}


def execute_tools(state: AgentState) -> dict[str, list[BaseMessage]]:
    """工具执行节点"""
    last_message = state["messages"][-1]
    updates = {"messages": []}
    if hasattr(last_message, "tool_calls"):
        for tool_call in last_message.tool_calls:
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
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END
```

### Workflow Construction

```python
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", execute_tools)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")
app = workflow.compile()
```

## Usage

```python
# 1. Initialize tools
tools = [bash, read_file, write_file, edit_file, todo_manager]
tools_by_name = {t.name: t for t in tools}
model_with_tools = model.bind_tools(tools)

# 2. Run workflow
inputs = {"messages": [HumanMessage(content=query)], "current_todo": []}
for chunk in app.stream(inputs, stream_mode="updates", version="v2"):
    for node_name, node_update in chunk["data"].items():
        if "messages" in node_update:
            for msg in node_update["messages"]:
                msg.pretty_print()
```