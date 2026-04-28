import asyncio
import json
import operator
import subprocess
from pathlib import Path
from typing import Annotated, List, Literal, TypedDict, Dict, Any, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver, InMemorySaver
from langchain.agents import create_agent

# =============================================================================================
# 1. 配置模型
# =============================================================================================
# 使用 init_chat_model 灵活配置模型，确保与本地 vLLM 端口对接
model = init_chat_model(
    base_url="http://localhost:8001/v1",
    api_key="vllm-no-key",
    model="Qwen_agent",
    model_provider="openai",
)

# =============================================================================================
# 2. 定义核心工具集 (原子能力层)
# =============================================================================================
WORKDIR = Path.cwd()


@tool
def bash(command: str) -> str:
    """
    在当前工作目录运行 shell 命令并返回输出。
    用于创建文件、列出目录、运行脚本或执行系统命令。

    Args:
        command: The full shell command to execute (e.g., 'ls -la' or 'touch index.py').
    """

    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=120)
        return (r.stdout + r.stderr).strip()[:5000]
    except Exception as e:
        return f"Error: {e}"


@tool
def read_file(path: str) -> str:
    """
    读取文件内容。在编辑文件前必须先读取，以获取正确的上下文。
    """
    try:
        p = (WORKDIR / path).resolve()
        return p.read_text(encoding="utf-8")[:5000]
    except Exception as e:
        return f"Error: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """
    在指定路径创建一个文件并写入内容。
    如果你需要保存代码、笔记或配置文件，请使用此工具。
    它会自动创建不存在的中间目录。

    Args:
        path: 相对路径字符串，包含文件名。
        content: 要写入文件的完整文本内容。

    Returns:
        成功或错误的提示消息。
    """
    try:
        p = (WORKDIR / path).resolve()
        if not str(p).startswith(str(WORKDIR.resolve())):
            return "Error: 越界访问！不准操作工作目录以外的文件。"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """
    通过查找并替换的方式修改现有文件的一小部分内容。
    这是修改已有代码的首选方式，因为它更安全且高效。

    Args:
        path: 文件相对路径。
        old_text: 要被替换的原始代码片段（必须完全匹配）。
        new_text: 替换后的新代码片段。
    """
    try:
        p = (WORKDIR / path).resolve()
        content = p.read_text(encoding="utf-8")
        occurrence = content.count(old_text)
        if occurrence == 0:
            return "Error: 找不到指定的文本。请先 read_file 确认内容。"
        if occurrence > 1:
            return f"Error: 找到 {occurrence} 处匹配。请提供唯一的上下文片段。"
        new_content = content.replace(old_text, new_text, 1)
        p.write_text(new_content, encoding="utf-8")
        return f"Successfully edited {path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def todo_manager(items: List[dict]) -> str:
    """
    更新当前会话的任务列表

    主动使用此工具来跟踪进度和管理任务执行。

    ## 何时使用此工具
    在以下场景中主动使用此工具：
    1. 收到复杂的多步骤任务时 - 立即分解为子任务
    2. 开始执行任务时 - 将任务标记为 in_progress
    3. 完成任务后 - 将任务标记为 completed
    4. 遇到错误时 - 将任务标记为 failed 并记录错误

    ## 任务状态管理
    1. **任务状态**: 使用这些状态来跟踪进度：
       - pending: 任务尚未开始
       - in_progress: 当前正在执行（同一时间最多3个）
       - completed: 任务成功完成
       - failed: 任务遇到错误

    2. **任务管理规则**:
       - 实时更新任务状态
       - 同一时间最多1个任务处于 in_progress
       - 必须按顺序处理任务
       - 任务失败时，标记为 failed 并包含错误详情

    3. **任务完成要求**:
       - 只有在完全完成时才标记为 completed
       - 如果遇到错误，标记为 failed
       - 绝不要在以下情况标记为 completed：
         * 实现不完整
         * 遇到未解决的错误
         * 找不到必要的文件或依赖


    Args:
    items: 任务对象列表。每个对象必须严格包含以下键：
           - 'id': 任务编号 (如 "1")
           - 'text': 任务描述内容
           - 'status': 状态，只能是 "pending", "in_progress", "completed","failed"之一。
    """
    status_headers = {
        "in_progress": "🔄进行中:",
        "pending": "⏳待处理:",
        "completed": "✅已完成:",
        "failed": "❌失败:"
    }
    # 2. 逐行构造结果
    lines = []
    for item in items:
        raw_status = item.get("status", "❌error")
        # 获取对应的标题，如果模型传错了，保底显示 pending
        header = status_headers.get(raw_status, "❌error")

        tid = item.get("id", "❌id?")
        text = item.get("text", "❌无内容")

        # 拼接成你要求的格式：状态标题 + #ID + 内容
        lines.append(f"{header} #{tid} {text}")

    if not lines:
        return "任务列表为空。"

    summary = "\n".join(lines)
    return f"--- 当前任务面板 ---\n{summary}"


# =============================================================================================
# 3. 异步状态与 MCP 接入
# =============================================================================================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    current_todo: List[dict]


_agent_instances = {}
_tools_by_name = {}
_model_with_tools = None


async def fetch_mcp_tools():
    """连接远程 MCP 服务器并获取动态工具"""
    mcp_servers = {
        "langchain_docs": {
            "url": "https://docs.langchain.com/mcp",
            "transport": "http"
        }
    }
    client = MultiServerMCPClient(mcp_servers)
    try:
        return await client.get_tools()
    except Exception as e:
        print(f"[!] MCP 连接失败: {e}")
        return []


# =============================================================================================
# 4. 定义图节点 (异步管理层)
# =============================================================================================

async def call_model(state: AgentState) -> Dict:
    """主 Agent (Manager) 节点：负责分析、规划、委派"""
    todo_status = json.dumps(state.get("current_todo", []), ensure_ascii=False)

    system_prompt = SystemMessage(content=(
        f"你是一个高级编程协调员 (Manager)。当前工作路径: {WORKDIR}\n"
        f"当前任务计划进度: {todo_status}\n\n"
        "核心操作守则：\n"
        "1. 规划优先：面对复杂任务，必须先调用 'todo_manager' 编排分步计划。\n"
        "2. 专家委派：你自己不直接写代码或查文档。请通过 'task_tool' 指挥专家：\n"
        "   - 查阅 LangChain/LangGraph 技术资料 -> 调用 'tech-researcher'\n"
        "   - 编写、修改、测试代码或运行 Bash -> 调用 'coder'\n"
        "3. 状态闭环：开始执行步骤前更新为 'in_progress'，完成后更新为 'completed'。\n"
        "4. 严谨性：在委派 'coder' 修改文件前，必须确保自己或专家已调用过 'read_file'。"
    ))

    # 上下文管理：保留最近 12 条消息，避免超出模型 8192 token 限制
    messages = state["messages"]

    # 异步调用模型
    response = await _model_with_tools.ainvoke([system_prompt] + messages)
    return {"messages": [response]}


async def execute_tools(state: AgentState) -> Dict:
    """工具分发节点：异步执行工具并反馈结果"""
    last_message = state["messages"][-1]
    updates = {"messages": []}

    if hasattr(last_message, "tool_calls"):
        for tool_call in last_message.tool_calls:
            name = tool_call["name"]
            print(f"\n\033[33m[Manager 正在分派工具: {name}]\033[0m")

            if name == "todo_manager":
                updates["current_todo"] = tool_call["args"]["items"]

            tool_obj = _tools_by_name.get(name)
            if not tool_obj:
                observation = f"Error: 工具 '{name}' 未在系统中注册。"
            else:
                try:
                    # 异步执行工具
                    observation = await tool_obj.ainvoke(tool_call["args"])
                    # 输出截断保护
                    if isinstance(observation, str) and len(observation) > 10000:
                        observation = observation[:10000] + "\n... (内容过长，已自动截断)"
                except Exception as e:
                    observation = f"Error executing {name}: {e}"

            updates["messages"].append(ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"]
            ))
    return updates


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """条件边判断逻辑"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "__end__"


def store_memory(config, agent):
    import json

    # 1. 获取状态快照
    snapshot = agent.get_state(config)

    # 2. 准备要写入的数据结构
    serializable_data = {
        # 提取配置信息
        "config": {
            "thread_id": snapshot.config["configurable"].get("thread_id"),
            "checkpoint_id": snapshot.config["configurable"].get("checkpoint_id")
        },
        # 提取元数据
        "metadata": snapshot.metadata,
        # 提取消息列表
        "messages": []
    }

    # 3. 遍历并转换消息
    for msg in snapshot.values.get("messages", []):
        msg_dict = {
            "type": msg.type,
            "content": msg.content,
            "id": msg.id
        }

        # 如果有工具调用，也存下来
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            msg_dict["tool_calls"] = msg.tool_calls

        # 如果是 ToolMessage，存下 tool_call_id
        if msg.type == "tool":
            msg_dict["tool_call_id"] = msg.tool_call_id

        serializable_data["messages"].append(msg_dict)
    import os
    # 4. 写入 JSON 文件
    if config["configurable"].get("thread_id", "main"):
        name = config["configurable"].get("thread_id")
    else:
        name = "unknow_agent"
        print("❌没有配置agent存档,默认命名为unknow_agent")
    os.makedirs("./memory", exist_ok=True)
    with open(f"./memory/{name}_agent_state.json", "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, ensure_ascii=False, indent=4)

    print(f"状态已成功写入{config["configurable"].get("thread_id")}_agent_state.json")


# =============================================================================================
# 5. 主程序与子 Agent 初始化 (架构初始化层)
# =============================================================================================

async def main():
    global _tools_by_name, _model_with_tools, _agent_instances

    print("\033[34m[*] 系统正在启动...\033[0m")

    # A. 动态加载 MCP 工具
    mcp_tools = await fetch_mcp_tools()

    # B. 原子工具池
    all_atomic_tools = [bash, read_file, write_file, edit_file] + mcp_tools

    # C. 初始化专家 SubAgents (Executor 层)
    print("[*] 正在同步专家组 (Executor SubAgents) 配置...")
    sub_configs = [
        {
            "type": "tech-researcher",
            "prompt": "你是一个技术调研专家。请使用 MCP 工具深入搜索 LangChain 相关文档，为用户提供最新的 API 用法和代码示例。",
            "tools": [read_file] + mcp_tools
        },
        {
            "type": "coder",
            "prompt": "你是一个编程专家。你负责文件的创建 (write_file)、修改 (edit_file) 和代码运行 (bash)。请确保代码可读且高效。",
            "tools": [read_file, write_file, edit_file, bash]
        }
    ]

    checkpointer = InMemorySaver()
    for cfg in sub_configs:
        # 使用 create_react_agent 包装子 Agent 为独立的闭环 Graph
        agent_type = str(cfg["type"])
        _agent_instances[agent_type] = create_agent(
            model, checkpointer=checkpointer, tools=cfg["tools"], system_prompt=cfg["prompt"]
        )

    # D. 定义 TaskTool (Manager 委派专家的专属工具)
    @tool
    async def task_tool(description: str, subagent_type: str) -> str:
        """
        委派复杂的专项任务给专家处理。
        subagent_type 必须是 'tech-researcher' 或 'coder'。
        """
        agent = _agent_instances.get(subagent_type)
        if not agent:
            return f"Error: 找不到类型为 '{subagent_type}' 的专家。"

        print(f"\n\033[35m[系统] >>> 子 Agent ({subagent_type}) 开始工作...\033[0m")
        from datetime import datetime
        now = datetime.now()
        full_subagent_output = ""
        sub_agent_config = {
            "configurable": {
                "thread_id": f"{subagent_type}_{now}",
            }
        }

        # --- 核心改进：使用 astream 实时打印子 Agent 的过程 ---
        async for chunk in agent.astream({"messages": [HumanMessage(content=description)]}, config=sub_agent_config,
                                         stream_mode="updates"):
            for node, data in chunk.items():
                # 打印当前正在运行的子节点名（例如 'agent' 或 'tools'）
                print(f"  \033[34m└─ [{subagent_type}.{node}]\033[0m 正在处理...")

                if "messages" in data:
                    for msg in data["messages"]:
                        # 记录并实时展示子 Agent 的思考
                        if isinstance(msg, AIMessage) and msg.content:
                            print(f"    \033[37m思考: {msg.content}...\033[0m")
                        # 记录并实时展示子 Agent 调用的工具
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            for tc in msg.tool_calls:
                                print(f"    \033[32m工具调用: {tc['name']}\033[0m")
                        # 捕获最后的输出作为返回报告
                        full_subagent_output = msg.content
        store_memory(sub_agent_config, agent)
        print(f"\033[35m[系统] <<< 子 Agent ({subagent_type}) 任务完成。\033[0m")
        return f"--- SubAgent [{subagent_type}] 执行报告 ---\n\n{full_subagent_output}"

    # E. 配置主 Agent (Manager 层) 的可见工具
    # 主 Agent 不直接写代码，它只有：规划、委派、读取和查文档的能力
    manager_tools = [todo_manager, task_tool]
    _model_with_tools = model.bind_tools(manager_tools)

    # F. 全局工具索引 (供 execute_tools 节点查找)
    all_atomic_tools = all_atomic_tools + manager_tools
    _tools_by_name = {t.name: t for t in all_atomic_tools}

    # G. 构建 LangGraph 状态图
    builder = StateGraph(AgentState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", execute_tools)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue)
    builder.add_edge("tools", "agent")

    # 持久化记忆
    checkpointer = MemorySaver()
    app = builder.compile(checkpointer=checkpointer)

    print("\033[32m[OK] Nano Claude Code (Manager/Executor 架构) 已就绪。\033[0m")

    # H. 交互主循环
    session_config = {"configurable": {"thread_id": "manager_executor_v2"}}

    while True:
        try:
            print("===============================请输入您的需求or输入q,exit退出=================================")
            user_input = input("\n\033[36m >> \033[0m")
            if user_input.strip().lower() in ("q", "exit"):
                break
            if not user_input.strip():
                continue
        except (EOFError, KeyboardInterrupt):
            break

        initial_state = {"messages": [HumanMessage(content=user_input)], "current_todo": []}

        # 使用 astream 处理异步消息流
        async for chunk in app.astream(initial_state, session_config, stream_mode="updates"):
            for node_name, node_update in chunk.items():
                if "messages" in node_update:
                    for msg in node_update["messages"]:
                        # 过滤 ToolMessage 以保持终端简洁，只打印 AI 决策
                        print(f"\n--- 节点 [{node_name}] 输出 ---")
                        msg.pretty_print()

    store_memory(session_config, app)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
