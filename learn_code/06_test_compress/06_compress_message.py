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
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langgraph.prebuilt import ToolNode

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
    compression_history: Annotated[List[Dict[str, Any]], operator.add]


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


# =============================================================================================
#                                       Token监控
# =============================================================================================
def get_latest_token_usage(messages: List[BaseMessage]) -> int:
    """倒序查找最新的token使用情况

    这是一个优化的实现，避免遍历所有消息。
    从最新的消息开始查找，因为token usage信息通常在最近的AIMessage中。

    Args:
        messages: 消息列表

    Returns:
        最新的token使用总数
    """
    total_tokens = 0
    # 倒序扫描，找到最新的usage信息
    for msg in messages:
        # 注意：不同的LLM API返回的token统计格式可能不同。
        usage = msg.response_metadata.get('token_usage', {})

        if usage:
            # 精确计算：包含所有token类型
            total_tokens += usage.get('total_tokens', 0)
            if total_tokens > 0:
                return total_tokens
        else:
            if hasattr(msg, 'content') and msg.content:
                # 粗略估算：1 token ≈ 4 个字符（英文）或 1.5 个字符（中文）
                total_tokens += len(str(msg.content)) / 3

    # 如果没有找到usage信息，返回0
    return 0


def estimate_tokens(messages: List[BaseMessage]) -> int:
    """估算token数量（当没有精确token_usage时使用）

    粗略估算：1 token ≈ 4 个字符（英文）或 1.5 个字符（中文）

    Args:
        messages: 消息列表

    Returns:
        估算的token总数
    """
    total_chars = 0

    for msg in messages:
        if hasattr(msg, 'content') and msg.content:
            total_chars += len(str(msg.content))

    # 使用保守估算（假设中英文混合）
    return int(total_chars / 3)


def should_compression(state: AgentState):
    messages = state["messages"]
    # 1. 检查是否需要压缩
    if not needs_compression(messages):
        print("✅ Token使用率在安全范围内，无需压缩")
        return "agent"

    print("⚠️  Token使用率超过阈值，开始压缩...")
    return "compression"


def needs_compression(
        messages: List[BaseMessage],
        max_tokens: int = 163840,  # Claude Sonnet 3.5上下文窗口
        reserved_output: int = 4000,  # 为输出预留的token
        TOKEN_THRESHOLD: float = 0.92
) -> bool:
    """判断是否需要压缩

    综合考虑：
    1. 当前token使用量
    2. 模型上下文窗口大小
    3. 预留的输出空间

    Args:
        messages: 消息列表
        max_tokens: 最大token数
        reserved_output: 为输出预留的token数
        TOKEN_THRESHOLD:触发压缩阈值

    Returns:
        是否需要压缩
    """

    threshold = TOKEN_THRESHOLD
    # 获取当前token使用量
    current_tokens = get_latest_token_usage(messages)

    # 如果没有精确的usage信息，使用估算
    if current_tokens == 0:
        current_tokens = estimate_tokens(messages)
    print("✅ Token监控函数定义完成")
    # 计算可用token（减去预留的输出空间）
    available_tokens = max_tokens - reserved_output

    # 计算使用率
    usage_ratio = current_tokens / available_tokens

    print(f"📊 Token使用情况:")
    print(f"   当前: {current_tokens:,} tokens")
    print(f"   可用: {available_tokens:,} tokens")
    print(f"   使用率: {usage_ratio:.1%}")
    print(f"   阈值: {threshold:.1%}")

    return usage_ratio > threshold


# 8段式压缩提示词（完整版）
COMPRESSION_PROMPT = """你的任务是创建到目前为止对话的详细摘要，密切关注用户的明确请求和你之前的行动。
此摘要应彻底捕获技术细节、代码模式和架构决策，这些对于在不丢失上下文的情况下继续开发工作至关重要。

在提供最终摘要之前，将你的分析包装在<analysis>标签中，以组织你的思考并确保你涵盖了所有必要的要点。

你的摘要应包括以下部分:

1. **主要请求和意图**: 详细捕获用户的所有明确请求和意图
   - 用户明确说了什么？
   - 任务的核心目标是什么？
   - 有哪些隐含的需求？

2. **关键技术概念**: 列出讨论的所有重要技术概念、技术和框架
   - 使用了哪些技术栈？
   - 涉及哪些核心概念？
   - 有哪些重要的架构决策？

3. **文件和代码段**: 枚举检查、修改或创建的特定文件和代码段
   - 特别注意最近的消息
   - 包含完整的代码片段（如适用）
   - 记录文件路径和关键行号

4. **错误和修复**: 列出你遇到的所有错误以及修复方法
   - 具体的错误信息
   - 解决方案和原因
   - 特别注意用户的反馈

5. **问题解决**: 记录已解决的问题和任何正在进行的故障排除工作
   - 解决了哪些难题？
   - 采用了什么方法？
   - 还有哪些未解决的问题？

6. **所有用户消息**: 列出所有非工具结果的用户消息
   - 这些对于理解用户的反馈和变化的意图至关重要
   - 按时间顺序列出
   - 注意用户态度和需求的变化

7. **待处理任务**: 概述你明确被要求处理的任何待处理任务
   - 哪些任务还没完成？
   - 优先级如何？
   - 有哪些依赖关系？

8. **当前工作**: 详细描述在此摘要请求之前正在进行的确切工作
   - 最后在做什么？
   - 进展到哪一步了？
   - 下一步计划做什么？

9. **可选的下一步**: 列出与你最近正在做的工作相关的下一步
   - 逻辑上的下一步是什么？
   - 有哪些可能的方向？

请确保摘要足够详细，使得另一个AI助手（或你自己在新会话中）能够无缝地继续这个对话和工作。
"""
from langchain_core.messages import RemoveMessage
import datetime


class CompressionRecord(TypedDict):
    """压缩记录"""
    timestamp: str
    messages_removed: int
    messages_kept: int
    tokens_before: int
    tokens_after: int
    summary_content: str


async def compression_node(state: dict) -> dict:
    """上下文压缩节点

    工作流程：
    1. 检查是否需要压缩
    2. 如果不需要，直接返回空字典
    3. 如果需要，调用LLM生成摘要
    4. 删除旧消息，保留最近消息和摘要
    5. 记录压缩历史到compression_history

    Args:
        state: 当前状态

    Returns:
        更新后的状态（messages字段和compression_history字段）
    """
    messages = state.get("messages", [])

    print("\n" + "=" * 60)
    print("🗜️  压缩节点执行")
    print("=" * 60)

    # 记录压缩前的token数量
    tokens_before = get_latest_token_usage(messages)

    # 2. 调用LLM生成压缩摘要
    try:
        print("📝 生成8段式压缩摘要...")

        # 构建压缩请求
        compression_messages = [
            SystemMessage(content=COMPRESSION_PROMPT),
            *messages  # 包含完整对话历史
        ]

        # 调用LLM
        summary_response = await model.ainvoke(compression_messages)
        summary_content = summary_response.content

        print("=" * 60)
        print(f"✅ 摘要生成完成，长度: {len(summary_content)} 字符，内容如下：")
        print("=" * 60)
        print(summary_content)

        # 3. 决定保留多少最近消息
        # 策略：保留最近3条消息
        keep_recent = 3

        if len(messages) <= keep_recent:
            print("⚠️  消息数量较少，不进行压缩")
            return {}

        # 4. 构建新的消息列表
        # 删除旧消息，保留最近的消息
        messages_to_remove = messages[:-keep_recent]
        recent_messages = messages[-keep_recent:]

        # 如果拆分点下一条是ToolMessage, 将该条message也删除
        if recent_messages and isinstance(recent_messages[0], ToolMessage):
            messages_to_remove.append(recent_messages[0])
            recent_messages = recent_messages[1:]

        # 创建摘要消息
        summary_message = AIMessage(
            content=f"""📋 **对话摘要** (压缩于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})

            {summary_content}

            ---
            *以上是之前 {len(messages_to_remove)} 条消息的摘要，以下是最近的对话*
            """
        )

        print(f"🗑️  删除旧消息: {len(messages_to_remove)} 条")
        print(f"📌 保留最近消息: {len(recent_messages)} 条")
        print(f"📄 添加摘要: 1 条")

        # 5. 返回更新
        # 使用RemoveMessage删除旧消息
        remove_messages = [RemoveMessage(id=msg.id) for msg in messages_to_remove if hasattr(msg, 'id')]

        # 计算压缩后的token数量
        new_messages_for_count = recent_messages + [summary_message]
        tokens_after = estimate_tokens(new_messages_for_count)

        # 6. 创建压缩记录
        compression_record: CompressionRecord = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "messages_removed": len(messages_to_remove),
            "messages_kept": len(recent_messages),
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "summary_content": summary_content[:200] + "..." if len(summary_content) > 200 else summary_content
        }

        # 7. 获取现有的压缩历史并添加新记录
        compression_history = state.get("compression_history", [])

        print(f"📊 压缩统计:")
        print(f"   压缩前: {tokens_before:,} tokens")
        print(f"   压缩后: {tokens_after:,} tokens")
        print(
            f"   节省: {tokens_before - tokens_after:,} tokens ({(tokens_before - tokens_after) / tokens_before * 100:.1f}%)")

        # 返回更新的完整列表
        return {
            "messages": remove_messages + [summary_message],
            "compression_history": compression_history + [compression_record]
        }

    except Exception as e:
        print(f"❌ 压缩失败: {e}")
        print("继续执行，不进行压缩")
        return {}


# =============================================================================================
# 5. 主程序与子 Agent 初始化 (架构初始化层)
# =============================================================================================
def create_managed_agent(name: str, tools: list, system_prompt: str, checkpointer):
    """
    创建一个带有压缩和持久化能力的自定义 Agent
    """
    workflow = StateGraph(AgentState)

    # 绑定工具到模型
    model_with_tools = model.bind_tools(tools)

    # 定义模型调用节点
    async def call_agent_model(state: AgentState):
        todo_info = f"\n当前TODO: {json.dumps(state.get('current_todo', []), ensure_ascii=False)}" if state.get(
            'current_todo') else ""
        sys_msg = SystemMessage(content=f"{system_prompt}{todo_info}")
        resp = await model_with_tools.ainvoke([sys_msg] + state["messages"])
        return {"messages": [resp]}

    # 定义工具执行节点
    tool_node = ToolNode(tools)

    # 条件判断
    def should_continue(state: AgentState):
        last = state["messages"][-1]
        return "tools" if last.tool_calls else END

    # 构建图结构
    workflow.add_node("compression", compression_node)
    workflow.add_node("agent", call_agent_model)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")

    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_conditional_edges("tools", should_compression, ["compression", "agent"])
    workflow.add_edge("compression", "agent")  # 工具执行后再次检查压缩

    return workflow.compile(checkpointer=checkpointer)


async def main():
    global _tools_by_name, _model_with_tools, _agent_instances
    checkpointer = MemorySaver()
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

    for cfg in sub_configs:
        # 使用 create_react_agent 包装子 Agent 为独立的闭环 Graph
        agent_type = str(cfg["type"])
        _agent_instances[agent_type] = create_managed_agent(
            model, tools=cfg["tools"], system_prompt=cfg["prompt"], checkpointer=checkpointer
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

        full_subagent_output = ""
        import datetime
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sub_config2 = {"configurable": {"thread_id": f"{now_str}_{subagent_type}"}}

        # --- 核心改进：使用 astream 实时打印子 Agent 的过程 ---
        async for chunk in agent.astream({"messages": [HumanMessage(content=description)]}, config=sub_config2,
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
    builder.add_node("compression", compression_node)  # 新增：压缩节点

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue)
    builder.add_edge("tools", "agent")
    # tools执行后回到compression检查
    builder.add_conditional_edges("tools", should_compression, ["compression", "agent"])
    builder.add_edge("compression", "agent")

    # 持久化记忆
    checkpointer = MemorySaver()
    app = builder.compile(checkpointer=checkpointer)
    png_data = app.get_graph(xray=True).draw_mermaid_png()

    with open("compress_agent_graph.png", "wb") as f:
        f.write(png_data)

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
                if "messages" in node_update and isinstance(node_update, dict) and "messages" in node_update:
                    for msg in node_update["messages"]:
                        # 过滤 ToolMessage 以保持终端简洁，只打印 AI 决策
                        print(f"\n--- 节点 [{node_name}] 输出 ---")
                        msg.pretty_print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
