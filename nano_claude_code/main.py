import asyncio
import re
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import json
from datetime import datetime
from dotenv import load_dotenv
import redis.asyncio as aioredis

load_dotenv()
from core import prompt_ui
from typing import Dict, List, Literal
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.types import Command, interrupt

from core.state import AgentState
from core.config import get_model, REDIS_URI, REDIS_TTL
from core.memory_store import MemoryStore
from core.tools import WORKDIR
from coder_agent.agent import create_coder_agent
from researcher_agent.agent import create_researcher_agent
from researcher_agent.tools import get_mcp_tools
from reviewer_agent.agent import create_reviewer_agent
from memory import LongTermMemory
from memory import (
    create_compression_node,
    create_warn_node,
    make_token_router,
    route_after_warn,
)
from intent_agent.agent import create_intent_node


async def _handle_interrupt(val: str) -> str:
    """统一处理所有中断类型，返回 resume 值。"""
    # ── 需求确认问答（意图分析，可多轮）─────────────────────────
    if "❓" in val:
        print(f"\n\033[36m{val}\033[0m")
        ans = input("\n回答（回车=完成回答/生成文档，skip=跳过分析）: ").strip()
        return ans if ans else "done"

    # ── 需求文档确认（意图分析第二轮）────────────────────────────
    if "📄" in val:
        print(f"\n\033[36m{val}\033[0m")
        choice = await prompt_ui.select(
            "请选择操作：",
            [
                ("✅  确认文档，开始执行", "ok"),
                ("✏️   有补充修改意见", "__doc_feedback__"),
                ("🚫  跳过分析，直接执行", "skip"),
            ],
        )
        if choice == "__doc_feedback__":
            return input("请输入补充意见: ").strip()
        return choice

    # ── 计划确认反馈输入（二级中断）──────────────────────────────
    if val == "__feedback_input__":
        return input("请输入修改意见: ").strip()

    # ── 任务规划确认 ─────────────────────────────────────────────
    if "📋" in val:
        return await prompt_ui.select(
            "请选择操作：",
            [
                ("✅  确认，开始执行", ""),
                ("✏️   提供反馈，重新规划", "__feedback__"),
                ("🚫  取消任务", "abort"),
            ],
        )

    # ── 工具执行确认 ─────────────────────────────────────────────
    if "🔐" in val:
        print(f"\n{val}")
        return await prompt_ui.select(
            "请选择操作：",
            [
                ("✅  确认执行", "y"),
                ("❌  跳过（Agent 重新考虑）", ""),
                ("🚫  中止本次任务", "abort"),
            ],
        )

    # ── 上下文压缩提醒 ───────────────────────────────────────────
    print(f"\n\033[33m{val}\033[0m")
    return await prompt_ui.select(
        "请选择操作：",
        [
            ("跳过", "skip"),
            ("立即压缩", "compress"),
        ],
    )


@tool
def todo_manager(items: List[dict]) -> str:
    """更新当前会话的任务列表

    ## 任务对象必须包含以下键：
    - 'id': 任务编号 (如 "1")
    - 'text': 任务描述内容 (注意是 'text' 而不是 'description')
    - 'status': 状态，只能是 "pending", "in_progress", "completed", "failed" 之一。
    """
    status_headers = {
        "in_progress": "🔄进行中:",
        "pending": "⏳待处理:",
        "completed": "✅已完成:",
        "failed": "❌失败:",
    }
    lines = []
    for item in items:
        raw_status = item.get("status", "❌error")
        header = status_headers.get(raw_status, "❌error")
        tid = item.get("id", "❌id?")
        text = item.get("text") or item.get("description", "❌无内容")
        lines.append(f"{header} #{tid} {text}")
    if not lines:
        return "任务列表为空。"
    return f"--- 当前任务面板 ---\n" + "\n".join(lines)


# ==========================================================================
# task_tool 仅作为"虚拟工具"存在 —— 其 tool schema 用于让 Manager LLM 理解
# 可用的子 agent 类型及其参数。实际执行在子图 wrapper 节点中完成。
# 此函数不会被 tools 节点调用（tools_by_name 中不包含它）。
# ==========================================================================
@tool
def task_tool(description: str, subagent_type: str) -> str:
    """
    委派复杂的专项任务给专家处理。
    subagent_type 必须是 'tech-researcher'、'coder' 或 'reviewer'。
    """
    return ""  # 虚拟工具，实际执行在子图节点中


# ==========================================================================
# 子图 wrapper 工厂函数 —— 为每个子 agent 生成原生子图挂载节点
# ==========================================================================
def create_subagent_wrapper(agent, subagent_type: str, thread_id: str):
    """
    创建一个异步节点函数，该函数：
    1. 从父图 state 中提取 Manager 发出的 task_tool 调用参数
    2. 构造子图初始 state（隔离的 messages）
    3. 以独立 thread_id 隔离记忆的方式调用子图
    4. 将子图输出作为 ToolMessage 返回，闭合 Manager 的 tool_call
    """

    async def run_subagent(state: AgentState) -> Dict:
        # 从消息历史中找到最近的 task_tool 调用
        task_desc = ""
        tool_call_id = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
                for tc in msg.tool_calls:
                    if (
                        tc["name"] == "task_tool"
                        and tc["args"].get("subagent_type") == subagent_type
                    ):
                        task_desc = tc["args"]["description"]
                        tool_call_id = tc["id"]
                        break
            if task_desc:
                break

        if not task_desc:
            return {
                "messages": [
                    ToolMessage(
                        content=f"Error: 未找到 {subagent_type} 子图的任务描述",
                        tool_call_id="unknown",
                    )
                ]
            }

        print(f"\n\033[35m[系统] >>> 子 Agent ({subagent_type}) 开始工作...\033[0m")

        sub_config = {"configurable": {"thread_id": thread_id}}
        sub_stream_input = {
            "messages": [HumanMessage(content=task_desc)],
            "current_todo": [],
            "compress_choice": "",
        }

        full_output = ""
        while True:
            interrupted = False
            async for chunk in agent.astream(
                sub_stream_input, sub_config, stream_mode="updates"
            ):
                if "__interrupt__" in chunk:
                    for intr in chunk["__interrupt__"]:
                        val = str(intr.value)
                        print(f"\n\033[35m[子Agent: {subagent_type}]\033[0m")
                        raw = await _handle_interrupt(val)
                        sub_stream_input = Command(resume=raw)
                        interrupted = True
                    break

                for node, data in chunk.items():
                    print(f"  \033[34m└─ [{subagent_type}.{node}]\033[0m 正在处理...")
                    if data and "messages" in data:
                        for msg in data["messages"]:
                            if isinstance(msg, AIMessage) and msg.content:
                                preview = msg.content[:200].replace("\n", " ")
                                print(f"    \033[37m思考: {preview}...\033[0m")
                            if isinstance(msg, AIMessage) and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    print(f"    \033[32m工具调用: {tc['name']}\033[0m")
                            if isinstance(msg, AIMessage):
                                full_output = msg.content

            if not interrupted:
                break

        print(f"\033[35m[系统] <<< 子 Agent ({subagent_type}) 任务完成。\033[0m")

        return {
            "messages": [
                ToolMessage(
                    content=f"--- SubAgent [{subagent_type}] 执行报告 ---\n\n{full_output}",
                    tool_call_id=tool_call_id,
                )
            ]
        }

    return run_subagent


_SESSION_RE = re.compile(r'(session_[^:]+?)(_coder|_tech-researcher|_reviewer)?(?=:|$)')


async def _scan_base_sessions(db_uri: str) -> list[str]:
    """从 Redis 扫描所有 session_ 开头的基础会话 ID，倒序返回。"""
    r = aioredis.from_url(db_uri, decode_responses=True)
    try:
        all_keys = await r.keys("*")
        base_ids = set()
        for k in all_keys:
            m = _SESSION_RE.search(k)
            if m:
                base_ids.add(m.group(1))
        return sorted(base_ids, reverse=True)
    finally:
        await r.aclose()


async def _scan_mysql_sessions(memory_store) -> list[str]:
    """从 MySQL 扫描所有归档的会话 ID，倒序返回。"""
    try:
        threads = await memory_store.list_threads()
        return sorted([t["thread_id"] for t in threads], reverse=True)
    except Exception:
        return []


async def _restore_from_mysql(thread_id: str, memory_store: MemoryStore, app, session_config: dict) -> bool:
    """从 MySQL 加载历史消息注入 State，重建 Redis checkpoint。

    Returns: True 表示成功从 MySQL 恢复了历史上下文。
    """
    messages = await memory_store.load(thread_id)
    if not messages:
        return False
    print(f"\033[33m[记忆] 从 MySQL 恢复了 {len(messages)} 条历史消息，重建 Redis checkpoint。\033[0m")
    # 用一条空消息触发 LangGraph checkpoint 建立
    initial_state = {"messages": list(messages)}
    try:
        await app.aupdate_state(session_config, initial_state)
    except Exception:
        # aupdate_state 可能不可用，回退到 stream 方式
        async for _ in app.astream(
            {"messages": []}, session_config, stream_mode="updates"
        ):
            pass
    return True


async def _generate_title(model, messages) -> str:
    """用 LLM 根据对话内容生成简短标题。"""
    try:
        lines = []
        for msg in messages[:10]:
            content = msg.content if hasattr(msg, "content") else ""
            if isinstance(content, list):
                content = json.dumps(content, ensure_ascii=False)
            role = getattr(msg, "type", "?").upper()
            lines.append(f"[{role}]: {str(content)[:200]}")
        prompt = (
            "请用不超过15个字概括以下对话的核心任务，只输出标题本身，不要标点符号和引号：\n\n"
            + "\n".join(lines)
        )
        resp = await model.ainvoke([HumanMessage(content=prompt)])
        return resp.content.strip()[:100]
    except Exception:
        return ""


async def _archive_to_mysql(thread_id: str, checkpointer, memory_store: MemoryStore, model=None) -> None:
    """将 Redis checkpoint 中的消息归档到 MySQL。"""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = await checkpointer.aget(config)
    except Exception as e:
        print(f"\033[33m[记忆] 读取 Redis checkpoint 失败: {e}\033[0m")
        return

    if snapshot is None:
        return

    channel_values = getattr(snapshot, "channel_values", {})
    if isinstance(channel_values, dict):
        messages = channel_values.get("messages", [])
    else:
        messages = getattr(channel_values, "messages", [])

    if not messages:
        return

    title = None
    if model is not None:
        title = await _generate_title(model, messages)
        if title:
            print(f"\033[36m[记忆] 会话标题：{title}\033[0m")

    try:
        await memory_store.archive(thread_id, messages, title=title)
        print(f"\033[32m[记忆] 已将会话 {thread_id} 的 {len(messages)} 条消息归档到 MySQL。\033[0m")
    except Exception as e:
        print(f"\033[31m[记忆] 归档到 MySQL 失败: {e}\033[0m")


async def main():
    print("\033[34m[*] 系统正在启动...\033[0m")
    model = get_model()

    import os; os.environ.setdefault("REDIS_URL", REDIS_URI)  # redisvl 后台任务需要

    # 初始化 MySQL MemoryStore（可选，失败则降级为 Redis-only）
    memory_store = None
    try:
        from core.config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
        memory_store = await MemoryStore.create(
            host=MYSQL_HOST, port=MYSQL_PORT,
            user=MYSQL_USER, password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
        )
        print("\033[32m[OK] MySQL 记忆存储已连接。\033[0m")
    except Exception as e:
        print(f"\033[33m[!] MySQL 不可用 ({e})，将仅使用 Redis 记忆。\033[0m")

    try:
        async with AsyncRedisSaver.from_conn_string(REDIS_URI, ttl=REDIS_TTL) as checkpointer, LongTermMemory(
            REDIS_URI, memory_store=memory_store
        ) as ltm:
            cached_memories = await ltm.load_all()
        if cached_memories:
            print(f"[*] 已加载 {len(cached_memories)} 条长期记忆。")

        print("[*] 正在连接 MCP 服务器...")
        mcp_tools = await get_mcp_tools()
        if mcp_tools:
            names = [t.name for t in mcp_tools]
            print(f"\033[32m[OK] MCP 工具汇总：共 {len(mcp_tools)} 个: {names}\033[0m")
        else:
            print(
                "\033[33m[!] 所有 MCP 服务器均不可用，research agent 将仅使用 read_file\033[0m"
            )

        existing = await _scan_base_sessions(REDIS_URI)
        choices = [(f"▶  恢复: {sid}", f"resume:{sid}") for sid in existing]

        # 扫描 MySQL 归档会话，把 Redis 中已不存在的加进来
        mysql_sessions = []
        _restored_from_mysql = False
        if memory_store is not None:
            threads = await memory_store.list_threads()
            mysql_sessions = [t for t in threads if t["thread_id"] not in existing]
            for t in mysql_sessions:
                title_str = f"「{t['title']}」" if t.get("title") else t["thread_id"]
                choices.append((
                    f"🗄  {title_str}  ({t['message_count']}条消息  {str(t['updated_at'])[:16]})",
                    f"mysql:{t['thread_id']}"
                ))

        choices.append(("✨  新建会话", "__new__"))
        pick = await prompt_ui.select("请选择会话：", choices)

        if pick.startswith("resume:"):
            session_id = pick[len("resume:"):]
            project_name = re.sub(r'_\d{8}_\d{4}$', '', session_id[len("session_"):])
            print(f"\033[32m[OK] 恢复会话: {session_id}\033[0m")
        elif pick.startswith("mysql:"):
            session_id = pick[len("mysql:"):]
            project_name = re.sub(r'_\d{8}_\d{4}$', '', session_id[len("session_"):])
            print(f"\033[32m[OK] 从 MySQL 恢复会话: {session_id}\033[0m")
            _restored_from_mysql = True
        else:
            print("\n请输入项目名称（如：天气agent、search_tool，直接回车则仅用时间）:")
            project_name = input(" >> ").strip()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            session_id = (
                f"session_{project_name}_{timestamp}"
                if project_name
                else f"session_{timestamp}"
            )
        print(f"[*] 会话 ID: {session_id}")

        print("[*] 正在同步专家组 (Executor SubAgents) 配置...")
        coder_agent = create_coder_agent(model, checkpointer)
        researcher_agent = create_researcher_agent(model, checkpointer, mcp_tools)
        reviewer_agent = create_reviewer_agent(model, checkpointer)

        # =====================================================================
        # Manager 工具配置
        # task_tool 用于 LLM 路由选择（会出现在 tool schema 中），
        # 但不在 tools_by_name 中 —— 路由在条件边中拦截，不进入 tools 节点
        # =====================================================================
        manager_tools = [todo_manager, task_tool]
        model_with_tools = model.bind_tools(manager_tools)
        tools_by_name = {"todo_manager": todo_manager}  # 只有真正的工具

        # =====================================================================
        # 图节点定义
        # =====================================================================
        async def call_manager_model(state: AgentState) -> Dict:
            current_todo = state.get("current_todo") or []
            extra_state: Dict = {}

            # 恢复会话时 current_todo 可能为空（checkpoint 未保存或崩溃前未执行）
            # 从消息历史中找最后一次 todo_manager 调用来重建任务列表
            if not current_todo:
                for msg in reversed(state["messages"]):
                    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                        for tc in msg.tool_calls:
                            if tc["name"] == "todo_manager" and tc["args"].get("items"):
                                current_todo = tc["args"]["items"]
                                extra_state["current_todo"] = current_todo
                                print(f"\033[33m[Manager] 从消息历史恢复任务列表（{len(current_todo)} 条）\033[0m")
                                break
                    if current_todo:
                        break

            todo_status = json.dumps(current_todo, ensure_ascii=False)
            mem_section = ltm.format_for_prompt(cached_memories)
            system_content = (
                f"你是一个高级编程协调员 (Manager)。当前工作路径: {WORKDIR}\n"
            )
            if mem_section:
                system_content += f"\n{mem_section}\n"
            system_content += (
                f"当前任务计划进度: {todo_status}\n\n"
                "核心操作守则：\n"
                "1. 规划优先：面对复杂任务，必须先调用 'todo_manager' 编排分步计划。\n"
                "2. 专家委派：你自己不直接写代码或查文档。请通过 'task_tool' 指挥专家：\n"
                "   - 查阅技术资料 -> 调用 'tech-researcher'\n"
                "   - 编写、修改代码或执行只读 Bash -> 调用 'coder'\n"
                "   - 审查代码质量、运行测试 -> 调用 'reviewer'\n"
                "3. 状态闭环：开始执行步骤前更新为 'in_progress'，完成后更新为 'completed'。\n"
                "4. 审查强制：'coder' 完成代码编写/修改后，你必须立即调用 'reviewer' 对该代码进行审查。\n"
                "   reviewer 会运行测试、检查代码质量。只有 reviewer 确认通过后，才能将任务标记为 completed。\n"
                "5. 严谨性：在委派 'coder' 修改文件前，必须确保自己或专家已调用过 'read_file'。\n"
                "6. 项目目录：创建新项目时，必须先在工作路径下建立以项目名称命名的根目录（如 {WORKDIR}/my_project/），"
                "所有文件都放在该目录内，禁止直接将项目文件散落在工作路径根目录下。"
            )
            system_prompt = SystemMessage(content=system_content)
            messages = state["messages"]
            response = await model_with_tools.ainvoke([system_prompt] + messages)
            return {"messages": [response], **extra_state}

        async def execute_todo_only(state: AgentState) -> Dict:
            """仅执行 todo_manager 工具；task_tool 调用已被条件边拦截"""
            last_message = state["messages"][-1]
            updates = {"messages": []}
            if hasattr(last_message, "tool_calls"):
                for tool_call in last_message.tool_calls:
                    name = tool_call["name"]
                    if name == "task_tool":
                        continue  # 防御性跳过

                    print(f"\n\033[33m[Manager 正在执行工具: {name}]\033[0m")

                    if name == "todo_manager":
                        items = tool_call["args"]["items"]
                        statuses = {it.get("status") for it in items}
                        # 仅当所有条目都是 pending 时才视为初次规划，触发确认弹窗
                        # 后续状态更新（含 in_progress/completed）不再触发
                        is_initial_plan = bool(items) and statuses == {"pending"}
                        updates["current_todo"] = items
                        updates["plan_needs_confirm"] = is_initial_plan

                    tool_obj = tools_by_name.get(name)
                    if not tool_obj:
                        observation = f"Error: 工具 '{name}' 未在系统中注册。"
                    else:
                        try:
                            observation = await tool_obj.ainvoke(tool_call["args"])
                            if (
                                isinstance(observation, str)
                                and len(observation) > 10000
                            ):
                                observation = (
                                    observation[:10000] + "\n... (内容过长，已自动截断)"
                                )
                        except Exception as e:
                            observation = f"Error executing {name}: {e}"

                    updates["messages"].append(
                        ToolMessage(
                            content=str(observation), tool_call_id=tool_call["id"]
                        )
                    )
            return updates

        # =====================================================================
        # 条件路由：解析 Manager 的 tool_calls 决定下一步
        # =====================================================================
        def route_after_agent(
            state: AgentState,
        ) -> Literal["coder", "researcher", "reviewer", "tools", "__end__"]:
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                for tc in last_message.tool_calls:
                    if tc["name"] == "task_tool":
                        target = tc["args"].get("subagent_type", "")
                        if target in ("coder", "tech-researcher", "reviewer"):
                            return (
                                "researcher" if target == "tech-researcher" else target
                            )
                # 其他 tool calls（如 todo_manager）→ tools 节点
                return "tools"
            return "__end__"

        async def plan_confirm_node(state: AgentState) -> Dict:
            """中断节点：展示任务规划，等待用户通过方向键选择。"""
            todos = state.get("current_todo", [])
            icons = {
                "pending": "⏳",
                "in_progress": "🔄",
                "completed": "✅",
                "failed": "❌",
            }
            lines = [
                f"  {icons.get(t.get('status',''), '•')} #{t.get('id')} {t.get('text', t.get('description', ''))}"
                for t in todos
            ]
            plan_text = "\n".join(lines)
            choice = interrupt(f"📋 任务规划已生成：\n\n{plan_text}")
            result: Dict = {"plan_needs_confirm": False}
            if choice == "__feedback__":
                feedback = interrupt("__feedback_input__")
                feedback_text = str(feedback).strip()
                if feedback_text:
                    result["messages"] = [
                        HumanMessage(
                            content=f"[规划反馈] {feedback_text}，请根据此反馈重新调整任务规划。"
                        )
                    ]
            elif choice == "abort":
                result["messages"] = [
                    HumanMessage(content="用户已取消任务，请终止执行并告知用户。")
                ]
            return result

        def route_after_tools_node(state: AgentState) -> str:
            """tools 执行后：若刚生成全新计划则先确认，否则走 token 路由。"""
            if state.get("plan_needs_confirm"):
                return "plan_confirm"
            return _router(state)

        def route_after_plan_confirm_node(state: AgentState) -> str:
            """plan_confirm 后：用户给了反馈则回 agent 重新规划，否则走 token 路由继续执行。"""
            msgs = state.get("messages", [])
            if (
                msgs
                and isinstance(msgs[-1], HumanMessage)
                and "[规划反馈]" in str(msgs[-1].content)
            ):
                return "agent"
            return _router(state)

        # =====================================================================
        # 构建主图 (Manager + 原生挂载的子图节点)
        # =====================================================================
        _router = make_token_router()

        builder = StateGraph(AgentState)

        # 主节点
        builder.add_node("intent_analysis", create_intent_node(model, project_name))
        builder.add_node("agent", call_manager_model)
        builder.add_node("tools", execute_todo_only)
        builder.add_node("plan_confirm", plan_confirm_node)
        builder.add_node("warn", create_warn_node())
        builder.add_node("compress", create_compression_node(model))

        # 子图 wrapper 节点 —— 子图作为父图的节点挂载
        builder.add_node(
            "coder",
            create_subagent_wrapper(coder_agent, "coder", f"{session_id}_coder"),
        )
        builder.add_node(
            "researcher",
            create_subagent_wrapper(
                researcher_agent, "tech-researcher", f"{session_id}_tech-researcher"
            ),
        )
        builder.add_node(
            "reviewer",
            create_subagent_wrapper(
                reviewer_agent, "reviewer", f"{session_id}_reviewer"
            ),
        )

        # 边
        builder.add_edge(START, "intent_analysis")
        builder.add_edge("intent_analysis", "agent")
        builder.add_conditional_edges(
            "agent",
            route_after_agent,
            {
                "coder": "coder",
                "researcher": "researcher",
                "reviewer": "reviewer",
                "tools": "tools",
                "__end__": "__end__",
            },
        )
        # 子图 / tools 执行完后经 token 路由决定：warn / compress / agent
        _route_map = {"compress": "compress", "warn": "warn", "agent": "agent"}
        builder.add_conditional_edges("coder", _router, _route_map)
        builder.add_conditional_edges("researcher", _router, _route_map)
        builder.add_conditional_edges("reviewer", _router, _route_map)
        # tools 先判断是否需要计划确认，再走 token 路由
        builder.add_conditional_edges(
            "tools",
            route_after_tools_node,
            {**_route_map, "plan_confirm": "plan_confirm"},
        )
        builder.add_conditional_edges(
            "plan_confirm",
            route_after_plan_confirm_node,
            {**_route_map, "agent": "agent"},
        )
        builder.add_conditional_edges(
            "warn", route_after_warn, {"compress": "compress", "agent": "agent"}
        )
        builder.add_edge("compress", "agent")

        app = builder.compile(checkpointer=checkpointer)

        # --- 保存主图到本地 ---
        try:
            graph_image = app.get_graph().draw_mermaid_png()
            with open("manager_graph.png", "wb") as f:
                f.write(graph_image)
            print("\033[32m[OK] 主图 (Manager) 已保存至 manager_graph.png\033[0m")
        except Exception as e:
            print(f"\033[33m[!] 无法生成 PNG (可能缺少 graphviz): {e}\033[0m")
            try:
                mermaid_text = app.get_graph().draw_mermaid()
                with open("manager_graph.mmd", "w") as f:
                    f.write(mermaid_text)
                print(
                    "\033[32m[OK] 主图 (Manager) Mermaid 已保存至 manager_graph.mmd\033[0m"
                )
            except Exception as e2:
                print(f"\033[31m[!] 保存 Mermaid 也失败: {e2}\033[0m")

        print("\033[32m[OK] Nano Claude Code (Manager/Executor 架构) 已就绪。\033[0m")
        session_config = {"configurable": {"thread_id": session_id}}

        # ── MySQL 恢复：将归档消息注入 State，重建 Redis checkpoint ──────
        if _restored_from_mysql and memory_store is not None:
            await _restore_from_mysql(session_id, memory_store, app, session_config)

        # ── 恢复会话时检测未完成任务，询问是否自动继续 ────────────────────
        _auto_input = None
        if pick.startswith("resume:") or _restored_from_mysql:
            try:
                snap = await app.aget_state(session_config)
                saved_todo: list = []
                if snap and snap.values:
                    saved_todo = snap.values.get("current_todo") or []
                    # checkpoint 里没有 → 从消息历史重建
                    if not saved_todo:
                        for msg in reversed(snap.values.get("messages", [])):
                            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                                for tc in msg.tool_calls:
                                    if tc["name"] == "todo_manager" and tc["args"].get("items"):
                                        saved_todo = tc["args"]["items"]
                                        break
                            if saved_todo:
                                break

                pending = [t for t in saved_todo if t.get("status") in ("pending", "in_progress")]
                if pending:
                    _icons = {"in_progress": "🔄", "pending": "⏳"}
                    print("\n\033[33m[*] 检测到以下未完成任务：\033[0m")
                    for t in pending[:6]:
                        print(f"  {_icons.get(t.get('status',''), '•')} #{t.get('id')} {t.get('text', t.get('description',''))}")
                    if len(pending) > 6:
                        print(f"  ... 共 {len(pending)} 条未完成")
                    _choice = await prompt_ui.select(
                        "是否自动继续执行未完成的任务？",
                        [
                            ("▶  是，继续执行", "continue"),
                            ("✏️   否，手动输入", "manual"),
                        ],
                    )
                    if _choice == "continue":
                        _auto_input = "请继续完成未完成的任务，从当前进度继续，不需要重新规划。"
            except Exception as _e:
                print(f"\033[33m[!] 检测恢复状态失败: {_e}，跳过自动继续\033[0m")

        # =====================================================================
        # 交互主循环
        # =====================================================================
        while True:
            try:
                print("=" * 60)
                if _auto_input:
                    user_input = _auto_input
                    _auto_input = None
                    print(f"\033[33m[自动继续] >> {user_input}\033[0m")
                else:
                    print("请输入您的需求 (输入 q/exit 退出):")
                    user_input = input("\n\033[36m >> \033[0m")
                    if user_input.strip().lower() in ("q", "exit"):
                        break
                    if not user_input.strip():
                        continue
            except (EOFError, KeyboardInterrupt):
                break

            stream_input = {
                "messages": [HumanMessage(content=user_input)],
                "compress_choice": "",
                "plan_needs_confirm": False,
            }
            while True:
                interrupted = False
                async for chunk in app.astream(
                    stream_input, session_config, stream_mode="updates"
                ):
                    if "__interrupt__" in chunk:
                        for intr in chunk["__interrupt__"]:
                            val = str(intr.value)
                            raw = await _handle_interrupt(val)
                            stream_input = Command(resume=raw)
                            interrupted = True
                        break

                    for node_name, node_update in chunk.items():
                        if node_update and "messages" in node_update:
                            for msg in node_update["messages"]:
                                print(f"\n--- 节点 [{node_name}] 输出 ---")
                                msg.pretty_print()

                if not interrupted:
                    break

        _MIN_MSGS_FOR_MEMORY = 6  # 少于此条数视为会话过短，跳过记忆提炼
        try:
            snap = await app.aget_state(session_config)
            msgs = snap.values.get("messages", [])
            if len(msgs) < _MIN_MSGS_FOR_MEMORY:
                print(
                    f"\n[*] 会话消息不足 {_MIN_MSGS_FOR_MEMORY} 条（当前 {len(msgs)} 条），跳过记忆提炼。"
                )
            else:
                print("\n[*] 正在提炼本次会话的长期记忆...")
                new_mems = await ltm.extract_and_save(model, msgs)
                if new_mems:
                    print(f"[OK] 已保存 {len(new_mems)} 条新记忆:")
                    for m in new_mems:
                        print(f"  [{m['type']}] {m['name']}: {m['description']}")
                else:
                    print("[*] 本次会话无需提炼新记忆。")
        except Exception as e:
            print(f"[!] 记忆提炼过程出错: {e}")

        # ── 归档完整对话到 MySQL ────────────────────────────────────────
        if memory_store is not None:
            print("\n[*] 正在归档会话到 MySQL...")
            await _archive_to_mysql(session_id, checkpointer, memory_store, model=model)

    finally:
        if memory_store is not None:
            await memory_store.close()
            print("\033[32m[OK] MySQL 连接已关闭。\033[0m")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        # 确保 MemoryStore 被关闭（asyncio.run 内部异常时也能清理）
        pass
