import asyncio
import json
import re
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from datetime import datetime
from typing import Dict, List, Literal

import redis.asyncio as aioredis
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command, interrupt

load_dotenv()

from core import prompt_ui
from core.agent_loader import list_skills
from core.config import REDIS_TTL, REDIS_URI, get_model
from core.memory_store import MemoryStore
from core.spawn_tool import create_spawn_tool, init_spawn_tool
from core.state import AgentState
from core.tools import WORKDIR, build_tool_registry
from intent_agent.agent import create_intent_tool
from memory import (
    LongTermMemory,
    create_compression_node,
    create_warn_node,
    make_token_router,
    route_after_warn,
)
from core.mcp import get_mcp_tools


# ── 中断统一处理 ──────────────────────────────────────────────────────────────

async def _handle_interrupt(val: str) -> str:
    """统一处理所有 interrupt 类型，返回 resume 值。"""
    if "❓" in val:
        print(f"\n\033[36m{val}\033[0m")
        ans = input("\n回答（回车=完成/skip=跳过分析）: ").strip()
        return ans if ans else "done"

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

    if val == "__feedback_input__":
        return input("请输入修改意见: ").strip()

    if "📋" in val:
        return await prompt_ui.select(
            "请选择操作：",
            [
                ("✅  确认，开始执行", ""),
                ("✏️   提供反馈，重新规划", "__feedback__"),
                ("🚫  取消任务", "abort"),
            ],
        )

    # 上下文压缩提醒
    print(f"\n\033[33m{val}\033[0m")
    return await prompt_ui.select(
        "请选择操作：",
        [("跳过", "skip"), ("立即压缩", "compress")],
    )


# ── todo_manager 工具 ─────────────────────────────────────────────────────────

@tool
def todo_manager(items: List[dict]) -> str:
    """更新当前会话的任务列表。

    每个任务对象必须包含：
    - 'id': 任务编号（如 "1"）
    - 'text': 任务描述
    - 'status': "pending" | "in_progress" | "completed" | "failed"
    """
    _icons = {
        "in_progress": "🔄进行中:",
        "pending":     "⏳待处理:",
        "completed":   "✅已完成:",
        "failed":      "❌失败:",
    }
    lines = []
    for item in items:
        status = item.get("status", "❌error")
        lines.append(
            f"{_icons.get(status, '❌error')} #{item.get('id', '?')} "
            f"{item.get('text') or item.get('description', '无内容')}"
        )
    return "--- 当前任务面板 ---\n" + "\n".join(lines) if lines else "任务列表为空。"


# ── Redis 会话管理 ─────────────────────────────────────────────────────────────

_SESSION_RE = re.compile(r'(session_[^:]+?)(_coder|_tech-researcher|_reviewer)?(?=:|$)')


async def _scan_base_sessions(db_uri: str) -> list[str]:
    r = aioredis.from_url(db_uri, decode_responses=True)
    try:
        base_ids = set()
        for k in await r.keys("*"):
            m = _SESSION_RE.search(k)
            if m:
                base_ids.add(m.group(1))
        return sorted(base_ids, reverse=True)
    finally:
        await r.aclose()


async def _restore_from_mysql(
    thread_id: str, memory_store: MemoryStore, app, session_config: dict
) -> bool:
    messages = await memory_store.load(thread_id)
    if not messages:
        return False
    print(f"\033[33m[记忆] 从 MySQL 恢复了 {len(messages)} 条历史消息。\033[0m")
    try:
        await app.aupdate_state(session_config, {"messages": list(messages)})
    except Exception:
        async for _ in app.astream({"messages": []}, session_config, stream_mode="updates"):
            pass
    return True


async def _generate_title(model, messages) -> str:
    try:
        lines = []
        for msg in messages[:10]:
            content = msg.content if hasattr(msg, "content") else ""
            if isinstance(content, list):
                content = json.dumps(content, ensure_ascii=False)
            lines.append(f"[{getattr(msg, 'type', '?').upper()}]: {str(content)[:200]}")
        resp = await model.ainvoke([HumanMessage(
            content="请用不超过15个字概括以下对话的核心任务，只输出标题本身，不要标点和引号：\n\n"
                    + "\n".join(lines)
        )])
        return resp.content.strip()[:100]
    except Exception:
        return ""


async def _archive_to_mysql(
    thread_id: str, checkpointer, memory_store: MemoryStore, model=None
) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = await checkpointer.aget(config)
    except Exception as e:
        print(f"\033[33m[记忆] 读取 checkpoint 失败: {e}\033[0m")
        return
    if snapshot is None:
        return
    channel_values = getattr(snapshot, "channel_values", {})
    messages = (channel_values if isinstance(channel_values, dict) else {}).get("messages", [])
    if not messages:
        return
    title = await _generate_title(model, messages) if model else None
    if title:
        print(f"\033[36m[记忆] 会话标题：{title}\033[0m")
    try:
        await memory_store.archive(thread_id, messages, title=title)
        print(f"\033[32m[记忆] 已归档 {len(messages)} 条消息到 MySQL。\033[0m")
    except Exception as e:
        print(f"\033[31m[记忆] 归档失败: {e}\033[0m")


# ── 主程序 ────────────────────────────────────────────────────────────────────

async def main():
    print("\033[34m[*] 系统正在启动...\033[0m")
    model = get_model()

    import os
    os.environ.setdefault("REDIS_URL", REDIS_URI)

    # MySQL MemoryStore
    memory_store = None
    try:
        from core.config import (
            MYSQL_DATABASE,
            MYSQL_HOST,
            MYSQL_PASSWORD,
            MYSQL_PORT,
            MYSQL_USER,
        )
        memory_store = await MemoryStore.create(
            host=MYSQL_HOST, port=MYSQL_PORT,
            user=MYSQL_USER, password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
        )
        print("\033[32m[OK] MySQL 记忆存储已连接。\033[0m")
    except Exception as e:
        print(f"\033[33m[!] MySQL 不可用 ({e})，仅使用 Redis 记忆。\033[0m")

    try:
        async with AsyncRedisSaver.from_conn_string(REDIS_URI, ttl={"default_ttl": REDIS_TTL}) as checkpointer, \
                   LongTermMemory(REDIS_URI, memory_store=memory_store) as ltm:

            # 加载长期记忆
            cached = await ltm.load_all()
            if cached:
                print(f"[*] 已加载 {len(cached)} 条长期记忆。")

            # 初始化 MCP 工具
            print("[*] 正在连接 MCP 服务器...")
            mcp_tools = await get_mcp_tools()
            if mcp_tools:
                print(f"\033[32m[OK] MCP 工具：{len(mcp_tools)} 个\033[0m")
            else:
                print("\033[33m[!] MCP 服务器不可用，将使用基础工具集。\033[0m")

            # 构建工具注册表
            build_tool_registry(mcp_tools)

            # 初始化 spawn_tool
            init_spawn_tool(model, mcp_tools)
            spawn_agents = create_spawn_tool()

            # 打印已加载的 skill
            skill_defs = list_skills()
            print(f"[*] 已加载 {len(skill_defs)} 个 Skill: "
                  f"{[s['name'] for s in skill_defs]}")

            # ── 会话选择 ───────────────────────────────────────────────────
            existing = await _scan_base_sessions(REDIS_URI)
            choices = [(f"▶  恢复: {sid}", f"resume:{sid}") for sid in existing]

            _restored_from_mysql = False
            if memory_store is not None:
                threads = await memory_store.list_threads()
                mysql_only = [t for t in threads if t["thread_id"] not in existing]
                for t in mysql_only:
                    title_str = f"「{t['title']}」" if t.get("title") else t["thread_id"]
                    choices.append((
                        f"🗄  {title_str}  ({t['message_count']}条  {str(t['updated_at'])[:16]})",
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
                _restored_from_mysql = True
                print(f"\033[32m[OK] 从 MySQL 恢复: {session_id}\033[0m")
            else:
                print("\n请输入项目名称（直接回车仅用时间）:")
                project_name = input(" >> ").strip()
                ts = datetime.now().strftime("%Y%m%d_%H%M")
                session_id = f"session_{project_name}_{ts}" if project_name else f"session_{ts}"

            print(f"[*] 会话 ID: {session_id}")

            # ── Supervisor 工具配置 ────────────────────────────────────────
            intent_tool = create_intent_tool(model, project_name)
            manager_tools = [todo_manager, intent_tool, spawn_agents]
            model_with_tools = model.bind_tools(manager_tools)
            tools_by_name = {t.name: t for t in manager_tools}

            # ── 图节点定义 ─────────────────────────────────────────────────
            async def call_manager_model(state: AgentState) -> Dict:
                current_todo = state.get("current_todo") or []

                # 从消息历史恢复 todo（checkpoint 崩溃时兜底）
                if not current_todo:
                    for msg in reversed(state["messages"]):
                        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                            for tc in msg.tool_calls:
                                if tc["name"] == "todo_manager" and tc["args"].get("items"):
                                    current_todo = tc["args"]["items"]
                                    break
                        if current_todo:
                            break

                todo_status = json.dumps(current_todo, ensure_ascii=False)

                # 混合检索相关长期记忆
                last_human = next(
                    (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
                )
                query = last_human.content if last_human else ""
                if isinstance(query, list):
                    query = " ".join(str(p) for p in query)
                mem_section = await ltm.search_for_prompt(query)

                # 动态生成 skill 目录（每次调用时从文件系统读取，支持热更新）
                fresh_skills = list_skills()
                skill_catalog = "\n".join(
                    f"   - `{s['name']}`: {s['description']}" for s in fresh_skills
                )

                system_content = (
                    f"你是一个高级编程协调员 (Supervisor)。当前工作路径: {WORKDIR}\n"
                )
                if mem_section:
                    system_content += f"\n{mem_section}\n"
                system_content += (
                    f"当前任务进度: {todo_status}\n\n"
                    "核心操作守则：\n"
                    "0. 需求判断：收到任务后先判断需求是否明确。\n"
                    "   调用 'analyze_intent' 的条件（需求不确定）：\n"
                    "   - 用户描述了目标但没说怎么做（「加个分享功能」「支持多用户」）\n"
                    "   - 新功能涉及数据结构、接口设计或与现有代码的集成方式未定\n"
                    "   - 即使项目已有代码，后续方向仍有多种可能性\n"
                    "   直接执行的条件（需求已明确）：\n"
                    "   - 修 bug、改样式、重构指定模块等目标和边界清晰的任务\n"
                    "   - 用户已提供足够细节（接口、字段、行为均已说明）\n"
                    "1. 规划优先：面对复杂任务，先调用 'todo_manager' 编排分步计划。\n"
                    "2. 专家委派：通过 'spawn_agents' 并行启动专家执行任务，你自己不写代码。\n"
                    "   每个 Agent 通过 skills 列表注入身份和领域知识：\n"
                    f"{skill_catalog}\n"
                    "   身份 skill（coder/researcher/reviewer）决定 agent 的角色和行为约束。\n"
                    "   领域 skill（react-patterns 等）提供具体技术知识，可叠加。\n"
                    "   示例: skills=[\"coder\",\"react-patterns\"] → 前端开发；skills=[\"reviewer\"] → 代码审查\n"
                    "3. 状态闭环：启动子 Agent 前更新 todo 为 in_progress，完成后更新为 completed。\n"
                    "4. 审查强制：代码编写完成后，必须通过 skills=[\"reviewer\"] Agent 审查后才能标记完成。\n"
                    f"5. 项目目录：新项目所有文件必须放在 {WORKDIR}/<项目名>/ 下，不散落在根目录。"
                )

                response = await model_with_tools.ainvoke(
                    [SystemMessage(content=system_content)] + state["messages"]
                )
                extra = {"current_todo": current_todo} if not state.get("current_todo") else {}
                return {"messages": [response], **extra}

            async def execute_tools(state: AgentState) -> Dict:
                """执行 Manager 调用的所有工具（todo_manager + spawn_agents）。"""
                last_message = state["messages"][-1]
                updates: Dict = {"messages": []}

                for tool_call in getattr(last_message, "tool_calls", []):
                    name = tool_call["name"]
                    print(f"\n\033[33m[Manager 工具: {name}]\033[0m")

                    # todo_manager：同步更新任务状态
                    if name == "todo_manager":
                        items = tool_call["args"].get("items", [])
                        statuses = {it.get("status") for it in items}
                        updates["current_todo"] = items
                        updates["plan_needs_confirm"] = bool(items) and statuses == {"pending"}

                    tool_obj = tools_by_name.get(name)
                    if not tool_obj:
                        obs = f"Error: 工具 '{name}' 未注册。"
                    else:
                        try:
                            obs = await tool_obj.ainvoke(tool_call["args"])
                            if isinstance(obs, str) and len(obs) > 10000:
                                obs = obs[:10000] + "\n...(截断)"
                        except Exception as e:
                            obs = f"Error: {e}"

                    updates["messages"].append(
                        ToolMessage(content=str(obs), tool_call_id=tool_call["id"])
                    )
                return updates

            async def plan_confirm_node(state: AgentState) -> Dict:
                """展示任务规划，等待用户确认。"""
                todos = state.get("current_todo", [])
                icons = {"pending": "⏳", "in_progress": "🔄", "completed": "✅", "failed": "❌"}
                lines = [
                    f"  {icons.get(t.get('status', ''), '•')} "
                    f"#{t.get('id')} {t.get('text', t.get('description', ''))}"
                    for t in todos
                ]
                choice = interrupt(f"📋 任务规划：\n\n" + "\n".join(lines))
                result: Dict = {"plan_needs_confirm": False}
                if choice == "__feedback__":
                    feedback = interrupt("__feedback_input__")
                    if str(feedback).strip():
                        result["messages"] = [HumanMessage(
                            content=f"[规划反馈] {feedback}，请根据此反馈重新调整任务规划。"
                        )]
                elif choice == "abort":
                    result["messages"] = [HumanMessage(
                        content="用户已取消任务，请终止执行并告知用户。"
                    )]
                return result

            # ── 路由函数 ───────────────────────────────────────────────────
            def route_after_agent(state: AgentState) -> Literal["tools", "__end__"]:
                last = state["messages"][-1]
                return "tools" if getattr(last, "tool_calls", None) else "__end__"

            _router = make_token_router()

            def route_after_tools(state: AgentState) -> str:
                if state.get("plan_needs_confirm"):
                    return "plan_confirm"
                return _router(state)

            def route_after_plan_confirm(state: AgentState) -> str:
                msgs = state.get("messages", [])
                if msgs and isinstance(msgs[-1], HumanMessage) \
                        and "[规划反馈]" in str(msgs[-1].content):
                    return "agent"
                return _router(state)

            # ── 构建图 ─────────────────────────────────────────────────────
            _route_map = {"compress": "compress", "warn": "warn", "agent": "agent"}

            builder = StateGraph(AgentState)
            builder.add_node("agent",        call_manager_model)
            builder.add_node("tools",        execute_tools)
            builder.add_node("plan_confirm", plan_confirm_node)
            builder.add_node("warn",         create_warn_node())
            builder.add_node("compress",     create_compression_node(model))

            builder.add_edge(START, "agent")
            builder.add_conditional_edges("agent", route_after_agent,
                                          {"tools": "tools", "__end__": "__end__"})
            builder.add_conditional_edges("tools", route_after_tools,
                                          {**_route_map, "plan_confirm": "plan_confirm"})
            builder.add_conditional_edges("plan_confirm", route_after_plan_confirm,
                                          {**_route_map, "agent": "agent"})
            builder.add_conditional_edges("warn", route_after_warn,
                                          {"compress": "compress", "agent": "agent"})
            builder.add_edge("compress", "agent")

            app = builder.compile(checkpointer=checkpointer)

            # 保存图可视化
            try:
                img = app.get_graph().draw_mermaid_png()
                with open("manager_graph.png", "wb") as f:
                    f.write(img)
                print("\033[32m[OK] 主图已保存至 manager_graph.png\033[0m")
            except Exception:
                try:
                    with open("manager_graph.mmd", "w") as f:
                        f.write(app.get_graph().draw_mermaid())
                    print("\033[32m[OK] 主图 Mermaid 已保存至 manager_graph.mmd\033[0m")
                except Exception:
                    pass

            print("\033[32m[OK] Supervisor Agent 已就绪。\033[0m")
            session_config = {"configurable": {"thread_id": session_id}}

            # MySQL 会话恢复
            if _restored_from_mysql and memory_store is not None:
                await _restore_from_mysql(session_id, memory_store, app, session_config)

            # 检测未完成任务，询问是否自动继续
            _auto_input = None
            if pick.startswith("resume:") or _restored_from_mysql:
                try:
                    snap = await app.aget_state(session_config)
                    saved_todo: list = []
                    if snap and snap.values:
                        saved_todo = snap.values.get("current_todo") or []
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
                        _ico = {"in_progress": "🔄", "pending": "⏳"}
                        print("\n\033[33m[*] 检测到未完成任务：\033[0m")
                        for t in pending[:6]:
                            print(f"  {_ico.get(t.get('status', ''), '•')} "
                                  f"#{t.get('id')} {t.get('text', t.get('description', ''))}")
                        choice = await prompt_ui.select(
                            "是否自动继续执行？",
                            [("▶  是，继续执行", "continue"), ("✏️   否，手动输入", "manual")],
                        )
                        if choice == "continue":
                            _auto_input = "请继续完成未完成的任务，从当前进度继续，不需要重新规划。"
                except Exception as e:
                    print(f"\033[33m[!] 检测恢复状态失败: {e}\033[0m")

            # ── 交互主循环 ─────────────────────────────────────────────────
            while True:
                try:
                    print("=" * 60)
                    if _auto_input:
                        user_input = _auto_input
                        _auto_input = None
                        print(f"\033[33m[自动继续] >> {user_input}\033[0m")
                    else:
                        print("请输入需求 (q/exit 退出):")
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
                    async for chunk in app.astream(stream_input, session_config, stream_mode="updates"):
                        if "__interrupt__" in chunk:
                            for intr in chunk["__interrupt__"]:
                                raw = await _handle_interrupt(str(intr.value))
                                stream_input = Command(resume=raw)
                                interrupted = True
                            break
                        for node_name, node_update in chunk.items():
                            if node_update and "messages" in node_update:
                                for msg in node_update["messages"]:
                                    print(f"\n--- [{node_name}] ---")
                                    msg.pretty_print()
                    if not interrupted:
                        break

            # ── 退出时提炼记忆 + 归档 ──────────────────────────────────────
            _MIN_MSGS = 6
            try:
                snap = await app.aget_state(session_config)
                msgs = snap.values.get("messages", [])
                if len(msgs) < _MIN_MSGS:
                    print(f"\n[*] 消息不足 {_MIN_MSGS} 条，跳过记忆提炼。")
                else:
                    print("\n[*] 正在提炼长期记忆...")
                    new_mems = await ltm.extract_and_save(model, msgs)
                    if new_mems:
                        print(f"[OK] 保存了 {len(new_mems)} 条新记忆。")
                    else:
                        print("[*] 无需提炼新记忆。")
            except Exception as e:
                print(f"[!] 记忆提炼出错: {e}")

    finally:
        if memory_store is not None:
            print("\n[*] 正在归档会话到 MySQL...")
            await _archive_to_mysql(session_id, checkpointer, memory_store, model=model)
            await memory_store.close()
            print("\033[32m[OK] MySQL 连接已关闭。\033[0m")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
