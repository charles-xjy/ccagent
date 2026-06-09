import asyncio
import re
import sys
import os
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# 直接运行时确保父包可寻址
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
import redis.asyncio as redis
from memory.long_term_memory import LongTermMemory, MEMORY_TYPES, _TYPE_LABELS
from core import prompt_ui
from core.config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
from core.memory_store import MemoryStore

DB_URI = "redis://10.129.107.145:6379"

# session_<项目名> 格式，提取 base session ID（去除 _coder/_tech-researcher/_reviewer 后缀）
_ROLE_SUFFIX_RE = re.compile(r'(_coder|_tech-researcher|_reviewer)$')
_SESSION_RE = re.compile(r'(session_[^:]+)')

# 旧格式 thread ID 前缀
_LEGACY_BASE = "manager_executor_v2"
_LEGACY_THREAD_IDS = [
    _LEGACY_BASE,
    f"{_LEGACY_BASE}_coder",
    f"{_LEGACY_BASE}_tech-researcher",
    f"{_LEGACY_BASE}_reviewer",
]

_ROLE_SUFFIXES = [
    ("",                  "主 Agent (Manager)"),
    ("_coder",            "编程专家 (Coder)"),
    ("_tech-researcher",  "调研专家 (Researcher)"),
    ("_reviewer",         "代码审核专家 (Reviewer)"),
]


async def _scan_sessions(r) -> tuple:
    """
    扫描 Redis，返回 (base_sessions, legacy_ids)：
      base_sessions — session_<项目名>_YYYYMMDD_HHMM 基础 ID 列表，倒序（最新在前）
      legacy_ids    — 旧格式 thread ID 列表（仅返回 Redis 中实际存在的）
    """
    all_keys = await r.keys("*")

    # 新格式：提取 session id，剥离 agent 角色后缀得到 base ID
    base_ids = set()
    for k in all_keys:
        m = _SESSION_RE.search(k)
        if m:
            base_ids.add(_ROLE_SUFFIX_RE.sub("", m.group(1)))

    # 旧格式：检查 Redis 里是否真的有对应 key
    legacy = [tid for tid in _LEGACY_THREAD_IDS
              if any(tid in k for k in all_keys)]

    return sorted(base_ids, reverse=True), legacy


# ─────────────────────────────────────────────────────────────
# 内部工具函数
# ─────────────────────────────────────────────────────────────


async def _collect_checkpoints(checkpointer: AsyncRedisSaver, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    checkpoints = []
    try:
        async for cp in checkpointer.alist(config):
            checkpoints.append(cp)
    except Exception as e:
        print(f"  [警告] 列出检查点时出错: {e}")
    return checkpoints


def _get_messages_from_checkpoint(checkpoint):
    try:
        if isinstance(checkpoint, dict):
            channel_values = checkpoint.get("channel_values", {})
        else:
            channel_values = getattr(checkpoint, "channel_values", {})
        if isinstance(channel_values, dict):
            return channel_values.get("messages", [])
        return getattr(channel_values, "messages", [])
    except Exception:
        return []


async def _show_messages(messages, start_label: int = 1):
    if not messages:
        print("消息列表为空。")
        return
    for i, msg in enumerate(messages, start=start_label):
        content = msg.content if hasattr(msg, "content") else msg.get("content", "")
        if isinstance(content, list):
            content_str = json.dumps(content, ensure_ascii=False, indent=2)
        else:
            content_str = str(content)

        msg_type = getattr(msg, "type", "unknown")
        extra_info = ""
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            extra_info += (
                f" [工具调用: {', '.join(tc.get('name','?') for tc in msg.tool_calls)}]"
            )
        if hasattr(msg, "name") and msg.name:
            extra_info += f" [工具名: {msg.name}]"

        print(f"\n{'─' * 60}")
        print(f"#{i} [{msg_type.upper()}]{extra_info}")
        print(f"{'─' * 60}")
        print(content_str)

    print(f"\n{'─' * 60}")
    print(f"(共 {len(messages)} 条消息)")


async def _confirm(prompt_text: str) -> bool:
    return await prompt_ui.select(prompt_text, [("确认", "y"), ("取消", "n")]) == "y"


# ─────────────────────────────────────────────────────────────
# 查看记忆
# ─────────────────────────────────────────────────────────────


async def view_memory(checkpointer: AsyncRedisSaver, thread_id: str):
    checkpoints = await _collect_checkpoints(checkpointer, thread_id)
    if not checkpoints:
        print(f"\n[!] 线程 '{thread_id}' 没有找到任何记忆。")
        return

    messages = []
    for cp in checkpoints:
        msgs = _get_messages_from_checkpoint(cp.checkpoint)
        if msgs:
            messages = msgs
            break

    total = len(messages)
    print(f"\n=== 线程 '{thread_id}' ===")
    print(f"检查点数量: {len(checkpoints)}  |  最新有效检查点消息数: {total}")

    while True:
        choice = await prompt_ui.select(
            "查看方式：",
            [
                (f"查看全部消息 ({total} 条)", "1"),
                ("查看最后 N 条", "2"),
                ("查看前 N 条", "3"),
                ("查看指定范围 (第 X~Y 条)", "4"),
                ("选择某个检查点查看", "5"),
                ("返回", "q"),
            ],
        )

        if choice == "q":
            break

        elif choice == "1":
            if total == 0:
                print("所有检查点中都没有消息。")
            else:
                await _show_messages(messages)

        elif choice == "2":
            try:
                n = int(input(f"显示最后几条 (1-{total}): ").strip())
                if 1 <= n <= total:
                    await _show_messages(messages[-n:], start_label=total - n + 1)
                else:
                    print(f"请输入 1~{total} 之间的数字。")
            except ValueError:
                print("请输入有效数字。")

        elif choice == "3":
            try:
                n = int(input(f"显示前几条 (1-{total}): ").strip())
                if 1 <= n <= total:
                    await _show_messages(messages[:n])
                else:
                    print(f"请输入 1~{total} 之间的数字。")
            except ValueError:
                print("请输入有效数字。")

        elif choice == "4":
            try:
                x = int(input(f"起始 (1-{total}): ").strip())
                y = int(input(f"结束 ({x}-{total}): ").strip())
                if 1 <= x <= y <= total:
                    await _show_messages(messages[x - 1 : y], start_label=x)
                else:
                    print(f"请确保 1 ≤ 起始 ≤ 结束 ≤ {total}。")
            except ValueError:
                print("请输入有效数字。")

        elif choice == "5":
            cp_choices = []
            for idx, cp in enumerate(checkpoints):
                cp_id = cp.config.get("configurable", {}).get(
                    "checkpoint_id", f"cp_{idx}"
                )
                short_id = str(cp_id)[:16]
                n_msgs = len(_get_messages_from_checkpoint(cp.checkpoint))
                cp_choices.append((f"[{idx}] {short_id}…  消息数={n_msgs}", str(idx)))
            cp_choices.append(("取消", "q"))

            sel = await prompt_ui.select("选择检查点：", cp_choices)
            if sel == "q":
                continue
            cp = checkpoints[int(sel)]
            cp_messages = _get_messages_from_checkpoint(cp.checkpoint)
            cp_id = cp.config.get("configurable", {}).get("checkpoint_id", "?")
            print(f"\n=== 检查点 {cp_id} (索引 {sel}) ===")
            await _show_messages(cp_messages)


# ─────────────────────────────────────────────────────────────
# 删除检查点
# ─────────────────────────────────────────────────────────────


async def delete_checkpoints(checkpointer: AsyncRedisSaver, thread_id: str):
    checkpoints = await _collect_checkpoints(checkpointer, thread_id)
    if not checkpoints:
        print(f"\n[!] 线程 '{thread_id}' 没有找到任何检查点。")
        return

    while True:
        choice = await prompt_ui.select(
            f"删除检查点 (共 {len(checkpoints)} 个)：",
            [
                ("选择删除指定检查点", "1"),
                ("仅保留最近 N 个（删除其余）", "2"),
                ("删除全部", "3"),
                ("返回", "q"),
            ],
        )

        if choice == "q":
            break

        elif choice == "1":
            cp_choices = []
            for idx, cp in enumerate(checkpoints):
                cp_id = cp.config.get("configurable", {}).get(
                    "checkpoint_id", f"cp_{idx}"
                )
                short_id = str(cp_id)[:16]
                n_msgs = len(_get_messages_from_checkpoint(cp.checkpoint))
                cp_choices.append((f"[{idx}] {short_id}…  消息数={n_msgs}", str(idx)))
            cp_choices.append(("取消", "q"))

            sel = await prompt_ui.select("选择要删除的检查点：", cp_choices)
            if sel == "q":
                continue

            target = checkpoints[int(sel)]
            if await _confirm(f"确认删除检查点 [{sel}]？"):
                deleted = await _delete_checkpoint_list([target])
                print(f"[OK] 已删除 1 个检查点（清理 {deleted} 个 Redis 键）。")
                checkpoints = await _collect_checkpoints(checkpointer, thread_id)

        elif choice == "2":
            max_keep = len(checkpoints) - 1
            if max_keep < 1:
                print("检查点数量不足，无需操作。")
                continue
            try:
                raw_n = input(f"保留最近几个 (1-{max_keep}，q 取消): ").strip()
                if raw_n.lower() == "q":
                    continue
                n = int(raw_n)
                if not 1 <= n <= max_keep:
                    print(f"请输入 1~{max_keep} 之间的数字。")
                    continue
                targets = checkpoints[n:]
                if await _confirm(
                    f"将删除 {len(targets)} 个旧检查点，保留最近 {n} 个，确认？"
                ):
                    deleted = await _delete_checkpoint_list(targets)
                    print(
                        f"[OK] 已删除 {len(targets)} 个旧检查点（清理 {deleted} 个 Redis 键）。"
                    )
                    checkpoints = await _collect_checkpoints(checkpointer, thread_id)
            except ValueError:
                print("请输入有效数字。")

        elif choice == "3":
            if await _confirm(f"确定删除 '{thread_id}' 的全部检查点？"):
                await delete_memory(checkpointer, thread_id)
                break


async def _delete_checkpoint_list(checkpoints: list) -> int:
    r = redis.from_url(DB_URI)
    total = 0
    for cp in checkpoints:
        cp_id = cp.config.get("configurable", {}).get("checkpoint_id", "")
        if not cp_id:
            continue
        keys = await r.keys(f"*{cp_id}*")
        if keys:
            await r.delete(*keys)
            total += len(keys)
    await r.aclose()
    return total


async def delete_memory(checkpointer: AsyncRedisSaver, thread_id: str):
    deleted_via_api = False
    try:
        if hasattr(checkpointer, "adelete_thread"):
            await checkpointer.adelete_thread(thread_id)
            deleted_via_api = True
        elif hasattr(checkpointer, "delete_thread"):
            fn = checkpointer.delete_thread
            await fn(thread_id) if asyncio.iscoroutinefunction(fn) else fn(thread_id)
            deleted_via_api = True
    except NotImplementedError:
        pass
    except Exception as e:
        print(f"调用标准 API 删除出错: {e}")

    try:
        r = redis.from_url(DB_URI)
        keys = await r.keys(f"*{thread_id}*")
        if keys:
            await r.delete(*keys)
            print(f"\n[OK] 已清理 Redis 中 '{thread_id}' 相关的 {len(keys)} 个键。")
        elif deleted_via_api:
            print(f"\n[OK] 已通过 API 删除线程 '{thread_id}'。")
        else:
            print(f"\n[!] 未找到 '{thread_id}' 相关数据。")
        await r.aclose()
    except Exception as e:
        print(f"\n[Error] 手动清理 Redis 键失败: {e}")


# ─────────────────────────────────────────────────────────────
# 长期记忆管理
# ─────────────────────────────────────────────────────────────


async def manage_long_term_memories(ltm: LongTermMemory):
    while True:
        memories = await ltm.load_all()

        choice = await prompt_ui.select(
            f"长期记忆管理 (共 {len(memories)} 条)：",
            [
                ("查看全部记忆", "1"),
                ("删除指定记忆", "2"),
                ("清空全部", "3"),
                ("返回", "q"),
            ],
        )

        if choice == "q":
            break

        elif choice == "1":
            if not memories:
                print("暂无长期记忆。")
                continue
            by_type = {}
            for m in memories:
                by_type.setdefault(m["type"], []).append(m)
            for t in MEMORY_TYPES:
                if t not in by_type:
                    continue
                print(f"\n【{_TYPE_LABELS[t]}】")
                for m in by_type[t]:
                    print(f"  ID={m['id']}  name={m['name']}")
                    print(f"  描述: {m['description']}")
                    print(f"  内容: {m['content']}")
                    print(f"  创建: {m['created_at']}")
                    print()

        elif choice == "2":
            if not memories:
                print("暂无记忆可删除。")
                continue
            mem_choices = [
                (f"[{m['type']}] {m['name']} — {m['description']}", str(i))
                for i, m in enumerate(memories)
            ]
            mem_choices.append(("取消", "q"))
            sel = await prompt_ui.select("选择要删除的记忆：", mem_choices)
            if sel == "q":
                continue
            m = memories[int(sel)]
            if await _confirm(f"确认删除 '{m['name']}'？"):
                await ltm.delete(m["id"])
                print(f"[OK] 已删除: {m['name']}")

        elif choice == "3":
            if await _confirm("确认清空全部长期记忆？"):
                await ltm.delete_all()
                print("[OK] 已清空全部长期记忆。")


# ─────────────────────────────────────────────────────────────
# MySQL 归档管理
# ─────────────────────────────────────────────────────────────


async def view_mysql_archives(memory_store: MemoryStore):
    """查看 MySQL 中归档的完整对话历史。"""
    while True:
        threads = await memory_store.list_threads()
        if not threads:
            print("\n[!] MySQL 中没有归档的会话。")
            input("按回车返回...")
            return

        choices = []
        for t in threads:
            label = (
                f"{t['thread_id']}  —  {t['message_count']} 条消息"
                f"  ({t['updated_at'].strftime('%Y-%m-%d %H:%M') if hasattr(t['updated_at'], 'strftime') else str(t['updated_at'])})"
            )
            choices.append((label, f"view:{t['thread_id']}"))
        choices += [("删除某个归档", "__delete__"), ("返回", "q")]

        choice = await prompt_ui.select(
            f"MySQL 归档列表 (共 {len(threads)} 个会话)：",
            choices,
        )

        if choice == "q":
            return

        if choice == "__delete__":
            del_choices = [
                (f"{t['thread_id']} ({t['message_count']} 条消息)", f"del:{t['thread_id']}")
                for t in threads
            ]
            del_choices.append(("取消", "q"))
            sel = await prompt_ui.select("选择要删除的归档：", del_choices)
            if sel == "q":
                continue
            tid = sel[len("del:"):]
            if await _confirm(f"确认删除归档 '{tid}'？此操作不可恢复。"):
                await memory_store.delete(tid)
                print(f"[OK] 已删除归档: {tid}")
            continue

        # 查看某个归档的完整消息
        tid = choice[len("view:"):]
        messages = await memory_store.load(tid)
        if not messages:
            print(f"\n[!] 无法加载 '{tid}' 的消息。")
            continue

        print(f"\n=== MySQL 归档: {tid} ===")
        print(f"消息总数: {len(messages)}")

        while True:
            total = len(messages)
            view_choice = await prompt_ui.select(
                "查看方式：",
                [
                    (f"查看全部消息 ({total} 条)", "1"),
                    ("查看最后 N 条", "2"),
                    ("查看前 N 条", "3"),
                    ("查看指定范围 (第 X~Y 条)", "4"),
                    ("返回", "q"),
                ],
            )

            if view_choice == "q":
                break
            elif view_choice == "1":
                await _show_messages(messages)
            elif view_choice == "2":
                try:
                    n = int(input(f"显示最后几条 (1-{total}): ").strip())
                    if 1 <= n <= total:
                        await _show_messages(messages[-n:], start_label=total - n + 1)
                except ValueError:
                    print("请输入有效数字。")
            elif view_choice == "3":
                try:
                    n = int(input(f"显示前几条 (1-{total}): ").strip())
                    if 1 <= n <= total:
                        await _show_messages(messages[:n])
                except ValueError:
                    print("请输入有效数字。")
            elif view_choice == "4":
                try:
                    x = int(input(f"起始 (1-{total}): ").strip())
                    y = int(input(f"结束 ({x}-{total}): ").strip())
                    if 1 <= x <= y <= total:
                        await _show_messages(messages[x - 1 : y], start_label=x)
                except ValueError:
                    print("请输入有效数字。")


# ─────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────


async def main():
    # 初始化 MySQL（可选，失败降级）
    memory_store = None
    try:
        memory_store = await MemoryStore.create(
            host=MYSQL_HOST, port=MYSQL_PORT,
            user=MYSQL_USER, password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
        )
        print("[OK] MySQL 归档已连接。\n")
    except Exception as e:
        print(f"[!] MySQL 不可用 ({e})，归档管理功能将跳过。\n")

    async with AsyncRedisSaver.from_conn_string(DB_URI) as checkpointer, \
               LongTermMemory(DB_URI) as ltm:
        while True:
            r_scan = redis.from_url(DB_URI, decode_responses=True)
            base_sessions, legacy_ids = await _scan_sessions(r_scan)
            await r_scan.aclose()

            total = len(base_sessions) + len(legacy_ids)
            session_choices = [(f"{sid}", f"new:{sid}") for sid in base_sessions]
            if legacy_ids:
                session_choices += [(f"[旧] {tid}", f"legacy:{tid}") for tid in legacy_ids]
            session_choices += [("长期记忆管理", "__ltm__")]
            if memory_store is not None:
                session_choices.append(("MySQL 归档管理", "__mysql__"))
            session_choices.append(("退出", "q"))

            choice = await prompt_ui.select(
                f"记忆管理器 — 共 {total} 条会话记录：",
                session_choices,
            )

            if choice == "q":
                break

            if choice == "__ltm__":
                await manage_long_term_memories(ltm)
                continue

            if choice == "__mysql__":
                if memory_store is not None:
                    await view_mysql_archives(memory_store)
                continue

            if choice.startswith("legacy:"):
                tid = choice[len("legacy:"):]
                action = await prompt_ui.select(
                    f"[旧] {tid}：",
                    [("查看记忆", "1"), ("删除记忆", "2"), ("返回", "q")],
                )
                if action == "1":
                    await view_memory(checkpointer, tid)
                elif action == "2":
                    await delete_checkpoints(checkpointer, tid)
                continue

            # 新格式：展示子 Agent 列表
            base_sid = choice[len("new:"):]
            agent_choices = [
                (f"{label}  ({base_sid}{suffix})", f"{base_sid}{suffix}")
                for suffix, label in _ROLE_SUFFIXES
            ]
            agent_choices.append(("返回", "q"))

            agent_choice = await prompt_ui.select(
                f"会话 {base_sid} — 选择 Agent：", agent_choices
            )
            if agent_choice == "q":
                continue

            action = await prompt_ui.select(
                f"{agent_choice}：",
                [("查看记忆", "1"), ("删除记忆", "2"), ("返回", "q")],
            )
            if action == "1":
                await view_memory(checkpointer, agent_choice)
            elif action == "2":
                await delete_checkpoints(checkpointer, agent_choice)

    if memory_store is not None:
        await memory_store.close()
        print("[OK] MySQL 连接已关闭。")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
