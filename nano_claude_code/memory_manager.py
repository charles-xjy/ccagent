import asyncio
import json
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
import redis.asyncio as redis

DB_URI = "redis://10.129.107.145:6379"

THREADS = {
    "1": ("主 Agent (Manager)", "manager_executor_v2"),
    "2": ("编程专家 (Coder)", "manager_executor_v2_coder"),
    "3": ("调研专家 (Researcher)", "manager_executor_v2_tech-researcher"),
    "4": ("代码审核专家 (Reviewer)", "manager_executor_v2_reviewer"),
}


async def _collect_checkpoints(checkpointer: AsyncRedisSaver, thread_id: str):
    """收集线程的所有检查点"""
    config = {"configurable": {"thread_id": thread_id}}
    checkpoints = []
    try:
        async for cp in checkpointer.alist(config):
            checkpoints.append(cp)
    except Exception as e:
        print(f"  [警告] 列出检查点时出错: {e}")
    return checkpoints


def _get_messages_from_checkpoint(checkpoint):
    """从 checkpoint 对象中提取消息列表（兼容 dict 和 dataclass）。"""
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
    """完整打印消息列表，不截断内容"""
    if not messages:
        print("消息列表为空。")
        return
    for i, msg in enumerate(messages, start=start_label):
        content = msg.content if hasattr(msg, "content") else msg.get("content", "")
        if isinstance(content, list):
            content_str = json.dumps(content, ensure_ascii=False, indent=2)
        elif isinstance(content, str):
            content_str = content
        else:
            content_str = str(content)

        msg_type = getattr(msg, "type", "unknown")
        extra_info = ""
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_names = [tc.get("name", "?") for tc in msg.tool_calls]
            extra_info += f" [工具调用: {', '.join(tool_names)}]"
        if hasattr(msg, "name") and msg.name:
            extra_info += f" [工具名: {msg.name}]"

        print(f"\n{'─' * 60}")
        print(f"#{i} [{msg_type.upper()}]{extra_info}")
        print(f"{'─' * 60}")
        print(content_str)

    print(f"\n{'─' * 60}")
    print(f"(共 {len(messages)} 条消息)")


async def view_memory(checkpointer: AsyncRedisSaver, thread_id: str):
    checkpoints = await _collect_checkpoints(checkpointer, thread_id)
    if not checkpoints:
        print(f"\n[!] 线程 '{thread_id}' 没有找到任何记忆。")
        return

    # 从最新往前找，找到第一个包含消息的检查点
    messages = []
    for cp in reversed(checkpoints):
        msgs = _get_messages_from_checkpoint(cp.checkpoint)
        if msgs:
            messages = msgs
            break

    total_messages = len(messages)
    print(f"\n=== 线程 '{thread_id}' 的记忆 ===")
    print(f"总检查点(状态快照)数量: {len(checkpoints)}")
    print(f"最近有效检查点中的消息总数: {total_messages}")

    while True:
        print(f"\n--- 查看选项 ---")
        print("1. 查看全部消息 (完整不截断)")
        print("2. 查看最后 N 条消息")
        print("3. 查看前 N 条消息")
        print("4. 查看指定范围 (第 X 到第 Y 条)")
        print("5. 选择一个检查点查看其消息")
        print("0. 返回上一级")

        choice = input("\n请选择 (0-5): ").strip()

        if choice == "0":
            break
        elif choice == "1":
            if total_messages == 0:
                print("所有检查点中都没有消息。")
            else:
                await _show_messages(messages, start_label=1)
        elif choice == "2":
            try:
                n = int(input(f"显示最后几条? (1-{total_messages}): ").strip())
                if 1 <= n <= total_messages:
                    await _show_messages(messages[-n:], start_label=total_messages - n + 1)
                else:
                    print(f"请输入 1 到 {total_messages} 之间的数字。")
            except ValueError:
                print("请输入有效数字。")
        elif choice == "3":
            try:
                n = int(input(f"显示前几条? (1-{total_messages}): ").strip())
                if 1 <= n <= total_messages:
                    await _show_messages(messages[:n], start_label=1)
                else:
                    print(f"请输入 1 到 {total_messages} 之间的数字。")
            except ValueError:
                print("请输入有效数字。")
        elif choice == "4":
            try:
                x = int(input(f"起始索引 (1-{total_messages}): ").strip())
                y = int(input(f"结束索引 ({x}-{total_messages}): ").strip())
                if 1 <= x <= y <= total_messages:
                    await _show_messages(messages[x - 1 : y], start_label=x)
                else:
                    print(f"请确保 1 <= 起始 <= 结束 <= {total_messages}。")
            except ValueError:
                print("请输入有效数字。")
        elif choice == "5":
            print(f"\n--- 可用检查点 (共 {len(checkpoints)}) ---")
            for idx, cp in enumerate(checkpoints):
                cp_id = cp.config.get("configurable", {}).get("checkpoint_id", f"cp_{idx}")
                short_id = cp_id[:16] if isinstance(cp_id, str) else str(cp_id)[:16]
                cp_messages = _get_messages_from_checkpoint(cp.checkpoint)
                print(f"  {idx}. checkpoint_id={short_id}...  消息数={len(cp_messages)}")

            try:
                cp_idx = int(input(f"\n选择检查点索引 (0-{len(checkpoints)-1}): ").strip())
                if 0 <= cp_idx < len(checkpoints):
                    cp = checkpoints[cp_idx]
                    cp_messages = _get_messages_from_checkpoint(cp.checkpoint)
                    cp_id = cp.config.get("configurable", {}).get("checkpoint_id", "?")
                    print(f"\n=== 检查点 {cp_id} (索引 {cp_idx}) ===")
                    print(f"消息数: {len(cp_messages)}")
                    await _show_messages(cp_messages, start_label=1)
                else:
                    print(f"请输入 0 到 {len(checkpoints)-1} 之间的数字。")
            except ValueError:
                print("请输入有效数字。")
        else:
            print("无效选项。")


async def delete_memory(checkpointer: AsyncRedisSaver, thread_id: str):
    deleted_via_api = False
    try:
        if hasattr(checkpointer, "adelete_thread"):
            await checkpointer.adelete_thread(thread_id)
            deleted_via_api = True
        elif hasattr(checkpointer, "delete_thread"):
            if asyncio.iscoroutinefunction(checkpointer.delete_thread):
                await checkpointer.delete_thread(thread_id)
            else:
                checkpointer.delete_thread(thread_id)
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
            print(f"\n[OK] 成功清理 Redis 中与线程 '{thread_id}' 相关的 {len(keys)} 个底层键。")
        elif deleted_via_api:
            print(f"\n[OK] 成功通过 API 删除线程 '{thread_id}' 的记忆。")
        else:
            print(f"\n[!] 没有在 Redis 中找到 '{thread_id}' 相关的记忆数据。")
        await r.aclose()
    except Exception as e:
        print(f"\n[Error] 手动清理 Redis Keys 失败: {e}")


async def main():
    async with AsyncRedisSaver.from_conn_string(DB_URI) as checkpointer:
        while True:
            print("\n================ 记忆管理器 ================")
            for key, (name, tid) in THREADS.items():
                print(f"{key}. {name} (ID: {tid})")
            print("q. 退出")
            print("============================================")

            choice = input("请选择要操作的 Agent (1-4) 或输入 q 退出: ").strip().lower()
            if choice == "q":
                break

            if choice not in THREADS:
                print("无效的选项，请重新输入。")
                continue

            name, tid = THREADS[choice]
            print(f"\n当前选中: {name}")
            print("1. 查看记忆")
            print("2. 删除记忆")

            action = input("请选择操作 (1/2): ").strip()
            if action == "1":
                await view_memory(checkpointer, tid)
            elif action == "2":
                confirm = input(f"确定要删除 {name} 的所有记忆吗？(y/n): ").strip().lower()
                if confirm == "y":
                    await delete_memory(checkpointer, tid)
            else:
                print("无效的操作。")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
