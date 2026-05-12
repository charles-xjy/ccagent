import asyncio
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
import redis.asyncio as redis

DB_URI = "redis://10.129.107.145:6379"

THREADS = {
    "1": ("主 Agent (Manager)", "manager_executor_v2"),
    "2": ("编程专家 (Coder)", "manager_executor_v2_coder"),
    "3": ("调研专家 (Researcher)", "manager_executor_v2_tech-researcher"),
}

async def view_memory(checkpointer: AsyncRedisSaver, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    
    # 获取最新的检查点
    latest = await checkpointer.aget_tuple(config)
    if not latest:
        print(f"\n[!] 线程 '{thread_id}' 没有找到任何记忆。")
        return

    # 获取所有检查点历史
    count = 0
    async for _ in checkpointer.alist(config):
        count += 1
        
    print(f"\n=== 线程 '{thread_id}' 的记忆 ===")
    print(f"总检查点(状态快照)数量: {count}")
    
    if latest.checkpoint and "channel_values" in latest.checkpoint:
        messages = latest.checkpoint["channel_values"].get("messages", [])
        if not messages:
            print("消息列表为空。")
            return
            
        print(f"\n--- 最近的消息上下文 (共 {len(messages)} 条) ---")
        # 仅显示最后 5 条消息避免刷屏
        for i, msg in enumerate(messages[-5:], start=max(1, len(messages)-4)):
            content = msg.content
            if isinstance(content, str) and len(content) > 200:
                content = content[:200] + "... (已截断)"
            elif not isinstance(content, str):
                # 可能是 list 形式的 tool_calls 等
                content = str(content)
            
            print(f"{i}. [{msg.type.upper()}] {content}\n")
            
        if len(messages) > 5:
            print(f"(仅显示最后 5 条消息，还有 {len(messages)-5} 条历史消息)")
    else:
        print("最新状态中没有消息数据。")

async def delete_memory(checkpointer: AsyncRedisSaver, thread_id: str):
    # LangGraph 中不同版本的 Saver 删除 API 不完全一致
    # 优先尝试标准 API
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

    # Fallback: 如果标准 API 不起作用或没抛出异常但没删干净，我们用 Redis 客户端直接清空对应的 Keys
    try:
        r = redis.from_url(DB_URI)
        # LangGraph RedisSaver 一般把 thread_id 拼在 key 中
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
            
            choice = input("请选择要操作的 Agent (1-3) 或输入 q 退出: ").strip().lower()
            if choice == 'q':
                break
            
            if choice not in THREADS:
                print("无效的选项，请重新输入。")
                continue
                
            name, tid = THREADS[choice]
            print(f"\n当前选中: {name}")
            print("1. 查看记忆")
            print("2. 删除记忆")
            
            action = input("请选择操作 (1/2): ").strip()
            if action == '1':
                await view_memory(checkpointer, tid)
            elif action == '2':
                confirm = input(f"确定要删除 {name} 的所有记忆吗？(y/n): ").strip().lower()
                if confirm == 'y':
                    await delete_memory(checkpointer, tid)
            else:
                print("无效的操作。")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
