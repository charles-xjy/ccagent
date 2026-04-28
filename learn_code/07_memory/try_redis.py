import asyncio
from dataclasses import dataclass
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.store.redis.aio import AsyncRedisStore
from langgraph.runtime import Runtime
import uuid


async def main():
    model = init_chat_model(
        base_url="http://localhost:8001/v1",
        api_key="vllm-no-key",
        model="Qwen_agent",
        model_provider="openai",
    )

    @dataclass
    class Context:
        user_id: str

    async def call_model(
            state: MessagesState,
            runtime: Runtime[Context],
    ):
        user_id = runtime.context.user_id
        namespace = ("memories", user_id)
        memories = await runtime.store.asearch(namespace, query=str(state["messages"][-1].content))
        info = "\n".join([d.value["data"] for d in memories])
        system_msg = f"You are a helpful assistant talking to the user. User info: {info}"

        # Store new memories if the user asks the model to remember
        last_message = state["messages"][-1]
        if "remember" in last_message.content.lower():
            memory = "User name is Bob"
            await runtime.store.aput(namespace, str(uuid.uuid4()), {"data": memory})

        response = await model.ainvoke(
            [{"role": "system", "content": system_msg}] + state["messages"]
        )
        return {"messages": response}

    DB_URI = "redis://localhost:6379"

    async with (
        AsyncRedisStore.from_conn_string(DB_URI) as store,
        AsyncRedisSaver.from_conn_string(DB_URI) as checkpointer,
    ):
        await checkpointer.asetup()
        await store.setup()

        builder = StateGraph(MessagesState, context_schema=Context)
        builder.add_node(call_model)
        builder.add_edge(START, "call_model")

        graph = builder.compile(
            checkpointer=checkpointer,
            store=store,
        )

        config = {"configurable": {"thread_id": "1"}}
        async for chunk in graph.astream(
                {"messages": [{"role": "user", "content": "Hi! Remember: my name is Bob"}]},
                config,
                stream_mode="values",
                context=Context(user_id="1"),
        ):
            chunk["messages"][-1].pretty_print()

        config = {"configurable": {"thread_id": "2"}}
        async for chunk in graph.astream(
                {"messages": [{"role": "user", "content": "what is my name?"}]},
                config,
                stream_mode="values",
                context=Context(user_id="1"),
        ):
            chunk["messages"][-1].pretty_print()


from redis.asyncio import Redis


async def print_redis_raw_data():
    # 直接连接 Redis 客户端
    r = Redis.from_url("redis://localhost:6379", decode_responses=True)

    print("\n--- [Redis] 原始键值对列表 ---")
    keys = await r.keys("*")

    for key in keys:
        key_type = await r.type(key)
        print(f"Key: {key} ({key_type})")

        # 如果是 Hash 类型（LangGraph 常用的存储方式）
        if key_type == "hash":
            data = await r.hgetall(key)
            # 注意：这里的数据通常是序列化后的二进制，打印出来可能不直观
            print(f"  Value (Hash Keys): {list(data.keys())}")
        elif key_type == "string":
            val = await r.get(key)
            print(f"  Value: {val[:50]}...")  # 只打印前50个字符防止刷屏

    await r.aclose()


async def dump_all_memories():
    DB_URI = "redis://localhost:6379"
    async with AsyncRedisStore.from_conn_string(DB_URI) as store:
        # 搜索所有用户 (user_id="1") 的记忆
        # ("memories",) 是根命名空间
        memories = await store.asearch(("memories",))

        print(f"\n{'=' * 20} 长期存储内容 {'=' * 20}")
        if not memories:
            print("当前数据库中没有存入任何记忆。")
        for m in memories:
            print(f"命名空间: {m.namespace}")
            print(f"条目 ID: {m.key}")
            print(f"数据内容: {m.value}")
            print("-" * 40)


if __name__ == "__main__":
    # asyncio.run(main())
    asyncio.run(print_redis_raw_data())
    asyncio.run(dump_all_memories())
