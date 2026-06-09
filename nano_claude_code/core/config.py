from langchain.chat_models import init_chat_model

def get_model():
    return init_chat_model(
        base_url="http://10.129.107.145:8001/v1",
        api_key="vllm-no-key",
        model="Qwen_agent",
        model_provider="openai",
    )


# Redis — hot storage for active LangGraph checkpoints
REDIS_URI = "redis://10.129.107.145:6379"
REDIS_TTL = 604800  # 秒，传给 AsyncRedisSaver 的 ttl={"default_ttl": REDIS_TTL}

# MySQL — cold storage for archived conversation history
MYSQL_HOST = "10.129.107.145"
MYSQL_PORT = 3306
MYSQL_USER = "ccagent"
MYSQL_PASSWORD = "ccagent123"
MYSQL_DATABASE = "ccagent"
