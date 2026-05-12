from langchain.chat_models import init_chat_model

def get_model():
    return init_chat_model(
        base_url="http://10.129.107.145:8001/v1",
        api_key="vllm-no-key",
        model="Qwen_agent",
        model_provider="openai",
    )
