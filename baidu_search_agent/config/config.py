"""
配置管理器模块
统一模型配置、环境变量管理、API 路由
"""
import os
from typing import Optional
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod

# ==================== 模型配置类 ====================
class ModelConfig(BaseModel):
    """模型配置基类"""
    api_base: str = Field(default="https://api.deepseek.com", description="API 基础地址")
    api_key: str = Field(default="", description="API 密钥")
    model_name: str = Field(default="deepseek-chat", description="模型名称")
    platform: str = Field(default="deepseek", description="平台类型: deepseek|silicon")
    
    @classmethod
    def from_env(cls) -> "ModelConfig":
        """从环境变量加载配置"""
        platform = os.getenv("MODE_PLATFORM", "deepseek")
        model_name = os.getenv("MODEL_NAME", "deepseek-chat")
        
        if platform == "silicon":
            api_base = os.getenv("SILICON_BASE", "https://api.siliconflow.cn/v1")
            api_key = os.getenv("SILICON_API_KEY", "")
        else:
            api_base = os.getenv("MODEL_BASE", "https://api.deepseek.com")
            api_key = os.getenv("API_KEY", "")
            
        return cls(api_base=api_base, api_key=api_key, model_name=model_name, platform=platform)
    
    @property
    def headers(self) -> dict:
        """获取请求头"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def get_url(self, endpoint: str) -> str:
        """获取完整 URL"""
        return f"{self.api_base}/{endpoint}"

# ==================== LLM 封装基类 ====================
class LLMWrapper(ABC):
    """LLM 封装基类"""
    
    def __init__(self, base: str, key: str, name: str):
        self.api_base = base
        self.api_key = key
        self.model_name = name
    
    @abstractmethod
    def chat(self, messages: list[dict]) -> str:
        """聊天接口"""
        pass
    
    @abstractmethod
    def complete(self, prompt: str) -> str:
        """完成接口（单轮对话）"""
        pass

class DeepSeekWrapper(LLMWrapper):
    """DeepSeek 模型封装"""
    def __init__(self, base: str, key: str, name: str = "deepseek-chat"):
        super().__init__(base, key, name)
    
    def chat(self, messages: list[dict]) -> str:
        """DeepSeek 聊天接口"""
        url = self.api_base.strip("/")
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7
        }
        # API 调用逻辑...
        return ""
    
    def complete(self, prompt: str) -> str:
        """单轮完成"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages)

class SiliconFlowWrapper(LLMWrapper):
    """硅基流动模型封装"""
    def __init__(self, base: str, key: str, name: str = "deepseek-chat"):
        super().__init__(base, key, name)
    
    def chat(self, messages: list[dict]) -> str:
        """硅基流动聊天接口"""
        url = self.api_base.strip("/")
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7
        }
        # API 调用逻辑...
        return ""
    
    def complete(self, prompt: str) -> str:
        """单轮完成"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages)

def get_llm(config: ModelConfig) -> LLMWrapper:
    """根据配置获取对应的 LLM 实例"""
    if config.platform == "silicon":
        return SiliconFlowWrapper(config.api_base, config.api_key, config.model_name)
    else:
        return DeepSeekWrapper(config.api_base, config.api_key, config.model_name)