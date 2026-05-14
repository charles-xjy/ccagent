"""
LLM Wrapper - LLM 封装
提供统一的 LLM 接口封装
"""

from typing import Optional, Any, Dict, List
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
import httpx


class LLMResponse(BaseModel):
    """LLM 响应"""
    content: str
    model: str
    usage: Dict[str, int] = Field(default_factory=dict)


class LLMWrapper(ABC):
    """LLM 封装基类"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
    
    @abstractmethod
    async def invoke(self, prompts: List[str]) -> LLMResponse:
        """调用 LLM"""
        pass
    
    async def chat(self, message: str) -> LLMResponse:
        """聊天"""
        return await self.invoke([message])
    
    async def stream_chat(self, message: str) -> Any:
        """流式聊天"""
        # 实现流式响应
        pass


class BaiduLLMWrapper(LLMWrapper):
    """百度 LLM 封装"""
    
    def __init__(self, api_key: str, model: str = "ERNIE-Bot-4.0"):
        super().__init__(model_name=model, api_key=api_key)
        self.base_url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkbench"
    
    async def invoke(self, prompts: List[str]) -> LLMResponse:
        """调用百度 LLM"""
        response = await httpx.get(
            self.base_url,
            params={
                "bar": str(len(prompts)),
                "scene": 2,
                "prompt": ",".join(prompts),
                "api_key": self.api_key
            }
        )
        
        return LLMResponse(
            content=response.text,
            model=self.model_name,
            usage={"prompt_tokens": 0, "completion_tokens": 0}
        )
