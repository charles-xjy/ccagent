"""
Model Router - 模型路由器
路由到合适的 LLM 模型
"""

from typing import Optional, Dict, Any
from enum import Enum
from langgraph.graph import StateGraph
from pydantic import BaseModel


class ModelType(str, Enum):
    """模型类型枚举"""
    ERNIE_BOT = "ernie_bot"
    ERNIE_BOT_4 = "ernie_bot_4"
    ERNIE_BOT_VLARGE = "ernie_bot_vlarge"
    CUSTOM = "custom"


class ModelRouter:
    """模型路由器"""
    
    def __init__(self, models: Optional[Dict[str, Any]] = None):
        self.models = models or {}
        self.default_model = self._get_default_model()
    
    def _get_default_model(self) -> ModelType:
        """获取默认模型"""
        return ModelType.ERNIE_BOT
    
    def route(self, request: Dict[str, Any]) -> str:
        """路由到合适的模型"""
        model_key = request.get("model")
        
        if model_key:
            return model_key.upper()
        
        return self.default_model.value
    
    def validate(self, request: Dict[str, Any]) -> bool:
        """验证请求"""
        return bool(request.get("api_key") and request.get("system_prompt"))
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """获取模型配置"""
        return {
            "model_name": model_name,
            "temperature": request.get("temperature", 0.7),
            "max_tokens": request.get("max_tokens", 2048)
        }
