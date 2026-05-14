"""
Application Settings
应用配置
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """应用全局配置"""
    
    # Application settings
    app_name: str = "Baidu Search Agent"
    app_version: str = "1.0.0"
    
    # Baidu Search settings
    baidu_api_key: Optional[str] = None
    baidu_access_token: Optional[str] = None
    search_timeout: int = 30
    search_retries: int = 3
    max_search_results: int = 5
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[str] = "logs/app.log"
    
    # LLM settings (can be overridden by environment)
    llm_provider: str = "baidu"
    llm_api_key: Optional[str] = None
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048
    
    # MCP settings
    mcp_enabled: bool = True
    mcp_timeout: int = 30
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
