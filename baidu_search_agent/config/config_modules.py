"""
配置模块（Modules）
提供配置工具函数和辅助类
"""
import os
from typing import Optional, Dict, Any
from urllib.parse import urlparse
import time

# ==================== 配置验证工具 ====================

def parse_api_base(api_base: str) -> str:
    """
    解析 API 基础 URL
    
    Args:
        api_base: 原始 API URL
        
    Returns:
        解析后的基础 URL（去除路径）
    """
    parsed = urlparse(api_base)
    return f"{parsed.scheme}://{parsed.netloc}"

def validate_api_url(url: str) -> bool:
    """
    验证 API URL 格式
    
    Args:
        url: 待验证的 URL
        
    Returns:
        是否有效的 URL 布尔值
    """
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url)
        return bool(parsed.scheme and parsed.netloc)
    except Exception:
        return False

# ==================== 重试控制 ====================

class RetryConfig:
    """配置重试逻辑"""
    
    MAX_RETRIES: int = 3
    INITIAL_DELAY: float = 1.0
    MAX_DELAY: float = 30.0
    BACKOFF_FACTOR: float = 2.0
    
    def __init__(self, max_retries: int = 3, delay: float = 1.0):
        self.max_retries = max_retries
        self.initial_delay = delay
    
    def get_delay(self, attempt: int) -> float:
        """
        计算第 attempt 次重试的延迟时间（指数退避）
        
        Args:
            attempt: 当前尝试次数
            
        Returns:
            延迟时间
        """
        delay = self.initial_delay * (self.BACKOFF_FACTOR ** attempt)
        return min(delay, self.MAX_DELAY)
    
    def can_retry(self, attempt: int) -> bool:
        """
        判断是否可以继续重试
        
        Args:
            attempt: 当前尝试次数
            
        Returns:
            是否可以重试
        """
        return attempt < self.max_retries
    
    def wait(self, attempt: int) -> None:
        """
        等待指定时长后重试
        
        Args:
            attempt: 当前尝试次数
        """
        delay = self.get_delay(attempt)
        time.sleep(delay)
        print(f"重试 [{attempt + 1}/{self.max_retries}]，等待 {delay:.2f} 秒")

# ==================== 环境变量加载 ====================

def load_env_vars(env_dict: dict) -> dict:
    """
    加载环境变量，优先使用环境变量，其次使用默认值
    
    Args:
        env_dict: 默认值字典
        
    Returns:
        加载后的环境变量字典
    """
    loaded = {}
    for key, default in env_dict.items():
        value = os.getenv(key.upper())
        if value:
            try:
                # 尝试解析为数字
                if '.' in value:
                    loaded[key] = float(value)
                else:
                    loaded[key] = int(value)
            except (ValueError, TypeError):
                loaded[key] = value
        else:
            loaded[key] = default
    return loaded

def check_env_variables(required_vars: Dict[str, str]) -> Dict[str, Any]:
    """
    检查必需的环境变量是否存在
    
    Args:
        required_vars: 必需环境变量名称列表
        
    Returns:
        检查结果字典 {变量名: (是否存在, 值或错误信息)}
    """
    result = {}
    missing_vars = []
    
    for var_name, var_value in required_vars.items():
        exists = var_value in os.environ
        if exists:
            result[var_name] = (True, os.environ[var_value])
        else:
            missing_vars.append(var_name)
            result[var_name] = (False, f"环境变量 {var_name} 未设置")
    
    if missing_vars:
        print(f"警告：缺少以下环境变量：{', '.join(missing_vars)}")
    
    return result

# ==================== 配置状态 ====================

class ConfigStatus:
    """配置文件加载状态"""
    LOADING = "LOADING"
    READY = "READY"
    ERROR = "ERROR"
    
    @classmethod
    def check_status(cls) -> str:
        """检查配置状态"""
        return cls.READY if cls._initialized else cls.LOADING
    
    @classmethod
    def initialize(cls, validate: bool = True) -> bool:
        """
        初始化配置状态
        
        Args:
            validate: 是否验证配置
            
        Returns:
            初始化是否成功
        """
        if validate and cls._initialized:
            return True
        
        if validate and not cls._validate():
            return False
        
        cls._initialized = True
        print("配置模块初始化完成")
        return True
    
    @classmethod
    def _validate(cls) -> bool:
        """内部验证逻辑"""
        # 这里可以添加更多验证逻辑
        return True
    
    _initialized = False

# ==================== 辅助函数 ====================

def deep_default_dict(d: Dict, defaults: Dict) -> Dict:
    """
    将默认值合并到字典
    
    Args:
        d: 目标字典
        defaults: 默认值字典
        
    Returns:
        新字典（保留 d 的所有键，缺失键使用 defaults 的值）
    """
    for key, default in defaults.items():
        if key not in d:
            d[key] = default
    return dict(d)

def safe_get(d: dict, key: str, default: Any = None) -> Any:
    """
    安全地从字典中获取值
    
    Args:
        d: 目标字典
        key: 键名
        default: 默认值
        
    Returns:
        获取的值或默认值
    """
    if isinstance(d, dict):
        return d.get(key, default)
    return default

# ==================== 导出 ====================

__all__ = [
    "parse_api_base",
    "validate_api_url",
    "RetryConfig",
    "load_env_vars",
    "check_env_variables",
    "ConfigStatus",
    "deep_default_dict",
    "safe_get"
]
