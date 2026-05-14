"""
Logger - 日志工具
提供统一的日志记录功能
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from loguru import logger
import os


class LoggerManager:
    """日志管理器"""
    
    def __init__(self, name: str, log_level: str = "INFO", log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # 避免重复添加处理器
        if not self.logger.handlers:
            self._setup_handlers(log_file)
    
    def _setup_handlers(self, log_file: Optional[str] = None):
        """设置处理器"""
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        
        # 格式化器
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 文件处理器
        file_path = log_file if log_file else f"{Path.home()}/logs/.ccagent.log"
        
        Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def get_logger(self) -> logging.Logger:
        """获取日志记录器"""
        return self.logger


def setup_logger(name: str = "baidu_search_agent", log_level: str = "INFO"):
    """设置全局日志"""
    mgr = LoggerManager(name, log_level)
    return mgr.get_logger()
