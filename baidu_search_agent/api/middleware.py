"""
API Middleware - API 中间件
处理请求和响应中间件
"""

from typing import Dict, Any, Request
from fastapi import FastAPI, Request, Response, HTTPException
from pydantic import BaseModel
import time


class APIMiddleware:
    """API 中间件"""
    
    def __init__(self, app: FastAPI):
        self.app = app
    
    def add_middleware(self):
        """添加中间件"""
        
        @self.app.middleware("http")
        async def log_request(request: Request, call_next):
            start_time = time.time()
            
            # 记录请求信息
            request_id = f"{id(request)}-{time.time()}"
            
            response = await call_next(request)
            
            process_time = time.time() - start_time
            
            # 返回响应
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
        
        @self.app.exception_handler(Exception)
        async def global_exception_handler(request: Request, exc: Exception):
            error_data = {
                "error": True,
                "message": str(exc),
                "request_id": f"{id(request)}-{time.time()}"
            }
            return Response(
                content=str(error_data),
                status_code=500
            )
    
    async def validate_request(self, request: Request, required_fields: list) -> Dict[str, Any]:
        """验证请求"""
        data = await request.json() if request.method == "POST" else {}
        missing_fields = [f for f in required_fields if f not in data]
        
        if missing_fields:
            raise HTTPException(status_code=400, detail=f"Missing fields: {missing_fields}")
        
        return data
    
    async def log_request(self, request: Request, response: Response):
        """记录请求"""
        pass


class ResponseMiddleware:
    """响应中间件"""
    
    @staticmethod
    async def apply(app: FastAPI):
        """应用响应中间件"""
        
        @app.middleware("http")
        async def wrap_hijack(request: Request, call_next):
            response = await call_next(request)
            return response
