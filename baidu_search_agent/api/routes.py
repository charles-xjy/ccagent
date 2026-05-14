"""
API Routes -路由定义
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

app = FastAPI(
    title="Baidu Search Agent API",
    description="智能搜索代理服务",
    version="1.0.0"
)


class SearchRequest(BaseModel):
    """搜索请求"""
    query: str
    max_results: Optional[int] = 5
    search_type: Optional[str] = "keyword"
    
    class Config:
        protected_namespaces = ()


class SearchResponse(BaseModel):
    """搜索响应"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    total_results: int = 0
    error_msg: Optional[str] = None
    usage: Optional[Dict[str, int]] = None


async def stream_search_response(response):
    """流式响应生成器"""
    yield '{"type": "search-query", "data": "' + response['query'] + '"}'
    yield '{"type": "results", "data": ' + str(response.get('results', [])) + '}'
    yield '{"type": "complete"}'
