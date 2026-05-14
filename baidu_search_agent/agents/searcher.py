"""
Searcher - 搜索器
Baidu 搜索功能封装，基于 MCP 协议
"""

import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from httpx import AsyncClient, Timeout
import re


class SearchResult(BaseModel):
    """搜索结果"""
    title: str = Field(..., description="结果标题")
    url: str = Field(..., description="结果链接")
    snippet: str = Field(default="", description="摘要信息")
    source: str = Field(default="", description="来源域名")
    timestamp: Optional[str] = Field(default=None, description="时间戳")
    relevance_score: float = Field(default=1.0, description="相关性评分")


class SearchRequest(BaseModel):
    """搜索请求"""
    query: str = Field(..., description="搜索关键词")
    max_results: int = Field(default=5, description="最大返回结果数")
    search_type: str = Field(default="keyword", description="搜索类型")
    timestamp: str = Field(default="", description="时间范围")


class BaiduSearcher:
    """Baidu 搜索引擎封装"""
    
    def __init__(self, api_key: str, access_token: str, timeout: int = 30):
        self.api_key = api_key
        self.access_token = access_token
        self.timeout = timeout
        self.base_url = "https://aip.baidubce.com/rest/2.0/search"
        
        # 初始化 HTTP 客户端
        self._session = AsyncClient(timeout=Timeout(timeout))
    
    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]:
        """执行搜索"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        
        params = {
            "q": query,
            "fmt": "json"
        }
        
        try:
            async with self._session.get(self.base_url, params=params, headers=headers) as response:
                response.raise_for_status()
                result = response.json()
                return self._parse_results(result, max_results)
        except Exception as e:
            raise Exception(f"搜索失败：{str(e)}")
    
    def _parse_results(self, raw_data: Dict, max_results: int) -> List[Dict[str, Any]]:
        """解析搜索结果"""
        results = []
        
        for item in raw_data.get("data", [])[:max_results]:
            result = {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
                "source": self._extract_domain(item.get("url", "")),
                "timestamp": item.get("timestamp"),
                "relevance_score": self._calculate_relevance(item, query)
            }
            results.append(result)
        
        return results
    
    def _extract_domain(self, url: str) -> str:
        """提取域名"""
        if "//" in url:
            domain = url.split("//")[1].split("/")[0]
            return domain
        return url.split("/")[0] if "/" in url else url
    
    def _calculate_relevance(self, item: Dict, query: str) -> float:
        """计算相关性评分"""
        query_lower = query.lower()
        title = item.get("title", "").lower()
        snippet = item.get("snippet", "").lower()
        
        score = 0.0
        
        # 标题匹配
        if query_lower in title:
            score += 8.0
        elif any(qw in title for qw in query_lower.split()):
            score += 5.0
        
        # 摘要匹配
        if query_lower in snippet:
            score += 5.0
        else:
            score += 2.0
        
        return min(score, 10.0)
    
    async def keyword_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """关键词搜索"""
        return await self.search(query, max_results)
    
    async def image_search(self, query: str, count: int = 10) -> List[Dict[str, Any]]:
        """图片搜索"""
        pass
    
    async def voice_search(self, query: str, duration: int = 3) -> List[Dict[str, Any]]:
        """语音搜索"""
        pass
