"""
Summarizer - 内容摘要器
生成最终摘要
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field


class SearchSummary(BaseModel):
    """搜索结果摘要"""
    question: str
    answer: str
    summary: str
    key_points: List[str] = Field(default_factory=list)
    sources: List[Dict[str, str]] = Field(default_factory=list)
    confidence: float = 1.0


class SummarizeResult(BaseModel):
    """摘要结果"""
    original: str
    summary: str
    word_count: int
    reduction_ratio: float


class Summarizer:
    """内容摘要器"""
    
    def __init__(self):
        pass
    
    def create_summary(self, query: str, results: List[Dict[str, Any]]) -> SearchSummary:
        """创建搜索摘要"""
        if not results:
            return SearchSummary(
                question=query,
                answer="未找到相关结果",
                summary="抱歉，没有找到有用的信息。请尝试其他关键词。",
                confidence=0.0
            )
        
        # 提取关键信息
        answer = self._extract_answer(query, results)
        summary = self._generate_summary(query, results)
        key_points = self._extract_key_points(results)
        sources = self._extract_sources(results)
        
        return SearchSummary(
            question=query,
            answer=answer,
            summary=summary,
            key_points=key_points,
            sources=sources,
            confidence=self._calculate_confidence(results)
        )
    
    def _extract_answer(self, query: str, results: List[Dict]) -> str:
        """提取答案"""
        if not results:
            return "未找到答案"
        
        # 选择最相关的结果
        best_result = max(results, key=lambda x: x.get("relevance_score", 0))
        
        if query.startswith("如何") or query.startswith("怎么"):
            return best_result.get("snippet", "暂无可用信息")
        
        return best_result.get("title", "") + " - " + best_result.get("snippet", "")[:100]
    
    def _generate_summary(self, query: str, results: List[Dict]) -> str:
        """生成摘要"""
        snippets = [r["snippet"] for r in results[:3] if r.get("snippet")]
        if not snippets:
            return "暂无可用信息"
        
        return "综合搜索结果:\n" + "\n".join(snippets[:3])
    
    def _extract_key_points(self, results: List[Dict]) -> List[str]:
        """提取关键点"""
        points = []
        
        for result in results[:5]:
            title = result.get("title", "")[:50]
            snippet = result.get("snippet", "")[:50]
            points.append(f"{title}: {snippet}")
        
        return points[:5]
    
    def _extract_sources(self, results: List[Dict]) -> List[str]:
        """提取来源"""
        return [{"title": r.get("title", ""), "url": r.get("url", "")} for r in results[:3]]
    
    def _calculate_confidence(self, results: List[Dict]) -> float:
        """计算置信度"""
        if not results:
            return 0.0
        
        # 基于结果多样性和质量计算置信度
        sources_diversity = len(set([r["url"].split("/")[2] for r in results[:5] if "/" in r.get("url", "")]))
        
        # 高质量域名权重
        good_domains = [d for r in results if "." in r.get("url", "") and r["url"].split("/")[2] for d in ".".join(r["url"].split("/")[2].split(".")[-2:])]
        
        score = sources_diversity * 20
        if sources_diversity >= 3:
            score += 30
        if sources_diversity >= 5:
            score += 20
        
        return min(score * 5, 100.0)
