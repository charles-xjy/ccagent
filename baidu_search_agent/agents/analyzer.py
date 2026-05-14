"""
Analyzer - 结果分析器
分析搜索结果
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field


class SearchAnalysis(BaseModel):
    """搜索结果分析"""
    relevance_score: float = Field(default=0.0, description="相关性评分")
    quality_score: float = Field(default=0.0, description="质量评分")
    credibility_score: float = Field(default=0.0, description="可信度评分")
    summary_bullets: List[str] = Field(default_factory=list, description="要点总结")
    key_findings: List[str] = Field(default_factory=list, description="关键发现")


class AnalyzeResult(BaseModel):
    """分析结果"""
    query: str
    analysis: SearchAnalysis
    insights: List[str] = Field(default_factory=list)


class Analyzer:
    """结果分析器"""
    
    def __init__(self):
        pass
    
    def analyze(self, results: List[Dict[str, Any]], query: str) -> AnalyzeResult:
        """分析搜索结果"""
        # 计算相关性评分
        relevance = self._calculate_relevance(results, query)
        
        # 计算质量评分
        quality = self._calculate_quality(results)
        
        # 计算可信度评分
        credibility = self._calculate_credibility(results)
        
        # 生成要点总结
        summary_bullets = self._generate_summary(results)
        
        # 提取关键发现
        key_findings = self._extract_key_findings(results, query)
        
        return AnalyzeResult(
            query=query,
            analysis=SearchAnalysis(
                relevance_score=relevance,
                quality_score=quality,
                credibility_score=credibility,
                summary_bullets=summary_bullets
            ),
            insights=key_findings
        )
    
    def _calculate_relevance(self, results: List[Dict], query: str) -> float:
        """计算相关性评分"""
        if not results:
            return 0.0
        
        scores = []
        query_lower = query.lower()
        
        for result in results:
            score = self._result_relevance_score(result, query_lower)
            scores.append(score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _result_relevance_score(self, result: Dict, query: str) -> float:
        """计算单个结果的相关性"""
        title = result.get("title", "").lower()
        snippet = result.get("snippet", "").lower()
        
        if any(kw in title or kw in snippet for kw in query.split()):
            return 8.0
        elif query in title or query in snippet:
            return 6.0
        elif result.get("url", "").lower() in query.lower():
            return 4.0
        else:
            return 2.0
    
    def _calculate_quality(self, results: List[Dict]) -> float:
        """计算质量评分"""
        quality_factors = [
            lambda r: 5.0 if r.get("source", "").lower().find(".gov") != -1 else 3.0,
            lambda r: 4.0 if r.get("timestamp") and not r["timestamp"].year < 2022 else 2.0,
            lambda r: 3.0 if len(r.get("snippet", "")) > 50 else 1.0,
        ]
        
        if not results:
            return 0.0
        
        scores = []
        for factor in quality_factors:
            scores.append(sum(factor(r) for r in results) / len(results))
        
        return sum(scores) / len(scores)
    
    def _calculate_credibility(self, results: List[Dict]) -> float:
        """计算可信度评分"""
        domains_seen = set()
        for r in results:
            url = r.get("url", "")
            domain = url.split("//")[-1].split("/")[0] if "//" in url else ""
            domains_seen.add(domain.lower())
        
        credibility_map = {
            ".gov": 10.0,
            ".edu": 8.0,
            ".org": 6.0,
            ".com": 4.0,
            ".net": 3.0,
            ".cn": 5.0,
        }
        
        total = sum(credibility_map.get(d, 2.0) for d in domains_seen)
        return min(total * 5, 100.0) / len(domains_seen) if domains_seen else 0.0
    
    def _generate_summary(self, results: List[Dict]) -> List[str]:
        """生成要点总结"""
        bullets = []
        
        # 统计信息
        total_results = len(results)
        bullets.append(f"共检索到 {total_results} 个结果")
        
        # 摘要
        snippets = [r.get("snippet", "")[:100] for r in results[:3] if r.get("snippet")]
        bullets.append("主要信息：" + " | ".join(snippets[:3]) if snippets else "暂无摘要")
        
        return bullets
    
    def _extract_key_findings(self, results: List[Dict], query: str) -> List[str]:
        """提取关键发现"""
        findings = []
        
        # 提取标题中的关键词
        for result in results[:5]:
            findings.append(f'"{' '.join(result["title"].split()[:3])}')"
        
        # 提取源域名
        domains = set([r["url"].split("/")[2] for r in results if "/" in r.get("url", "")])
        findings.append(f"信息来源：{', '.join(domains)}")
        
        return findings[:5]
