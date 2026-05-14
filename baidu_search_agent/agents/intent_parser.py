"""
Intent Parser - 意图解析器
解析用户搜索意图
"""

import re
from typing import Tuple, Dict, Optional
from pydantic import BaseModel, Field
from enum import Enum


class IntentType(str, Enum):
    """意图类型枚举"""
    FACTUAL = "factual"
    COMPARATIVE = "comparative"
    PROBLEM_SOLUTION = "problem_solution"
    NEWS = "news"
    LOCATION = "location"
    RECOMMENDATION = "recommendation"
    GENERAL_SEARCH = "general"


class Intent(BaseModel):
    """搜索结果意图"""
    type: IntentType = Field(default=IntentType.GENERAL_SEARCH, description="意图类型")
    query_original: Optional[str] = None
    entities: list = Field(default_factory=list)
    focus_keywords: list = Field(default_factory=list)
    priority: int = Field(default=1)


class SearchIntent(BaseModel):
    """搜索意图"""
    user_query: str
    detected_intent: IntentType
    search_keywords: list = Field(default_factory=list)
    required_fields: list = Field(default_factory=list)
    forbidden_terms: list = Field(default_factory=list)


class IntentParser:
    """意图解析器"""
    
    INTENT_PATTERNS = {
        IntentType.FACTUAL: [
            r"是什么", r"为什么", r"怎么样", r"含义是", r"定义是"
        ],
        IntentType.COMPARATIVE: [
            r"对比", r"区别", r"比较", r"与.*不同", r"和.*哪个好"
        ],
        IntentType.PROBLEM_SOLUTION: [
            r"怎么.*", r"如何.*", r"解决.*", r"有哪些方法"
        ],
        IntentType.NEWS: [
            r"最新消息", r"最新新闻", r"2024.*", r"近日", r"今天"
        ],
        IntentType.LOCATION: [
            r"地址是", r"去哪里", r"位置.*", r"附近", r"在哪里"
        ],
        IntentType.RECOMMENDATION: [
            r"推荐", r"最好", r"值得.*", r"哪个.*好", r"最好的"
        ]
    }
    
    def __init__(self):
        pass
    
    def parse(self, user_query: str) -> SearchIntent:
        """解析用户查询意图"""
        query_lower = user_query.lower()
        
        # 检测主要意图
        detected_intent = self._detect_intent(user_query)
        
        # 提取关键词
        keywords = self._extract_keywords(user_query)
        
        # 提取实体
        entities = self._extract_entities(user_query)
        
        return SearchIntent(
            user_query=user_query,
            detected_intent=detected_intent,
            search_keywords=keywords,
            entities=entities,
            required_fields=["intent_type", "relevant_info"],
            forbidden_terms=[]
        )
    
    def _detect_intent(self, query: str) -> IntentType:
        """检测用户意图"""
        for intent_type, patterns in self.INTENT_PATTERNS.items():
            if any(re.search(pattern, query, re.IGNORECASE) for pattern in patterns):
                return intent_type
        return IntentType.GENERAL_SEARCH
    
    def _extract_keywords(self, query: str):
        """提取关键词"""
        # 去除停用词并分割
        tokens = query.split()
        keywords = [t for t in tokens if len(t) > 1]
        return list(set(keywords))[:5]
    
    def _extract_entities(self, query: str):
        """提取实体"""
        patterns = [
            r"[a-zA-Z\u4e00-\u9fa5]+(\d+)?",  # 单词
            r"\d+(\.\d+)?",  # 数字
            r"[a-z]{2,}(/(\w+))?",  # 英文域名
        ]
        
        entity_groups = set()
        for pattern in patterns:
            entity_groups.update(re.findall(pattern, query, re.IGNORECASE))
        
        return list(entity_groups)[:5]
