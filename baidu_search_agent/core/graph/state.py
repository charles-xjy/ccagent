"""
LangGraph 状态定义模块
定义 Search Agent 的全局状态共享结构
"""
from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from pydantic import field_validator

# ==================== 状态枚举 ====================
class AgentStatus(str, Enum):
    """Agent 生命周期状态"""
    INITIAL = "INITIAL"          # 初始
    SEARCHING = "SEARCHING"      # 搜索中
    ANALYZING = "ANALYZING"      # 分析中
    SUMMARIZING = "SUMMARIZING"  # 总结中
    DONE = "DONE"                # 完成
    FAILED = "FAILED"            # 失败

# ==================== 事件类型 ====================
class EventType(str, Enum):
    SEARCH = "search"          # 新搜索
    RESEARCH = "research"       # 研究结果
    CLARIFICATION = "clarification"  # 需要澄清
    SUMMARY = "summary"         # 总结生成
    ERROR = "error"            # 错误事件

# ==================== 结果数据结构 ====================
class SearchResult(BaseModel):
    """单条搜索结果"""
    url: str = Field(..., description="结果 URL")
    title: str = Field(..., description="结果标题")
    snippet: str = Field(..., description="摘要片段")
    source_name: str = Field(
        description="来源网站名称",
        examples=["baidu.com", "technews.com"]
    )
    is_sponsored: bool = Field(default=False, description="是否为广告")
    relevance_score: float = Field(
        default=0.5,
        ge=0, le=1,
        description="相关性评分 (0-1)"
    )
    
    # ========= 字段验证 ==========
    @field_validator('relevance_score')
    @classmethod
    def validate_score(cls, v):
        if v < 0 or v > 1:
            raise ValueError("relevance_score 必须在 0-1 之间")
        return v

class SearchQuery(BaseModel):
    """搜索请求序列"""
    query: str = Field(..., description="搜索关键词")
    timestamp: datetime = Field(default_factory=datetime.now, description="搜索时间")
    intent: str = Field(default="general", description="搜索意图类型")
    source: str = Field(default="baidu", description="搜索引擎来源")

class AnalysisData(BaseModel):
    """分析中间数据"""
    facts: List[str] = Field(default_factory=list, description="关键事实列表")
    uncertainties: Optional[List[str]] = Field(
        default=None,
        description="不确定的陈述（需要引用说明）"
    )
    _summary_action: bool = Field(
        default=False,
        description="是否触发了总结操作"
    )
    
    # ========= 字段验证 ==========
    @field_validator('uncertainties')
    @classmethod
    def validate_uncertainties(cls, v):
        if v and not isinstance(v, list):
            raise ValueError("uncertainties 必须是列表类型")
        return v

# ==================== 主状态 Schema ====================
class SearchState(BaseModel):
    """
    LangGraph 状态共享状态（State Graph）
    
    此状态在以下节点间流转：
    - initial_parse
    - intent_parser
    - search_node
    - analyze_node
    - synthesize_node
    """
    
    # ========== 基础字段 ==========
    session_id: str = Field(
        default="",
        description="唯一会话 ID"
    )
    user_input: str = Field(
        default="",
        description="用户原始输入"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="会话开始时间"
    )
    
    # ========== 搜索相关字段 ==========
    queries: List[SearchQuery] = Field(
        default_factory=list,
        description="搜索关键词序列（支持多轮搜索）"
    )
    search_results: List[SearchResult] = Field(
        default_factory=list,
        description="所有搜索结果集合"
    )
    current_query: Optional[str] = Field(
        default=None,
        description="当前正在处理的查询"
    )
    
    # ========== 分析相关字段 ==========
    analysis_data: Optional[AnalysisData] = Field(
        default=None,
        description="中间分析数据（提取事实、识别不确定性）"
    )
    analysis_history: List[str] = Field(
        default_factory=list,
        description="分析过程历史（用于 tracing）"
    )
    
    # ========== 总结相关字段 ==========
    final_answer: Optional[str] = Field(
        default=None,
        description="最终结构化答案"
    )
    sources: List[str] = Field(
        default_factory=list,
        description="引用的源链接列表"
    )
    summary_metadata: Optional[dict] = Field(
        default=None,
        description="总结元数据（confidence, tokens used）"
    )
    
    # ========== 控制流程字段 ==========
    status: AgentStatus = Field(
        default=AgentStatus.INITIAL,
        description="当前 Agent 状态"
    )
    max_retries: int = Field(
        default=3,
        description="最大重试次数"
    )
    retry_count: int = Field(
        default=0,
        description="已重试次数"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="错误信息"
    )
    
    # ========== 消息上下文字段 ==========
    messages: List[dict] = Field(
        default_factory=list,
        description="对话消息历史（format: {'role': 'user'|'assistant', 'content': str}）"
    )
    
    # ========== 进阶控制字段 ==========
    follow_up_probated: int = Field(
        default=0,
        description="已触发的追问次数"
    )
    
    # ==================== Traceing & Validation Helpers ====================
    def can_advance(self) -> bool:
        """判断是否可以推进到下一个状态"""
        if self.status == AgentStatus.DONE:
            return False
        if self.status == AgentStatus.FAILED:
            return False
        if self.retry_count >= self.max_retries:
            return False
        return True
    
    def add_search_result(self, result: SearchResult) -> "SearchState":
        """添加搜索结果（兼容 LangGraph 更新模式）"""
        return self.__class__(
            **self.model_dump(),
            search_results=self.search_results + [result]
        )
    
    def add_query(self, query: str) -> "SearchState":
        """添加新的搜索查询"""
        return self.__class__(
            **self.model_dump(),
            queries=self.queries + [SearchQuery(query=query, intent="general", source="baidu")]
        )
    
    def set_status(self, new_status: AgentStatus) -> "SearchState":
        """设置新状态"""
        return self.__class__(
            **self.model_dump(),
            status=new_status
        )

# ==================== 状态动作定义（辅助函数）====================
def create_initial_state(session_id: str = "default") -> SearchState:
    """创建初始状态"""
    return SearchState(
        session_id=session_id,
        user_input="",
        status=AgentStatus.INITIAL
    )