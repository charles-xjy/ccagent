"""
LangGraph Search Agent 状态定义模块

提供搜索代理的全局状态 Schema，用于状态管理和上下文传递
"""

from enum import Enum
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    """代理生命周期状态枚举"""
    
    INITIAL = "INITIAL"          # 初始状态 - 等待用户输入
    SEARCHING = "SEARCHING"      # 搜索中 - 正在执行 MCP 搜索
    ANALYZING = "ANALYZING"      # 分析中 - 正在分析搜索结果
    SUMMARIZING = "SUMMARIZING"  # 总结中 - 正在生成最终答案
    DONE = "DONE"                # 完成 - 任务完成
    FAILED = "FAILED"            # 失败 - 任务失败


class SearchResult(BaseModel):
    """MCP 搜索项模型"""
    source: str = Field(..., description="搜索源名称")
    title: Optional[str] = Field(None, description="标题")
    url: Optional[str] = Field(None, description="URL")
    summary: Optional[str] = Field(None, description="内容摘要")
    metadata: Optional[dict] = Field(default_factory=dict, description="额外元数据")
    raw_content: Optional[str] = Field(None, description="原始内容（可选）")
    score: Optional[float] = Field(None, description="相关性评分 0-1")


class SearchQuery(BaseModel):
    """搜索查询请求模型"""
    query: str = Field(..., description="搜索关键词")
    source: str = Field(..., description="指定或自动选择的搜索源")
    depth: int = Field(default=1, ge=1, le=5, description="搜索深度层级")
    use_serp: bool = Field(default=False, description="是否启用近似搜索")
    context: Optional[str] = Field(None, description="上下文信息")


class ClarifyingQuestion(BaseModel):
    """澄清问题模型"""
    question: str = Field(..., description="需要澄清的问题")
    type: str = Field(default="missing_info", description="问题类型")
    suggested_answers: Optional[List[str]] = Field(None, description="建议答案")
    required: bool = Field(default=False, description="是否必需回答")


class AgentHistoryItem(BaseModel):
    """对话历史项"""
    role: str = Field(..., description="角色 (user/assistant/system)")
    content: str = Field(..., description="消息内容")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    is_function_call: bool = Field(False, description="是否为函数调用")


class AnalysisData(BaseModel):
    """分析中间数据容器"""
    search_iterations: int = Field(default=0, description="搜索迭代次数")
    best_matches: List[SearchResult] = Field(default_factory=list, description="最佳匹配项")
    synthesis_notes: List[str] = Field(default_factory=list, description="综合笔记")
    extracted_facts: List[dict] = Field(default_factory=list, description="提取的事实")
    processing_steps: List[dict] = Field(default_factory=list, description="处理步骤记录")


class SearchState(BaseModel):
    """
    LangGraph 搜索代理状态模型
    
    用于在多轮对话和状态机中始终保持一致的状态上下文
    """
    
    # 基础标识
    session_id: str = Field(..., description="会话唯一标识")
    user_input: str = Field(..., description="用户原始输入")
    creation_time: datetime = Field(default_factory=datetime.now, description="创建时间")
    
    # 查询信息
    query: str = Field(..., description="当前搜索查询")
    
    # 交互管理
    clarifying_questions: List[ClarifyingQuestion] = Field(
        default_factory=list,
        description="需要澄清的问题列表"
    )
    history: List[AgentHistoryItem] = Field(
        default_factory=list,
        description="对话消息历史"
    )
    
    # 搜索与结果
    search_results: List[SearchResult] = Field(
        default_factory=list,
        description="存储 MCP 搜索结果（支持多轮）"
    )
    current_search_query: Optional[SearchQuery] = Field(
        default_factory=SearchQuery,
        description="当前搜索查询对象"
    )
    
    # 分析与处理
    analysis_data: Optional[AnalysisData] = Field(
        default_factory=AnalysisData,
        description="分析中间数据"
    )
    final_answer: Optional[str] = Field(
        None,
        description="最终总结答案"
    )
    
    # 状态管理
    status: AgentStatus = Field(default=AgentStatus.INITIAL, description="当前状态")
    
    # 重试控制
    retry_count: int = Field(default=0, ge=0, description="重试次数")
    max_retries: int = Field(default=3, ge=1, le=10, description="最大重试次数")
    
    class Config:
        """Pydantic 配置"""
        json_schema_extra = {
            "title": "Search Agent State",
            "description": "用于 LangGraph 搜索代理的全局状态管理"
        }

    @property
    def is_complete(self) -> bool:
        """判断状态是否完成"""
        return self.status == AgentStatus.DONE
    
    @property
    def is_failed(self) -> bool:
        """判断状态是否失败"""
        return self.status == AgentStatus.FAILED
    
    @property
    def is_awaiting_clarification(self) -> bool:
        """判断是否在等待用户澄清"""
        return self.status == AgentStatus.INITIAL and len(self.clarifying_questions) > 0
    
    def has_pending_questions(self) -> bool:
        """是否有待处理的澄清问题"""
        return len(self.clarifying_questions) > 0
    
    def add_message(self, role: str, content: str, is_function_call: bool = False) -> None:
        """
        添加对话消息到历史记录
        
        Args:
            role: 消息角色 (user/assistant/system)
            content: 消息内容
            is_function_call: 是否为函数调用
        """
        self.history.append(AgentHistoryItem(
            role=role,
            content=content,
            is_function_call=is_function_call
        ))
    
    def add_search_result(self, source: str, **kwargs) -> None:
        """
        添加搜索结果
        
        Args:
            source: 搜索源名称
            **kwargs: 结果详情 (title, url, summary, metadata, raw_content, score)
        """
        result = SearchResult(source=source, **{k: v for k, v in kwargs.items() if v is not None})
        self.search_results.append(result)
    
    def add_clarifying_question(self, question: str, question_type: str = "missing_info",
                                 required: bool = False, suggested_answers: Optional[List[str]] = None) -> None:
        """
        添加澄清问题
        
        Args:
            question: 需要澄清的问题
            question_type: 问题类型
            required: 是否必需回答
            suggested_answers: 建议答案列表
        """
        self.clarifying_questions.append(ClarifyingQuestion(
            question=question,
            type=question_type,
            suggested_answers=suggested_answers,
            required=required
        ))
    
    def advance_status(self, new_status: AgentStatus) -> None:
        """
        推进状态
        
        Args:
            new_status: 新的状态值
        """
        self.status = new_status
        
        # 根据状态设置默认值
        if new_status == AgentStatus.DONE:
            if not self.final_answer and self.search_results:
                # 自动生成总结默认文本
                self.final_answer = self._generate_default_summary()
        elif new_status in [AgentStatus.DONE, AgentStatus.FAILED]:
            # 完成状态不允许再添加问题
            self.clarifying_questions.clear()
    
    def _generate_default_summary(self) -> str:
        """生成默认总结（简化版调用 LangChain/Neo4j 会生成详细版）"""
        count = len(self.search_results)
        if count == 0:
            return "未找到相关搜索结果，请重新搜索或提供更多上下文。"
        elif count == 1:
            result = self.search_results[0]
            return f"找到 {count} 条结果。主要内容：{result.summary or str(result)[:200]}..."
        else:
            return f"共找到 {count} 条结果。已综合所有来源信息生成答案。"
    
    def reset(self) -> None:
        """
        重置状态（保留 session_id 和 user_input）
        
        常用于新一轮搜索但保持会话连续性
        """
        self.status = AgentStatus.INITIAL
        self.query = ""
        self.search_results.clear()
        self.clarifying_questions.clear()
        self.history.clear()
        self.final_answer = None
        self.analysis_data = AnalysisData()
        self.retry_count = 0
        
        # 保留的用户输入和会话 ID
        # self.user_input 保留
        # self.session_id 保留
