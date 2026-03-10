"""
LangGraph 状态定义模块
定义工作流中的状态结构
"""

from typing import TypedDict, List, Optional, Dict, Any
from pydantic import BaseModel, Field


class GraphState(TypedDict):
    """
    LangGraph 工作流状态定义
    """
    # 用户问题
    question: str
    # 重写后的查询
    rewritten_question: Optional[str]
    # 检索到的文档
    documents: List[Dict[str, Any]]
    # 生成的答案
    generation: Optional[str]
    # 是否需要网络搜索
    web_search: str  # "yes" or "no"
    # 检索迭代次数
    iteration: int
    # 相关性评分
    relevance_scores: Optional[List[float]]
    # 答案评分
    answer_grade: Optional[str]
    # 错误信息
    error: Optional[str]
    # 路由决策
    route: Optional[str]
    # 参考来源
    sources: Optional[List[Dict[str, Any]]]


class RetrievalState(BaseModel):
    """检索状态"""
    question: str
    top_k: int = 5
    filter_expr: Optional[str] = None
    use_hybrid: bool = True


class GenerationState(BaseModel):
    """生成状态"""
    question: str
    context: str
    prompt_template: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 2000


class GradingState(BaseModel):
    """评估状态"""
    question: str
    document: str
    grade: Optional[str] = None
    score: Optional[float] = None
    reason: Optional[str] = None


class RoutingState(BaseModel):
    """路由状态"""
    question: str
    route: str  # "knowledge_base", "web_search", "direct_answer"
    confidence: float = 0.0
    reason: Optional[str] = None
