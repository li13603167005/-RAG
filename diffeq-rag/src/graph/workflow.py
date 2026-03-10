"""
LangGraph 工作流构建模块
构建完整的Corrective + Adaptive RAG工作流
"""

import logging
import uuid  # 用于自动生成会话档案号
from typing import Optional, Dict, Any, Callable

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import GraphState
from .nodes import (
    retrieve_documents,
    grade_documents,
    transform_query,
    generate_answer,
    hallucination_grader,
    route_question,
    evaluate_answer_quality,
    web_search
)

logger = logging.getLogger(__name__)


class RAGWorkflow:
    """
    RAG工作流管理器
    构建和执行Corrective + Adaptive RAG流程
    """

    def __init__(
        self,
        retriever,
        llm,
        max_iterations: int = 3,
        grade_threshold: float = 0.5,
        use_hallucination_check: bool = True,
        use_query_rewrite: bool = True,
        use_routing: bool = True
    ):
        """
        初始化RAG工作流

        Args:
            retriever: 检索器实例
            llm: LLM实例
            max_iterations: 最大迭代次数
            grade_threshold: 相关性评分阈值
            use_hallucination_check: 是否使用幻觉检测
            use_query_rewrite: 是否使用查询重写
            use_routing: 是否使用路由决策
        """
        self.retriever = retriever
        self.llm = llm
        self.max_iterations = max_iterations
        self.grade_threshold = grade_threshold
        self.use_hallucination_check = use_hallucination_check
        self.use_query_rewrite = use_query_rewrite
        self.use_routing = use_routing

        self.graph = None

    def build_graph(self) -> StateGraph:
        """
        构建工作流图

        Returns:
            编译后的LangGraph
        """
        # 创建图
        workflow = StateGraph(GraphState)

        # 添加节点
        if self.use_routing:
            # 将路由节点命名为 router_node 避免与状态键 route 冲突
            workflow.add_node("router_node", self._route_node)

        workflow.add_node("retrieve", lambda state: retrieve_documents(state, self.retriever))
        workflow.add_node("grade_documents", lambda state: grade_documents(state, self.llm, self.grade_threshold))
        
        # 核心修复：新增包装函数，强制迭代次数 +1，打破死循环
        def _transform_query_wrapper(state):
            result = transform_query(state, self.llm)
            if isinstance(result, dict):
                result["iteration"] = state.get("iteration", 0) + 1
            return result

        workflow.add_node("transform_query", _transform_query_wrapper)

        workflow.add_node("generate", lambda state: generate_answer(state, self.llm))

        if self.use_hallucination_check:
            workflow.add_node("hallucination_grader", lambda state: hallucination_grader(state, self.llm))

        workflow.add_node("evaluate_quality", lambda state: evaluate_answer_quality(state, self.llm))

        # 设置入口点
        if self.use_routing:
            workflow.set_entry_point("router_node")
        else:
            workflow.set_entry_point("retrieve")

        # 添加边
        if self.use_routing:
            # 路由决策边
            workflow.add_conditional_edges(
                "router_node",
                self._route_decision,
                {
                    "knowledge_base": "retrieve",
                    "web_search": "generate",  # Web搜索作为备用
                    "direct_answer": "generate"
                }
            )

        # 检索 -> 评估
        workflow.add_edge("retrieve", "grade_documents")

        # 评估 -> 生成（如果相关文档足够）
        workflow.add_conditional_edges(
            "grade_documents",
            self._grade_decision,
            {
                "generate": "generate",
                "rewrite": "transform_query"
            }
        )

        # 查询重写 -> 重新检索（循环）
        workflow.add_edge("transform_query", "retrieve")

        # 生成 -> 幻觉检测
        if self.use_hallucination_check:
            workflow.add_edge("generate", "hallucination_grader")

            # 幻觉检测 -> 质量评估
            workflow.add_conditional_edges(
                "hallucination_grader",
                self._hallucination_decision,
                {
                    "retry": "generate",
                    "quality": "evaluate_quality"
                }
            )
        else:
            workflow.add_edge("generate", "evaluate_quality")

        # 质量评估 -> 结束
        workflow.add_edge("evaluate_quality", END)

        # 编译图
        checkpointer = MemorySaver()
        self.graph = workflow.compile(checkpointer=checkpointer)

        logger.info("RAG工作流构建完成")
        return self.graph

    def _route_node(self, state: GraphState) -> Dict[str, Any]:
        """路由节点"""
        return route_question(state, self.llm)

    def _route_decision(self, state: GraphState) -> str:
        """路由决策"""
        route = state.get("route", "knowledge_base")

        if route == "web_search":
            return "web_search"
        elif route == "direct_answer":
            return "direct_answer"
        else:
            return "knowledge_base"

    def _grade_decision(self, state: GraphState) -> str:
        """评估决策"""
        documents = state.get("documents", [])
        iteration = state.get("iteration", 0)

        # 如果有相关文档，生成答案
        if documents:
            return "generate"

        # 如果没有相关文档且未超过最大迭代次数，重写查询
        if iteration < self.max_iterations:
            return "rewrite"

        # 超过最大迭代次数，生成答案（可能质量不高）
        return "generate"

    def _hallucination_decision(self, state: GraphState) -> str:
        """幻觉检测决策"""
        grade = state.get("answer_grade", "unknown")

        # 👇 核心修复：防止大模型在“检查幻觉”这一步无限互相折磨
        # 无论有没有幻觉，都先进入下一步 quality 评估并输出给用户，打破死循环
        if grade == "supported":
            return "quality"
        elif grade == "not_supported":
            return "quality"  # 👈 原本是 "retry"，现在改为强制往下走
        else:
            return "quality"

    def run(self, question: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        运行工作流

        Args:
            question: 用户问题
            config: 运行配置

        Returns:
            工作流结果
        """
        if not self.graph:
            self.build_graph()

        # 为 MemorySaver 自动分配 thread_id
        if config is None:
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        elif "configurable" not in config:
            config["configurable"] = {"thread_id": str(uuid.uuid4())}
        elif "thread_id" not in config["configurable"]:
            config["configurable"]["thread_id"] = str(uuid.uuid4())

        initial_state = {
            "question": question,
            "documents": [],
            "generation": None,
            "web_search": "no",
            "iteration": 0,
            "relevance_scores": [],
            "answer_grade": None,
            "error": None,
            "route": None,
            "sources": []
        }

        try:
            result = self.graph.invoke(initial_state, config=config)
            return result

        except Exception as e:
            logger.error(f"工作流执行失败: {e}")
            return {
                "question": question,
                "generation": f"处理问题时发生错误：{str(e)}",
                "documents": [],
                "error": str(e)
            }

    async def run_async(self, question: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        异步运行工作流

        Args:
            question: 用户问题
            config: 运行配置

        Returns:
            工作流结果
        """
        if not self.graph:
            self.build_graph()

        # 为 MemorySaver 自动分配 thread_id
        if config is None:
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        elif "configurable" not in config:
            config["configurable"] = {"thread_id": str(uuid.uuid4())}
        elif "thread_id" not in config["configurable"]:
            config["configurable"]["thread_id"] = str(uuid.uuid4())

        initial_state = {
            "question": question,
            "documents": [],
            "generation": None,
            "web_search": "no",
            "iteration": 0,
            "relevance_scores": [],
            "answer_grade": None,
            "error": None,
            "route": None,
            "sources": []
        }

        try:
            result = await self.graph.ainvoke(initial_state, config=config)
            return result

        except Exception as e:
            logger.error(f"工作流执行失败: {e}")
            return {
                "question": question,
                "generation": f"处理问题时发生错误：{str(e)}",
                "documents": [],
                "error": str(e)
            }


class SimpleRAGWorkflow:
    """
    简化的RAG工作流
    不使用LangGraph的直接实现
    """

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def run(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        简单的RAG流程

        Args:
            question: 用户问题
            top_k: 检索数量

        Returns:
            结果字典
        """
        # 1. 检索
        results = self.retriever.retrieve(question, top_k=top_k)

        # 2. 构建上下文
        context = "\n\n".join([
            f"[文档 {i+1}]\n{result.text}"
            for i, result in enumerate(results)
        ])

        # 3. 生成答案
        prompt = f"""基于以下参考文档回答用户问题。

参考文档：
{context}

问题：{question}

回答："""

        response = self.llm.invoke([{"type": "human", "content": prompt}])

        return {
            "question": question,
            "generation": response.content,
            "documents": [
                {
                    "id": r.id,
                    "text": r.text,
                    "score": r.score,
                    "metadata": r.metadata
                }
                for r in results
            ],
            "sources": [
                {
                    "id": r.id,
                    "score": r.score,
                    "metadata": r.metadata
                }
                for r in results
            ]
        }


def create_rag_workflow(
    retriever,
    llm,
    max_iterations: int = 3,
    use_advanced: bool = True,
    **kwargs
) -> RAGWorkflow:
    """
    创建RAG工作流的工厂函数

    Args:
        retriever: 检索器
        llm: LLM
        max_iterations: 最大迭代次数
        use_advanced: 是否使用高级工作流

    Returns:
        RAG工作流实例
    """
    if use_advanced:
        return RAGWorkflow(
            retriever=retriever,
            llm=llm,
            max_iterations=max_iterations,
            **kwargs
        )
    else:
        return SimpleRAGWorkflow(retriever=retriever, llm=llm)