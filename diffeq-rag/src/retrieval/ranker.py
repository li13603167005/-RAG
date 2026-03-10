"""
重排序模块 - RRF融合与LLM重排序
提供检索结果的重排序功能
"""

import os
import json
from typing import List, Dict, Any, Optional, Union  # <-- 修复：在这里加上了 Union
import logging

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import numpy as np

from .milvus_client import SearchResult

logger = logging.getLogger(__name__)


class RRFReRanker:
    """
    基于Reciprocal Rank Fusion的重排序器
    融合多路检索结果
    """

    def __init__(self, k: int = 60):
        """
        初始化RRF重排序器

        Args:
            k: RRF参数
        """
        self.k = k

    def rerank(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        使用RRF融合重排序

        Args:
            dense_results: 稠密向量检索结果
            sparse_results: 稀疏向量检索结果
            top_k: 返回数量

        Returns:
            重排序后的结果列表
        """
        if not dense_results and not sparse_results:
            return []

        if not dense_results:
            return sparse_results[:top_k]

        if not sparse_results:
            return dense_results[:top_k]

        # 构建排名字典
        dense_rank = {result.id: rank for rank, result in enumerate(dense_results)}
        sparse_rank = {result.id: rank for rank, result in enumerate(sparse_results)}

        # 计算RRF分数
        rrf_scores = {}

        for result in dense_results:
            rank = dense_rank[result.id]
            score = 1.0 / (self.k + rank + 1)
            rrf_scores[result.id] = rrf_scores.get(result.id, 0) + score

        for result in sparse_results:
            rank = sparse_rank[result.id]
            score = 1.0 / (self.k + rank + 1)
            rrf_scores[result.id] = rrf_scores.get(result.id, 0) + score

        # 合并结果对象
        all_results = {r.id: r for r in dense_results + sparse_results}

        # 按RRF分数排序
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # 构建最终结果
        reranked = []
        for result_id in sorted_ids[:top_k]:
            original_result = all_results[result_id]
            reranked.append(SearchResult(
                id=original_result.id,
                text=original_result.text,
                score=rrf_scores[result_id],
                distance=original_result.distance,
                metadata=original_result.metadata
            ))

        return reranked


class LLMReRanker:
    """
    基于LLM的重排序器
    使用LLM评估检索结果与查询的相关性
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        grading_prompt: str = None
    ):
        """
        初始化LLM重排序器

        Args:
            llm: LangChain LLM实例
            grading_prompt: 评分Prompt
        """
        self.llm = llm
        self.grading_prompt = grading_prompt or self._default_grading_prompt()

    def _default_grading_prompt(self) -> str:
        """默认的评分Prompt"""
        return """请评估以下文档与用户问题的相关性，给出0-1的分数。

用户问题：{question}

文档内容：
{document}

相关性分数（0-1）："""

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        使用LLM重排序

        Args:
            query: 查询文本
            results: 检索结果列表
            top_k: 返回数量

        Returns:
            重排序后的结果列表
        """
        if not results:
            return []

        # 限制并行请求数量
        batch_size = 10
        graded_results = []

        for i in range(0, len(results), batch_size):
            batch = results[i:i + batch_size]
            batch_grades = self._grade_batch(query, batch)
            graded_results.extend(batch_grades)

        # 按LLM评分排序
        sorted_results = sorted(
            graded_results,
            key=lambda x: x.get("llm_score", 0),
            reverse=True
        )

        # 转换为SearchResult格式
        reranked = []
        for item in sorted_results[:top_k]:
            reranked.append(SearchResult(
                id=item["id"],
                text=item["text"],
                score=item.get("llm_score", 0),
                distance=item.get("distance", 0),
                metadata=item.get("metadata", {})
            ))

        return reranked

    def _grade_batch(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[Dict[str, Any]]:
        """批量评分"""
        graded = []

        for result in results:
            try:
                prompt = self.grading_prompt.format(
                    question=query,
                    document=result.text[:2000]  # 限制长度
                )

                response = self.llm.invoke([HumanMessage(content=prompt)])

                # 解析分数
                score = self._parse_score(response.content)

                graded.append({
                    "id": result.id,
                    "text": result.text,
                    "llm_score": score,
                    "distance": result.distance,
                    "metadata": result.metadata
                })

            except Exception as e:
                logger.warning(f"评分失败: {e}")
                # 如果评分失败，使用原始分数
                graded.append({
                    "id": result.id,
                    "text": result.text,
                    "llm_score": result.score,
                    "distance": result.distance,
                    "metadata": result.metadata
                })

        return graded

    def _parse_score(self, response: str) -> float:
        """从LLM响应中解析分数"""
        try:
            # 尝试提取数字
            import re
            numbers = re.findall(r'0\.\d+|\d\.\d+|\d', response)
            if numbers:
                score = float(numbers[0])
                return min(max(score, 0), 1)  # 限制在0-1之间
        except:
            pass

        return 0.5  # 默认分数


class CrossEncoderReRanker:
    """
    基于Cross-Encoder的重排序器
    使用预训练的Cross-Encoder模型进行重排序
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        初始化Cross-Encoder重排序器

        Args:
            model_name: 模型名称
        """
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
            self.available = True
        except ImportError:
            logger.warning("Cross-Encoder模型不可用")
            self.model = None
            self.available = False

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        使用Cross-Encoder重排序

        Args:
            query: 查询文本
            results: 检索结果列表
            top_k: 返回数量

        Returns:
            重排序后的结果列表
        """
        if not results:
            return []

        if not self.available:
            return results[:top_k]

        # 准备查询-文档对
        pairs = [(query, result.text) for result in results]

        # 预测相关性分数
        scores = self.model.predict(pairs)

        # 组合结果
        scored_results = [
            (result, score) for result, score in zip(results, scores)
        ]

        # 按分数排序
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # 转换为SearchResult格式
        reranked = []
        for result, score in scored_results[:top_k]:
            reranked.append(SearchResult(
                id=result.id,
                text=result.text,
                score=float(score),
                distance=result.distance,
                metadata=result.metadata
            ))

        return reranked


class EnsembleReRanker:
    """
    集成重排序器
    结合多种重排序策略
    """

    def __init__(
        self,
        rrf_ranker: Optional[RRFReRanker] = None,
        llm_ranker: Optional[LLMReRanker] = None,
        cross_encoder_ranker: Optional[CrossEncoderReRanker] = None
    ):
        self.rrf_ranker = rrf_ranker
        self.llm_ranker = llm_ranker
        self.cross_encoder_ranker = cross_encoder_ranker

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        strategy: str = "ensemble",
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        重排序检索结果

        Args:
            query: 查询文本
            results: 检索结果列表
            strategy: 重排序策略 ("rrf", "llm", "cross_encoder", "ensemble")
            top_k: 返回数量

        Returns:
            重排序后的结果列表
        """
        if not results:
            return []

        if strategy == "rrf" or len(results) < 3:
            # RRF不需要查询，只基于排名
            return results[:top_k]

        if strategy == "llm" and self.llm_ranker:
            return self.llm_ranker.rerank(query, results, top_k)

        if strategy == "cross_encoder" and self.cross_encoder_ranker:
            return self.cross_encoder_ranker.rerank(query, results, top_k)

        if strategy == "ensemble":
            # 集成策略：先用Cross-Encoder快速筛选，再用LLM精细排序
            if self.cross_encoder_ranker:
                results = self.cross_encoder_ranker.rerank(query, results, top_k * 2)

            if self.llm_ranker:
                results = self.llm_ranker.rerank(query, results, top_k)

            return results

        return results[:top_k]


def create_reranker(
    reranker_type: str = "ensemble",
    llm: Optional[ChatOpenAI] = None
) -> Union[RRFReRanker, LLMReRanker, CrossEncoderReRanker, EnsembleReRanker]:
    """
    创建重排序器的工厂函数

    Args:
        reranker_type: 重排序器类型
        llm: LLM实例

    Returns:
        重排序器实例
    """
    if reranker_type == "rrf":
        return RRFReRanker()

    if reranker_type == "llm" and llm:
        return LLMReRanker(llm)

    if reranker_type == "cross_encoder":
        return CrossEncoderReRanker()

    if reranker_type == "ensemble":
        return EnsembleReRanker(
            rrf_ranker=RRFReRanker(),
            llm_ranker=LLMReRanker(llm) if llm else None,
            cross_encoder_ranker=CrossEncoderReRanker()
        )

    return RRFReRanker()