"""
RAG评估模块
使用RAGAS框架进行自动化评估
"""

import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_relevancy
)
from datasets import Dataset
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EvalQuestion:
    """评估问题数据类"""
    question: str
    ground_truth: str
    context: Optional[str] = None


class RAGEvaluator:
    """
    RAG系统评估器
    使用RAGAS框架评估系统性能
    """

    def __init__(
        self,
        llm,
        embeddings,
        eval_questions: Optional[List[EvalQuestion]] = None
    ):
        """
        初始化评估器

        Args:
            llm: LLM实例
            embeddings: 嵌入模型实例
            eval_questions: 评估问题列表
        """
        self.llm = llm
        self.embeddings = embeddings
        self.eval_questions = eval_questions or []

    def add_eval_question(self, question: EvalQuestion):
        """添加评估问题"""
        self.eval_questions.append(question)

    def evaluate_retriever(
        self,
        retriever,
        top_k: int = 5
    ) -> Dict[str, float]:
        """
        评估检索器性能

        Args:
            retriever: 检索器实例
            top_k: 检索数量

        Returns:
            评估指标
        """
        results = []

        for eval_q in self.eval_questions:
            # 检索文档
            retrieved_docs = retriever.retrieve(eval_q.question, top_k=top_k)

            # 提取检索到的上下文
            context = "\n\n".join([doc.text for doc in retrieved_docs])

            results.append({
                "question": eval_q.question,
                "ground_truth": eval_q.ground_truth,
                "retrieved_context": context,
                "retrieved_docs": [doc.text for doc in retrieved_docs]
            })

        # 计算检索指标
        context_precisions = []
        for result in results:
            # 简单计算：检索到的文档中有多少包含答案相关关键词
            precision = self._calculate_context_precision(
                result["retrieved_docs"],
                result["ground_truth"]
            )
            context_precisions.append(precision)

        return {
            "context_precision": sum(context_precisions) / len(context_precisions) if context_precisions else 0,
            "num_questions": len(results)
        }

    def evaluate_generator(
        self,
        generator_func,
        use_ragas: bool = False
    ) -> Dict[str, float]:
        """
        评估生成器性能

        Args:
            generator_func: 生成函数
            use_ragas: 是否使用RAGAS

        Returns:
            评估指标
        """
        if use_ragas and self.llm and self.embeddings:
            return self._evaluate_with_ragas(generator_func)
        else:
            return self._evaluate_basic(generator_func)

    def _evaluate_with_ragas(self, generator_func) -> Dict[str, float]:
        """使用RAGAS进行评估"""
        # 准备RAGAS数据集
        eval_data = []

        for eval_q in self.eval_questions:
            # 生成答案
            result = generator_func(eval_q.question)

            eval_data.append({
                "question": eval_q.question,
                "answer": result.get("generation", ""),
                "ground_truth": eval_q.ground_truth,
                "contexts": [doc["text"] for doc in result.get("documents", [])]
            })

        # 创建数据集
        dataset = Dataset.from_list(eval_data)

        # 运行评估
        try:
            eval_result = evaluate(
                dataset=dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall
                ],
                llm=self.llm,
                embeddings=self.embeddings
            )

            return {
                "faithfulness": eval_result["faithfulness"],
                "answer_relevancy": eval_result["answer_relevancy"],
                "context_precision": eval_result["context_precision"],
                "context_recall": eval_result["context_recall"]
            }

        except Exception as e:
            logger.error(f"RAGAS评估失败: {e}")
            return self._evaluate_basic(generator_func)

    def _evaluate_basic(self, generator_func) -> Dict[str, float]:
        """基础评估（不使用RAGAS）"""
        results = []

        for eval_q in self.eval_questions:
            result = generator_func(eval_q.question)

            # 计算基本指标
            answer = result.get("generation", "")
            ground_truth = eval_q.ground_truth

            # 答案相似度（简化版）
            answer_sim = self._calculate_similarity(answer, ground_truth)

            # 答案长度比
            length_ratio = len(answer) / len(ground_truth) if ground_truth else 0

            results.append({
                "answer_similarity": answer_sim,
                "length_ratio": length_ratio,
                "has_answer": len(answer) > 0,
                "answer": answer,
                "ground_truth": ground_truth
            })

        # 计算平均指标
        return {
            "avg_answer_similarity": sum(r["answer_similarity"] for r in results) / len(results),
            "avg_length_ratio": sum(r["length_ratio"] for r in results) / len(results),
            "completion_rate": sum(r["has_answer"] for r in results) / len(results)
        }

    def evaluate_full_system(
        self,
        retriever,
        llm
    ) -> Dict[str, Any]:
        """
        评估完整RAG系统

        Args:
            retriever: 检索器
            llm: LLM

        Returns:
            完整评估报告
        """
        # 1. 评估检索器
        retrieval_metrics = self.evaluate_retriever(retriever)

        # 2. 评估生成器
        def generate_func(question):
            docs = retriever.retrieve(question)
            context = "\n\n".join([doc.text for doc in docs])

            from langchain.schema import HumanMessage
            prompt = f"基于以下上下文回答问题：\n\n{context}\n\n问题：{question}"
            response = llm.invoke([HumanMessage(content=prompt)])

            return {
                "generation": response.content,
                "documents": [{"text": doc.text} for doc in docs]
            }

        generation_metrics = self.evaluate_generator(generate_func)

        # 3. 组合结果
        report = {
            "retrieval": retrieval_metrics,
            "generation": generation_metrics,
            "overall": {
                "context_precision": retrieval_metrics.get("context_precision", 0),
                "answer_quality": generation_metrics.get("avg_answer_similarity", 0),
                "completion_rate": generation_metrics.get("completion_rate", 0)
            }
        }

        return report

    def _calculate_context_precision(
        self,
        retrieved_docs: List[str],
        ground_truth: str
    ) -> float:
        """
        计算上下文精确度

        Args:
            retrieved_docs: 检索到的文档
            ground_truth: 标准答案

        Returns:
            精确度分数
        """
        if not retrieved_docs or not ground_truth:
            return 0.0

        # 提取关键词
        gt_keywords = set(ground_truth.lower().split())

        relevant_count = 0
        for doc in retrieved_docs:
            doc_keywords = set(doc.lower().split())
            # 计算交集
            overlap = gt_keywords & doc_keywords
            if len(overlap) > 0:
                relevant_count += 1

        return relevant_count / len(retrieved_docs)

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算文本相似度（简化版）

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            相似度分数
        """
        if not text1 or not text2:
            return 0.0

        # 使用简单的词集合相似度
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)


class DifferentialEquationEvaluator:
    """
    微分方程领域专用评估器
    针对数学领域的特殊评估指标
    """

    def __init__(self):
        self.math_keywords = [
            "sobolev", "空间", "嵌入", "定理", "不等式",
            "算子", "范数", "导数", "偏微分", "常微分",
            "椭圆", "抛物", "双曲", "弱解", "强解"
        ]

    def evaluate_math_accuracy(
        self,
        generated_answer: str,
        ground_truth: str
    ) -> Dict[str, Any]:
        """
        评估数学答案准确性

        Args:
            generated_answer: 生成的答案
            ground_truth: 标准答案

        Returns:
            评估结果
        """
        # 检查是否包含数学关键词
        gen_has_math = any(kw in generated_answer.lower() for kw in self.math_keywords)
        gt_has_math = any(kw in ground_truth.lower() for kw in self.math_keywords)

        # 检查公式
        has_formulas = "$" in generated_answer or "$$" in generated_answer

        return {
            "contains_math_concepts": gen_has_math,
            "expected_math_concepts": gt_has_math,
            "contains_formulas": has_formulas,
            "math_concept_match": gen_has_math == gt_has_math
        }

    def evaluate_definition_accuracy(
        self,
        generated_answer: str,
        required_definitions: List[str]
    ) -> float:
        """
        评估定义准确性

        Args:
            generated_answer: 生成的答案
            required_definitions: 需要的定义列表

        Returns:
            准确度分数
        """
        matched = 0
        for definition in required_definitions:
            if definition.lower() in generated_answer.lower():
                matched += 1

        return matched / len(required_definitions) if required_definitions else 0


def load_eval_questions(file_path: str) -> List[EvalQuestion]:
    """
    从JSON文件加载评估问题

    Args:
        file_path: 文件路径

    Returns:
        评估问题列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = []
    for item in data:
        questions.append(EvalQuestion(
            question=item["question"],
            ground_truth=item["ground_truth"]
        ))

    return questions


def save_eval_results(results: Dict[str, Any], output_path: str):
    """
    保存评估结果

    Args:
        results: 评估结果
        output_path: 输出路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# 示例评估问题
DEFAULT_EVAL_QUESTIONS = [
    EvalQuestion(
        question="什么是Sobolev空间？",
        ground_truth="Sobolev空间是函数及其弱导数都属于L^p空间的函数空间。"
    ),
    EvalQuestion(
        question="Sobolev嵌入定理的内容是什么？",
        ground_truth="Sobolev嵌入定理建立了Sobolev空间与常规函数空间之间的嵌入关系。"
    ),
    EvalQuestion(
        question="H^1空间的定义是什么？",
        ground_truth="H^1是Sobolev空间，表示一阶弱导数平方可积的函数空间。"
    ),
    EvalQuestion(
        question="Poincare不等式的条件是什么？",
        ground_truth="Poincare不等式通常要求函数在有界域上且边界值为零。"
    ),
    EvalQuestion(
        question="椭圆型偏微分方程的特征是什么？",
        ground_truth="椭圆型PDE的特征是没有实特征线，解通常是光滑的。"
    )
]
