"""
LangGraph 工作流节点定义
定义检索、评估、生成等核心节点
"""

import json
import logging
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate

from .state import GraphState
from ..retrieval.milvus_client import SearchResult

logger = logging.getLogger(__name__)


# =====================================================
# 检索节点
# =====================================================
def retrieve_documents(state: GraphState, retriever) -> GraphState:
    """
    检索相关文档

    Args:
        state: 当前状态
        retriever: 检索器实例

    Returns:
        更新后的状态
    """
    question = state.get("rewritten_question") or state["question"]

    logger.info(f"开始检索: {question}")

    try:
        # 执行检索
        results = retriever.retrieve(question)

        # 转换为文档格式
        documents = []
        for result in results:
            documents.append({
                "id": result.id,
                "text": result.text,
                "score": result.score,
                "metadata": result.metadata
            })

        logger.info(f"检索完成，找到 {len(documents)} 个相关文档")

        return {
            "documents": documents,
            "iteration": state.get("iteration", 0) + 1
        }

    except Exception as e:
        logger.error(f"检索失败: {e}")
        return {
            "documents": [],
            "error": str(e),
            "iteration": state.get("iteration", 0) + 1
        }


# =====================================================
# 文档评估节点
# =====================================================
def grade_documents(state: GraphState, llm: ChatOpenAI, threshold: float = 0.5) -> GraphState:
    """
    评估检索到的文档与问题的相关性

    Args:
        state: 当前状态
        llm: LLM实例
        threshold: 相关性阈值

    Returns:
        更新后的状态
    """
    question = state["question"]
    documents = state.get("documents", [])

    if not documents:
        logger.warning("没有文档需要评估")
        return {
            "relevance_scores": [],
            "web_search": "yes"
        }

    grading_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""你是一个专业的数学文献评估专家。
判断检索到的文档是否与用户问题相关。
只需要判断相关性，不需要回答问题。
返回JSON格式：{"relevant": true/false, "score": 0.0-1.0, "reason": "原因"}"""),
        HumanMessage(content="""用户问题：{question}

文档内容：
{document}

请评估相关性：""")
    ])

    relevant_docs = []
    relevance_scores = []

    for doc in documents:
        try:
            prompt = grading_prompt.format(
                question=question,
                document=doc["text"][:2000]
            )

            response = llm.invoke([HumanMessage(content=prompt)])
            result = json.loads(response.content)

            score = result.get("score", 0)
            is_relevant = result.get("relevant", False) or score >= threshold

            if is_relevant:
                relevant_docs.append(doc)

            relevance_scores.append(score)

        except Exception as e:
            logger.warning(f"评估文档失败: {e}")
            relevance_scores.append(0.5)  # 默认分数

    logger.info(f"文档评估完成: {len(relevant_docs)}/{len(documents)} 相关")

    # 判断是否需要网络搜索
    web_search = "yes" if len(relevant_docs) == 0 else "no"

    return {
        "documents": relevant_docs,
        "relevance_scores": relevance_scores,
        "web_search": web_search
    }


# =====================================================
# 查询重写节点
# =====================================================
def transform_query(state: GraphState, llm: ChatOpenAI) -> GraphState:
    """
    重写查询以改善检索效果

    Args:
        state: 当前状态
        llm: LLM实例

    Returns:
        更新后的状态
    """
    question = state["question"]

    transform_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""你是一个专业的数学查询优化专家。
将用户的原始问题重写为更适合向量检索的查询。
要求：
1. 补充潜在的同义词和相关概念
2. 使查询更加精确和明确
3. 保持核心数学概念不变
直接返回重写后的查询，不要添加解释。"""),
        HumanMessage(content="原始问题：{question}\n\n重写后的查询：")
    ])

    try:
        prompt = transform_prompt.format(question=question)
        response = llm.invoke([HumanMessage(content=prompt)])

        rewritten_question = response.content.strip()
        logger.info(f"查询重写: {question} -> {rewritten_question}")

        return {"rewritten_question": rewritten_question}

    except Exception as e:
        logger.warning(f"查询重写失败: {e}")
        return {"rewritten_question": question}


# =====================================================
# 答案生成节点
# =====================================================
def generate_answer(
    state: GraphState,
    llm: ChatOpenAI,
    prompt_template: Optional[str] = None
) -> GraphState:
    """
    生成答案

    Args:
        state: 当前状态
        llm: LLM实例
        prompt_template: 自定义Prompt模板

    Returns:
        更新后的状态
    """
    question = state["question"]
    documents = state.get("documents", [])

    if not documents:
        return {
            "generation": "抱歉，我没有找到相关的文档来回答您的问题。",
            "sources": []
        }

    # 构建上下文
    context = "\n\n".join([
        f"[文档 {i+1}]\n{doc['text']}"
        for i, doc in enumerate(documents)
    ])

    # 使用自定义Prompt或默认Prompt
    if prompt_template:
        prompt = prompt_template.format(context=context, question=question)
    else:
        prompt = f"""基于以下参考文档回答用户问题。
如果文档中没有足够信息，请明确说明。

参考文档：
{context}

问题：{question}

要求：
1. 基于提供的文档内容进行回答
2. 使用准确的数学术语和公式
3. 在回答末尾注明参考来源
4. 不要编造不在文档中的信息

回答："""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])

        # 提取参考来源
        sources = [
            {
                "id": doc["id"],
                "score": doc.get("score", 0),
                "metadata": doc.get("metadata", {})
            }
            for doc in documents
        ]

        return {
            "generation": response.content,
            "sources": sources
        }

    except Exception as e:
        logger.error(f"答案生成失败: {e}")
        return {
            "generation": f"生成答案时发生错误：{str(e)}",
            "sources": [],
            "error": str(e)
        }


# =====================================================
# 幻觉检测节点
# =====================================================
def hallucination_grader(state: GraphState, llm: ChatOpenAI) -> GraphState:
    """
    检测生成答案中是否存在幻觉

    Args:
        state: 当前状态
        llm: LLM实例

    Returns:
        更新后的状态
    """
    question = state["question"]
    generation = state.get("generation", "")
    documents = state.get("documents", [])

    if not generation:
        return {"answer_grade": "not_supported"}

    if not documents:
        return {"answer_grade": "not_supported"}

    # 构建文档摘要
    doc_summary = "\n".join([doc["text"][:500] for doc in documents])

    grading_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""你是一个专业的数学知识验证专家。
判断生成的答案是否完全基于提供的文档内容，是否存在幻觉（编造）信息。

评估标准：
1. 答案中的所有事实性陈述是否都能在文档中找到依据
2. 答案是否有过度推断或错误解释

返回JSON格式：{"grade": "supported|not_supported|partially_supported", "score": 0.0-1.0, "reason": "原因"}"""),
        HumanMessage(content="""用户问题：{question}

提供的文档：
{documents}

生成的答案：
{generation}

请判断答案是否存在幻觉：""")
    ])

    try:
        prompt = grading_prompt.format(
            question=question,
            documents=doc_summary,
            generation=generation
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        result = json.loads(response.content)

        grade = result.get("grade", "not_supported")
        score = result.get("score", 0)

        logger.info(f"幻觉检测结果: {grade} (score: {score})")

        return {
            "answer_grade": grade,
            "error": None if grade == "supported" else result.get("reason")
        }

    except Exception as e:
        logger.warning(f"幻觉检测失败: {e}")
        return {"answer_grade": "unknown", "error": str(e)}


# =====================================================
# 路由决策节点
# =====================================================
def route_question(state: GraphState, llm: ChatOpenAI) -> GraphState:
    """
    决定问题的路由方式

    Args:
        state: 当前状态
        llm: LLM实例

    Returns:
        更新后的状态
    """
    question = state["question"]

    routing_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""你是一个智能路由专家。
决定如何处理用户的问题。

可能的路由选项：
1. knowledge_base: 使用知识库回答（适用于专业数学概念）
2. web_search: 需要联网搜索（适用于最新研究或知识库中没有的信息）
3. direct_answer: 可以直接回答（适用于简单问题）

返回JSON格式：{"route": "knowledge_base|web_search|direct_answer", "confidence": 0.0-1.0, "reason": "原因"}"""),
        HumanMessage(content="问题：{question}\n\n请给出路由决策：")
    ])

    try:
        prompt = routing_prompt.format(question=question)
        response = llm.invoke([HumanMessage(content=prompt)])
        result = json.loads(response.content)

        route = result.get("route", "knowledge_base")
        confidence = result.get("confidence", 0.5)

        logger.info(f"路由决策: {route} (confidence: {confidence})")

        return {
            "route": route,
            "error": None
        }

    except Exception as e:
        logger.warning(f"路由决策失败: {e}")
        return {"route": "knowledge_base", "error": str(e)}


# =====================================================
# 答案质量评估节点
# =====================================================
def evaluate_answer_quality(
    state: GraphState,
    llm: ChatOpenAI
) -> GraphState:
    """
    评估答案质量

    Args:
        state: 当前状态
        llm: LLM实例

    Returns:
        更新后的状态
    """
    question = state["question"]
    generation = state.get("generation", "")

    if not generation:
        return {"answer_grade": "low_quality"}

    quality_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""你是一个专业的数学答案质量评估专家。
评估答案的完整性和准确性。

评估维度：
1. 准确性：答案是否正确
2. 完整性：答案是否涵盖了问题的所有方面
3. 专业性：是否使用了准确的数学术语

返回JSON格式：{"accuracy": 0.0-1.0, "completeness": 0.0-1.0, "overall": 0.0-1.0, "feedback": "改进建议"}"""),
        HumanMessage(content="""问题：{question}

答案：{answer}

请给出质量评估：""")
    ])

    try:
        prompt = quality_prompt.format(question=question, answer=generation)
        response = llm.invoke([HumanMessage(content=prompt)])
        result = json.loads(response.content)

        overall = result.get("overall", 0.5)
        grade = "high_quality" if overall >= 0.7 else "low_quality"

        return {
            "answer_grade": grade,
            "error": None
        }

    except Exception as e:
        logger.warning(f"质量评估失败: {e}")
        return {"answer_grade": "unknown", "error": str(e)}


# =====================================================
# Web搜索节点（占位符）
# =====================================================
def web_search(state: GraphState, search_api) -> GraphState:
    """
    执行Web搜索

    Args:
        state: 当前状态
        search_api: 搜索API实例

    Returns:
        更新后的状态
    """
    question = state["question"]

    try:
        # 使用搜索API
        results = search_api.search(question)

        # 提取文本
        documents = []
        for result in results:
            documents.append({
                "id": result.get("id", ""),
                "text": result.get("snippet", ""),
                "score": result.get("score", 0),
                "metadata": {
                    "source": result.get("source", "web"),
                    "url": result.get("url", "")
                }
            })

        return {"documents": documents}

    except Exception as e:
        logger.error(f"Web搜索失败: {e}")
        return {"documents": [], "error": str(e)}
