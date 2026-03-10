"""
嵌入模型模块 - BGE-M3 嵌入向量生成
支持稠密向量和稀疏向量的生成
"""

import os
os.environ['HF_HOME'] = r'D:\Anaconda\envs\diffeq-rag\models_cache'
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging

from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    嵌入向量生成器
    支持BGE-M3模型的稠密向量和稀疏向量生成
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        use_fp16: bool = False
    ):
        """
        初始化嵌入模型

        Args:
            model_name: 模型名称
            device: 设备 ("cpu" 或 "cuda")
            use_fp16: 是否使用半精度
        """
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16 and device == "cuda"

        logger.info(f"加载嵌入模型: {model_name}")
        self.model = BGEM3FlagModel(model_name, use_fp16=self.use_fp16)
        logger.info("嵌入模型加载完成")

    def embed_documents(self, texts: List[str]) -> Dict[str, Any]:
        """
        对文档进行嵌入

        Args:
            texts: 文档文本列表

        Returns:
            包含稠密向量和稀疏向量的字典
        """
        logger.info(f"开始嵌入 {len(texts)} 个文档")

        # BGE-M3 同时生成稠密和稀疏向量
        results = self.model.encode(
            texts,
            batch_size=32,
            max_length=1024,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False
        )

        return {
            "dense_vectors": results["dense_vecs"].tolist(),
            "lexical_weights": results["lexical_weights"],  # 稀疏向量权重
            "normalize": results["normalize"]
        }

    def embed_query(self, text: str) -> Dict[str, Any]:
        """
        对查询进行嵌入

        Args:
            text: 查询文本

        Returns:
            包含稠密向量和稀疏向量的字典
        """
        result = self.model.encode(
            [text],
            batch_size=1,
            max_length=512,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False
        )

        return {
            "dense_vector": result["dense_vecs"][0].tolist(),
            "lexical_weights": result["lexical_weights"][0],
            "normalize": result["normalize"]
        }

    def get_dense_embedding(self, text: str) -> List[float]:
        """
        获取稠密向量嵌入

        Args:
            text: 文本

        Returns:
            稠密向量
        """
        result = self.embed_query(text)
        return result["dense_vector"]

    def get_sparse_embedding(self, text: str) -> Dict[int, float]:
        """
        获取稀疏向量嵌入 (用于BM25)

        Args:
            text: 文本

        Returns:
            稀疏向量 (token_id -> weight)
        """
        result = self.embed_query(text)
        return result["lexical_weights"]

    def get_embedding_dim(self) -> int:
        """获取嵌入向量的维度"""
        return 1024  # BGE-M3 的维度


class BM25Encoder:
    """
    BM25 稀疏向量编码器
    用于传统的关键词检索
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        初始化BM25编码器

        Args:
            k1: BM25参数
            b: BM25参数
        """
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = {}
        self.idf = {}
        self.doc_len = []
        self.corpus = []

    def fit(self, corpus: List[str]):
        """
        拟合语料库

        Args:
            corpus: 文档列表
        """
        logger.info(f"拟合BM25模型，语料库大小: {len(corpus)}")

        self.corpus_size = len(corpus)
        self.corpus = corpus

        # 计算文档长度
        self.doc_len = [len(doc.split()) for doc in corpus]
        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 0

        # 计算文档频率
        for doc in corpus:
            freq = {}
            for word in set(doc.split()):
                freq[word] = freq.get(word, 0) + 1
            self.doc_freqs.update(freq)

        # 计算IDF
        for word, freq in self.doc_freqs.items():
            self.idf[word] = np.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)

        logger.info("BM25模型拟合完成")

    def encode_queries(self, queries: List[str]) -> List[Dict[int, float]]:
        """
        对查询进行编码

        Args:
            queries: 查询列表

        Returns:
            稀疏向量列表
        """
        results = []
        for query in queries:
            scores = self._score(query)
            # 转换为稀疏向量格式
            sparse_vector = {int(k): float(v) for k, v in scores.items() if v > 0}
            results.append(sparse_vector)

        return results

    def _score(self, query: str) -> Dict[str, float]:
        """计算查询与每个文档的BM25分数"""
        query_terms = query.split()
        scores = {}

        for i, doc in enumerate(self.corpus):
            doc_terms = doc.split()
            doc_len = self.doc_len[i]

            score = 0.0
            for term in query_terms:
                if term not in self.idf:
                    continue

                tf = doc_terms.count(term)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += self.idf[term] * numerator / denominator

            scores[str(i)] = score

        return scores


class HybridEmbedder:
    """
    混合嵌入器
    同时生成稠密向量和稀疏向量
    """

    def __init__(
        self,
        dense_model: EmbeddingModel,
        sparse_model: Optional[BM25Encoder] = None
    ):
        self.dense_model = dense_model
        self.sparse_model = sparse_model

    def embed_documents(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        对文档进行混合嵌入

        Args:
            texts: 文档列表

        Returns:
            混合嵌入列表
        """
        # 稠密向量
        dense_results = self.dense_model.embed_documents(texts)

        # 稀疏向量
        sparse_results = []
        if self.sparse_model:
            # 需要先拟合
            if self.sparse_model.corpus_size == 0:
                self.sparse_model.fit(texts)
            sparse_results = self.sparse_model.encode_queries(texts)

        # 合并结果
        results = []
        for i in range(len(texts)):
            result = {
                "dense_vector": dense_results["dense_vectors"][i],
                "sparse_vector": sparse_results[i] if sparse_results else {}
            }
            results.append(result)

        return results

    def embed_query(self, text: str) -> Dict[str, Any]:
        """
        对查询进行混合嵌入

        Args:
            text: 查询文本

        Returns:
            混合嵌入
        """
        dense_result = self.dense_model.embed_query(text)

        sparse_result = {}
        if self.sparse_model:
            sparse_results = self.sparse_model.encode_queries([text])
            sparse_result = sparse_results[0]

        return {
            "dense_vector": dense_result["dense_vector"],
            "sparse_vector": sparse_result
        }


def create_embedding_model(
    model_name: str = "BAAI/bge-m3",
    device: str = "cpu"
) -> EmbeddingModel:
    """
    创建嵌入模型的工厂函数

    Args:
        model_name: 模型名称
        device: 设备

    Returns:
        嵌入模型实例
    """
    return EmbeddingModel(model_name=model_name, device=device)


def create_bm25_encoder() -> BM25Encoder:
    """
    创建BM25编码器

    Returns:
        BM25编码器实例
    """
    return BM25Encoder()
