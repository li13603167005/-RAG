"""
Milvus 向量数据库客户端
支持混合向量检索 (Dense + Sparse)
"""

import os
import json
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import logging

import numpy as np
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
    MilvusException
)

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """检索结果数据类"""
    id: str
    text: str
    score: float
    distance: float
    metadata: Dict[str, Any]


class MilvusClient:
    """
    Milvus 向量数据库客户端
    支持稠密向量、稀疏向量和混合检索
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        user: str = "",
        password: str = "",
        collection_name: str = "diffeq_knowledge_base"
    ):
        """
        初始化Milvus客户端

        Args:
            host: Milvus主机地址
            port: Milvus端口
            user: 用户名
            password: 密码
            collection_name: 集合名称
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.collection_name = collection_name
        self.collection = None
        self._connected = False

    def connect(self):
        """建立与Milvus的连接"""
        if self._connected:
            logger.warning("已经连接到Milvus")
            return

        logger.info(f"连接Milvus: {self.host}:{self.port}")

        try:
            connections.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password
            )
            self._connected = True
            logger.info("Milvus连接成功")
        except MilvusException as e:
            logger.error(f"Milvus连接失败: {e}")
            raise

    def disconnect(self):
        """断开与Milvus的连接"""
        if self._connected:
            connections.disconnect("default")
            self._connected = False
            logger.info("Milvus连接已断开")

    def create_collection(
        self,
        dimension: int = 1024,
        description: str = "Differential Equation Knowledge Base",
        enable_dynamic_field: bool = True
    ):
        """
        创建集合

        Args:
            dimension: 向量维度
            description: 集合描述
            enable_dynamic_field: 是否启用动态字段
        """
        if utility.has_collection(self.collection_name):
            logger.warning(f"集合 {self.collection_name} 已存在")
            self.collection = Collection(self.collection_name)
            return

        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]

        # 创建schema
        schema = CollectionSchema(
            fields=fields,
            description=description,
            enable_dynamic_field=enable_dynamic_field
        )

        # 创建集合
        self.collection = Collection(
            name=self.collection_name,
            schema=schema,
            using="default"
        )

        logger.info(f"集合 {self.collection_name} 创建成功")

        # 创建索引
        self.create_indexes()

    def create_indexes(self):
        """创建索引"""
        if not self.collection:
            raise ValueError("集合未初始化")

        # 稠密向量索引 - HNSW
        dense_index_params = {
            "index_type": "HNSW",
            "metric_type": "L2",
            "params": {
                "M": 16,
                "efConstruction": 256
            }
        }
        self.collection.create_index(
            field_name="dense_vector",
            index_params=dense_index_params
        )
        logger.info("稠密向量索引(HNSW)创建成功")

        # 稀疏向量索引 - Sparsen BF
        sparse_index_params = {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP",
            "params": {
                "inverted_index": {}
            }
        }
        self.collection.create_index(
            field_name="sparse_vector",
            index_params=sparse_index_params
        )
        logger.info("稀疏向量索引创建成功")

    def insert(
        self,
        texts: List[str],
        dense_vectors: List[List[float]],
        sparse_vectors: List[Dict[int, float]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        插入数据

        Args:
            texts: 文本列表
            dense_vectors: 稠密向量列表
            sparse_vectors: 稀疏向量列表
            metadata: 元数据列表

        Returns:
            插入的ID列表
        """
        if not self.collection:
            raise ValueError("集合未初始化")

        # 准备数据
        chunk_ids = [f"chunk_{i}" for i in range(len(texts))]
        metadatas = metadata or [{}] * len(texts)

        # 转换为稀疏向量格式
        from pymilvus import SparseFloatVector
        sparse_data = [
            SparseFloatVector(weights) if weights else SparseFloatVector({})
            for weights in sparse_vectors
        ]

        data = [
            chunk_ids,
            texts,
            dense_vectors,
            sparse_data,
            metadatas
        ]

        # 插入数据
        result = self.collection.insert(data)
        self.collection.flush()

        logger.info(f"成功插入 {len(texts)} 条数据")
        return [str(id) for id in result.primary_keys]

    def search_dense(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_expr: Optional[str] = None
    ) -> List[SearchResult]:
        """
        稠密向量检索

        Args:
            query_vector: 查询向量
            top_k: 返回数量
            filter_expr: 过滤表达式

        Returns:
            检索结果列表
        """
        if not self.collection:
            raise ValueError("集合未初始化")

        search_params = {
            "metric_type": "L2",
            "params": {
                "ef": 128
            }
        }

        results = self.collection.search(
            data=[query_vector],
            anns_field="dense_vector",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=["chunk_id", "text", "metadata"]
        )

        return self._parse_results(results)

    def search_sparse(
        self,
        query_vector: Dict[int, float],
        top_k: int = 5,
        filter_expr: Optional[str] = None
    ) -> List[SearchResult]:
        """
        稀疏向量检索

        Args:
            query_vector: 查询稀疏向量
            top_k: 返回数量
            filter_expr: 过滤表达式

        Returns:
            检索结果列表
        """
        if not self.collection:
            raise ValueError("集合未初始化")

        from pymilvus import SparseFloatVector

        search_params = {
            "metric_type": "IP",
            "params": {}
        }

        results = self.collection.search(
            data=[SparseFloatVector(query_vector)],
            anns_field="sparse_vector",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=["chunk_id", "text", "metadata"]
        )

        return self._parse_results(results)

    def search_hybrid(
        self,
        dense_vector: List[float],
        sparse_vector: Dict[int, float],
        top_k: int = 5,
        rrf_k: int = 60,
        filter_expr: Optional[str] = None
    ) -> List[SearchResult]:
        """
        混合检索 (RRF融合)

        Args:
            dense_vector: 稠密查询向量
            sparse_vector: 稀疏查询向量
            top_k: 返回数量
            rrf_k: RRF参数
            filter_expr: 过滤表达式

        Returns:
            融合后的检索结果列表
        """
        # 并行执行两路检索
        dense_results = self.search_dense(
            dense_vector,
            top_k=top_k * 2,  # 获取更多结果以便融合
            filter_expr=filter_expr
        )

        sparse_results = self.search_sparse(
            sparse_vector,
            top_k=top_k * 2,
            filter_expr=filter_expr
        )

        # RRF融合
        fused_results = self._rrf_fusion(
            dense_results,
            sparse_results,
            k=rrf_k,
            top_k=top_k
        )

        return fused_results

    def _rrf_fusion(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        k: int = 60,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Reciprocal Rank Fusion (RRF) 融合

        Args:
            dense_results: 稠密检索结果
            sparse_results: 稀疏检索结果
            k: RRF参数
            top_k: 返回数量

        Returns:
            融合后的结果列表
        """
        # 构建排名字典
        dense_rank = {result.id: rank for rank, result in enumerate(dense_results)}
        sparse_rank = {result.id: rank for rank, result in enumerate(sparse_results)}

        # 计算RRF分数
        rrf_scores = {}

        # 处理稠密结果
        for result in dense_results:
            rank = dense_rank[result.id]
            score = 1.0 / (k + rank + 1)
            rrf_scores[result.id] = rrf_scores.get(result.id, 0) + score

        # 处理稀疏结果
        for result in sparse_results:
            rank = sparse_rank[result.id]
            score = 1.0 / (k + rank + 1)
            rrf_scores[result.id] = rrf_scores.get(result.id, 0) + score

        # 合并结果
        all_results = {r.id: r for r in dense_results + sparse_results}

        # 按RRF分数排序
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # 构建最终结果
        fused = []
        for i, result_id in enumerate(sorted_ids[:top_k]):
            original_result = all_results[result_id]
            fused.append(SearchResult(
                id=original_result.id,
                text=original_result.text,
                score=rrf_scores[result_id],
                distance=original_result.distance,
                metadata=original_result.metadata
            ))

        return fused

    def _parse_results(self, results) -> List[SearchResult]:
        """解析检索结果"""
        parsed = []

        for hits in results:
            for hit in hits:
                parsed.append(SearchResult(
                    id=hit.entity.get("chunk_id"),
                    text=hit.entity.get("text"),
                    score=hit.distance,
                    distance=hit.distance,
                    metadata=hit.entity.get("metadata", {})
                ))

        return parsed

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        if not self.collection:
            return {}

        stats = {
            "name": self.collection_name,
            "num_entities": self.collection.num_entities,
            "indexes": self.collection.indexes
        }

        return stats

    def delete_collection(self):
        """删除集合"""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            logger.info(f"集合 {self.collection_name} 已删除")

    def load_collection(self):
        """加载集合到内存"""
        if self.collection:
            self.collection.load()
            logger.info("集合已加载")

    def release_collection(self):
        """释放集合"""
        if self.collection:
            self.collection.release()
            logger.info("集合已释放")


class HybridRetriever:
    """
    混合检索器
    封装了Milvus客户端的混合检索功能
    """

    def __init__(
        self,
        milvus_client: MilvusClient,
        embedding_model,
        top_k: int = 5,
        rrf_k: int = 60
    ):
        self.milvus_client = milvus_client
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """
        检索相关文档

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            检索结果列表
        """
        # 生成嵌入向量
        embeddings = self.embedding_model.embed_query(query)

        # 混合检索
        results = self.milvus_client.search_hybrid(
            dense_vector=embeddings["dense_vector"],
            sparse_vector=embeddings["sparse_vector"],
            top_k=top_k or self.top_k,
            rrf_k=self.rrf_k
        )

        return results

    def retrieve_with_filter(
        self,
        query: str,
        filter_expr: str,
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        带过滤条件的检索

        Args:
            query: 查询文本
            filter_expr: 过滤表达式
            top_k: 返回数量

        Returns:
            检索结果列表
        """
        embeddings = self.embedding_model.embed_query(query)

        results = self.milvus_client.search_hybrid(
            dense_vector=embeddings["dense_vector"],
            sparse_vector=embeddings["sparse_vector"],
            top_k=top_k or self.top_k,
            rrf_k=self.rrf_k,
            filter_expr=filter_expr
        )

        return results


def create_milvus_client(
    host: str = "localhost",
    port: int = 19530,
    collection_name: str = "diffeq_knowledge_base"
) -> MilvusClient:
    """
    创建Milvus客户端的工厂函数

    Args:
        host: 主机地址
        port: 端口
        collection_name: 集合名称

    Returns:
        Milvus客户端实例
    """
    client = MilvusClient(
        host=host,
        port=port,
        collection_name=collection_name
    )
    client.connect()
    return client
