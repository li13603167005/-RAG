"""
配置加载工具
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Config:
    """配置类"""
    # Milvus配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    collection_name: str = "diffeq_knowledge_base"
    dimension: int = 1024

    # Embedding配置
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"

    # LLM配置
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2000

    # 检索配置
    top_k: int = 5
    retrieval_threshold: float = 0.5
    rrf_k: int = 60

    # 分块配置
    chunk_size: int = 512
    chunk_overlap: int = 128
    min_chunk_length: int = 50

    # LangGraph配置
    max_iterations: int = 3
    web_search_threshold: float = 0.3

    @classmethod
    def from_yaml(cls, file_path: str) -> "Config":
        """
        从YAML文件加载配置

        Args:
            file_path: YAML文件路径

        Returns:
            Config实例
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        return cls(**data)

    @classmethod
    def from_env(cls) -> "Config":
        """
        从环境变量加载配置

        Returns:
            Config实例
        """
        return cls(
            milvus_host=os.getenv("MILVUS_HOST", "localhost"),
            milvus_port=int(os.getenv("MILVUS_PORT", "19530")),
            collection_name=os.getenv("COLLECTION_NAME", "diffeq_knowledge_base"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu"),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o"),
            top_k=int(os.getenv("TOP_K", "5")),
            rrf_k=int(os.getenv("RRF_K", "60"))
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "milvus_host": self.milvus_host,
            "milvus_port": self.milvus_port,
            "collection_name": self.collection_name,
            "dimension": self.dimension,
            "embedding_model": self.embedding_model,
            "embedding_device": self.embedding_device,
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "top_k": self.top_k,
            "retrieval_threshold": self.retrieval_threshold,
            "rrf_k": self.rrf_k,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_chunk_length": self.min_chunk_length,
            "max_iterations": self.max_iterations,
            "web_search_threshold": self.web_search_threshold
        }


def load_config(config_path: Optional[str] = None) -> Config:
    """
    加载配置

    Args:
        config_path: 配置文件路径

    Returns:
        Config实例
    """
    if config_path and Path(config_path).exists():
        return Config.from_yaml(config_path)
    else:
        return Config.from_env()
