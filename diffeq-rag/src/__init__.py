"""
DiffEq-RAG: 微分方程概念领域RAG知识库系统
"""

__version__ = "1.0.0"
__author__ = "DiffEq-RAG Team"

from .ingestion.parser import DocumentParser, LaTeXAwareTextSplitter, SemanticChunker
from .ingestion.embedding import EmbeddingModel, BM25Encoder, HybridEmbedder
from .retrieval.milvus_client import MilvusClient, HybridRetriever
from .graph.workflow import RAGWorkflow, create_rag_workflow

__all__ = [
    "DocumentParser",
    "LaTeXAwareTextSplitter",
    "SemanticChunker",
    "EmbeddingModel",
    "BM25Encoder",
    "HybridEmbedder",
    "MilvusClient",
    "HybridRetriever",
    "RAGWorkflow",
    "create_rag_workflow"
]
