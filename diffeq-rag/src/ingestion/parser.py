"""
文档解析模块 - 高保真PDF解析与LaTeX敏感的分块策略
支持PDF、PPT、Word等格式的文档解析
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import numpy as np

# 核心依赖修复
import fitz  # PyMuPDF 的导入名必须是 fitz

# ==========================================
# 🚨 终极环境急救补丁 (Monkey Patch)
# 强行解决 unstructured 与 pdfminer 版本不合导致的 ImportError
# ==========================================
import pdfminer
try:
    import pdfminer.pdfparser
    # 如果 pdfparser 里没有 PSSyntaxError，我们强行塞一个进去
    if not hasattr(pdfminer.pdfparser, 'PSSyntaxError'):
        class PSSyntaxError(Exception): 
            pass
        pdfminer.pdfparser.PSSyntaxError = PSSyntaxError
except ImportError:
    pass
# ==========================================

from unstructured.partition.pdf import partition_pdf
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.docx import partition_docx

logger = logging.getLogger(__name__)

class DocumentParser:
    """
    高保真文档解析器
    专门优化数学文档中的LaTeX公式提取
    """

    def __init__(
        self,
        extract_images: bool = False,
        extract_tables: bool = True,
        hi_res_model: bool = True
    ):
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.hi_res_model = hi_res_model  # 修复：之前这里是 self = hi_res_model.hi_res_model (错误)

        # LaTeX公式模式
        self.inline_math_pattern = re.compile(r'\$([^\$]+)\$')
        self.display_math_pattern = re.compile(r'\$\$([^\$]+)\$\$')

    def parse_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """解析PDF文件，结合布局分析与元数据提取"""
        logger.info(f"开始解析PDF文件: {file_path}")

        # 使用 Unstructured 进行布局分析
        # 注意：infer_table_structure 在新版 unstructured 中用于提取表格内容
        elements = partition_pdf(
            filename=file_path,
            extract_images=self.extract_images,
            infer_table_structure=self.extract_tables,
            strategy="hi_res" if self.hi_res_model else "fast"
        )

        # 使用 PyMuPDF (fitz) 提取元数据 - 修复：之前写的是 PyMuPDF.open
        doc = fitz.open(file_path)
        metadata = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "creator": doc.metadata.get("creator", ""),
            "pages": len(doc)
        }
        doc.close()

        documents = []
        for idx, element in enumerate(elements):
            doc_data = {
                "id": f"{os.path.basename(file_path)}_{idx}",
                "type": element.category,
                "text": self._clean_text(element.text),
                "metadata": {
                    **metadata,
                    "element_id": idx,
                    "source_file": os.path.basename(file_path),
                    "page_number": getattr(element.metadata, "page_number", None)
                }
            }
            documents.append(doc_data)

        logger.info(f"PDF解析完成，共提取 {len(documents)} 个元素")
        return documents

    def _clean_text(self, text: str) -> str:
        """清洗文本并保护公式"""
        if not text: return ""

        # 1. 保护公式：使用唯一占位符
        display_formulas = self.display_math_pattern.findall(text)
        for i, f in enumerate(display_formulas):
            text = text.replace(f"$${f}$$", f"__DP_MATH_{i}__")

        inline_formulas = self.inline_math_pattern.findall(text)
        for i, f in enumerate(inline_formulas):
            text = text.replace(f"${f}$", f"__IL_MATH_{i}__")

        # 2. 清洗多余空格
        text = re.sub(r'\s+', ' ', text).strip()

        # 3. 还原公式
        for i, f in enumerate(display_formulas):
            text = text.replace(f"__DP_MATH_{i}__", f"$${f}$$")
        for i, f in enumerate(inline_formulas):
            text = text.replace(f"__IL_MATH_{i}__", f"${f}$")

        return text

    # parse_pptx 和 parse_docx 逻辑类似，保持基础实现
    def parse_pptx(self, file_path: str) -> List[Dict[str, Any]]:
        elements = partition_pptx(filename=file_path)
        return self._elements_to_docs(elements, file_path)

    def parse_docx(self, file_path: str) -> List[Dict[str, Any]]:
        elements = partition_docx(filename=file_path)
        return self._elements_to_docs(elements, file_path)

    def _elements_to_docs(self, elements, file_path):
        return [{
            "id": f"{os.path.basename(file_path)}_{i}",
            "type": el.category,
            "text": self._clean_text(el.text),
            "metadata": {"source_file": os.path.basename(file_path), "element_id": i}
        } for i, el in enumerate(elements)]

    def parse_any(self, file_path: str) -> List[Dict[str, Any]]:
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf": return self.parse_pdf(file_path)
        if ext in [".pptx", ".ppt"]: return self.parse_pptx(file_path)
        if ext in [".docx", ".doc"]: return self.parse_docx(file_path)
        raise ValueError(f"不支持的文件类型: {ext}")


class LaTeXAwareTextSplitter:
    """LaTeX感知的文本分块器"""
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128, min_chunk_length: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        self.display_math_pattern = re.compile(r'\$\$[\s\S]+?\$\$')
        self.inline_math_pattern = re.compile(r'\$[^\$]+\$')

    def split_text(self, text: str) -> List[str]:
        if not text or len(text) < self.min_chunk_length:
            return [text] if text else []

        # 逻辑：公式保护 -> 分块 -> 还原
        placeholders = []
        def _save(m):
            placeholder = f"__F_{len(placeholders)}__"
            placeholders.append((placeholder, m.group(0)))
            return placeholder

        protected = self.display_math_pattern.sub(_save, text)
        protected = self.inline_math_pattern.sub(_save, protected)

        # 简单分块逻辑
        raw_chunks = [protected[i:i + self.chunk_size] for i in range(0, len(protected), self.chunk_size - self.chunk_overlap)]
        
        # 还原
        final_chunks = []
        for chunk in raw_chunks:
            for p, original in placeholders:
                chunk = chunk.replace(p, original)
            if len(chunk.strip()) >= self.min_chunk_length:
                final_chunks.append(chunk)
        return final_chunks

    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_chunks = []
        for doc in documents:
            chunks = self.split_text(doc.get("text", ""))
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "id": f"{doc['id']}_chunk_{i}",
                    "text": chunk,
                    "metadata": {**doc["metadata"], "chunk_id": i}
                })
        return all_chunks


class SemanticChunker:
    """基于语义相似度的文本分块器"""
    def __init__(self, embedding_model, threshold: float = 0.6, max_chunk_size: int = 800):
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.max_chunk_size = max_chunk_size

    def split_text(self, text: str) -> List[str]:
        # 1. 按句拆分
        sentences = re.split(r'(?<=[。！？\.])\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences: return []

        # 2. 计算句子 Embedding
        embeddings = self.embedding_model.embed_documents(sentences)
        
        chunks = []
        current_chunk = sentences[0]
        
        for i in range(len(sentences) - 1):
            # 计算相邻句子的余弦相似度
            sim = self._cosine_similarity(embeddings[i], embeddings[i+1])
            
            # 如果相似度太低 或 长度超限，则断开
            if sim < self.threshold or len(current_chunk) > self.max_chunk_size:
                chunks.append(current_chunk)
                current_chunk = sentences[i+1]
            else:
                current_chunk += " " + sentences[i+1]
        
        chunks.append(current_chunk)
        return chunks

    def _cosine_similarity(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))