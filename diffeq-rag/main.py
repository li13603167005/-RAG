#deepseek API性价比更高，且支持openai api接口格式，所以本项目使用deepseek。
import os
import sys
import nltk
nltk.download('averaged_perceptron_tagger')
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1" 


# 👇 新增：自动告诉 Tesseract 语言包在哪里
tessdata_dir = os.path.join(sys.prefix, 'share', 'tessdata')
if not os.path.exists(tessdata_dir):
    # 如果 share 目录下没有，尝试找 Library 目录
    tessdata_dir = os.path.join(sys.prefix, 'Library', 'bin', 'tessdata')
    
os.environ["TESSDATA_PREFIX"] = tessdata_dir

import json
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from src.ingestion.parser import DocumentParser, LaTeXAwareTextSplitter
from src.ingestion.embedding import create_embedding_model
from src.retrieval.milvus_client import create_milvus_client, HybridRetriever
from src.retrieval.ranker import create_reranker
from src.graph.workflow import create_rag_workflow
from src.graph.state import GraphState

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """应用配置"""
    # Milvus配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    collection_name: str = "diffeq_knowledge_base"

    # Embedding配置
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"

    # LLM配置
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # 检索配置
    top_k: int = 5
    rrf_k: int = 60

    class Config:
        env_file = ".env"


settings = Settings()

# 初始化FastAPI
app = FastAPI(
    title="微分方程RAG知识库API",
    description="基于GPT-4o的微分方程概念领域RAG知识库系统",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# 数据模型
# =====================================================
class QueryRequest(BaseModel):
    """查询请求"""
    question: str = Field(..., description="用户问题")
    top_k: Optional[int] = Field(5, description="返回结果数量")
    use_rerank: Optional[bool] = Field(True, description="是否使用重排序")
    use_advanced_workflow: Optional[bool] = Field(True, description="是否使用高级工作流")


class QueryResponse(BaseModel):
    """查询响应"""
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    documents: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class DocumentInsertRequest(BaseModel):
    """文档插入请求"""
    texts: List[str]
    metadata: Optional[List[Dict[str, Any]]] = None


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    services: Dict[str, str]


# =====================================================
# 全局变量
# =====================================================
embedding_model = None
milvus_client = None
retriever = None
reranker = None
rag_workflow = None
llm = None


def initialize_services():
    """初始化服务"""
    global embedding_model, milvus_client, retriever, reranker, rag_workflow, llm

    logger.info("开始初始化服务...")

    # 初始化Embedding模型
    try:
        embedding_model = create_embedding_model(
            model_name=settings.embedding_model,
            device=settings.embedding_device
        )
        logger.info("Embedding模型初始化完成")
    except Exception as e:
        logger.warning(f"Embedding模型初始化失败: {e}")

    # 初始化Milvus客户端
    try:
        milvus_client = create_milvus_client(
            host=settings.milvus_host,
            port=settings.milvus_port,
            collection_name=settings.collection_name
        )
        milvus_client.create_collection()
        milvus_client.load_collection()
        logger.info("Milvus客户端初始化完成")
    except Exception as e:
        logger.warning(f"Milvus客户端初始化失败: {e}")
        milvus_client = None

    # 初始化检索器
    if embedding_model and milvus_client:
        retriever = HybridRetriever(
            milvus_client=milvus_client,
            embedding_model=embedding_model,
            top_k=settings.top_k,
            rrf_k=settings.rrf_k
        )
        logger.info("检索器初始化完成")

    # 初始化LLM
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model='deepseek-chat',
            temperature=0.0,
            api_key='sk-0462bd395cdc4a86ac6abc8454124aed',
            base_url="https://api.deepseek.com"
        )
        logger.info("LLM初始化完成")
    except Exception as e:
        logger.warning(f"LLM初始化失败: {e}")

    # 初始化RAG工作流
    if retriever and llm:
        rag_workflow = create_rag_workflow(
            retriever=retriever,
            llm=llm,
            max_iterations=3,
            use_advanced=True
        )
        rag_workflow.build_graph()
        logger.info("RAG工作流初始化完成")

    logger.info("服务初始化完成")


# =====================================================
# API路由
# =====================================================
@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    initialize_services()


@app.get("/", tags=["Root"])
async def root():
    """根路径"""
    return {
        "message": "微分方程RAG知识库API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"], response_model=HealthResponse)
async def health_check():
    """健康检查"""
    services = {
        "embedding": "ok" if embedding_model else "not_initialized",
        "milvus": "ok" if milvus_client else "not_initialized",
        "llm": "ok" if llm else "not_initialized",
        "rag_workflow": "ok" if rag_workflow else "not_initialized"
    }

    return HealthResponse(
        status="healthy" if all(v == "ok" for v in services.values()) else "degraded",
        version="1.0.0",
        services=services
    )


@app.post("/query", tags=["Query"], response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    查询接口

    Args:
        request: 查询请求

    Returns:
        查询响应
    """
    if not rag_workflow:
        raise HTTPException(status_code=503, detail="RAG工作流未初始化")

    try:
        # 使用高级工作流
        if request.use_advanced_workflow:
            result = rag_workflow.run(request.question)
        else:
            # 使用简单RAG
            result = rag_workflow.retriever.retrieve(request.question, top_k=request.top_k)
            context = "\n\n".join([doc["text"] for doc in result])

            from langchain_openai import ChatOpenAI 
            from langchain.schema import HumanMessage

            llm = ChatOpenAI(model='deepseek-chat',
             api_key='sk-0462bd395cdc4a86ac6abc8454124aed',
             base_url="https://api.deepseek.com/v1",
             temperature=0.0)
            prompt = f"基于以下文档回答问题：\n\n{context}\n\n问题：{request.question}\n\n回答："
            response = llm.invoke([HumanMessage(content=prompt)])
            result = {
                "question": request.question,
                "generation": response.content,
                "documents": result,
                "sources": result
            }

        return QueryResponse(
            question=result.get("question", request.question),
            answer=result.get("generation", "无法生成答案"),
            sources=result.get("sources", []),
            documents=result.get("documents", []),
            metadata={
                "route": result.get("route", "knowledge_base"),
                "answer_grade": result.get("answer_grade", "unknown"),
                "iteration": result.get("iteration", 0)
            }
        )

    except Exception as e:
        logger.error(f"查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", tags=["Documents"])
async def upload_documents(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chunk_size: Optional[int] = Form(512),
    chunk_overlap: Optional[int] = Form(128)
):
    """
    上传文档

    Args:
        background_tasks: 后台任务
        file: 上传的文件
        chunk_size: 块大小
        chunk_overlap: 块重叠大小

    Returns:
        上传结果
    """
    # 保存上传的文件
    upload_dir = Path("data/raw")
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / file.filename

    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        # 后台处理文档
        background_tasks.add_task(
            process_document,
            str(file_path),
            chunk_size,
            chunk_overlap
        )

        return {
            "status": "processing",
            "file_name": file.filename,
            "message": "文档已上传，正在处理中..."
        }

    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def process_document(file_path: str, chunk_size: int, chunk_overlap: int):
    """处理上传的文档"""
    try:
        logger.info(f"开始处理文档: {file_path}")

        # 解析文档
        parser = DocumentParser()
        documents = parser.parse_any(file_path)

        # 分块
        splitter = LaTeXAwareTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(documents)

        logger.info(f"文档处理完成，共 {len(chunks)} 个chunk")

        # TODO: 嵌入并存储到Milvus

    except Exception as e:
        logger.error(f"文档处理失败: {e}")


@app.get("/collection/stats", tags=["Collection"])
async def get_collection_stats():
    """获取集合统计信息"""
    if not milvus_client:
        raise HTTPException(status_code=503, detail="Milvus未初始化")

    try:
        stats = milvus_client.get_collection_stats()
        return stats

    except Exception as e:
        logger.error(f"获取集合统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/collection/create", tags=["Collection"])
async def create_collection(dimension: int = 1024):
    """创建集合"""
    if not milvus_client:
        raise HTTPException(status_code=503, detail="Milvus未初始化")

    try:
        milvus_client.create_collection(dimension=dimension)
        return {"status": "success", "message": f"集合 {settings.collection_name} 创建成功"}

    except Exception as e:
        logger.error(f"创建集合失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/collection/load", tags=["Collection"])
async def load_collection():
    """加载集合到内存"""
    if not milvus_client:
        raise HTTPException(status_code=503, detail="Milvus未初始化")

    try:
        milvus_client.load_collection()
        return {"status": "success", "message": "集合已加载"}

    except Exception as e:
        logger.error(f"加载集合失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/collection", tags=["Collection"])
async def delete_collection():
    """删除集合"""
    if not milvus_client:
        raise HTTPException(status_code=503, detail="Milvus未初始化")

    try:
        milvus_client.delete_collection()
        return {"status": "success", "message": f"集合 {settings.collection_name} 已删除"}

    except Exception as e:
        logger.error(f"删除集合失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
