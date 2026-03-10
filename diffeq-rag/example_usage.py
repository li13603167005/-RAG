"""
使用示例
演示如何使用微分方程RAG系统
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.parser import DocumentParser, LaTeXAwareTextSplitter
from src.ingestion.embedding import create_embedding_model
from src.retrieval.milvus_client import create_milvus_client, HybridRetriever
from src.graph.workflow import create_rag_workflow
from src.utils.config import Config


def example_1_parse_document():
    """示例1：解析PDF文档"""
    print("=" * 50)
    print("示例1：解析PDF文档")
    print("=" * 50)

    parser = DocumentParser()

    # 解析PDF（需要实际文件）
    # documents = parser.parse_pdf("data/raw/sample.pdf")
    # print(f"解析完成，共 {len(documents)} 个元素")

    print("文档解析功能就绪")
    print()


def example_2_chunk_text():
    """示例2：文本分块"""
    print("=" * 50)
    print("示例2：LaTeX感知文本分块")
    print("=" * 50)

    # 测试文本
    text = """
    定义 (Sobolev空间): Sobolev空间 $H^s(\\Omega)$ 是函数及其直到阶数 $s$ 的弱导数都属于 $L^2(\\Omega)$ 的函数空间。

    范数定义为：$$\\|u\\|_{H^s(\\Omega)}^2 = \\sum_{|\\alpha| \\leq s} \\int_\\Omega |D^\\alpha u|^2 dx$$

    定理 (Sobolev嵌入定理): 当 $s > n/2$ 时，$H^s(\\Omega)$ 嵌入到 $C^{0,\\gamma}(\\Omega)$，其中 $\\gamma = s - n/2$。
    """

    splitter = LaTeXAwareTextSplitter(
        chunk_size=512,
        chunk_overlap=128
    )

    chunks = splitter.split_text(text)
    print(f"分块完成，共 {len(chunks)} 个chunk")
    for i, chunk in enumerate(chunks[:2]):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk[:200] + "...")

    print()


def example_3_embedding():
    """示例3：生成嵌入向量"""
    print("=" * 50)
    print("示例3：生成嵌入向量")
    print("=" * 50)

    # 创建嵌入模型（需要GPU或足够内存）
    # embedding_model = create_embedding_model(
    #     model_name="BAAI/bge-m3",
    #     device="cpu"
    # )

    # 生成查询嵌入
    # query = "什么是Sobolev空间？"
    # embedding = embedding_model.embed_query(query)
    # print(f"稠密向量维度: {len(embedding['dense_vector'])}")
    # print(f"稀疏向量词数: {len(embedding['sparse_vector'])}")

    print("嵌入模型功能就绪（需要安装模型权重）")
    print()


def example_4_milvus_retrieval():
    """示例4：Milvus检索"""
    print("=" * 50)
    print("示例4：Milvus混合检索")
    print("=" * 50)

    # 连接Milvus
    # client = create_milvus_client(
    #     host="localhost",
    #     port=19530
    # )
    # client.connect()

    # 创建集合
    # client.create_collection(dimension=1024)

    # 插入数据（示例）
    # texts = ["Sobolev空间是...", "嵌入定理表明..."]
    # dense_vectors = [[0.1] * 1024, [0.2] * 1024]
    # sparse_vectors = [{0: 0.5, 1: 0.3}, {2: 0.4}]
    # client.insert(texts, dense_vectors, sparse_vectors)

    # 检索
    # retriever = HybridRetriever(client, embedding_model)
    # results = retriever.retrieve("Sobolev空间定义")

    print("Milvus检索功能就绪（需要Milvus服务运行）")
    print()


def example_5_rag_workflow():
    """示例5：RAG工作流"""
    print("=" * 50)
    print("示例5：LangGraph RAG工作流")
    print("=" * 50)

    # 初始化组件（需要实际配置）
    # embedding_model = create_embedding_model()
    # milvus_client = create_milvus_client()
    # retriever = HybridRetriever(milvus_client, embedding_model)
    #
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-4o")

    # 创建工作流
    # workflow = create_rag_workflow(
    #     retriever=retriever,
    #     llm=llm,
    #     max_iterations=3,
    #     use_advanced=True
    # )
    #
    # # 构建图
    # workflow.build_graph()
    #
    # # 运行
    # result = workflow.run("什么是Sobolev嵌入定理？")
    # print(f"答案: {result['generation']}")
    # print(f"来源数量: {len(result['sources'])}")

    print("RAG工作流功能就绪（需要完整配置）")
    print()


def example_6_evaluation():
    """示例6：系统评估"""
    print("=" * 50)
    print("示例6：RAGAS评估")
    print("=" * 50)

    from src.eval.ragas_eval import DEFAULT_EVAL_QUESTIONS

    print("预定义评估问题：")
    for i, q in enumerate(DEFAULT_EVAL_QUESTIONS[:3]):
        print(f"\n问题 {i+1}: {q.question}")
        print(f"标准答案: {q.ground_truth[:50]}...")

    print("\n评估功能就绪")
    print()


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("微分方程RAG系统 - 使用示例")
    print("=" * 60 + "\n")

    # 运行各个示例
    example_1_parse_document()
    example_2_chunk_text()
    example_3_embedding()
    example_4_milvus_retrieval()
    example_5_rag_workflow()
    example_6_evaluation()

    print("=" * 60)
    print("所有示例运行完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
