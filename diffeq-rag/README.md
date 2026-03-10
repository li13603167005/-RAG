
# 微分方程概念领域 RAG 知识库系统

基于 GPT-4o 的高精度、可溯源的私有化 RAG 系统，专门针对微分方程、偏微分方程、Sobolev 空间、嵌入定理等领域构建。

## 项目概述

本系统旨在解决大语言模型在高阶微分方程概念领域存在的知识滞后与高幻觉问题。传统方案问答准确率仅 39%，本系统通过以下技术手段实现高精度知识检索与问答：

- **混合向量检索架构**：结合稠密向量（语义）和稀疏向量（关键词）
- **LangGraph 动态工作流**：实现 Corrective + Adaptive RAG
- **双层评估机制**：检索相关性评估 + 答案生成前评估

## 技术架构

### 核心技术栈

- **LangChain + LangGraph**: 工作流编排
- **Milvus**: 向量数据库（Docker 部署）
- **BGE-M3**: 嵌入向量生成
- **GPT-4o**: 大语言模型
- **RAGAS**: 评估框架
- **PyMuPDF + UnstructuredLoader**: 文档解析

### 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI API Layer                     │
├─────────────────────────────────────────────────────────────┤
│                    LangGraph Workflow Engine                │
│  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────────┐  │
│  │  Route  │→ │ Retrieve │→ │  Grade  │→ │   Generate   │  │
│  └─────────┘  └──────────┘  └─────────┘  └──────────────┘  │
│       │            │             │              │           │
│       │            │             │              ▼           │
│       │            │             │      ┌─────────────┐      │
│       │            │             └─────→│  Hallucina- │      │
│       │            │                    │   tion      │      │
│       │            │                    │   Grader    │      │
│       │            │                    └─────────────┘      │
│       │            │                                       │
│       ▼            ▼                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Hybrid Retrieval (RRF Fusion)              │   │
│  │    Dense Vector (BGE)  +  Sparse Vector (BM25)    │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                      Milvus Vector DB                       │
├─────────────────────────────────────────────────────────────┤
│              Document Processing Pipeline                   │
│  PDF/PPT/Word → Parser → LaTeX Splitter → Embedding       │
└─────────────────────────────────────────────────────────────┘
```

## 项目结构

```
diffeq-rag/
├── config/                    # 配置文件
│   ├── config.yaml           # 系统配置
│   └── prompt_templates.py   # Prompt 模板
├── data/                      # 数据目录
│   ├── raw/                  # 原始文档
│   ├── processed/            # 处理后的数据
│   └── eval_questions.json   # 评估问题
├── docker/                    # Docker 配置
│   └── docker-compose.yml   # Milvus 部署
├── src/                       # 源代码
│   ├── ingestion/           # 文档解析与嵌入
│   │   ├── parser.py        # PDF/PPT/Word 解析
│   │   ├── chunker.py       # LaTeX 感知分块
│   │   └── embedding.py     # BGE 嵌入向量
│   ├── retrieval/           # 检索层
│   │   ├── milvus_client.py # Milvus 客户端
│   │   └── ranker.py       # RRF 重排序
│   ├── graph/               # LangGraph 工作流
│   │   ├── state.py        # 状态定义
│   │   ├── nodes.py        # 工作流节点
│   │   └── workflow.py     # 工作流构建
│   ├── eval/               # 评估模块
│   │   └── ragas_eval.py  # RAGAS 评估
│   └── utils/              # 工具模块
├── notebooks/               # Jupyter notebooks
├── main.py                  # FastAPI 入口
├── Dockerfile               # 应用容器化
├── requirements.txt         # Python 依赖
└── README.md               # 项目文档
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd diffeq-rag

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制环境变量文件
cp .env.example .env

# 编辑 .env 文件，填入您的 API Key
OPENAI_API_KEY=your_openai_api_key_here
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

### 3. 启动 Milvus

```bash
# 使用 Docker Compose 启动 Milvus
cd docker
docker-compose up -d
```

### 4. 启动 API 服务

```bash
# 开发环境
python main.py

# 或使用 uvicorn
uvicorn main:app --reload --port 8000
```

### 5. 使用 API

```bash
# 查询示例
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是Sobolev空间？"}'
```

## 核心功能

### 文档解析与分块

- 支持 PDF、PPT、Word 等多种格式
- LaTeX 公式保护机制
- 语义分块策略

### 混合向量检索

```python
# 稠密向量 + 稀疏向量 + RRF 融合
results = retriever.hybrid_search(
    query="Sobolev空间定义",
    top_k=5,
    rrf_k=60
)
```

### LangGraph 工作流

```python
# 构建 RAG 工作流
workflow = create_rag_workflow(
    retriever=retriever,
    llm=llm,
    max_iterations=3,
    use_advanced=True
)

# 运行工作流
result = workflow.run("什么是Sobolev嵌入定理？")
```

### 评估系统

```python
# 使用 RAGAS 评估
evaluator = RAGEvaluator(llm=llm, embeddings=embeddings)
report = evaluator.evaluate_full_system(retriever, llm)
```

## API 接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/` | GET | 根路径 |
| `/health` | GET | 健康检查 |
| `/query` | POST | 问答接口 |
| `/upload` | POST | 文档上传 |
| `/collection/stats` | GET | 集合统计 |
| `/collection/create` | POST | 创建集合 |

## 部署

### Docker 部署

```bash
# 构建应用镜像
docker build -t diffeq-rag:latest .

# 运行应用
docker run -d -p 8000:8000 --env-file .env diffeq-rag:latest
```

### Docker Compose 完整部署

```yaml
# docker-compose.yml
version: '3.8'
services:
  milvus:
    image: milvusdb/milvus:v2.4.0
    # ...

  app:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - milvus
```

## 评估指标

系统支持以下评估指标：

- **Context Precision**: 检索精确度
- **Faithfulness**: 答案忠实度
- **Answer Relevancy**: 答案相关性
- **Context Recall**: 上下文召回率

## 扩展指南

### 添加新的数据源

1. 在 `src/ingestion/parser.py` 中添加新的解析器
2. 实现对应的分块策略
3. 更新 `main.py` 中的上传接口

### 自定义工作流

1. 修改 `src/graph/nodes.py` 中的节点逻辑
2. 在 `src/graph/workflow.py` 中重新构建图
3. 配置 Prompt 模板
