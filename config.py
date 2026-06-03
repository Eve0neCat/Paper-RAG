# -*- coding: utf-8 -*-
"""
config.py
=========
全局配置中心。所有模块（解析 / 向量化 / 检索 / RAG / 评估）均从此处读取配置，
避免散落的硬编码。优先从环境变量读取，便于部署时覆盖。
"""

import os
from pathlib import Path

# ----------------------------------------------------------------------------
# 基础路径
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"            # 原始上传文献
OUTPUT_DIR = BASE_DIR / "output"             # 解析中间结果（JSON / MD）
MEMORY_DIR = BASE_DIR / "memory_storage"     # 长期记忆向量落盘
TEMPLATE_DIR = BASE_DIR / "templates"        # 前端模板

for _d in (UPLOAD_DIR, OUTPUT_DIR, MEMORY_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# 阿里云通义千问（DashScope）大模型
# ----------------------------------------------------------------------------
# 申请地址: https://dashscope.console.aliyun.com/
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-***")
# 主回答模型 / 查询改写模型，可使用 qwen-max / qwen-plus / qwen-turbo
QWEN_CHAT_MODEL = os.getenv("QWEN_CHAT_MODEL", "qwen-plus")
QWEN_REWRITE_MODEL = os.getenv("QWEN_REWRITE_MODEL", "qwen-plus")
QWEN_TEMPERATURE = float(os.getenv("QWEN_TEMPERATURE", "0.1"))


# ----------------------------------------------------------------------------
# LlamaParse（高精度 PDF 解析）
# ----------------------------------------------------------------------------
# 申请地址: https://cloud.llamaindex.ai/
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "llx-Hsm0oQIfHCYIjsv3wvK2up9Xu8C36YGTJ0Wcld8JozZvI7Cl")


# ----------------------------------------------------------------------------
# 嵌入模型（bge-m3）与重排模型（bge-reranker-v2-m3）
# ----------------------------------------------------------------------------
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))          # bge-m3 输出维度为 1024
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cuda")          # cpu / cuda


# ----------------------------------------------------------------------------
# Milvus（Milvus Lite 本地模式）
# ----------------------------------------------------------------------------
# 使用 Milvus Lite：直接落盘到 .db 文件，无需额外部署服务端
MILVUS_URI = os.getenv("MILVUS_URI", str(OUTPUT_DIR / "milvus_paper.db"))
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "paper_chunks")
# 长期记忆使用独立的 collection / 库文件
MEMORY_MILVUS_URI = os.getenv("MEMORY_MILVUS_URI", str(MEMORY_DIR / "milvus_memory.db"))
MEMORY_COLLECTION = os.getenv("MEMORY_COLLECTION", "long_term_memory")


# ----------------------------------------------------------------------------
# 分块（父子节点）参数
# ----------------------------------------------------------------------------
PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", "1536"))  # 父节点最大字符
CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", "384"))     # 子节点目标字符
CHILD_CHUNK_OVERLAP = int(os.getenv("CHILD_CHUNK_OVERLAP", "64"))


# ----------------------------------------------------------------------------
# 检索参数
# ----------------------------------------------------------------------------
ROUTE_TOP_K = int(os.getenv("ROUTE_TOP_K", "3"))         # 路由命中的章节数
ROUTE_MAX_LEVEL = int(os.getenv("ROUTE_MAX_LEVEL", "2"))  # 路由只在 <= 该级别的标题节点中检索（一级/二级）
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "10"))
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "10"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))       # 重排后保留数

# 融合策略: "rrf" 或 "weighted"
FUSION_MODE = os.getenv("FUSION_MODE", "rrf")
RRF_K = int(os.getenv("RRF_K", "60"))                    # RRF 平滑常数
WEIGHT_BM25 = float(os.getenv("WEIGHT_BM25", "0.4"))     # 加权融合：BM25 权重
WEIGHT_VECTOR = float(os.getenv("WEIGHT_VECTOR", "0.6")) # 加权融合：向量权重

# Auto-Merge（上下文自愈）阈值：同一父节点下命中子节点比例超过该值则合并
AUTO_MERGE_RATIO = float(os.getenv("AUTO_MERGE_RATIO", "0.5"))
AUTO_MERGE_MIN_CHILDREN = int(os.getenv("AUTO_MERGE_MIN_CHILDREN", "2"))


# ----------------------------------------------------------------------------
# 记忆参数
# ----------------------------------------------------------------------------
SHORT_TERM_MAX_TURNS = int(os.getenv("SHORT_TERM_MAX_TURNS", "6"))   # 短期对话轮数
LONG_TERM_TOP_K = int(os.getenv("LONG_TERM_TOP_K", "3"))            # 召回长期记忆条数
