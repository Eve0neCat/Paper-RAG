# -*- coding: utf-8 -*-
"""
chunk_embedding.py
==================
父子分块（Parent-Child）切分 + bge-m3 向量化 + Milvus 本地存储封装。

核心思想（对应需求 1 与需求 4 的基础设施）：
- 章节(section) -> 父节点(Parent)：保留较大上下文，供"上下文自愈/Auto-Merge"回填。
- 父节点 -> 子节点(Child)：细粒度切分，作为向量库与 BM25 的检索单元。
- 路由节点(Route)：仅由 <= ROUTE_MAX_LEVEL 的标题（一级/二级）聚合而成，
  单独建一个 Milvus collection，供"路由检索"先行定位章节。

每个子节点 metadata 关键字段：
    doc_id, section_id, section_title, section_path,
    parent_id, route_section_id, level, child_order

落盘：
- 向量 -> Milvus（Milvus Lite .db 文件）
- 父节点全文 -> docstore JSON（OUTPUT_DIR/<doc_id>.parents.json）
- 子节点语料 -> OUTPUT_DIR/<doc_id>.children.json（供 BM25 重建）
"""

import json
import uuid
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.node_parser import SentenceSplitter

import config


# ============================================================================ #
# 1. 父子分块
# ============================================================================ #
class HierarchicalChunker:
    """把解析得到的 sections 切成 父节点 / 子节点 / 路由节点。"""

    def __init__(self,
                 parent_chunk_size: int = config.PARENT_CHUNK_SIZE,
                 child_chunk_size: int = config.CHILD_CHUNK_SIZE,
                 child_overlap: int = config.CHILD_CHUNK_OVERLAP,
                 route_max_level: int = config.ROUTE_MAX_LEVEL):
        self.route_max_level = route_max_level
        self.parent_splitter = SentenceSplitter(
            chunk_size=parent_chunk_size, chunk_overlap=0
        )
        self.child_splitter = SentenceSplitter(
            chunk_size=child_chunk_size, chunk_overlap=child_overlap
        )

    def build(self, sections: List[Dict]) -> Tuple[List[TextNode], Dict[str, Dict], List[TextNode]]:
        """返回 (child_nodes, parent_store, route_nodes)。"""
        child_nodes: List[TextNode] = []
        parent_store: Dict[str, Dict] = {}     # parent_id -> {text, metadata, child_ids}
        route_index: Dict[str, Dict] = {}       # route_section_id -> 聚合信息

        current_route_id: Optional[str] = None  # 最近的一/二级标题作为路由归属

        for sec in sections:
            # 更新当前路由归属：遇到 <= route_max_level 的标题即切换
            if sec["level"] <= self.route_max_level:
                current_route_id = sec["section_id"]
                route_index[current_route_id] = {
                    "section_id": current_route_id,
                    "doc_id": sec["doc_id"],
                    "title": sec["title"],
                    "path": sec["path"],
                    "level": sec["level"],
                    "titles": [sec["title"]],   # 聚合自身 + 子孙标题
                    "preview": "",
                }
            route_id = current_route_id or sec["section_id"]

            # 把子孙标题汇入路由聚合，丰富路由语义信号
            if current_route_id and current_route_id in route_index:
                if sec["level"] > self.route_max_level:
                    route_index[current_route_id]["titles"].append(sec["title"])
                if not route_index[current_route_id]["preview"] and sec["content"]:
                    route_index[current_route_id]["preview"] = sec["content"][:300]

            content = (sec.get("content") or "").strip()
            if not content:
                continue

            # 1) 章节 -> 父节点（过长则继续切，保证父节点不至于超过 LLM 友好上限）
            parent_chunks = self.parent_splitter.split_text(content)
            for p_i, p_text in enumerate(parent_chunks):
                parent_id = f"{sec['section_id']}::p{p_i:02d}"
                parent_meta = {
                    "doc_id": sec["doc_id"],
                    "section_id": sec["section_id"],
                    "section_title": sec["title"],
                    "section_path": " > ".join(sec["path"]),
                    "route_section_id": route_id,
                    "level": sec["level"],
                }
                child_ids: List[str] = []

                # 2) 父节点 -> 子节点
                child_chunks = self.child_splitter.split_text(p_text)
                for c_i, c_text in enumerate(child_chunks):
                    child_id = f"{parent_id}::c{c_i:03d}"
                    node = TextNode(
                        id_=child_id,
                        text=c_text,
                        metadata={
                            **parent_meta,
                            "parent_id": parent_id,
                            "child_order": c_i,
                            "node_type": "child",
                        },
                        # 关联父节点，便于 Auto-Merge 回溯
                        relationships={
                            NodeRelationship.PARENT: RelatedNodeInfo(node_id=parent_id)
                        },
                    )
                    # 避免把全部 metadata 注入向量文本，控制干扰
                    node.excluded_embed_metadata_keys = [
                        "parent_id", "route_section_id", "child_order",
                        "node_type", "doc_id", "level",
                    ]
                    node.excluded_llm_metadata_keys = ["parent_id", "child_order", "node_type"]
                    child_nodes.append(node)
                    child_ids.append(child_id)

                parent_store[parent_id] = {
                    "parent_id": parent_id,
                    "text": p_text,
                    "metadata": parent_meta,
                    "child_ids": child_ids,
                }

        # 路由节点：标题 + 路径 + 子孙标题 + 正文预览，提供丰富路由语义
        route_nodes: List[TextNode] = []
        for rid, info in route_index.items():
            route_text = (
                f"{' > '.join(info['path'])}\n"
                f"{' ; '.join(info['titles'])}\n"
                f"{info['preview']}"
            ).strip()
            route_nodes.append(TextNode(
                id_=f"route::{rid}",
                text=route_text,
                metadata={
                    "doc_id": info["doc_id"],
                    "route_section_id": rid,
                    "section_title": info["title"],
                    "section_path": " > ".join(info["path"]),
                    "level": info["level"],
                    "node_type": "route",
                },
            ))

        return child_nodes, parent_store, route_nodes


# ============================================================================ #
# 2. 向量化 + Milvus 存储
# ============================================================================ #
class VectorStoreManager:
    """封装 bge-m3 嵌入与 Milvus（Lite）双 collection（子节点库 / 路由库）。"""

    def __init__(self,
                 milvus_uri: str = config.MILVUS_URI,
                 chunk_collection: str = config.MILVUS_COLLECTION,
                 embed_dim: int = config.EMBED_DIM):
        self.milvus_uri = milvus_uri
        self.chunk_collection = chunk_collection
        self.route_collection = f"{chunk_collection}_route"
        self.embed_dim = embed_dim

        self.embed_model = self._load_embed_model()

        # 延迟创建：首次建库时实例化
        self._chunk_index = None
        self._route_index = None

    # -------------------------------------------------------------- #
    def _load_embed_model(self):
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.core import Settings

        embed = HuggingFaceEmbedding(
            model_name=config.EMBED_MODEL_NAME,   # BAAI/bge-m3
            device=config.EMBED_DEVICE,
            max_length=512,
            normalize=True,
        )
        Settings.embed_model = embed
        return embed

    def _make_vector_store(self, collection_name: str, overwrite: bool):
        from llama_index.vector_stores.milvus import MilvusVectorStore
        return MilvusVectorStore(
            uri=self.milvus_uri,
            collection_name=collection_name,
            dim=self.embed_dim,
            overwrite=overwrite,
            # 余弦相似度，配合 bge-m3 归一化向量
            similarity_metric="COSINE",
        )

    # -------------------------------------------------------------- #
    def build(self,
              child_nodes: List[TextNode],
              route_nodes: List[TextNode],
              overwrite: bool = True):
        """将子节点与路由节点分别写入两个 Milvus collection 并建索引。"""
        from llama_index.core import VectorStoreIndex, StorageContext

        # 子节点向量库
        chunk_vs = self._make_vector_store(self.chunk_collection, overwrite)
        chunk_ctx = StorageContext.from_defaults(vector_store=chunk_vs)
        self._chunk_index = VectorStoreIndex(
            child_nodes, storage_context=chunk_ctx, embed_model=self.embed_model
        )

        # 路由向量库
        route_vs = self._make_vector_store(self.route_collection, overwrite)
        route_ctx = StorageContext.from_defaults(vector_store=route_vs)
        self._route_index = VectorStoreIndex(
            route_nodes, storage_context=route_ctx, embed_model=self.embed_model
        )
        return self

    def load(self):
        """从已有 Milvus 文件加载索引（服务重启后复用）。"""
        from llama_index.core import VectorStoreIndex

        chunk_vs = self._make_vector_store(self.chunk_collection, overwrite=False)
        self._chunk_index = VectorStoreIndex.from_vector_store(
            chunk_vs, embed_model=self.embed_model
        )
        route_vs = self._make_vector_store(self.route_collection, overwrite=False)
        self._route_index = VectorStoreIndex.from_vector_store(
            route_vs, embed_model=self.embed_model
        )
        return self

    # -------------------------------------------------------------- #
    @property
    def chunk_index(self):
        if self._chunk_index is None:
            raise RuntimeError("chunk_index 尚未构建，请先调用 build() 或 load()")
        return self._chunk_index

    @property
    def route_index(self):
        if self._route_index is None:
            raise RuntimeError("route_index 尚未构建，请先调用 build() 或 load()")
        return self._route_index


# ============================================================================ #
# 3. 父节点 / 子语料 持久化（供 Auto-Merge 与 BM25 重建）
# ============================================================================ #
class DocStore:
    """轻量级父节点与子语料的本地持久化（JSON）。"""

    def __init__(self, output_dir: Path = config.OUTPUT_DIR):
        self.output_dir = Path(output_dir)

    def save(self, doc_id: str,
             parent_store: Dict[str, Dict],
             child_nodes: List[TextNode]):
        (self.output_dir / f"{doc_id}.parents.json").write_text(
            json.dumps(parent_store, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        children = [{
            "id": n.node_id,
            "text": n.text,
            "metadata": n.metadata,
        } for n in child_nodes]
        (self.output_dir / f"{doc_id}.children.json").write_text(
            json.dumps(children, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load_parents(self, doc_id: str) -> Dict[str, Dict]:
        p = self.output_dir / f"{doc_id}.parents.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def load_children(self, doc_id: str) -> List[TextNode]:
        p = self.output_dir / f"{doc_id}.children.json"
        if not p.exists():
            return []
        raw = json.loads(p.read_text(encoding="utf-8"))
        nodes = []
        for r in raw:
            n = TextNode(id_=r["id"], text=r["text"], metadata=r["metadata"])
            nodes.append(n)
        return nodes

    def load_all_parents(self) -> Dict[str, Dict]:
        merged: Dict[str, Dict] = {}
        for p in self.output_dir.glob("*.parents.json"):
            merged.update(json.loads(p.read_text(encoding="utf-8")))
        return merged

    def load_all_children(self) -> List[TextNode]:
        nodes: List[TextNode] = []
        for p in self.output_dir.glob("*.children.json"):
            for r in json.loads(p.read_text(encoding="utf-8")):
                nodes.append(TextNode(id_=r["id"], text=r["text"], metadata=r["metadata"]))
        return nodes


def ingest_document(sections: List[Dict],
                    vsm: Optional[VectorStoreManager] = None,
                    overwrite: bool = True) -> Dict:
    """一站式入库：分块 -> 向量化 -> Milvus + DocStore。

    返回统计信息供上层（app.py）使用。
    """
    doc_id = sections[0]["doc_id"]
    chunker = HierarchicalChunker()
    child_nodes, parent_store, route_nodes = chunker.build(sections)

    vsm = vsm or VectorStoreManager()
    vsm.build(child_nodes, route_nodes, overwrite=overwrite)

    DocStore().save(doc_id, parent_store, child_nodes)

    return {
        "doc_id": doc_id,
        "num_children": len(child_nodes),
        "num_parents": len(parent_store),
        "num_routes": len(route_nodes),
    }
