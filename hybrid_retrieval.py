# -*- coding: utf-8 -*-
"""
hybrid_retrieval.py
===================
混合检索核心，串联以下能力：

需求 2  路由检索 (Routing)   : 先在一级/二级标题(路由)节点中定位相关章节。
需求 4  深度召回             : 锁定章节后，在该范围内同时触发 BM25 + Milvus 向量检索。
需求 5  分数融合与重排       : RRF / 加权融合，再调用 bge-reranker-v2-m3 精排。
需求 6  上下文自愈 (AutoMerge): 命中子节点聚集于同一父节点时，回填为完整父节点内容。

对外主入口：HybridRetriever.retrieve(query) -> RetrievalResult
"""

import re
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable

from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.vector_stores import (
    MetadataFilters, MetadataFilter, FilterOperator, FilterCondition,
)

import config
from chunk_embedding import VectorStoreManager, DocStore


# ---------------------------------------------------------------------------- #
# 中英文混合分词器（BM25 用）。优先 jieba，缺失则回退到 CJK 单字 + 英文词。
# ---------------------------------------------------------------------------- #
def _build_tokenizer() -> Callable[[str], List[str]]:
    try:
        import jieba

        def tok(text: str) -> List[str]:
            text = text.lower()
            return [t for t in jieba.lcut(text) if t.strip()]
        return tok
    except Exception:
        cjk = re.compile(r"[\u4e00-\u9fff]")
        word = re.compile(r"[a-z0-9]+")

        def tok(text: str) -> List[str]:
            text = text.lower()
            tokens: List[str] = []
            for piece in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text):
                tokens.append(piece)
            return tokens
        return tok


_TOKENIZER = _build_tokenizer()


@dataclass
class RetrievalResult:
    query: str
    routed_sections: List[Dict] = field(default_factory=list)  # 命中的章节
    nodes: List[NodeWithScore] = field(default_factory=list)    # 最终上下文单元
    debug: Dict = field(default_factory=dict)

    def context_text(self) -> str:
        """拼接最终上下文，附带章节路径标注，供 LLM 生成答案。"""
        blocks = []
        for i, nw in enumerate(self.nodes, 1):
            path = nw.node.metadata.get("section_path", "")
            blocks.append(f"[片段{i} | {path}]\n{nw.node.get_content()}")
        return "\n\n".join(blocks)


class HybridRetriever:
    def __init__(self,
                 vsm: Optional[VectorStoreManager] = None,
                 docstore: Optional[DocStore] = None,
                 use_reranker: bool = True):
        self.vsm = vsm or VectorStoreManager().load()
        self.docstore = docstore or DocStore()
        self.parent_store: Dict[str, Dict] = self.docstore.load_all_parents()
        self.children_corpus: List[TextNode] = self.docstore.load_all_children()
        # child_id -> TextNode，便于按 id 取回
        self._child_by_id = {n.node_id: n for n in self.children_corpus}
        self._reranker = self._load_reranker() if use_reranker else None

    # ====================================================================== #
    # 需求 2：路由检索
    # ====================================================================== #
    def route(self, query: str, top_k: int = config.ROUTE_TOP_K) -> List[Dict]:
        retriever = self.vsm.route_index.as_retriever(similarity_top_k=top_k)
        hits = retriever.retrieve(query)
        sections = []
        for h in hits:
            sections.append({
                "route_section_id": h.node.metadata.get("route_section_id"),
                "section_title": h.node.metadata.get("section_title"),
                "section_path": h.node.metadata.get("section_path"),
                "score": float(h.score or 0.0),
            })
        return sections

    # ====================================================================== #
    # 需求 4：深度召回（BM25 + 向量），均限定在路由命中的章节范围内
    # ====================================================================== #
    def _route_filter(self, route_ids: List[str]) -> Optional[MetadataFilters]:
        if not route_ids:
            return None
        return MetadataFilters(
            filters=[MetadataFilter(
                key="route_section_id", value=route_ids, operator=FilterOperator.IN
            )],
            condition=FilterCondition.AND,
        )

    def vector_retrieve(self, query: str, route_ids: List[str],
                        top_k: int = config.VECTOR_TOP_K) -> List[NodeWithScore]:
        retriever = self.vsm.chunk_index.as_retriever(
            similarity_top_k=top_k,
            filters=self._route_filter(route_ids),
        )
        return retriever.retrieve(query)

    def bm25_retrieve(self, query: str, route_ids: List[str],
                      top_k: int = config.BM25_TOP_K) -> List[NodeWithScore]:
        # 仅在路由命中的章节内构建 BM25（论文规模可接受，按需即时构建）
        if route_ids:
            corpus = [n for n in self.children_corpus
                      if n.metadata.get("route_section_id") in set(route_ids)]
        else:
            corpus = self.children_corpus
        if not corpus:
            return []

        from llama_index.retrievers.bm25 import BM25Retriever
        bm25 = BM25Retriever.from_defaults(
            nodes=corpus,
            similarity_top_k=min(top_k, len(corpus)),
            tokenizer=_TOKENIZER,
        )
        return bm25.retrieve(query)

    # ====================================================================== #
    # 需求 5：分数融合
    # ====================================================================== #
    @staticmethod
    def _rrf_fuse(result_lists: List[List[NodeWithScore]],
                  k: int = config.RRF_K) -> List[NodeWithScore]:
        """Reciprocal Rank Fusion：分数 = Σ 1/(k + rank)。"""
        scores: Dict[str, float] = {}
        node_map: Dict[str, NodeWithScore] = {}
        for results in result_lists:
            for rank, nw in enumerate(results):
                nid = nw.node.node_id
                scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + rank + 1)
                node_map[nid] = nw
        fused = [NodeWithScore(node=node_map[nid].node, score=s)
                 for nid, s in scores.items()]
        fused.sort(key=lambda x: x.score, reverse=True)
        return fused

    @staticmethod
    def _weighted_fuse(bm25: List[NodeWithScore], vector: List[NodeWithScore],
                       w_bm25: float = config.WEIGHT_BM25,
                       w_vec: float = config.WEIGHT_VECTOR) -> List[NodeWithScore]:
        """加权融合：各路分数 min-max 归一化后加权求和。"""
        def norm(results: List[NodeWithScore]) -> Dict[str, float]:
            if not results:
                return {}
            vals = [float(r.score or 0.0) for r in results]
            lo, hi = min(vals), max(vals)
            span = (hi - lo) or 1e-9
            return {r.node.node_id: (float(r.score or 0.0) - lo) / span for r in results}

        bm_n, vec_n = norm(bm25), norm(vector)
        node_map: Dict[str, NodeWithScore] = {}
        for r in bm25 + vector:
            node_map[r.node.node_id] = r

        scores: Dict[str, float] = {}
        for nid in set(bm_n) | set(vec_n):
            scores[nid] = w_bm25 * bm_n.get(nid, 0.0) + w_vec * vec_n.get(nid, 0.0)

        fused = [NodeWithScore(node=node_map[nid].node, score=s)
                 for nid, s in scores.items()]
        fused.sort(key=lambda x: x.score, reverse=True)
        return fused

    def fuse(self, bm25: List[NodeWithScore], vector: List[NodeWithScore],
             mode: str = config.FUSION_MODE) -> List[NodeWithScore]:
        if mode == "weighted":
            return self._weighted_fuse(bm25, vector)
        return self._rrf_fuse([bm25, vector])

    # ====================================================================== #
    # 需求 5：重排（bge-reranker-v2-m3）
    # ====================================================================== #
    def _load_reranker(self):
        try:
            from FlagEmbedding import FlagReranker
            model = FlagReranker(config.RERANK_MODEL_NAME, use_fp16=False)
            return model
        except Exception as e:  # 依赖缺失时优雅降级（跳过重排）
            print(f"[WARN] 重排模型加载失败，将跳过 Rerank：{e}")
            return None

    def rerank(self, query: str, nodes: List[NodeWithScore],
               top_n: int = config.RERANK_TOP_N) -> List[NodeWithScore]:
        if not nodes:
            return []
        if self._reranker is None:
            return nodes[:top_n]
        pairs = [[query, nw.node.get_content()] for nw in nodes]
        raw = self._reranker.compute_score(pairs, normalize=True)
        if not isinstance(raw, list):
            raw = [raw]
        reranked = [NodeWithScore(node=nw.node, score=float(s))
                    for nw, s in zip(nodes, raw)]
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_n]

    # ====================================================================== #
    # 需求 6：上下文自愈（Auto-Merge）
    # ====================================================================== #
    def auto_merge(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """命中子节点聚集于同一父节点时，替换合并为完整父节点内容。"""
        # 按 parent_id 聚合命中的子节点
        groups: Dict[str, List[NodeWithScore]] = {}
        for nw in nodes:
            pid = nw.node.metadata.get("parent_id")
            if pid:
                groups.setdefault(pid, []).append(nw)

        merged: List[NodeWithScore] = []
        consumed_child_ids = set()

        for pid, hit_children in groups.items():
            parent = self.parent_store.get(pid)
            total = len(parent["child_ids"]) if parent else 0
            hit = len(hit_children)
            ratio = (hit / total) if total else 0.0

            should_merge = (
                parent is not None
                and hit >= config.AUTO_MERGE_MIN_CHILDREN
                and ratio >= config.AUTO_MERGE_RATIO
            )
            if should_merge:
                # 合并：用父节点全文替换这些子节点，分数取子节点最高分
                best = max(float(c.score or 0.0) for c in hit_children)
                parent_node = TextNode(
                    id_=pid,
                    text=parent["text"],
                    metadata={**parent["metadata"], "node_type": "merged_parent",
                              "merged_from": hit},
                )
                merged.append(NodeWithScore(node=parent_node, score=best))
                for c in hit_children:
                    consumed_child_ids.add(c.node.node_id)

        # 保留未被合并的子节点，维持原有顺序
        for nw in nodes:
            if nw.node.node_id not in consumed_child_ids:
                merged.append(nw)

        merged.sort(key=lambda x: x.score, reverse=True)
        return merged

    # ====================================================================== #
    # 顶层：完整检索流水线
    # ====================================================================== #
    def retrieve(self, query: str,
                 fusion_mode: str = config.FUSION_MODE,
                 enable_auto_merge: bool = True) -> RetrievalResult:
        # 1) 路由
        routed = self.route(query)
        route_ids = [s["route_section_id"] for s in routed if s["route_section_id"]]

        # 2) 深度召回（BM25 + 向量）
        bm25_hits = self.bm25_retrieve(query, route_ids)
        vector_hits = self.vector_retrieve(query, route_ids)

        # 3) 融合
        fused = self.fuse(bm25_hits, vector_hits, mode=fusion_mode)

        # 4) 重排
        reranked = self.rerank(query, fused)

        # 5) 自愈合并
        final = self.auto_merge(reranked) if enable_auto_merge else reranked

        return RetrievalResult(
            query=query,
            routed_sections=routed,
            nodes=final,
            debug={
                "route_ids": route_ids,
                "bm25_hits": len(bm25_hits),
                "vector_hits": len(vector_hits),
                "fused": len(fused),
                "reranked": len(reranked),
                "final": len(final),
                "fusion_mode": fusion_mode,
            },
        )


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "本文提出的方法的核心创新点是什么？"
    hr = HybridRetriever()
    res = hr.retrieve(q)
    print("路由命中章节：")
    for s in res.routed_sections:
        print(f"  - {s['section_path']}  ({s['score']:.3f})")
    print(f"\n调试：{res.debug}\n")
    print("最终上下文片段：")
    for nw in res.nodes:
        print(f"  [{nw.score:.3f}] {nw.node.metadata.get('section_path')}")
