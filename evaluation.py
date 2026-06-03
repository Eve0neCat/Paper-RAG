# -*- coding: utf-8 -*-
r"""
evaluation.py
=============
检索评估流水线：自动构建评估集，并计算 Hit Rate@k 与 MRR。

流程：
1. 从已入库子节点中采样，调用通义千问为每个节点生成一个"该节点可独立回答"的问题，
   形成 (query, ground_truth_child_id, ground_truth_route_id) 评估样本。
2. 对每条 query 运行混合检索（关闭 Auto-Merge 以便对齐到子节点粒度），
   统计命中情况。
3. 同时评估"路由命中率"（路由是否定位到正确章节）。

指标：
- Hit Rate@k：top-k 中至少含 1 个相关结果的样本比例。
- MRR       ：首个相关结果排名倒数的平均值，\( \mathrm{MRR}=\frac{1}{N}\sum_i \frac{1}{rank_i} \)。
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional

import config
from chunk_embedding import VectorStoreManager, DocStore
from hybrid_retrieval import HybridRetriever


EVAL_PROMPT = (
    "下面是一篇学术论文中的一个片段。请基于该片段内容，提出一个具体的、"
    "仅凭该片段即可回答的问题。要求：只输出问题本身，不要答案、不要解释。\n\n"
    "片段：\n{chunk}\n\n问题："
)


# ============================================================================ #
# 1. 评估集构建
# ============================================================================ #
class EvalDatasetBuilder:
    def __init__(self, llm=None, docstore: Optional[DocStore] = None):
        self.docstore = docstore or DocStore()
        if llm is None:
            from llamaindex_rag import QwenLLM
            llm = QwenLLM(model_name=config.QWEN_REWRITE_MODEL)
        self.llm = llm

    def build(self, sample_size: int = 30, min_chars: int = 80,
              save_path: Optional[str] = None) -> List[Dict]:
        children = self.docstore.load_all_children()
        # 过滤过短片段，避免生成无意义问题
        candidates = [n for n in children if len(n.text) >= min_chars]
        random.shuffle(candidates)
        candidates = candidates[:sample_size]

        dataset: List[Dict] = []
        for node in candidates:
            try:
                question = self.llm.complete(
                    EVAL_PROMPT.format(chunk=node.text)
                ).strip().strip("？?。") + "？"
            except Exception as e:
                print(f"[WARN] 生成问题失败：{e}")
                continue
            dataset.append({
                "query": question,
                "gt_child_id": node.node_id,
                "gt_parent_id": node.metadata.get("parent_id"),
                "gt_route_id": node.metadata.get("route_section_id"),
                "gt_section_path": node.metadata.get("section_path"),
            })

        save_path = save_path or str(config.OUTPUT_DIR / "eval_dataset.json")
        Path(save_path).write_text(
            json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"评估集已构建：{len(dataset)} 条 -> {save_path}")
        return dataset


# ============================================================================ #
# 2. 指标计算
# ============================================================================ #
def hit_rate(ranks: List[Optional[int]]) -> float:
    """ranks 中非 None 表示命中（值为命中名次）。"""
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None) / len(ranks)


def mrr(ranks: List[Optional[int]]) -> float:
    if not ranks:
        return 0.0
    total = sum((1.0 / r) if r else 0.0 for r in ranks)
    return total / len(ranks)


# ============================================================================ #
# 3. 评估器
# ============================================================================ #
class RetrievalEvaluator:
    def __init__(self, retriever: Optional[HybridRetriever] = None):
        self.retriever = retriever or HybridRetriever(
            vsm=VectorStoreManager().load(), use_reranker=True
        )

    def evaluate(self, dataset: List[Dict], top_k: int = config.RERANK_TOP_N) -> Dict:
        retrieval_ranks: List[Optional[int]] = []  # 子节点级
        route_ranks: List[Optional[int]] = []      # 章节级

        for sample in dataset:
            # 关闭 Auto-Merge，保持子节点粒度以对齐 ground truth
            res = self.retriever.retrieve(sample["query"], enable_auto_merge=False)

            # —— 检索命中（child / parent 任一匹配视为命中）——
            rank = None
            for i, nw in enumerate(res.nodes[:top_k], 1):
                nid = nw.node.node_id
                pid = nw.node.metadata.get("parent_id")
                if nid == sample["gt_child_id"] or pid == sample["gt_parent_id"]:
                    rank = i
                    break
            retrieval_ranks.append(rank)

            # —— 路由命中 —— #
            r_rank = None
            for i, sec in enumerate(res.routed_sections, 1):
                if sec["route_section_id"] == sample["gt_route_id"]:
                    r_rank = i
                    break
            route_ranks.append(r_rank)

        metrics = {
            "num_samples": len(dataset),
            "top_k": top_k,
            "retrieval": {
                f"hit_rate@{top_k}": round(hit_rate(retrieval_ranks), 4),
                "mrr": round(mrr(retrieval_ranks), 4),
            },
            "routing": {
                f"hit_rate@{config.ROUTE_TOP_K}": round(hit_rate(route_ranks), 4),
                "mrr": round(mrr(route_ranks), 4),
            },
        }
        return metrics


def run_full_evaluation(sample_size: int = 30, top_k: int = config.RERANK_TOP_N) -> Dict:
    """一键评估：构建评估集 -> 运行检索 -> 输出指标。"""
    dataset = EvalDatasetBuilder().build(sample_size=sample_size)
    evaluator = RetrievalEvaluator()
    metrics = evaluator.evaluate(dataset, top_k=top_k)

    out = config.OUTPUT_DIR / "eval_metrics.json"
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    run_full_evaluation(sample_size=n)
