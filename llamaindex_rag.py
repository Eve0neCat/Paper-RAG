# -*- coding: utf-8 -*-
r"""
llamaindex_rag.py
=================
RAG 业务中枢，整合：

- 大模型封装           : 阿里云通义千问（DashScope）。
- 需求 3 查询改写      : QueryProcessor 结合多轮历史，把口语化提问重写为学术检索词。
- 长短期记忆管理       : MemoryManager
    * 短期：内存中按会话维护最近 N 轮对话。
    * 长期：将历史问答写入独立 Milvus collection，按语义召回相关历史。
- 流式编排            : RAGEngine.answer_stream() 以事件流形式逐步产出
    （改写结果 / 路由章节 / 答案 token / 完成）。

行内公式输出请使用 \(...\)，行间公式使用 \[...\]（已在系统提示中约束）。
"""

import time
import uuid
from collections import defaultdict, deque
from typing import List, Dict, Iterator, Optional

from llama_index.core.schema import TextNode
from llama_index.core.llms import ChatMessage, MessageRole

import config
from chunk_embedding import VectorStoreManager
from hybrid_retrieval import HybridRetriever, RetrievalResult


# ============================================================================ #
# 1. 通义千问大模型封装
# ============================================================================ #
class QwenLLM:
    """对 DashScope 通义千问的薄封装，提供补全与流式补全。"""

    def __init__(self,
                 model_name: str = config.QWEN_CHAT_MODEL,
                 api_key: str = config.DASHSCOPE_API_KEY,
                 temperature: float = config.QWEN_TEMPERATURE):
        from llama_index.llms.dashscope import DashScope
        self.llm = DashScope(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
        )

    def complete(self, prompt: str) -> str:
        return self.llm.complete(prompt).text

    def stream_complete(self, prompt: str) -> Iterator[str]:
        last = ""
        for chunk in self.llm.stream_complete(prompt):
            # DashScope 部分版本返回累计文本，统一转换为增量 delta
            text = chunk.text or ""
            delta = chunk.delta if getattr(chunk, "delta", None) else text[len(last):]
            last = text
            if delta:
                yield delta


# ============================================================================ #
# 2. 查询改写（需求 3）
# ============================================================================ #
class QueryProcessor:
    """结合多轮历史，把口语化/含指代的问题改写为自洽的学术检索 query。"""

    REWRITE_PROMPT = (
        "你是学术检索查询改写器。请根据【历史对话】消解【当前问题】中的指代"
        "（如\"它\"\"这个方法\"\"上面提到的\"），补全省略信息，并改写为一句"
        "自洽、信息完整、适合学术文献检索的查询语句。\n"
        "要求：只输出改写后的查询本身，不要任何解释、前缀或引号。\n\n"
        "【历史对话】\n{history}\n\n"
        "【当前问题】\n{question}\n\n"
        "改写后的检索查询："
    )

    def __init__(self, llm: Optional[QwenLLM] = None):
        # 改写使用更轻量的模型，降低时延与成本
        self.llm = llm or QwenLLM(model_name=config.QWEN_REWRITE_MODEL)

    def rewrite(self, question: str, history: List[Dict]) -> str:
        if not history:
            return question
        history_text = "\n".join(
            f"{'用户' if h['role'] == 'user' else '助手'}：{h['content']}"
            for h in history[-config.SHORT_TERM_MAX_TURNS:]
        )
        prompt = self.REWRITE_PROMPT.format(history=history_text, question=question)
        try:
            rewritten = self.llm.complete(prompt).strip()
            return rewritten or question
        except Exception as e:
            print(f"[WARN] 查询改写失败，使用原问题：{e}")
            return question


# ============================================================================ #
# 3. 长短期记忆
# ============================================================================ #
class MemoryManager:
    """短期：会话级滑动窗口；长期：Milvus 语义记忆。"""

    def __init__(self, embed_model,
                 max_turns: int = config.SHORT_TERM_MAX_TURNS):
        self.embed_model = embed_model
        self.max_turns = max_turns
        # session_id -> deque[{"role","content"}]
        self._short: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_turns * 2))
        self._long_index = self._init_long_term()

    # ---------------------- 短期记忆 ---------------------- #
    def get_short_term(self, session_id: str) -> List[Dict]:
        return list(self._short[session_id])

    def add_turn(self, session_id: str, user_msg: str, assistant_msg: str):
        self._short[session_id].append({"role": "user", "content": user_msg})
        self._short[session_id].append({"role": "assistant", "content": assistant_msg})
        self._write_long_term(session_id, user_msg, assistant_msg)

    # ---------------------- 长期记忆 ---------------------- #
    def _init_long_term(self):
        from llama_index.core import VectorStoreIndex
        from llama_index.vector_stores.milvus import MilvusVectorStore
        try:
            vs = MilvusVectorStore(
                uri=config.MEMORY_MILVUS_URI,
                collection_name=config.MEMORY_COLLECTION,
                dim=config.EMBED_DIM,
                overwrite=False,
                similarity_metric="COSINE",
            )
            return VectorStoreIndex.from_vector_store(vs, embed_model=self.embed_model)
        except Exception as e:
            print(f"[WARN] 长期记忆初始化失败：{e}")
            return None

    def _write_long_term(self, session_id: str, user_msg: str, assistant_msg: str):
        if self._long_index is None:
            return
        node = TextNode(
            id_=f"mem::{uuid.uuid4().hex}",
            text=f"问：{user_msg}\n答：{assistant_msg}",
            metadata={
                "session_id": session_id,
                "timestamp": time.time(),
                "node_type": "memory",
            },
        )
        try:
            self._long_index.insert_nodes([node])
        except Exception as e:
            print(f"[WARN] 长期记忆写入失败：{e}")

    def recall_long_term(self, query: str,
                         top_k: int = config.LONG_TERM_TOP_K) -> List[str]:
        if self._long_index is None:
            return []
        try:
            retriever = self._long_index.as_retriever(similarity_top_k=top_k)
            hits = retriever.retrieve(query)
            return [h.node.get_content() for h in hits]
        except Exception:
            return []


# ============================================================================ #
# 4. RAG 引擎
# ============================================================================ #
class RAGEngine:
    SYSTEM_PROMPT = (
        "你是一名严谨的学术文献阅读助理。请仅依据【检索到的文献片段】回答问题，"
        "做到忠实、准确、有条理；若文献中没有足够信息，请明确说明而不要臆造。\n"
        "回答时请在关键结论处标注其来源片段编号（如[片段2]）。\n"
        "涉及公式时，行内公式使用 \\(...\\)，行间公式使用 \\[...\\]。"
    )

    ANSWER_TEMPLATE = (
        "{system}\n\n"
        "【可能相关的历史记忆】\n{memory}\n\n"
        "【最近对话】\n{history}\n\n"
        "【检索到的文献片段】\n{context}\n\n"
        "【用户问题】\n{question}\n\n"
        "请给出准确、有条理的回答："
    )

    def __init__(self,
                 vsm: Optional[VectorStoreManager] = None,
                 use_reranker: bool = True):
        # 共享同一个 VectorStoreManager（含已加载的 bge-m3 嵌入模型）
        self.vsm = vsm or VectorStoreManager().load()
        self.retriever = HybridRetriever(vsm=self.vsm, use_reranker=use_reranker)
        self.llm = QwenLLM()
        self.query_processor = QueryProcessor()
        self.memory = MemoryManager(embed_model=self.vsm.embed_model)

    # ---------------------------------------------------------------- #
    def _build_prompt(self, question: str, retrieval: RetrievalResult,
                      history: List[Dict], memories: List[str]) -> str:
        history_text = "\n".join(
            f"{'用户' if h['role'] == 'user' else '助手'}：{h['content']}"
            for h in history
        ) or "（无）"
        memory_text = "\n".join(f"- {m}" for m in memories) or "（无）"
        return self.ANSWER_TEMPLATE.format(
            system=self.SYSTEM_PROMPT,
            memory=memory_text,
            history=history_text,
            context=retrieval.context_text() or "（未检索到相关片段）",
            question=question,
        )

    # ---------------------------------------------------------------- #
    def answer_stream(self, session_id: str, question: str) -> Iterator[Dict]:
        """事件流式回答，逐步 yield 事件字典，供 SSE 转发。"""
        # 1) 短期历史 + 长期记忆召回
        history = self.memory.get_short_term(session_id)
        memories = self.memory.recall_long_term(question)

        # 2) 查询改写（需求 3）
        rewritten = self.query_processor.rewrite(question, history)
        yield {"type": "rewrite", "original": question, "rewritten": rewritten}

        # 3) 混合检索（需求 2/4/5/6）
        retrieval = self.retriever.retrieve(rewritten)
        yield {"type": "route", "sections": retrieval.routed_sections}
        yield {"type": "retrieval", "debug": retrieval.debug,
               "num_context": len(retrieval.nodes)}

        # 4) 组装 prompt 并流式生成
        prompt = self._build_prompt(rewritten, retrieval, history, memories)
        answer_parts: List[str] = []
        try:
            for delta in self.llm.stream_complete(prompt):
                answer_parts.append(delta)
                yield {"type": "token", "text": delta}
        except Exception as e:
            err = f"生成回答时出错：{e}"
            answer_parts.append(err)
            yield {"type": "token", "text": err}

        answer = "".join(answer_parts)

        # 5) 更新记忆
        self.memory.add_turn(session_id, question, answer)

        # 6) 收尾事件，附带可供前端展示的来源
        sources = [{
            "section_path": nw.node.metadata.get("section_path"),
            "node_type": nw.node.metadata.get("node_type"),
            "score": float(nw.score or 0.0),
        } for nw in retrieval.nodes]
        yield {"type": "done", "answer": answer, "sources": sources}

    def answer(self, session_id: str, question: str) -> Dict:
        """非流式封装（便于评估 / 测试调用）。"""
        result = {"answer": "", "rewritten": question, "sources": []}
        for ev in self.answer_stream(session_id, question):
            if ev["type"] == "rewrite":
                result["rewritten"] = ev["rewritten"]
            elif ev["type"] == "done":
                result["answer"] = ev["answer"]
                result["sources"] = ev["sources"]
        return result


if __name__ == "__main__":
    engine = RAGEngine()
    sid = "demo-session"
    for q in ["这篇论文的主要贡献是什么？", "它和之前的方法相比有什么优势？"]:
        print(f"\n=== 用户：{q} ===")
        for ev in engine.answer_stream(sid, q):
            if ev["type"] == "token":
                print(ev["text"], end="", flush=True)
            elif ev["type"] == "rewrite":
                print(f"[改写] {ev['rewritten']}")
        print()
