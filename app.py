# -*- coding: utf-8 -*-
"""
app.py
======
FastAPI 后端主服务。

接口：
- GET  /                : 返回交互式 Web UI（templates/index.html）
- POST /api/upload      : 上传 PDF -> LlamaParse 解析 -> 父子分块入库
- POST /api/chat        : SSE 流式问答（事件：rewrite/route/retrieval/token/done）
- GET  /api/documents   : 已入库文档列表
- POST /api/session     : 新建会话，返回 session_id

前端通过 fetch + ReadableStream 读取 SSE 流（兼容 POST 请求体）。
"""

import json
import uuid
import shutil
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from document_parsing import DocumentParser
from chunk_embedding import ingest_document, VectorStoreManager

app = FastAPI(title="Paper-RAG", description="学术文献智能检索问答系统")


# ---------------------------------------------------------------------------- #
# 全局状态：延迟加载重模型，入库后刷新检索器
# ---------------------------------------------------------------------------- #
class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.vsm: Optional[VectorStoreManager] = None
        self.engine = None            # RAGEngine，首个文档入库后初始化
        self.documents: list = []     # 已入库文档元信息
        self.has_index = False

    def get_vsm(self) -> VectorStoreManager:
        if self.vsm is None:
            self.vsm = VectorStoreManager()  # 仅加载嵌入模型，索引稍后 build/load
        return self.vsm

    def refresh_engine(self):
        """文档入库后（重新）构建 RAG 引擎，使检索器获取最新语料。"""
        from llamaindex_rag import RAGEngine
        if self.engine is None:
            self.engine = RAGEngine(vsm=self.get_vsm())
        else:
            # 复用已加载模型，仅刷新检索器语料 / 索引句柄
            from hybrid_retrieval import HybridRetriever
            self.get_vsm().load()
            self.engine.retriever = HybridRetriever(vsm=self.get_vsm())
        self.has_index = True

    def get_engine(self):
        if self.engine is None:
            if not self.has_index:
                raise HTTPException(status_code=400, detail="请先上传并解析至少一篇文献。")
            self.refresh_engine()
        return self.engine


state = AppState()


# ---------------------------------------------------------------------------- #
# 数据模型
# ---------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    session_id: str
    question: str


# ---------------------------------------------------------------------------- #
# 路由：前端页面
# ---------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = config.TEMPLATE_DIR / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------- #
# 路由：新建会话
# ---------------------------------------------------------------------------- #
@app.post("/api/session")
async def new_session():
    return {"session_id": uuid.uuid4().hex}


# ---------------------------------------------------------------------------- #
# 路由：上传 + 解析 + 入库
# ---------------------------------------------------------------------------- #
@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件。")

    # 1) 保存上传文件
    dest = config.UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    def _process():
        # 2) 解析（LlamaParse + 标题层级）
        parser = DocumentParser()
        parsed = parser.parse(str(dest))
        # 3) 入库（父子分块 + bge-m3 + Milvus）。多文档场景下不覆盖既有库。
        overwrite = not state.has_index
        stats = ingest_document(parsed["sections"], vsm=state.get_vsm(),
                                overwrite=overwrite)
        return parsed, stats

    with state.lock:
        try:
            parsed, stats = _process()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"解析或入库失败：{e}")

        doc_meta = {
            "doc_id": parsed["doc_id"],
            "source": parsed["source"],
            "num_sections": parsed["num_sections"],
            **stats,
        }
        state.documents.append(doc_meta)
        state.refresh_engine()

    return {
        "status": "ok",
        "document": doc_meta,
        "heading_tree": parsed["heading_tree"],
    }


@app.get("/api/documents")
async def documents():
    return {"documents": state.documents}


# ---------------------------------------------------------------------------- #
# 路由：SSE 流式问答
# ---------------------------------------------------------------------------- #
@app.post("/api/chat")
async def chat(req: ChatRequest):
    engine = state.get_engine()

    def event_generator():
        try:
            for ev in engine.answer_stream(req.session_id, req.question):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:
            err = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 关闭 Nginx 缓冲，保证逐字下发
        },
    )


@app.get("/api/health")
async def health():
    return {"status": "ok", "has_index": state.has_index,
            "num_documents": len(state.documents)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
