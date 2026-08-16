# -*- coding: utf-8 -*-
"""
FastAPI 知识库检索接口（阿里云百炼版）。

用法（在 000code 目录下运行）：
    python -m uvicorn llm.server.app:app --host 0.0.0.0 --port 8000

接口：
    POST /api/v1/kb/search
    请求体: {"query": ""}
    返回: 统一格式 {"code": "0", "status": "success", "msg": "", "data": {...}}
"""
import asyncio
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field
from llama_index.core.llms import ChatMessage, MessageRole

from llm.util import config
from llm.agent.chat_engine import SYSTEM_PROMPT, PROMPT_TEMPLATE, build_llm, format_context
from llm.core.bocha_search import bocha_search
from llm.core.hybrid_retriever import HybridRetriever
from llm.core.ingest import has_new_docs, ingest, load_nodes, load_or_create_index
from llm.util.logger_pathlib import logger

_hybrid: HybridRetriever | None = None
_update_lock = threading.Lock()


def _sync_ingest() -> bool:
    """执行一次增量入库，成功且有新增返回 True；失败返回 False。"""
    global _hybrid
    if not config.AUTO_UPDATE:
        return False
    try:
        with _update_lock:
            n = ingest()
        if n > 0:
            _hybrid = None  # 索引已变化，下次检索时重建
            logger.info("自动更新：新增 {} 个节点，检索器已重置", n)
        return n > 0
    except Exception as e:
        logger.error("自动更新入库失败: {}", e)
        return False


async def _auto_update_loop() -> None:
    """后台轮询 docs 目录，发现新文档自动增量入库。"""
    while True:
        await asyncio.sleep(config.AUTO_UPDATE_INTERVAL)
        try:
            if has_new_docs():
                await asyncio.to_thread(_sync_ingest)
        except Exception as e:
            logger.warning("自动更新轮询异常: {}", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时先增量入库一次，再启动后台自动更新轮询。"""
    if config.AUTO_UPDATE:
        try:
            await asyncio.to_thread(_sync_ingest)
        except Exception as e:
            logger.error("启动时自动入库失败: {}", e)
    task = asyncio.create_task(_auto_update_loop())
    logger.info("自动更新已启用：每 {} 秒轮询 {} 目录", config.AUTO_UPDATE_INTERVAL, config.DOCS_DIR)
    yield
    task.cancel()


app = FastAPI(title="RAG 知识库检索服务", version="1.0.0", lifespan=lifespan)


class KBQuery(BaseModel):
    """知识库检索请求体。"""
    query: str = Field(..., description="检索问题", examples=["小麦蚜虫是什么病害"])


def _get_hybrid() -> HybridRetriever | None:
    """懒加载混合检索器（索引不存在时返回 None）。

    每次调用前先轻量检测 docs 目录是否有新文档：有则先增量入库，
    保证用户手动放入文件后，无需等待轮询也能在下次查询时命中。
    """
    global _hybrid
    if _hybrid is not None and config.AUTO_UPDATE:
        try:
            if has_new_docs():
                _sync_ingest()  # 有新文档时内部会将 _hybrid 置 None，走下方重建
        except Exception as e:
            logger.warning("查询前自动更新检测异常: {}", e)
    if _hybrid is not None:
        return _hybrid
    if config.AUTO_UPDATE:
        try:
            _sync_ingest()
        except Exception as e:
            logger.warning("索引初始化前自动入库异常: {}", e)
    index, _ = load_or_create_index()
    if index is None:
        logger.error("索引不存在，请先执行 ingest 入库")
        return None
    nodes = load_nodes()
    _hybrid = HybridRetriever(index, nodes)
    logger.info("知识库检索器初始化完成，节点数: {}", len(nodes))
    return _hybrid


@app.get("/")
def index():
    """首页。"""
    return {"code": "0", "status": "success", "msg": "服务器加载完成", "data": None}


@app.get("/api/v1/kb/health")
def health():
    """健康检查。"""
    hybrid = _get_hybrid()
    ok = hybrid is not None
    return {
        "code": "0" if ok else "1",
        "status": "success" if ok else "fail",
        "msg": "" if ok else "知识库索引未初始化，请先 ingest 入库",
        "data": {"model": config.CHAT_MODEL, "embed_model": config.EMBED_MODEL} if ok else None,
    }


@app.post("/api/v1/kb/search")
def kb_search(req: KBQuery):
    """知识库检索问答接口。

    流程：
    1. 混合检索相关片段（知识库）；
    2. 博查联网搜索同问题，检索到片段时作为补充上下文附加，未检索到时直接用联网结果；
    3. 拼装上下文 -> 百炼大模型回答。
    """
    query = (req.query or "").strip()
    if not query:
        return {"code": "1", "status": "fail", "msg": "query 不能为空", "data": None}

    hybrid = _get_hybrid()
    if hybrid is None:
        return {"code": "1", "status": "fail", "msg": "知识库索引未初始化，请先 ingest 入库", "data": None}

    try:
        nodes = hybrid.retrieve(query)

        # 博查联网搜索（失败不阻塞主流程，降级为仅知识库回答）
        web_context = ""
        try:
            web_context = bocha_search(query)
        except Exception as e:
            logger.error("博查联网搜索失败: {}", e)

        if nodes:
            context = format_context(nodes)
            if web_context:
                context += "\n\n===== 博查联网搜索结果 =====\n" + web_context
        else:
            # 知识库未检索到：直接用博查联网搜索
            if web_context:
                context = "===== 博查联网搜索结果 =====\n" + web_context
            else:
                return {
                    "code": "0",
                    "status": "success",
                    "msg": "success",
                    "data": {"answer": "知识库中没有检索到相关内容，联网搜索也未获取到结果，请换一个问题试试。"},
                }

        user_prompt = PROMPT_TEMPLATE.format(context=context, query=query)
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=user_prompt),
        ]

        llm = build_llm()
        resp = llm.chat(messages)
        answer = (resp.message.content or "").strip()

        data = {"answer": answer}
        if nodes:
            sources = [
                {
                    "file_name": n.node.metadata.get("file_name", "-"),
                    "md5": n.node.metadata.get("md5", "-"),
                    "score": round(n.score, 4),
                }
                for n in nodes
            ]
            data["sources"] = sources
        if web_context:
            data["web_search"] = True
        return {
            "code": "0",
            "status": "success",
            "msg": "success",
            "data": data,
        }
    except Exception as e:
        logger.error("知识库检索失败: {}", e)
        return {"code": "2", "status": "fail", "msg": f"检索异常: {e}", "data": None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
