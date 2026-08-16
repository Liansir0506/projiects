# -*- coding: utf-8 -*-
"""
文档入库模块：解析 -> MD5 去重存储 -> 分块 -> Chroma 向量索引。

流程：
1. 扫描 DOCS_DIR 下所有支持的文档；
2. 每个文件先经 md5_store 去重（重复文件跳过，不再解析）；
3. 解析为 Document，SentenceSplitter 分块；
4. 写入 Chroma（PERSIST_DIR 持久化）；
5. 缓存全量 nodes（nodes.pkl）供 BM25 关键词检索使用。
"""
import pickle
from pathlib import Path

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from llm.util import config, md5_store
from llm.chroma_loader import document_loader
from llm.util.logger_pathlib import logger

_NODES_CACHE = "nodes.pkl"

_COLLECTION_NAME = "rag_docs"


def build_embed_model() -> OpenAIEmbedding:
    """阿里云百炼 text-embedding-v3（OpenAI 兼容端点）。

    model 用枚举内的占位名通过构造校验，model_name 覆盖为百炼实际模型名。
    embed_batch_size=10：百炼兼容端点限制单次 batch <= 10。
    """
    return OpenAIEmbedding(
        model="text-embedding-ada-002",
        model_name=config.EMBED_MODEL,
        api_key=config.DASHSCOPE_KEY,
        api_base=config.EMBED_BASE_URL,
        dimensions=config.EMBEDDING_DIM,
        embed_batch_size=10,
    )


def _get_vector_store():
    config.PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(config.PERSIST_DIR))
    chroma_collection = chroma_client.get_or_create_collection(_COLLECTION_NAME)
    return ChromaVectorStore(chroma_collection=chroma_collection)


def load_or_create_index(nodes=None):
    """
    加载已有索引；传入 nodes 时重建（写库）。
    返回 (index, vector_store)；索引不存在时 index 为 None。
    """
    vector_store = _get_vector_store()
    embed_model = build_embed_model()
    if nodes:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(
            nodes, storage_context=storage_context, embed_model=embed_model
        )
        return index, vector_store
    try:
        index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=embed_model
        )
        return index, vector_store
    except Exception as e:
        logger.warning("加载已有索引失败（可能尚未入库）: {}", e)
        return None, vector_store


# ---------- nodes 缓存（供 BM25 全量检索） ----------

def load_nodes() -> list:
    cache = config.PERSIST_DIR / _NODES_CACHE
    if cache.exists():
        try:
            return pickle.loads(cache.read_bytes())
        except Exception as e:
            logger.warning("nodes 缓存读取失败: {}", e)
    return []


def _cache_nodes(nodes: list) -> None:
    """合并旧缓存与新增 nodes 后覆盖写回（增量入库场景）。"""
    cache = config.PERSIST_DIR / _NODES_CACHE
    merged = {n.node_id: n for n in load_nodes()}
    for n in nodes:
        merged[n.node_id] = n
    cache.write_bytes(pickle.dumps(list(merged.values())))
    logger.info("nodes 缓存已更新，共 {} 个节点", len(merged))


# ---------- 入库主流程 ----------

def has_new_docs(docs_dir=None) -> bool:
    """轻量检测 docs 目录中是否存在尚未入库的新文档（只算 MD5，不解析）。"""
    docs_dir = Path(docs_dir or config.DOCS_DIR)
    files = document_loader.scan_docs(docs_dir)
    if not files:
        return False
    records = md5_store.all_records()
    for f in files:
        if md5_store.file_md5(f) not in records:
            return True
    return False


def ingest(docs_dir=None) -> int:
    """扫描并入库文档，返回新增节点数。"""
    docs_dir = Path(docs_dir or config.DOCS_DIR)
    config.ensure_dirs()

    files = document_loader.scan_docs(docs_dir)
    if not files:
        logger.warning("没有找到可入库的文档: {}", docs_dir)
        return 0

    splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    documents, stored_count, skipped_count = [], 0, 0

    for f in files:
        rec = md5_store.store_file(f)
        if rec["existed"]:
            skipped_count += 1
            continue
        doc = document_loader.load_file(f)
        if doc is None:
            continue
        doc.metadata["md5"] = rec["md5"]
        documents.append(doc)
        stored_count += 1

    if not documents:
        logger.info("无新增文档，索引保持不变（跳过 {} 个已入库文件）", skipped_count)
        return 0

    nodes = splitter.get_nodes_from_documents(documents)
    load_or_create_index(nodes)
    _cache_nodes(nodes)

    logger.info(
        "入库完成: 新增 {} 个文档 / {} 个节点，跳过 {} 个重复文件",
        stored_count, len(nodes), skipped_count,
    )
    return len(nodes)


if __name__ == "__main__":
    n = ingest()
    print(f"新增节点数: {n}")
