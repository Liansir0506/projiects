# -*- coding: utf-8 -*-
"""
混合检索模块：向量检索（Chroma）+ 关键词检索（BM25）按 ALPHA 权重融合。

融合方式：
1. 两个检索器各自取 TOP_K*3 个候选；
2. 各自内部归一化（相对分数 = score / max_score，消除量纲差异）；
3. 融合分 = ALPHA * 向量分 + (1 - ALPHA) * BM25 分；
4. 按融合分排序取 TOP_K 返回。
"""
from typing import List, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.retrievers.bm25 import BM25Retriever

from llm.util import config
from llm.util.logger_pathlib import logger

try:
    import jieba


    def _tokenizer(text: str) -> List[str]:
        """中文分词（jieba 搜索模式）。"""
        return list(jieba.cut_for_search(text))


    logger.info("BM25 使用 jieba 中文分词")
except ImportError:
    def _tokenizer(text: str) -> List[str]:
        return text.split()


    logger.warning("未安装 jieba，BM25 使用空白分词（中文效果较差）")


class HybridRetriever(BaseRetriever):
    """向量 + BM25 加权融合检索器。"""

    def __init__(
            self,
            index: VectorStoreIndex,
            nodes: Optional[list] = None,
            alpha: Optional[float] = None,
            top_k: Optional[int] = None,
    ):
        self._index = index
        self._alpha = config.ALPHA if alpha is None else alpha
        self._top_k = config.TOP_K if top_k is None else top_k
        self._candidate_k = max(self._top_k * 3, 10)

        self._vector_retriever = index.as_retriever(
            similarity_top_k=self._candidate_k
        )
        if nodes is None:
            nodes = list(index.docstore.docs.values())
        if not nodes:
            logger.warning("nodes 为空，BM25 检索将无结果")
        self._bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=self._candidate_k,
            tokenizer=_tokenizer,
        )
        logger.info(
            "混合检索就绪: ALPHA={} TOP_K={} 候选={}", self._alpha, self._top_k, self._candidate_k
        )

    @staticmethod
    def _normalize(scores: dict) -> dict:
        """相对归一化：score / max_score，量纲统一到 [0, 1]。"""
        if not scores:
            return {}
        mx = max(scores.values()) or 1.0
        return {k: v / mx for k, v in scores.items()}

    def _retrieve(self, query_bundle: QueryBundle):
        query = query_bundle.query_str
        vec_nodes = self._vector_retriever.retrieve(query)
        bm25_nodes = self._bm25_retriever.retrieve(query)

        node_map = {n.node.node_id: n.node for n in vec_nodes + bm25_nodes}
        vec_scores = self._normalize({n.node.node_id: n.score for n in vec_nodes})
        bm25_scores = self._normalize({n.node.node_id: n.score for n in bm25_nodes})

        # ALPHA 加权融合
        fused ={}
        for nid in node_map:
            # 混合排序
            fused[nid] = (
                    self._alpha * vec_scores.get(nid, 0.0)
                    + (1.0 - self._alpha) * bm25_scores.get(nid, 0.0)
            )

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[: self._top_k]
        return [NodeWithScore(node=node_map[nid], score=s) for nid, s in ranked]
