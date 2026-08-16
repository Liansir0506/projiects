# -*- coding: utf-8 -*-
"""
博查 Web Search 联网搜索模块。

调用博查 Web Search API：
    POST https://api.bochaai.com/v1/web-search
    Header: Authorization: Bearer {BOCHA_API_KEY}
    Body: {"query": ..., "freshness": "noLimit", "summary": True, "count": N}

返回 data.webPages.value[]，每项含 name / url / snippet / summary，
本模块将其格式化为带序号、标题、链接、摘要的文本，供大模型拼装上下文。
"""
import requests

from llm.util import config
from llm.util.logger_pathlib import logger

BOCHA_API_URL = "https://api.bochaai.com/v1/web-search"


def bocha_search(query: str, count: int | None = None, freshness: str | None = None) -> str:
    """调用博查搜索，返回格式化文本（标题 + 链接 + 摘要）。失败时抛出异常。"""
    count = count or config.BOCHA_SEARCH_COUNT
    freshness = freshness or config.BOCHA_FRESHNESS
    payload = {
        "query": query,
        "freshness": freshness,
        "summary": True,
        "count": count,
    }
    headers = {
        "Authorization": f"Bearer {config.BOCHA_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(BOCHA_API_URL, json=payload, headers=headers, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 200:
        raise RuntimeError(f"博查搜索失败: code={data.get('code')}, msg={data.get('msg')}")

    results = (data.get("data") or {}).get("webPages", {}).get("value", [])
    if not results:
        logger.info("博查搜索无结果: {}", query)
        return ""

    lines = []
    for i, item in enumerate(results, 1):
        title = item.get("name", "")
        url = item.get("url", "")
        snippet = item.get("summary") or item.get("snippet") or ""
        lines.append(f"[{i}] {title}\n链接: {url}\n摘要: {snippet}\n")
    logger.info("博查搜索返回 {} 条结果: {}", len(results), query)
    return "\n".join(lines)
