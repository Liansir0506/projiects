# -*- coding: utf-8 -*-
"""
流式聊天模块：混合检索 + 阿里云百炼大模型，流式输出。

流程：
1. HybridRetriever 检索相关片段；
2. 拼装上下文 Prompt；
3. 百炼 OpenAI 兼容接口 stream_chat 流式生成；
4. 引用来源（文件名 + MD5）随上下文透传给模型。
"""
from pathlib import Path
from typing import Iterator, List, Tuple

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai_like import OpenAILike

from llm.util import config
from llm.core.hybrid_retriever import HybridRetriever
from llm.util.logger_pathlib import logger

# 基于 __file__ 定位 prompt 目录（.../llm/prompt），不依赖 cwd
_AGENT_DIR = Path(__file__).resolve().parent
_PROMPT_DIR = _AGENT_DIR.parent / "prompt"
SYSTEM_PROMPT = (_PROMPT_DIR / "system_prompt.txt").read_text(encoding="utf-8")
PROMPT_TEMPLATE = (_PROMPT_DIR / "prompt_template").read_text(encoding="utf-8")


def build_llm() -> OpenAILike:
    """阿里云百炼 OpenAI 兼容对话模型（OpenAILike 不做模型名白名单校验）。"""
    return OpenAILike(
        model=config.CHAT_MODEL,
        api_key=config.DASHSCOPE_KEY,
        api_base=config.CHAT_BASE_URL,
        temperature=config.TEMPERATURE,
        is_chat_model=True,
        is_function_calling_model=False,
        context_window=32768,
    )


def format_context(nodes) -> str:
    """将检索片段格式化为带来源标注的上下文。"""
    lines = []
    for i, n in enumerate(nodes, 1):
        fname = n.node.metadata.get("file_name", "-")
        md5 = n.node.metadata.get("md5", "-")
        lines.append(f"[{i}] 来源: {fname} (md5: {md5})\n{n.node.text}\n")
    return "\n".join(lines)


def stream_chat(
    hybrid: HybridRetriever,
    query: str,
    history: List[Tuple[str, str]] = None,
) -> Iterator[str]:
    """流式问答：逐 token 产出回答文本。"""
    nodes = hybrid.retrieve(query)
    if not nodes:
        yield "知识库中没有检索到相关内容，请先入库文档或换一个问题。"
        return

    context = format_context(nodes)
    user_prompt = PROMPT_TEMPLATE.format(context=context, query=query)

    llm = build_llm()
    messages = [ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT)]
    for q, a in (history or []):
        messages.append(ChatMessage(role=MessageRole.USER, content=q))
        messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=a))
    messages.append(ChatMessage(role=MessageRole.USER, content=user_prompt))

    logger.info("开始流式生成，检索到 {} 个片段", len(nodes))
    resp = llm.stream_chat(messages)
    for chunk in resp:
        yield chunk.message.content or ""
