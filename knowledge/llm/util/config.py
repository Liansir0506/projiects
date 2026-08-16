# -*- coding: utf-8 -*-
"""
RAG 配置模块（统一配置入口）。

项目逻辑与 llm 主项目一致：
1. yaml 配置驱动（yaml/api_keys.yaml）——所有大模型 API Key 与模型参数集中于此；
2. 基于 __file__ 定位项目根目录，不依赖 cwd；
3. 日志复用 llm.util.logger_pathlib 的 logger；
4. 聊天与 Embedding 模型统一走阿里云百炼（DashScope）。
"""
from pathlib import Path

import yaml

from llm.util.logger_pathlib import logger

# 当前文件所在目录: .../llm/util
_UTIL_DIR = Path(__file__).resolve().parent
# llm 项目目录: .../llm（数据目录统一收敛在 llm 下，项目只在 llm 中）
_LLM_DIR = _UTIL_DIR.parent
# 项目根目录: .../llm（数据目录基于它拼接）
_PROJECT_ROOT = _LLM_DIR
# 配置文件（所有 API Key 与模型参数统一存放）
_CONFIG_PATH = _LLM_DIR / "yaml" / "api_keys.yaml"


def _load_config() -> dict:
    if not _CONFIG_PATH.exists():
        logger.warning("配置文件不存在: {}", _CONFIG_PATH)
        return {}
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_CONFIG = _load_config()


def get(key: str, default=None):
    """读取配置项（key 不存在时返回 default）。"""
    return _CONFIG.get(key, default)


def get_path(key: str, default: str = "") -> Path:
    """读取相对路径配置并基于项目根目录拼接为绝对路径。"""
    rel = _CONFIG.get(key, default)
    if not rel:
        return _PROJECT_ROOT
    p = Path(rel)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return p


# ===== 常用配置项（模块级导出）=====
DOCS_DIR = get_path("DOCS_DIR", "chroma/docs")            # 原始文档目录
PERSIST_DIR = get_path("PERSIST_DIR", "chroma/index_storage")  # Chroma 索引目录
MD5_DIR = get_path("MD5_DIR", "chroma/md5_files")         # MD5 文件存储目录
CHUNK_SIZE = int(get("CHUNK_SIZE", 512))                  # 分块大小
CHUNK_OVERLAP = int(get("CHUNK_OVERLAP", 64))             # 分块重叠
ALPHA = float(get("ALPHA", 0.5))                          # 混合检索权重
TOP_K = int(get("TOP_K", 5))                              # 返回条数
TEMPERATURE = float(get("TEMPERATURE", 0.3))              # 采样温度

# ===== API Key（统一从 yaml/api_keys.yaml 读取）=====
DASHSCOPE_API_KEY = get("DASHSCOPE_API_KEY", "")          # 阿里云百炼（聊天 + Embedding）
BOCHA_API_KEY = get("BOCHA_API_KEY", "")                  # 博查 Web Search
MODELSCOPE_API_KEY = get("MODELSCOPE_API_KEY", "")        # 魔搭（备用）
SENSENOVA_API_KEY = get("SENSENOVA_API_KEY", "")          # 商汤（备用）

# ===== 博查联网搜索 =====
BOCHA_SEARCH_COUNT = int(get("BOCHA_SEARCH_COUNT", 5))     # 博查返回结果条数
BOCHA_FRESHNESS = get("BOCHA_FRESHNESS", "noLimit")        # 搜索时间范围: oneDay/oneWeek/oneMonth/oneYear/noLimit

# ===== 自动更新 =====
AUTO_UPDATE = bool(get("AUTO_UPDATE", True))              # 是否自动增量入库新文档
AUTO_UPDATE_INTERVAL = int(get("AUTO_UPDATE_INTERVAL", 60))  # 轮询 docs 目录间隔（秒）

CHAT_BASE_URL = get("CHAT_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
CHAT_MODEL = get("CHAT_MODEL", "qwen-plus")
EMBED_BASE_URL = get("EMBED_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
EMBED_MODEL = get("EMBED_MODEL", "text-embedding-v3")
EMBEDDING_DIM = int(get("EMBEDDING_DIM", 1024))
DASHSCOPE_KEY = DASHSCOPE_API_KEY


def ensure_dirs() -> None:
    """自动创建文档/索引/MD5 目录。"""
    for d in (DOCS_DIR, PERSIST_DIR, MD5_DIR):
        d.mkdir(parents=True, exist_ok=True)
    logger.debug("目录就绪: {} / {} / {}", DOCS_DIR, PERSIST_DIR, MD5_DIR)


if __name__ == "__main__":
    print("项目根:", _PROJECT_ROOT)
    print("文档目录:", DOCS_DIR)
    print("索引目录:", PERSIST_DIR)
    print("MD5 目录:", MD5_DIR)
    print("聊天模型:", CHAT_MODEL)
    print("Embedding 模型:", EMBED_MODEL)
    print("混合权重 ALPHA:", ALPHA)
