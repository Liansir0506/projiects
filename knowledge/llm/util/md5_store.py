# -*- coding: utf-8 -*-
"""
MD5 文件存储模块。

逻辑：
1. 每个入库文件按内容计算 MD5；
2. 以 <md5><原扩展名> 命名保存到 MD5_DIR，实现去重（同一内容只存一份）；
3. 维护 md5_index.json 台账：md5 -> {原始文件名, 存储路径, 大小, 入库时间}，
   便于溯源与增量入库判断。
"""
import hashlib
import json
import shutil
import time
from pathlib import Path

from llm.util.config import MD5_DIR
from llm.util.logger_pathlib import logger

_INDEX_FILE = "md5_index.json"


def _index_path() -> Path:
    return MD5_DIR / _INDEX_FILE


def _load_index() -> dict:
    p = _index_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("MD5 台账损坏，将重建: {}", e)
    return {}


def _save_index(index: dict) -> None:
    MD5_DIR.mkdir(parents=True, exist_ok=True)
    _index_path().write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """计算文件 MD5（大文件分块读取）。"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def store_file(src_path: Path) -> dict:
    """
    将文件按 MD5 命名存入 MD5_DIR 并登记台账。

    返回 {md5, stored_path, existed}：
    - existed=True  表示该文件已入库（去重跳过，不再解析）；
    - existed=False 表示本次新入库。
    """
    src_path = Path(src_path)
    md5 = file_md5(src_path)
    index = _load_index()

    if md5 in index:
        logger.debug("文件已入库（MD5 去重）: {}", src_path.name)
        return {"md5": md5, "stored_path": index[md5]["stored_path"], "existed": True}

    dst = MD5_DIR / f"{md5}{src_path.suffix.lower()}"
    MD5_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst)

    index[md5] = {
        "original_name": src_path.name,
        "stored_path": str(dst),
        "size": src_path.stat().st_size,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_index(index)
    logger.info("文件已按 MD5 入库: {} -> {}", src_path.name, dst.name)
    return {"md5": md5, "stored_path": str(dst), "existed": False}


def query_md5(md5: str) -> dict | None:
    """按 MD5 查询台账。"""
    return _load_index().get(md5)


def all_records() -> dict:
    """返回全部 MD5 台账。"""
    return _load_index()


if __name__ == "__main__":
    print("MD5 目录:", MD5_DIR)
    print("当前台账条数:", len(all_records()))
