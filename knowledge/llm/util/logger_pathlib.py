# -*- coding: utf-8 -*-
"""
基于 loguru 的日志配置模块（pathlib 路径版本）。

功能：
1. 日志同时输出到控制台（彩色）和 logger 目录下的多个文件（无颜色）；
2. 按日志等级分别写入 debug/info/warning/error 四个文件，并有一个总日志文件 all.log；
3. 日志文件路径直接用 pathlib 基于 __file__ 计算生成，不再依赖 path_tool.get_path 与 os.chdir；
4. logger 目录/文件不存在时自动创建；
5. 提供可直接 import 使用的 logger 对象。

用法：
    from llm.util.logger_pathlib import logger

    logger.info("这是一条信息日志")
    logger.debug("这是一条调试日志")
    logger.error("这是一条错误日志")
"""

import sys
from pathlib import Path

from loguru import logger

# 当前文件所在目录: .../llm/util
_CURRENT_DIR = Path(__file__).resolve().parent
# 日志目录: .../llm/logger
_LOG_DIR = _CURRENT_DIR.parent / "logger"
# logger 目录不存在时自动创建
_LOG_DIR.mkdir(parents=True, exist_ok=True)

# 控制台格式（彩色，使用 loguru 颜色标签）
CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

# 文件格式（纯文本，不带任何颜色标签）
FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} - "
    "{message}"
)

# 等级 -> 分等级文件名映射
_LEVEL_FILES = {
    "DEBUG": "debug.log",
    "INFO": "info.log",
    "WARNING": "warning.log",
    "ERROR": "error.log",
}


def _make_level_filter(level_name: str):
    """构造按等级精确匹配的 filter（ERROR 文件额外包含 CRITICAL）。"""
    if level_name == "ERROR":
        return lambda record: record["level"].name in ("ERROR", "CRITICAL")
    return lambda record: record["level"].name == level_name


def setup_logger() -> None:
    """配置 loguru 日志：移除默认 handler，添加控制台与文件 handler。"""
    # 移除 loguru 默认输出，避免重复打印
    logger.remove()

    # 控制台输出（彩色）
    logger.add(
        sys.stderr,
        format=CONSOLE_FORMAT,
        level="WARNING",
        colorize=True,
        enqueue=True,
    )

    # 总日志文件（全部等级，无颜色）
    logger.add(
        str(_LOG_DIR / "all.log"),
        format=FILE_FORMAT,
        level="DEBUG",
        encoding="utf-8",
        colorize=False,
        enqueue=True,
        rotation="10 MB",
        retention="7 days",
        backtrace=True,
        diagnose=True,
    )

    # 按等级分文件输出（无颜色，每个文件只接收对应等级）
    for level_name, file_name in _LEVEL_FILES.items():
        logger.add(
            str(_LOG_DIR / file_name),
            format=FILE_FORMAT,
            level=level_name,
            filter=_make_level_filter(level_name),
            encoding="utf-8",
            colorize=False,
            enqueue=True,
            rotation="10 MB",
            retention="7 days",
        )


# 模块导入时自动完成配置
setup_logger()


if __name__ == "__main__":
    # 简单自测
    logger.debug("DEBUG 级别测试")
    logger.info("INFO 级别测试")
    logger.warning("WARNING 级别测试")
    logger.error("ERROR 级别测试")
    logger.critical("CRITICAL 级别测试")
    logger.success("配置完成，日志目录: {}", _LOG_DIR)
