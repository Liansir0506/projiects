from pathlib import Path
from llm.util.logger_pathlib import logger


def get_path(path: str = ''):
    try:
        current_path = Path.cwd()  # 获取当前路径
        llm_path = current_path.parent

        return llm_path / (path.lstrip("/"))
    except Exception as e:
        logger.error(f"未找到对应文件，其中出现错误Error: {e}")


if __name__ == '__main__':
    print(get_path('yaml/test.yaml'))

