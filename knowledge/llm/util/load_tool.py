from pathlib import Path

import yaml

# 基于 __file__ 定位 llm 项目目录，不依赖 cwd
_LLM_DIR = Path(__file__).resolve().parent.parent
_CONFIG_PATH = str(_LLM_DIR / "yaml" / "api_keys.yaml")


def load_rag_yaml(config_path: str = _CONFIG_PATH, encoding: str = 'utf-8'):
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_agent_yaml(config_path: str = _CONFIG_PATH, encoding: str = 'utf-8'):
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


load_rag = load_rag_yaml()
load_agent = load_agent_yaml()

if __name__ == '__main__':
    print(load_agent['EMBED_MODEL'])
    print(load_rag['PERSIST_DIR'])
