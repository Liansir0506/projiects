# config.py
# 全局配置文件，所有可配置参数集中管理
# 后期调整阈值、路径等只需修改此文件

import os

# ---------- 图片校验配置 ----------
MAX_IMAGE_SIZE_MB = 10
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ---------- 置信度阈值（根据验证集调整） ----------
CONFIDENCE_THRESHOLD = 0.65  # 默认0.65，后期根据验证集Macro-F1调整

# ---------- 模型与设备 ----------
# 自动检测设备：若CUDA可用则使用GPU，否则CPU
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

# 真实模型路径（当前Mock模式不使用，但为后期预留）
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model.pth")

# ---------- 15类病害名称（2号提供，最终标准） ----------
CLASS_NAMES = [
    "Aphid",
    "Black Rust",
    "Blast",
    "Brown Rust",
    "Common Root Rot",
    "Fusarium Head Blight",
    "Healthy Wheat",
    "Leaf Blight",
    "Mildew",
    "Mite",
    "Septoria",
    "Smut",
    "Stem fly",
    "Tan spot",
    "Yellow Rust"
]

# ---------- 15类中英文映射（2号提供） ----------
CLASS_MAPPING = {
    "Aphid": "小麦蚜虫",
    "Black Rust": "小麦秆锈病",
    "Blast": "小麦瘟病",
    "Brown Rust": "小麦叶锈病",
    "Common Root Rot": "小麦根腐病",
    "Fusarium Head Blight": "小麦赤霉病",
    "Healthy Wheat": "健康小麦",
    "Leaf Blight": "小麦叶枯病",
    "Mildew": "小麦白粉病",
    "Mite": "小麦叶螨",
    "Septoria": "小麦叶斑病",
    "Smut": "小麦黑粉病",
    "Stem fly": "小麦茎蝇",
    "Tan spot": "小麦褐斑病",
    "Yellow Rust": "小麦条锈病"
}

# ---------- 其他配置 ----------
WEATHER_API_TIMEOUT = 5  # 秒，模拟超时