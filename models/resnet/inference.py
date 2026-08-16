# -*- coding: utf-8 -*-
"""
================================================================================
 ResNet50 小麦病害分类 — 推理示例（3 号交付物）
================================================================================

【功能说明】
    加载训练好的 ResNet50 模型权重，对单张小麦病害图片进行预测，
    输出 Top-3 预测结果（类别名、中文名、置信度）。

【使用方法】
    方式一：命令行传图片路径（推荐）
        python inference.py 图片路径
        例如：python inference.py test.jpg

    方式二：不带参数运行，交互式输入路径
        python inference.py

    方式三：作为模块被后端 adapter 调用
        from inference import ResNet50Predictor
        predictor = ResNet50Predictor()
        results = predictor.predict(pil_image)   # 返回 Top-3 列表

【模型权重路径】
    默认加载 ../weights/resnet50_best.pth（相对于本文件所在目录）。
    也可通过构造参数 model_path 指定。

【运行环境】
    pip install torch torchvision pillow
    （有 NVIDIA 显卡建议安装 GPU 版 torch，推理更快）
================================================================================
"""

import os
import sys
from typing import List, Dict, Optional

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ==============================================================================
# 类别配置（与训练时 ImageFolder 字母序完全一致，切勿调换顺序）
# ==============================================================================
NUM_CLASSES = 15
IMAGE_SIZE = 224

CLASS_NAMES = [
    "Aphid",                # 0
    "Black Rust",           # 1
    "Blast",                # 2
    "Brown Rust",           # 3
    "Common Root Rot",      # 4
    "Fusarium Head Blight", # 5
    "Healthy Wheat",        # 6
    "Leaf Blight",          # 7
    "Mildew",               # 8
    "Mite",                 # 9
    "Septoria",             # 10
    "Smut",                 # 11
    "Stem fly",             # 12
    "Tan spot",             # 13
    "Yellow Rust",          # 14
]

CLASS_NAMES_CN = {
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
    "Yellow Rust": "小麦条锈病",
}

# 默认权重路径：本文件所在目录的上级 weights/ 子目录
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "weights", "resnet50_best.pth"
)

# ==============================================================================
# 图片预处理（必须与训练时验证集的 val_transform 完全一致）
# ==============================================================================
_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet 均值
        std=[0.229, 0.224, 0.225],    # ImageNet 标准差
    ),
])


class ResNet50Predictor:
    """
    ResNet50 小麦病害分类预测器。

    用法：
        predictor = ResNet50Predictor()
        results = predictor.predict(pil_image)
        # results = [
        #     {"class_name": "Brown Rust", "display_name": "小麦叶锈病", "confidence": 0.9234, "rank": 1},
        #     {"class_name": "Yellow Rust", "display_name": "小麦条锈病", "confidence": 0.0512, "rank": 2},
        #     ...
        # ]
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        加载模型权重并初始化预测器。

        参数：
            model_path : 权重文件路径，默认为 ../weights/resnet50_best.pth
            device     : 推理设备，"cuda" 或 "cpu"，默认自动检测
        """
        self.model_path = model_path or os.path.normpath(_DEFAULT_MODEL_PATH)
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"找不到模型权重文件: {self.model_path}\n"
                f"请确认 resnet50_best.pth 已放置在 weights/ 目录下。"
            )

        # 构建与训练时完全相同的模型结构
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        num_features = self.model.fc.in_features  # 2048
        self.model.fc = nn.Linear(num_features, NUM_CLASSES)

        # 加载训练好的权重
        state_dict = torch.load(
            self.model_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model = self.model.to(self.device)
        print(f"[ResNet50] 模型已加载: {self.model_path}")
        print(f"[ResNet50] 推理设备: {self.device}")

    def predict(self, pil_img: Image.Image, top_k: int = 3) -> List[Dict]:
        """
        对一张 PIL 图片进行预测，返回 Top-K 结果。

        参数：
            pil_img : PIL.Image 格式的输入图片（RGB）
            top_k   : 返回前 K 个预测结果，默认 3

        返回：
            List[Dict]，每个元素包含：
                class_name   : 英文类别名
                display_name : 中文显示名
                confidence   : 置信度（0~1 浮点数）
                rank         : 排名（1 开始）
        """
        if not isinstance(pil_img, Image.Image):
            raise TypeError(f"输入必须是 PIL.Image，收到: {type(pil_img)}")

        # 统一转 RGB（防止灰度图 / RGBA 图通道数不对）
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        # 预处理 + 增加 batch 维度: [1, 3, 224, 224]
        input_tensor = _TRANSFORM(pil_img).unsqueeze(0).to(self.device)

        # 前向推理
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        # 取 Top-K
        top_k = min(top_k, NUM_CLASSES)
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)

        results = []
        for rank in range(top_k):
            idx = top_indices[0, rank].item()
            conf = top_probs[0, rank].item()
            class_name = CLASS_NAMES[idx]
            results.append({
                "class_name": class_name,
                "display_name": CLASS_NAMES_CN.get(class_name, class_name),
                "confidence": round(conf, 4),
                "rank": rank + 1,
            })
        return results

    def predict_file(self, image_path: str, top_k: int = 3) -> List[Dict]:
        """
        从文件路径加载图片并预测。
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"找不到图片文件: {image_path}")
        image = Image.open(image_path).convert("RGB")
        return self.predict(image, top_k=top_k)


# ==============================================================================
# 命令行入口
# ==============================================================================
def main():
    # 初始化预测器（只加载一次模型）
    predictor = ResNet50Predictor()

    # 获取图片路径
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("请输入图片路径（相对或绝对路径）：").strip().strip('"')

    if not os.path.exists(image_path):
        print(f"错误：找不到图片文件 -> {image_path}")
        sys.exit(1)

    print(f"\n正在预测: {image_path}")
    results = predictor.predict_file(image_path, top_k=3)

    # 输出结果
    print("\n" + "=" * 55)
    print(" 预测结果（Top-3）")
    print("=" * 55)
    for r in results:
        bar = "█" * int(r["confidence"] * 30)
        print(
            f"  Top{r['rank']}  {r['class_name']:<22s} "
            f"({r['display_name']})"
        )
        print(f"        置信度: {r['confidence']*100:5.2f}%  {bar}")
        print("-" * 55)
    print("=" * 55)


if __name__ == "__main__":
    main()
