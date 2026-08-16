# 模型目录（3 号交付物）

本目录包含小麦病害图像分类模型（ResNet50）的所有权重、配置、评估指标与推理代码。

## 目录结构

```
models/
├── resnet/
│   └── inference.py              # ResNet50 推理示例脚本（单张图片预测，Top-3 输出）
├── weights/
│   └── resnet50_best.pth         # ResNet50 最佳模型权重（~90MB）
├── training_config.yaml          # 训练配置（超参数、数据划分、增强策略等）
├── metrics.json                  # 验证集/测试集指标（逐类 Precision/Recall/F1、Macro-F1）
├── classification_report.txt     # 可读的评估报告文本
├── training_curves.png           # Loss 和 Macro-F1 训练曲线
├── confusion_matrix.png          # 测试集混淆矩阵
└── README.md                     # 本文件
```

## 交付物清单

| 交付物 | 文件 | 说明 |
|--------|------|------|
| 模型权重 | `weights/resnet50_best.pth` | ResNet50 迁移学习，15 分类，验证集 Macro-F1 ≥ 0.91 |
| 训练配置 | `training_config.yaml` | 完整超参数、数据划分、类别映射、环境依赖 |
| 验证集指标 | `metrics.json` / `classification_report.txt` | 逐类 P/R/F1、Accuracy、Macro-F1、Weighted-F1 |
| Loss 和 Macro-F1 曲线 | `training_curves.png` | 训练/验证 Loss 曲线 + Macro-F1 曲线 |
| 推理示例 | `resnet/inference.py` | 命令行单图预测 + 可被后端 import 的 Predictor 类 |

## 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision pillow pyyaml scikit-learn matplotlib
```

### 2. 命令行单张图片预测

```bash
cd models/resnet
python inference.py /path/to/wheat_image.jpg
```

输出示例：
```
=======================================================
 预测结果（Top-3）
=======================================================
  Top1  Brown Rust             (小麦叶锈病)
        置信度: 92.34%  ███████████████████████████
-------------------------------------------------------
  Top2  Yellow Rust            (小麦条锈病)
        置信度:  5.12%  █
-------------------------------------------------------
  Top3  Stem fly               (小麦茎蝇)
        置信度:  1.05%
=======================================================
```

### 3. 作为模块被后端调用

```python
from models.resnet.inference import ResNet50Predictor
from PIL import Image

predictor = ResNet50Predictor()
image = Image.open("test.jpg")
results = predictor.predict(image)  # 返回 Top-3 列表
```

返回格式（与后端 `predict_image` 接口兼容）：
```python
[
    {"class_name": "Brown Rust",  "display_name": "小麦叶锈病", "confidence": 0.9234, "rank": 1},
    {"class_name": "Yellow Rust", "display_name": "小麦条锈病", "confidence": 0.0512, "rank": 2},
    {"class_name": "Stem fly",    "display_name": "小麦茎蝇",   "confidence": 0.0105, "rank": 3},
]
```

## 模型性能

| 指标 | 验证集 (3541 张) | 测试集 (7082 张) |
|------|-----------------|-----------------|
| Accuracy | 0.9271 (92.71%) | 0.9192 (91.92%) |
| Macro-F1 | 0.9305 (93.05%) | 0.9206 (92.06%) |
| Weighted-F1 | 0.9277 (92.77%) | 0.9200 (92.00%) |

> 详细逐类指标请查看 `metrics.json` 或 `classification_report.txt`。

## 15 类标签（字母序，class_id 0~14）

| ID | 英文类别名 | 中文显示名 |
|----|-----------|-----------|
| 0 | Aphid | 小麦蚜虫 |
| 1 | Black Rust | 小麦秆锈病 |
| 2 | Blast | 小麦瘟病 |
| 3 | Brown Rust | 小麦叶锈病 |
| 4 | Common Root Rot | 小麦根腐病 |
| 5 | Fusarium Head Blight | 小麦赤霉病 |
| 6 | Healthy Wheat | 健康小麦 |
| 7 | Leaf Blight | 小麦叶枯病 |
| 8 | Mildew | 小麦白粉病 |
| 9 | Mite | 小麦叶螨 |
| 10 | Septoria | 小麦叶斑病 |
| 11 | Smut | 小麦黑粉病 |
| 12 | Stem fly | 小麦茎蝇 |
| 13 | Tan spot | 小麦褐斑病 |
| 14 | Yellow Rust | 小麦条锈病 |

## 训练说明

- **骨干网络**：ResNet50（ImageNet V2 预训练权重）
- **训练策略**：前 10 轮冻结主干只训分类头，第 10 轮解冻后余弦退火微调
- **数据划分**：35408 张图片，分层抽样 7:2:1（训练/测试/验证）
- **完整配置**：见 `training_config.yaml`
