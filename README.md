# 小麦病虫害智能诊断与长势评估系统

面向小麦田间管理的智能辅助决策系统：上传田间照片识别 15 类病虫害，输入环境 / 土壤 / 农事数据给出长势评分与干预建议，并提供天气风险提示、历史记录与统计。

> 本项目为个人视图模型练习项目：各模块分别交付（接口契约中对应统筹、数据处理、ResNet50、ConvNeXt、后端、前端、知识库、数据库 / 天气）。统一接口契约见 [`docs/接口契约.md`](docs/接口契约.md)。

## 功能概览

| 功能 | 说明 |
|---|---|
| **病害诊断** | 上传图片 → 图片质检（模糊 / 过暗 / 未检测到小麦，**质检不合格不调用模型推理**，直接返回错误码 1002）→ ResNet50 15 类分类（返回 Top-3 与置信度）→ 知识库病害介绍与防治建议 → 天气风险提示 |
| **长势评估** | 专家规则引擎 + 加权评分法：5 个生育期 × 7 个评分维度 → 综合评分、等级（壮苗 / 正常苗 / 弱苗 / 严重弱苗 / 旺苗）、主要与次要短板分析、针对性干预建议、趋势预判 |
| **历史与统计** | 诊断记录列表与单条详情、按类别 / 可信度 / 风险等级的统计、近期高风险记录 |
| **知识问答** | `knowledge/llm/` 内的 RAG 模块（LlamaIndex + Chroma 向量库 + BM25 混合检索 + 大模型对话 + 联网搜索），面向病虫害知识的检索问答 |

## 技术栈

- **后端 API**：Python + FastAPI + Uvicorn，接口统一前缀 `/api/v1/`，统一响应包装 `{code, msg, data}`
- **图像分类**：PyTorch + torchvision，ResNet50 迁移学习（ImageNet V2 预训练），15 分类
- **知识问答**：LlamaIndex + Chroma + BM25（jieba 中文分词）+ OpenAI 兼容大模型接口
- **外部合作方模拟接口**：Flask + Flask-CORS（天气 / 病虫害风险）
- **数据处理**：pandas + numpy + Pillow

## 快速开始

### 1. 启动后端 API

```bash
pip install -r backend/requirements.txt
python backend/main.py
```

- 服务地址：`http://127.0.0.1:8000`
- 交互式接口文档（Swagger）：`http://127.0.0.1:8000/docs`
- 健康检查：`GET http://127.0.0.1:8000/api/v1/health`

### 2. 模型单图推理

```bash
pip install torch torchvision pillow pyyaml
cd models/resnet
python inference.py /path/to/wheat_image.jpg
```

模型权重、训练配置、逐类指标与推理脚本说明见 [`models/README.md`](models/README.md)。

### 3. 外部合作方模拟接口（天气 / 病虫害风险）

```bash
pip install flask flask-cors
python database/partner_app.py
```

### 4. LLM 知识问答模块

```bash
pip install -r knowledge/llm/requirements.txt
# 先把 API Key 填入 knowledge/llm/yaml/api_keys.yaml（仓库内为占位符）
python knowledge/llm/server/app.py
```

### 5. 数据处理脚本

数据处理脚本需要 `data/合并数据集/` 目录（数据集不入仓库，需自备）。路径推导规则、命令行参数与常见问题见 [`database/路径配置说明.md`](database/路径配置说明.md)。

```bash
python database/merge_datasets.py --sources "数据集1=./data/数据集1" "archive=./data/archive" --output "./data/合并数据集"
python database/augment_offline_few_shot.py
python upgrade_class_mapping.py
python build_delivery.py
```

## 接口一览

前缀 `/api/v1/`，HTTP 状态码固定 200，业务结果由 `code` 区分（`0` 成功、`1001` 参数错误、`1002` 图片质检不合格、`1003` 记录不存在、`5000` 服务器内部错误）。

| 方法 | 路由 | 说明 |
|---|---|---|
| GET | `/api/v1/health` | 服务健康检查 |
| GET | `/api/v1/classes` | 15 类病害列表 |
| POST | `/api/v1/diagnoses` | 提交病害诊断（`multipart/form-data`：`image`、`region`、`growth_stage`、`field`） |
| GET | `/api/v1/diagnoses` | 诊断历史列表 |
| GET | `/api/v1/diagnoses/{id}` | 单条诊断详情 |
| GET | `/api/v1/statistics` | 数据统计 |
| POST | `/api/v1/growth/evaluate` | 长势综合评估 |
| GET | `/api/v1/growth/stages` | 支持的小麦生育期列表 |
| GET | `/api/v1/growth/weights` | 指定生育期的评分维度与权重 |
| POST | `/debug/set-low-confidence` | 调试用：模拟低置信度分支 |

请求 / 响应字段与示例详见 [`docs/接口契约.md`](docs/接口契约.md)（契约定稿）与 [`backend/接口文档.md`](backend/接口文档.md)（后端对接说明）。

## 数据与模型

### 数据集

- 溯源清单 `database/manifest.csv` 共 **42,826 条**记录，另有增强清单 `database/manifest_augmented.csv` **7,460 条**
- 图片总量约 **35,408 张**，按 `train : test : val ≈ 7 : 2 : 1` 分层抽样
- 类别目录为 ImageFolder 结构（每个子目录一类，按字母序编号）
- **数据集本身不入仓库**（`data/` 已在 `.gitignore` 中排除），克隆后需自行放入

### 模型指标（ResNet50，验证集 3,541 张）

| 指标 | 数值 |
|---|---|
| Accuracy | **0.9271**（92.71%） |
| Macro-F1 | **0.9305**（93.05%） |
| Weighted-F1 | 0.9277（92.77%） |

- 逐类 Precision / Recall / F1：`models/metrics.json`、`models/classification_report.txt`
- 训练曲线与混淆矩阵：`models/training_curves.png`、`models/confusion_matrix.png`
- 训练超参与数据划分：`models/training_config.yaml`

## 15 类病害与类别编号

**类别编号以本表为准**（来源：`models/training_config.yaml` 与 `backend/config.py`，与 ImageFolder 字母序一致）：

| class_id | class_name | 中文名称 |
|---:|---|---|
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

> ⚠️ [`docs/接口契约.md`](docs/接口契约.md) 正文中的 JSON 示例使用的是**早期版本的类别编号与英文名**（例如示例写 `Yellow Rust` 为 `class_id: 13`、写 `Powdery Mildew` 而代码中为 `Mildew`）。对接时请以上表和 `backend/config.py` 为准。

## 目录结构

```
wheat-disease-monitor/
├── backend/                       后端 API（FastAPI）
│   ├── main.py                       路由层：全部 /api/v1 接口
│   ├── config.py                     全局配置：图片限制、置信度阈值、15 类映射
│   ├── service/                      业务层：diagnose_service、growth_service
│   ├── adapter/external_adapter.py   适配层：封装模型 / 知识库 / 数据库 / 天气
│   ├── mock/mock_modules.py          模拟实现（各模块正式交付前使用）
│   ├── test/                         后端自测脚本
│   ├── 接口文档.md                   后端对接文档
│   └── docs/对接说明文档.md
├── models/                        模型模块（ResNet50）
│   ├── resnet/inference.py           推理脚本（命令行 + 可被后端 import 的 Predictor）
│   ├── weights/resnet50_best.pth     模型权重
│   ├── training_config.yaml          训练配置与类别列表（含 class_id）
│   ├── metrics.json                  验证集逐类指标
│   ├── classification_report.txt     可读评估报告
│   └── training_curves.png / confusion_matrix.png
├── database/                      数据与天气 / 合作方模块
│   ├── merge_datasets.py             多数据源合并
│   ├── augment_offline_few_shot.py   离线增强（小样本类）
│   ├── augment_online_transforms.py  在线增强模板（训练时 import）
│   ├── class_mapping.json            类别映射
│   ├── manifest.csv                  溯源清单（42,826 条）
│   ├── manifest_augmented.csv        增强清单
│   ├── data_processing_report.md     数据处理报告
│   ├── 路径配置说明.md               数据路径配置说明
│   └── partner_*/                    外部合作方模拟接口（天气 / 病虫害风险）
├── knowledge/                     知识库模块
│   ├── growth_rules/                 长势评估规则与干预建议库
│   └── llm/                          LLM / RAG 知识问答模块
├── docs/                          项目文档（接口契约、交付清单表）
├── frontend/                      前端（Streamlit 三页，当前为空占位）
├── tests/                         测试（当前为空占位）
├── build_delivery.py              交付物一键构建脚本
└── upgrade_class_mapping.py       类别映射升级脚本
```

## 已知限制

1. **后端目前在模拟（Mock）模式下运行**：模型、知识库、数据库、天气四类依赖通过 `backend/mock/mock_modules.py` 提供模拟实现，接口路径、请求参数与响应结构已按最终契约实现；真实模型权重与推理脚本已就绪（`models/`），接入位在 `backend/config.py` 的 `MODEL_PATH`。
2. **前端与测试未随仓库提交**：`frontend/`、`tests/` 目前只有 `.gitkeep` 占位。
3. **天气数据为离线样例**：`database/partner_*` 读取仓库内的 `forecast_weather.csv` 样例数据，不是实时气象接口。
4. **密钥需自行填写**：`knowledge/llm/yaml/api_keys.yaml` 与 `knowledge/llm/yaml/rag_dashscope.yaml` 中的 Key 均为占位符（`sk-xxxx…`），请填入自己的 Key；**不要把填好真实 Key 的文件提交到仓库**。
5. **数据集与测试集图片不入仓库**：仓库只保留代码、元数据与文档，图片需自备。

## 数据来源说明

- 图像数据整理自公开学术数据集（小麦病虫害图像分类）并做了类别合并与增强，图片本身不随仓库分发。
- `database/manifest.csv` 的 `source` / `destination` 两列保留了数据整理过程的本机绝对路径，用于溯源；如需对外分享可自行替换为相对路径。
- 天气样例数据与合作方接口均为课程项目内的模拟实现，不代表任何真实业务系统。

## 交付物对照

| 交付内容 | 位置 |
|---|---|
| 数据处理代码、类别映射、处理报告 | `database/` |
| ResNet50 权重、推理代码、类别顺序 | `models/` |
| ConvNeXt 权重、推理代码、类别顺序 | `models/convnext/`（待交付） |
| FastAPI 后端、接口文档 | `backend/` |
| Streamlit 三页前端 | `frontend/`（待交付） |
| 15 类病虫害知识库 | `knowledge/` |
| 天气风险 / 数据库 / 记录统计接口 | `database/partner_*`、后端 `/api/v1/statistics` |
| 测试报告、演示流程、PPT | 见 `docs/项目交付清单表.md` |
