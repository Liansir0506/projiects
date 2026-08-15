# 数据处理报告（2 号交付物）

**生成日期**：2026-08-15  
**负责成员**：2 号（数据处理）  

## 1. 项目背景

本数据集服务于小麦病害诊断系统（15 类分类），最终通过
`POST /api/v1/diagnoses` 接口返回 Top-3 预测结果，预测结构必须包含
`class_id`（整数 0~14）、`class_name`（英文类别名）、`display_name`（中文显示名）
三个字段。数据处理阶段的职责：多源命名统一、三划分一致、类别均衡、
字段与 7 号知识库 / 5 号前端 / 1 号接口契约保持一致。

## 2. 数据源与合并

| 项目 | 内容 |
|---|---|
| 数据源 1（数据集1） | 14 类 PascalCase 目录命名，已划分 train/val/test；**缺 Common Root Rot 根腐病** |
| 数据源 2（archive） | 15 类完整，valid/test 目录使用下划线小写并带后缀（`aphid_valid` / `aphid_test`），`Healthy` 统一更名为 `Healthy Wheat` |
| 合并后图片总数 | **42,826** 张 |

| source_dataset | 张数 |
|---|---:|
| archive | 14,154 |
| augment | 7,460 |
| 数据集1 | 21,212 |

## 3. 类别标准（15 类，按字母序分配 class_id 0~14）

| class_id | class_name | display_name | 原始源路径数 | 原 train | 增强后 train | val | test | 合计 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Aphid | 小麦蚜虫 | 4 | 1,579 | 2,000 | 214 | 147 | 2,361 |
| 1 | Black Rust | 小麦秆锈病 | 4 | 1,392 | 2,000 | 254 | 167 | 2,421 |
| 2 | Blast | 小麦瘟病 | 4 | 1,099 | 2,000 | 150 | 115 | 2,265 |
| 3 | Brown Rust | 小麦叶锈病 | 4 | 4,113 | 4,113 | 833 | 456 | 5,402 |
| 4 | Common Root Rot | 小麦根腐病 | 3 | 614 | 2,000 | 20 | 50 | 2,070 |
| 5 | Fusarium Head Blight | 小麦赤霉病 | 4 | 1,503 | 2,000 | 275 | 178 | 2,453 |
| 6 | Healthy Wheat | 健康小麦 | 4 | 3,544 | 3,544 | 747 | 413 | 4,704 |
| 7 | Leaf Blight | 小麦叶枯病 | 4 | 1,431 | 2,000 | 189 | 134 | 2,323 |
| 8 | Mildew | 小麦白粉病 | 4 | 1,855 | 2,000 | 241 | 160 | 2,401 |
| 9 | Mite | 小麦叶螨 | 4 | 1,360 | 2,000 | 180 | 130 | 2,310 |
| 10 | Septoria | 小麦叶斑病 | 4 | 2,141 | 2,141 | 306 | 193 | 2,640 |
| 11 | Smut | 小麦黑粉病 | 4 | 2,227 | 2,227 | 282 | 181 | 2,690 |
| 12 | Stem fly | 小麦茎蝇 | 4 | 398 | 2,000 | 67 | 73 | 2,140 |
| 13 | Tan spot | 小麦褐斑病 | 4 | 1,309 | 2,000 | 174 | 127 | 2,301 |
| 14 | Yellow Rust | 小麦条锈病 | 4 | 3,383 | 3,383 | 615 | 347 | 4,345 |
| **合计** | — | — | — | **27,948** | **35,408** | **4,547** | **2,871** | **42,826** |

## 4. 数据集拆分统计

| split | 张数 | 占比 |
|---|---:|---:|
| train | 35,408 | 82.68% |
| validation | 4,547 | 10.62% |
| test | 2,871 | 6.70% |
| **total** | **42,826** | **100.00%** |

* manifest.csv 交叉核对：
  - train:      35,408 / report 35,408
  - validation: 4,547 / report 4,547
  - test:       2,871 / report 2,871

## 5. 离线少样本数据增强（仅 train）

- **应用状态**：已应用 ✓
- **train 每类目标样本数**：`2000`
- **新增增强图片数**：**7,460 张**
- **增强管线**：`RandomResizedCrop(224,scale=0.8-1.0)+HFlip(p=0.5)+VFlip(p=0.3)+Rotate(±15)+Brightness/Contrast(±0.2)+GaussianBlur(p=0.3)`
- **备注**：只对 train 样本数 < target 的类别进行线下增强；val/test 保持不动；不修改 saturation/hue

增强策略要点：
- 只对 `train` 不足 2000 张的 10 类做离线增强；5 类充足样本（Brown Rust / Healthy / Septoria / Smut / Yellow Rust）跳过
- `val` / `test` 完全不动，保证评估基准不变
- **不调整饱和度 saturation 和色调 hue**：锈病颜色特征是分类的关键判别依据，改变会破坏语义
- 生成文件命名 `aug__s42__000000.jpg`，便于区分原图与增强图
- 随机种子固定 42，结果可复现

### 各类增强明细

| class_id | class_name | 原 train | 后 train | 新增 | 倍数 |
|---|---|---:|---:|---:|---:|
| 0 | Aphid | 1,579 | 2,000 | +421 | ×1.27 |
| 1 | Black Rust | 1,392 | 2,000 | +608 | ×1.44 |
| 2 | Blast | 1,099 | 2,000 | +901 | ×1.82 |
| 3 | Brown Rust | 4,113 | 4,113 | +0 | SKIP |
| 4 | Common Root Rot | 614 | 2,000 | +1,386 | ×3.26 |
| 5 | Fusarium Head Blight | 1,503 | 2,000 | +497 | ×1.33 |
| 6 | Healthy Wheat | 3,544 | 3,544 | +0 | SKIP |
| 7 | Leaf Blight | 1,431 | 2,000 | +569 | ×1.40 |
| 8 | Mildew | 1,855 | 2,000 | +145 | ×1.08 |
| 9 | Mite | 1,360 | 2,000 | +640 | ×1.47 |
| 10 | Septoria | 2,141 | 2,141 | +0 | SKIP |
| 11 | Smut | 2,227 | 2,227 | +0 | SKIP |
| 12 | Stem fly | 398 | 2,000 | +1,602 | ×5.03 |
| 13 | Tan spot | 1,309 | 2,000 | +691 | ×1.53 |
| 14 | Yellow Rust | 3,383 | 3,383 | +0 | SKIP |

## 6. 命名统一规范（与接口契约对齐）

| 字段 | 来源文件 | 接口契约对应 | 说明 |
|---|---|---|---|
| `class_id` 0~14 | `class_mapping.json` classes[].class_id | `predictions[].class_id` | 整数索引，按字母序分配 |
| `class_name` | `class_mapping.json` + train/val/test 目录名 | `predictions[].class_name` | 英文标准名，训练/接口/知识库三方匹配 |
| `display_name` | `class_mapping.json` | `predictions[].display_name` | 中文显示名，5 号前端展示给用户 |
| `source_paths[]` | `class_mapping.json` | — | 原始数据源所有命名变体（数据集1 缺 Common Root Rot，故 14 条 vs 15 条） |
| `manifest.csv` | `../data/合并数据集/manifest.csv` | — | 42,826 条溯源记录，source_dataset / split / class / source / destination 五列 |

## 7. 已知限制与备注

1. **数据集1 缺 Common Root Rot（普通根腐病）**：该类 3 个源路径全部来自 archive，
   数据集1 无此类别；其余 14 类均由数据集1 + archive 共同贡献。
2. **`manifest.csv` 体积约 8.1MB**：代码仓库不存储大体积 CSV，
   仓库交付 `manifest_augmented.csv` 作为增强索引；完整溯源清单保留在 `../data/合并数据集/`。
3. **class_name 含空格（如 `Black Rust`）**：PyTorch `ImageFolder` 完全支持，
   JSON 字符串中也合法，因此不强制改为下划线；若未来目录要规范化，可在 class_mapping.json
   中新增 `class_dir` 字段做映射，不动现有数据。
4. **val/test 仍不均衡**：离线增强只补齐 train，val/test（Stem fly 仅 57/57）保留真实分布；
   模型评估时应关注 Macro-F1 / Recall 而非整体 Accuracy。

## 8. 交付物清单（对应 docs/项目交付清单表.md 2 号条目）

| 交付物 | 仓库位置 | 状态 | 说明 |
|---|---|---|---|
| 类别映射表 class_mapping.json | `database/class_mapping.json` | ✅ 已交付 | v2.0 含 class_id / class_name / display_name / source_paths |
| 数据处理统计 JSON | `database/processing_report.json` | ✅ 已交付 | 含 class_counts、split_totals、augmentation_info |
| 数据处理报告 Markdown | `database/data_processing_report.md` | ✅ 已交付 | 本文件 |
| 数据集溯源清单 manifest.csv | `database/manifest.csv` | ✅ 已交付 | 42,826 条完整溯源记录（含 7,460 条增强记录） |
| 离线增强清单 manifest_augmented.csv | `database/manifest_augmented.csv` | ✅ 已交付 | 7,460 条增强源-目标映射 |
| 数据集合并代码 | `database/merge_datasets.py` | ✅ 已交付 | 纯 Python；支持多源命名对齐 + 类别映射 + 报告生成 |
| 在线训练增强模板 | `database/augment_online_transforms.py` | ✅ 已交付 | `get_train_transform(224)` 供 3/4 号模型训练使用 |
| 离线少样本增强脚本 | `database/augment_offline_few_shot.py` | ✅ 已交付 | 纯 PIL，无需 torchvision；已实际执行过 |
| 类别映射升级脚本 | `upgrade_class_mapping.py` | ✅ 已交付 | 将 v1 扁平映射升级为 v2 完整版 |
| 一键交付物构建脚本 | `build_delivery.py` | ✅ 已交付 | 执行本脚本可重新生成以上所有交付物 |
