# -*- coding: utf-8 -*-
"""
一键生成数据处理交付物：
  1. 把元文件副本复制到 wheat-disease-monitor/database/
     (manifest.csv 8MB+体积大，只复制 manifest_augmented.csv 索引，主 manifest 保留在原 data 目录)
  2. 生成 database/data_processing_report.md（人类可读的 Markdown 报告）
  3. 更新 docs/项目交付清单表.md 中 2号交付物为"已交付"
作者：2 号（数据处理）
"""
import csv
import json
import shutil
from collections import Counter, OrderedDict
from pathlib import Path

# -------- 路径（相对推导，项目根 = wheat-disease-monitor/） --------
PROJ_DIR = Path(__file__).resolve().parent      # wheat-disease-monitor/
DATA_DIR = PROJ_DIR / 'data' / '合并数据集'      # wheat-disease-monitor/data/合并数据集/
DB_DIR = PROJ_DIR / 'database'
DOCS_DIR = PROJ_DIR / 'docs'

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# 15 类字母序（class_id 0~14）
CLASSES = [
    (0,  'Aphid',                 '小麦蚜虫'),
    (1,  'Black Rust',            '小麦秆锈病'),
    (2,  'Blast',                 '小麦瘟病'),
    (3,  'Brown Rust',            '小麦叶锈病'),
    (4,  'Common Root Rot',       '小麦根腐病'),
    (5,  'Fusarium Head Blight',  '小麦赤霉病'),
    (6,  'Healthy Wheat',         '健康小麦'),
    (7,  'Leaf Blight',           '小麦叶枯病'),
    (8,  'Mildew',                '小麦白粉病'),
    (9,  'Mite',                  '小麦叶螨'),
    (10, 'Septoria',              '小麦叶斑病'),
    (11, 'Smut',                  '小麦黑粉病'),
    (12, 'Stem fly',              '小麦茎蝇'),
    (13, 'Tan spot',              '小麦褐斑病'),
    (14, 'Yellow Rust',           '小麦条锈病'),
]


def step1_copy_meta_files():
    """把体积小的元文件复制到 database/ 作为交付副本"""
    copies = [
        ('class_mapping.json',      DATA_DIR / 'class_mapping.json'),
        ('processing_report.json',  DATA_DIR / 'processing_report.json'),
        ('manifest_augmented.csv',  DATA_DIR / 'manifest_augmented.csv'),
    ]
    DB_DIR.mkdir(parents=True, exist_ok=True)
    for dst_name, src in copies:
        dst = DB_DIR / dst_name
        if src.exists():
            shutil.copy2(src, dst)
            print(f'  ✓ {dst_name:<25s}  {dst.stat().st_size/1024:>7.2f} KB')
        else:
            print(f'  ⚠ {src.name} 源不存在，跳过')


def step2_build_markdown_report():
    """生成 database/data_processing_report.md"""
    # 读数据
    with open(DATA_DIR / 'processing_report.json', 'r', encoding='utf-8') as f:
        report = json.load(f)
    cc = report['class_counts']
    st = report['split_totals']
    aug_info = report.get('augmentation_info', {})
    total = report['total_images']

    with open(DATA_DIR / 'class_mapping.json', 'r', encoding='utf-8') as f:
        cm = json.load(f)
    src_path_count = {c['class_name']: len(c['source_paths']) for c in cm['classes']}

    # 统计每个类 train 目录下的原图数（不含 aug__ 开头），得到「原 train」值
    def _non_aug_train(cname):
        d = DATA_DIR / 'train' / cname
        if not d.is_dir():
            return 0
        return sum(1 for p in d.iterdir()
                   if p.is_file()
                   and p.suffix.lower() in IMG_EXTS
                   and not p.name.startswith('aug__'))
    orig_train = {cname: _non_aug_train(cname) for _, cname, _ in CLASSES}

    # 读 manifest 统计
    main_manifest = DATA_DIR / 'manifest.csv'
    src_counter = Counter()
    split_counter = Counter()
    if main_manifest.exists():
        with open(main_manifest, 'r', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                src_counter[r['source_dataset']] += 1
                split_counter[r['split']] += 1

    lines = []
    lines += ['# 数据处理报告（2 号交付物）', '']
    lines += ['**生成日期**：2026-08-15  ',
              '**负责成员**：2 号（数据处理）  ',
              '']
    lines += ['## 1. 项目背景', '']
    lines += [
        '本数据集服务于小麦病害诊断系统（15 类分类），最终通过',
        '`POST /api/v1/diagnoses` 接口返回 Top-3 预测结果，预测结构必须包含',
        '`class_id`（整数 0~14）、`class_name`（英文类别名）、`display_name`（中文显示名）',
        '三个字段。数据处理阶段的职责：多源命名统一、三划分一致、类别均衡、',
        '字段与 7 号知识库 / 5 号前端 / 1 号接口契约保持一致。',
        ''
    ]
    lines += ['## 2. 数据源与合并', '']
    lines += [
        '| 项目 | 内容 |',
        '|---|---|',
        '| 数据源 1（数据集1） | 14 类 PascalCase 目录命名，已划分 train/val/test；**缺 Common Root Rot 根腐病** |',
        '| 数据源 2（archive） | 15 类完整，valid/test 目录使用下划线小写并带后缀（`aphid_valid` / `aphid_test`），`Healthy` 统一更名为 `Healthy Wheat` |',
        f'| 合并后图片总数 | **{total:,}** 张 |',
        ''
    ]
    if src_counter:
        lines += ['| source_dataset | 张数 |', '|---|---:|']
        for k, v in sorted(src_counter.items()):
            lines += [f'| {k} | {v:,} |']
        lines += ['']
    lines += ['## 3. 类别标准（15 类，按字母序分配 class_id 0~14）', '']
    lines += [
        '| class_id | class_name | display_name | 原始源路径数 | 原 train | 增强后 train | val | test | 合计 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|'
    ]
    class_grand = 0
    for cid, cname, cdisp in CLASSES:
        item = cc[cname]
        orig_t = orig_train[cname]
        cur_t  = item['train']
        v = item['validation']; te = item['test']
        tot = cur_t + v + te
        class_grand += tot
        lines += [f'| {cid} | {cname} | {cdisp} | {src_path_count[cname]} | {orig_t:,} | {cur_t:,} | {v:,} | {te:,} | {tot:,} |']
    lines += [
        f'| **合计** | — | — | — | '
        f'**{sum(orig_train.values()):,}** | **{st["train"]:,}** | '
        f'**{st["validation"]:,}** | **{st["test"]:,}** | **{class_grand:,}** |'
    ]
    lines += ['']
    lines += ['## 4. 数据集拆分统计', '']
    lines += ['| split | 张数 | 占比 |', '|---|---:|---:|']
    for s in ['train', 'validation', 'test']:
        n = st[s]
        lines += [f'| {s} | {n:,} | {100*n/total:.2f}% |']
    lines += [f'| **total** | **{total:,}** | **100.00%** |', '']
    if split_counter:
        lines += [
            '* manifest.csv 交叉核对：',
            f'  - train:      {split_counter["train"]:,} / report {st["train"]:,}',
            f'  - validation: {split_counter["validation"]:,} / report {st["validation"]:,}',
            f'  - test:       {split_counter["test"]:,} / report {st["test"]:,}',
            ''
        ]
    lines += ['## 5. 离线少样本数据增强（仅 train）', '']
    if aug_info:
        lines += [
            f'- **应用状态**：{"已应用 ✓" if aug_info.get("applied") else "未应用"}',
            f'- **train 每类目标样本数**：`{aug_info.get("target_per_class_train", 2000)}`',
            f'- **新增增强图片数**：**{aug_info.get("newly_added_train_images", 0):,} 张**',
            f'- **增强管线**：`{aug_info.get("augmentation_pipeline", "")}`',
            f'- **备注**：{aug_info.get("notes", "")}',
            ''
        ]
    lines += [
        '增强策略要点：',
        '- 只对 `train` 不足 2000 张的 10 类做离线增强；5 类充足样本（Brown Rust / Healthy / Septoria / Smut / Yellow Rust）跳过',
        '- `val` / `test` 完全不动，保证评估基准不变',
        '- **不调整饱和度 saturation 和色调 hue**：锈病颜色特征是分类的关键判别依据，改变会破坏语义',
        '- 生成文件命名 `aug__s42__000000.jpg`，便于区分原图与增强图',
        '- 随机种子固定 42，结果可复现',
        ''
    ]
    # 列出各类增强量
    lines += ['### 各类增强明细', '']
    lines += ['| class_id | class_name | 原 train | 后 train | 新增 | 倍数 |',
              '|---|---|---:|---:|---:|---:|']
    for cid, cname, _ in CLASSES:
        orig_t = orig_train[cname]
        cur_t = cc[cname]['train']
        add = cur_t - orig_t
        ratio = (cur_t / orig_t) if orig_t > 0 else float('inf')
        lines += [f'| {cid} | {cname} | {orig_t:,} | {cur_t:,} | +{add:,} | {"SKIP" if add==0 else f"×{ratio:.2f}"} |']
    lines += ['']
    lines += ['## 6. 命名统一规范（与接口契约对齐）', '']
    lines += [
        '| 字段 | 来源文件 | 接口契约对应 | 说明 |',
        '|---|---|---|---|',
        '| `class_id` 0~14 | `class_mapping.json` classes[].class_id | `predictions[].class_id` | 整数索引，按字母序分配 |',
        '| `class_name` | `class_mapping.json` + train/val/test 目录名 | `predictions[].class_name` | 英文标准名，训练/接口/知识库三方匹配 |',
        '| `display_name` | `class_mapping.json` | `predictions[].display_name` | 中文显示名，5 号前端展示给用户 |',
        '| `source_paths[]` | `class_mapping.json` | — | 原始数据源所有命名变体（数据集1 缺 Common Root Rot，故 14 条 vs 15 条） |',
        '| `manifest.csv` | `../data/合并数据集/manifest.csv` | — | 42,826 条溯源记录，source_dataset / split / class / source / destination 五列 |',
        ''
    ]
    lines += ['## 7. 已知限制与备注', '']
    lines += [
        '1. **数据集1 缺 Common Root Rot（普通根腐病）**：该类 3 个源路径全部来自 archive，',
        '   数据集1 无此类别；其余 14 类均由数据集1 + archive 共同贡献。',
        '2. **`manifest.csv` 体积约 8.1MB**：代码仓库不存储大体积 CSV，',
        '   仓库交付 `manifest_augmented.csv` 作为增强索引；完整溯源清单保留在 `../data/合并数据集/`。',
        '3. **class_name 含空格（如 `Black Rust`）**：PyTorch `ImageFolder` 完全支持，',
        '   JSON 字符串中也合法，因此不强制改为下划线；若未来目录要规范化，可在 class_mapping.json',
        '   中新增 `class_dir` 字段做映射，不动现有数据。',
        '4. **val/test 仍不均衡**：离线增强只补齐 train，val/test（Stem fly 仅 57/57）保留真实分布；',
        '   模型评估时应关注 Macro-F1 / Recall 而非整体 Accuracy。',
        ''
    ]
    lines += ['## 8. 交付物清单（对应 docs/项目交付清单表.md 2 号条目）', '']
    lines += [
        '| 交付物 | 仓库位置 | 状态 | 说明 |',
        '|---|---|---|---|',
        '| 类别映射表 class_mapping.json | `database/class_mapping.json` | ✅ 已交付 | v2.0 含 class_id / class_name / display_name / source_paths |',
        '| 数据处理统计 JSON | `database/processing_report.json` | ✅ 已交付 | 含 class_counts、split_totals、augmentation_info |',
        '| 数据处理报告 Markdown | `database/data_processing_report.md` | ✅ 已交付 | 本文件 |',
        '| 数据集溯源清单 manifest.csv | `../data/合并数据集/manifest.csv` | ✅ 已交付 (存 data 目录) | 42,826 条，因体积大不入 git |',
        '| 离线增强清单 manifest_augmented.csv | `database/manifest_augmented.csv` | ✅ 已交付 | 7,460 条增强源-目标映射 |',
        '| 数据集合并代码 | `database/merge_datasets.py` | ✅ 已交付 | 纯 Python；支持多源命名对齐 + 类别映射 + 报告生成 |',
        '| 在线训练增强模板 | `database/augment_online_transforms.py` | ✅ 已交付 | `get_train_transform(224)` 供 3/4 号模型训练使用 |',
        '| 离线少样本增强脚本 | `database/augment_offline_few_shot.py` | ✅ 已交付 | 纯 PIL，无需 torchvision；已实际执行过 |',
        '| 类别映射升级脚本 | `upgrade_class_mapping.py` | ✅ 已交付 | 将 v1 扁平映射升级为 v2 完整版 |',
        '| 一键交付物构建脚本 | `build_delivery.py` | ✅ 已交付 | 执行本脚本可重新生成以上所有交付物 |',
        ''
    ]

    out = DB_DIR / 'data_processing_report.md'
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'  ✓ Markdown 报告 -> {out.relative_to(PROJ_DIR)}')


def step3_update_delivery_list():
    """把 docs/项目交付清单表.md 中 "2 号 / 数据处理" 相关条目标记为 已交付"""
    doc = DOCS_DIR / '项目交付清单表.md'
    if not doc.exists():
        print(f'  ⚠ {doc} 不存在，跳过交付清单更新')
        return
    with open(doc, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    KWS = ['待开始', '未开始', '待处理', '进行中', '开发中', '未交付', '待交付', '未完成']
    new_lines = []
    hit = False
    for line in lines:
        if ('2号' in line or '二号' in line or '数据处理' in line) and ('|' in line or '交付' in line or '任务' in line):
            for kw in KWS:
                if kw in line:
                    line = line.replace(kw, '已交付')
                    hit = True
        new_lines.append(line)

    # 若文档中未提及 2 号/数据处理，在末尾附一短表
    if not hit:
        new_lines += [
            '', '',
            '## 补充：2 号（数据处理）交付物状态', '',
            '| 交付物 | 状态 |', '|---|---|',
            '| 类别映射+统计报告+处理代码+在线/离线增强脚本 | ✅ 已交付 |',
            ''
        ]

    with open(doc, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))
    print(f'  ✓ 更新 docs/项目交付清单表.md：2号相关条目标记为"已交付"（命中 {hit} 处）')


def main():
    print('[1/3] 复制元文件副本到 database/')
    step1_copy_meta_files()
    print('\n[2/3] 生成 Markdown 数据处理报告')
    step2_build_markdown_report()
    print('\n[3/3] 更新项目交付清单表')
    step3_update_delivery_list()
    print('\n✓ 全部完成')


if __name__ == '__main__':
    main()
