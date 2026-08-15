# -*- coding: utf-8 -*-
"""
小麦病害数据集合并与划分脚本（对应 manifest.csv 中的处理流程复现）

原始处理流程（复现逻辑）：
  输入：两个数据源
    1) 数据集1/  (已划分好 train/val/test，目录名 PascalCase，缺 Common Root Rot)
    2) archive/  (train/val/test 目录命名为：train/Aphid, valid/aphid_valid, test/aphid_test)
  输出：合并数据集/
    train/          （含 15 类标准名目录）
    validation/     （含 15 类标准名目录）
    test/           （含 15 类标准名目录）
    manifest.csv               溯源清单：源路径 -> 目标路径
    class_mapping.json         原始路径命名 -> 标准类别名
    processing_report.json     统计报告

使用（路径建议用相对项目根的相对路径）：
  python merge_datasets.py \
      --sources 数据集1=./data/数据集1  archive=./data/archive \
      --output  ./data/合并数据集

作者：2 号（数据处理）
"""
import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
SPLITS = ['train', 'validation', 'test']

# 原始路径前缀 -> 标准类别名（对应 class_mapping.json v1）
CLASS_MAPPING_RAW = {
    # ------- 数据集1 (14 类，缺 Common Root Rot) -------
    '数据集1/Aphid': 'Aphid',
    '数据集1/Black Rust': 'Black Rust',
    '数据集1/Blast': 'Blast',
    '数据集1/Brown Rust': 'Brown Rust',
    '数据集1/Fusarium Head Blight': 'Fusarium Head Blight',
    '数据集1/Healthy Wheat': 'Healthy Wheat',
    '数据集1/Leaf Blight': 'Leaf Blight',
    '数据集1/Mildew': 'Mildew',
    '数据集1/Mite': 'Mite',
    '数据集1/Septoria': 'Septoria',
    '数据集1/Smut': 'Smut',
    '数据集1/Stem fly': 'Stem fly',
    '数据集1/Tan spot': 'Tan spot',
    '数据集1/Yellow Rust': 'Yellow Rust',
    # ------- archive/train (15 类) -------
    'archive/train/Aphid': 'Aphid',
    'archive/train/Black Rust': 'Black Rust',
    'archive/train/Blast': 'Blast',
    'archive/train/Brown Rust': 'Brown Rust',
    'archive/train/Common Root Rot': 'Common Root Rot',
    'archive/train/Fusarium Head Blight': 'Fusarium Head Blight',
    'archive/train/Healthy': 'Healthy Wheat',
    'archive/train/Leaf Blight': 'Leaf Blight',
    'archive/train/Mildew': 'Mildew',
    'archive/train/Mite': 'Mite',
    'archive/train/Septoria': 'Septoria',
    'archive/train/Smut': 'Smut',
    'archive/train/Stem fly': 'Stem fly',
    'archive/train/Tan spot': 'Tan spot',
    'archive/train/Yellow Rust': 'Yellow Rust',
    # ------- archive/valid (后缀 _valid + 下划线小写) -------
    'archive/valid/aphid_valid': 'Aphid',
    'archive/valid/black_rust_valid': 'Black Rust',
    'archive/valid/blast_test_valid': 'Blast',
    'archive/valid/brown_rust_valid': 'Brown Rust',
    'archive/valid/common_root_rot_valid': 'Common Root Rot',
    'archive/valid/fusarium_head_blight_valid': 'Fusarium Head Blight',
    'archive/valid/healthy_valid': 'Healthy Wheat',
    'archive/valid/leaf_blight_valid': 'Leaf Blight',
    'archive/valid/mildew_valid': 'Mildew',
    'archive/valid/mite_valid': 'Mite',
    'archive/valid/septoria_valid': 'Septoria',
    'archive/valid/smut_valid': 'Smut',
    'archive/valid/stem_fly_valid': 'Stem fly',
    'archive/valid/tan_spot_valid': 'Tan spot',
    'archive/valid/yellow_rust_valid': 'Yellow Rust',
    # ------- archive/test (后缀 _test + 下划线小写) -------
    'archive/test/aphid_test': 'Aphid',
    'archive/test/black_rust_test': 'Black Rust',
    'archive/test/blast_test': 'Blast',
    'archive/test/brown_rust_test': 'Brown Rust',
    'archive/test/common_root_rot_test': 'Common Root Rot',
    'archive/test/fusarium_head_blight_test': 'Fusarium Head Blight',
    'archive/test/healthy_test': 'Healthy Wheat',
    'archive/test/leaf_blight_test': 'Leaf Blight',
    'archive/test/mildew_test': 'Mildew',
    'archive/test/mite_test': 'Mite',
    'archive/test/septoria_test': 'Septoria',
    'archive/test/smut_test': 'Smut',
    'archive/test/stem_fly_test': 'Stem fly',
    'archive/test/tan_spot_test': 'Tan spot',
    'archive/test/yellow_rust_test': 'Yellow Rust',
}

# 各数据源的 split 目录名映射（源 split 名 -> 标准 split 名）
SOURCE_SPLIT_MAP = {
    '数据集1': {'train': 'train', 'validation': 'validation', 'test': 'test'},
    'archive': {'train': 'train', 'valid': 'validation', 'test': 'test'},
}

# 合并输出文件名前缀（避免重名）
SOURCE_FILE_PREFIX = {
    '数据集1': 'dataset1__',
    'archive': 'archive__',
}


def list_dir_images(d: Path):
    """列出目录中的图片文件（稳定排序）"""
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def build_source_index(source_name: str, source_root: Path):
    """
    扫描某个数据源，建立 (split, 标准类别名) -> [图片路径列表] 的索引
    同时返回所有匹配到的 (原始映射键, 标准类名) 集合
    """
    idx = defaultdict(list)         # (split, class_name) -> [path]
    mapping_keys_found = set()      # {('数据集1/Aphid', 'Aphid'), ...}

    split_map = SOURCE_SPLIT_MAP[source_name]
    for src_split, std_split in split_map.items():
        split_dir = source_root / src_split
        if not split_dir.is_dir():
            # archive 有的把 train 图片放在 data/train，做兼容查找
            alt = source_root / 'data' / src_split
            if alt.is_dir():
                split_dir = alt
            else:
                continue
        for cls_folder in split_dir.iterdir():
            if not cls_folder.is_dir():
                continue
            # 构造 CLASS_MAPPING_RAW 的 key
            raw_key = f'{source_name}/{src_split}/{cls_folder.name}'
            if raw_key in CLASS_MAPPING_RAW:
                std_class = CLASS_MAPPING_RAW[raw_key]
                mapping_keys_found.add((raw_key, std_class))
                for img in list_dir_images(cls_folder):
                    idx[(std_split, std_class)].append(img)
            else:
                # 数据集1 没 split 目录层，直接就是类别目录，尝试直接匹配 source/class
                alt_key = f'{source_name}/{cls_folder.name}'
                if alt_key in CLASS_MAPPING_RAW:
                    std_class = CLASS_MAPPING_RAW[alt_key]
                    mapping_keys_found.add((alt_key, std_class))
                    for img in list_dir_images(cls_folder):
                        idx[(std_split, std_class)].append(img)

    return dict(idx), mapping_keys_found


def copy_with_prefix(src_img: Path, dst_dir: Path, prefix: str):
    """把图片复制到目标目录，文件名加前缀避免重名"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    new_name = prefix + src_img.name
    dst_path = dst_dir / new_name
    shutil.copy2(src_img, dst_path)
    return dst_path


def write_class_mapping(output_root: Path):
    """写出升级后的 class_mapping.json（含 class_id / display_name）"""
    # 字母序 15 类
    ordered_classes = sorted(list({v for v in CLASS_MAPPING_RAW.values()}))
    display_name_map = {
        'Aphid': '小麦蚜虫',
        'Black Rust': '小麦秆锈病',
        'Blast': '小麦瘟病',
        'Brown Rust': '小麦叶锈病',
        'Common Root Rot': '小麦根腐病',
        'Fusarium Head Blight': '小麦赤霉病',
        'Healthy Wheat': '健康小麦',
        'Leaf Blight': '小麦叶枯病',
        'Mildew': '小麦白粉病',
        'Mite': '小麦叶螨',
        'Septoria': '小麦叶斑病',
        'Smut': '小麦黑粉病',
        'Stem fly': '小麦茎蝇',
        'Tan spot': '小麦褐斑病',
        'Yellow Rust': '小麦条锈病',
    }
    # 按类聚合原始路径
    class_to_paths = defaultdict(list)
    for k, v in CLASS_MAPPING_RAW.items():
        class_to_paths[v].append(k)
    for cls in class_to_paths:
        class_to_paths[cls].sort()
    data = {
        'metadata': {
            'version': '2.0',
            'description': '类别映射表：原始数据源路径 -> 统一标准类别（含 class_id / class_name / display_name）',
            'total_classes': len(ordered_classes),
            'total_source_paths': len(CLASS_MAPPING_RAW),
            'class_id_order': 'alphabetical',
            'author': '2号-数据处理',
            'notes': [
                'class_id: 整数索引 0~14，模型训练标签与接口 predictions[].class_id 依据此值',
                'class_name: 英文类别名，对应 train/val/test 目录下的文件夹名',
                'display_name: 中文显示名，前端展示与 7 号知识库匹配依据',
                'source_paths: 该类别在原始数据源中的所有路径命名',
                '数据集1 缺少 Common Root Rot 类别，故该类只有 3 个 source_paths',
            ],
        },
        'classes': [
            {
                'class_id': i,
                'class_name': cls,
                'display_name': display_name_map[cls],
                'source_paths': class_to_paths[cls],
            }
            for i, cls in enumerate(ordered_classes)
        ],
    }
    out = output_root / 'class_mapping.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out, data


def write_processing_report(output_root: Path, split_counts: dict, class_counts: dict):
    """写出 processing_report.json"""
    ordered_classes = sorted(class_counts.keys())
    data = {
        'class_counts': {cls: class_counts[cls] for cls in ordered_classes},
        'split_totals': {s: split_counts[s] for s in SPLITS},
        'total_images': sum(split_counts.values()),
        'classes': ordered_classes,
        'source_file_prefixes': {
            'dataset1': SOURCE_FILE_PREFIX['数据集1'],
            'archive': SOURCE_FILE_PREFIX['archive'],
        },
    }
    out = output_root / 'processing_report.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out


def main():
    parser = argparse.ArgumentParser(description='合并两个小麦病害数据源 -> 标准 15 类 三划分数据集')
    parser.add_argument(
        '--sources',
        nargs='+', required=True,
        help='数据源列表，格式: 名称=路径  如: "数据集1=./data/数据集1_已划分" "archive=./data/archive"',
    )
    parser.add_argument('--output', type=str, required=True, help='合并输出根目录')
    args = parser.parse_args()

    # 解析 sources
    sources = {}
    for item in args.sources:
        name, path = item.split('=', 1)
        assert name in SOURCE_SPLIT_MAP, f'数据源名必须是 {" / ".join(SOURCE_SPLIT_MAP)}，收到: {name}'
        sources[name] = Path(path)
        assert sources[name].is_dir(), f'{name} 目录不存在: {sources[name]}'

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        (output_root / s).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    # class_counts[class][split] = count
    class_counts = defaultdict(lambda: {s: 0 for s in SPLITS})
    split_counts = {s: 0 for s in SPLITS}

    for source_name, source_root in sources.items():
        print(f'[扫描] {source_name}: {source_root}')
        idx, keys_found = build_source_index(source_name, source_root)
        print(f'  匹配到 {len(keys_found)} 条原始映射，覆盖 {len(idx)} 个 (split, class) 切片')
        prefix = SOURCE_FILE_PREFIX[source_name]
        for (split, cls_name), imgs in sorted(idx.items()):
            dst_dir = output_root / split / cls_name
            for img in imgs:
                dst_path = copy_with_prefix(img, dst_dir, prefix)
                manifest_rows.append({
                    'source_dataset': source_name,
                    'split': split,
                    'class': cls_name,
                    'source': str(img),
                    'destination': str(dst_path),
                })
                class_counts[cls_name][split] += 1
                split_counts[split] += 1
        print(f'  处理完成: {sum(len(v) for v in idx.values())} 张图片')

    # 写入 manifest.csv
    manifest_path = output_root / 'manifest.csv'
    with open(manifest_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['source_dataset', 'split', 'class', 'source', 'destination'])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f'✓ manifest.csv -> {manifest_path} ({len(manifest_rows)} 行)')

    # 写入 class_mapping.json
    cm_path, cm_data = write_class_mapping(output_root)
    print(f'✓ class_mapping.json -> {cm_path} ({len(cm_data["classes"])} 类)')

    # 写入 processing_report.json
    rp_path = write_processing_report(output_root, split_counts, class_counts)
    print(f'✓ processing_report.json -> {rp_path}')

    total = sum(split_counts.values())
    print('-' * 60)
    print(f'合并完成，共 {total} 张图 | train {split_counts["train"]} / val {split_counts["validation"]} / test {split_counts["test"]}')


if __name__ == '__main__':
    main()
