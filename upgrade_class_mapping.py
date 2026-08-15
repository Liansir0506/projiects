# -*- coding: utf-8 -*-
"""
class_mapping.json 升级脚本
作用：
  读取原始的"路径 -> 类别名"映射，升级为包含 class_id / class_name / display_name / source_paths 的完整版
  class_id 顺序与 processing_report.json 的 classes 数组保持一致（字母序 0~14）
路径约定：
  脚本默认与数据集同级放置在项目中：
    wheat-disease-monitor/upgrade_class_mapping.py
    data/合并数据集/class_mapping.json
    data/合并数据集/processing_report.json
  会自动根据自身位置推导路径；无需写死绝对路径。
作者：2 号（数据处理）
"""
import json
from collections import defaultdict
from pathlib import Path


def locate_data_dir() -> Path:
    """根据脚本位置推导 data/合并数据集 目录（相对路径：脚本所在 repo 根 -> data/合并数据集）"""
    script_dir = Path(__file__).resolve().parent   # wheat-disease-monitor/
    repo_root = script_dir.parent                   # 02NLP-project/
    return repo_root / 'data' / '合并数据集'


def main():
    data_dir = locate_data_dir()
    src_path = data_dir / 'class_mapping.json'
    report_path = data_dir / 'processing_report.json'
    out_path = src_path

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

    with open(src_path, 'r', encoding='utf-8') as f:
        source_to_class = json.load(f)
    print(f'读取原映射: {src_path}  ({len(source_to_class)} 条)')

    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)
    ordered_classes = report['classes']
    print(f'对齐 processing_report 的类别顺序（共 {len(ordered_classes)} 类）')

    # v2.0 格式时，原映射被替换成 classes[]，此时无需再升级
    if isinstance(source_to_class, dict) and 'classes' in source_to_class and 'metadata' in source_to_class:
        # 直接重映射：按原始 source_paths 聚合
        class_to_sources = defaultdict(list)
        for c in source_to_class['classes']:
            for sp in c.get('source_paths', []):
                class_to_sources[c['class_name']].append(sp)
    else:
        # v1.0 扁平格式：key=源路径, value=类别名
        class_to_sources = defaultdict(list)
        for sp, cn in source_to_class.items():
            class_to_sources[cn].append(sp)

    for cls in class_to_sources:
        class_to_sources[cls].sort()

    missing = [c for c in ordered_classes if c not in display_name_map]
    if missing:
        raise SystemExit(f'错误: display_name_map 缺少类别: {missing}')
    extra = [c for c in class_to_sources if c not in ordered_classes]
    if extra:
        raise SystemExit(f'错误: 原映射出现未知类别: {extra}')

    upgraded = {
        'metadata': {
            'version': '2.0',
            'description': '类别映射表：原始数据源路径 -> 统一标准类别（含 class_id / class_name / display_name）',
            'total_classes': len(ordered_classes),
            'total_source_paths': sum(len(class_to_sources[c]) for c in ordered_classes),
            'class_id_order': 'alphabetical, aligned with processing_report.json classes',
            'created_at': '2026-08-15',
            'author': '2号-数据处理',
            'notes': [
                'class_id: 整数索引 0~14，模型训练标签与接口 predictions[].class_id 依据此值',
                'class_name: 英文类别名，对应 train/val/test 目录下的文件夹名',
                'display_name: 中文显示名，前端展示与 7 号知识库匹配依据',
                'source_paths: 该类别在原始数据源中的所有路径（含命名差异）',
                '数据集1 缺少 Common Root Rot 类别，故该类只有 3 个 source_paths',
            ],
        },
        'classes': [
            {
                'class_id': i,
                'class_name': cn,
                'display_name': display_name_map[cn],
                'source_paths': class_to_sources[cn],
            }
            for i, cn in enumerate(ordered_classes)
        ],
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(upgraded, f, ensure_ascii=False, indent=2)

    print(f'\n升级完成 -> {out_path}')
    print(f'类别数: {len(upgraded["classes"])}')
    for c in upgraded['classes']:
        print(f'  {c["class_id"]:2d} | {c["class_name"]:<22s} | {c["display_name"]:<8s} | {len(c["source_paths"])} 个源路径')


if __name__ == '__main__':
    main()
