# -*- coding: utf-8 -*-
"""
离线少样本数据增强（纯 PIL + NumPy 实现，无需 torch/torchvision）
特性：
  1. 增强图直接写入 原始 train/<类别>/ 目录（放在源文件内）
  2. 只在每类样本数 < target_per_class 时生成
  3. 同时输出 manifest_augmented.csv（增强明细）并追加到主 manifest.csv
  4. 更新 processing_report.json 的 train 类别统计
  5. 可复现：固定随机种子
路径约定（项目根 = wheat-disease-monitor/）：
    wheat-disease-monitor/               <- 项目根
    ├── database/augment_offline_few_shot.py   (本脚本)
    └── data/合并数据集/                  <- 数据集根目录
        ├── train/|validation/|test/
        ├── manifest.csv
        └── processing_report.json
  脚本自动根据自身位置推导数据集根目录（向上 1 级到项目根，再进 data/合并数据集），
  也可用 --dataset_root 覆盖。详见 database/路径配置说明.md。
作者：2 号（数据处理）
"""
import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def default_dataset_root() -> Path:
    """根据本脚本位置推导数据集根目录（项目根 = wheat-disease-monitor/）"""
    # __file__ = wheat-disease-monitor/database/augment_offline_few_shot.py
    script_dir = Path(__file__).resolve().parent          # wheat-disease-monitor/database/
    project_root = script_dir.parent                       # wheat-disease-monitor/
    default = project_root / 'data' / '合并数据集'          # wheat-disease-monitor/data/合并数据集/
    return default


def list_images(d: Path):
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


# ---------- 以下是用纯 PIL 手写的等价变换（不依赖 torchvision）----------

def pil_random_resized_crop(img: Image.Image, size=224, scale=(0.8, 1.0), ratio=(0.9, 1.1)) -> Image.Image:
    """等价 torchvision.transforms.RandomResizedCrop"""
    W, H = img.size
    for _ in range(10):
        s = random.uniform(*scale) * W * H
        r = random.uniform(*ratio)
        cw = int(round((s * r) ** 0.5))
        ch = int(round((s / r) ** 0.5))
        if cw <= W and ch <= H and cw > 0 and ch > 0:
            x = random.randint(0, W - cw)
            y = random.randint(0, H - ch)
            return img.crop((x, y, x + cw, y + ch)).resize((size, size), Image.BILINEAR)
    # fallback: center crop + resize
    side = min(W, H)
    left = (W - side) // 2
    top = (H - side) // 2
    return img.crop((left, top, left + side, top + side)).resize((size, size), Image.BILINEAR)


def pil_random_flip(img: Image.Image, hv: str, p=0.5) -> Image.Image:
    if random.random() < p:
        if hv == 'h':
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        elif hv == 'v':
            return img.transpose(Image.FLIP_TOP_BOTTOM)
    return img


def pil_random_rotation(img: Image.Image, degrees=15) -> Image.Image:
    deg = random.uniform(-degrees, degrees)
    return img.rotate(deg, resample=Image.BILINEAR, expand=False, fillcolor=(128, 128, 128))


def pil_color_jitter(img: Image.Image, brightness=0.2, contrast=0.2) -> Image.Image:
    """只调亮度+对比度，不碰饱和度/色调"""
    from PIL import ImageEnhance
    if brightness != 0:
        factor = 1.0 + random.uniform(-brightness, brightness)
        img = ImageEnhance.Brightness(img).enhance(factor)
    if contrast != 0:
        factor = 1.0 + random.uniform(-contrast, contrast)
        img = ImageEnhance.Contrast(img).enhance(factor)
    return img


def apply_pipeline(img: Image.Image, size=224) -> Image.Image:
    """串行应用整套在线增强等价变换"""
    img = pil_random_resized_crop(img, size=size)
    img = pil_random_flip(img, 'h', p=0.5)
    img = pil_random_flip(img, 'v', p=0.3)
    img = pil_random_rotation(img, degrees=15)
    img = pil_color_jitter(img, brightness=0.2, contrast=0.2)
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))
    return img


# ---------- 主增强逻辑 ----------

def augment_one_class(
    src_class_dir: Path,
    target: int,
    class_name: str,
    img_size: int,
    seed: int,
) -> list[dict]:
    """
    对单个类别生成增强图，直接写入 src_class_dir 目录。
    返回增强记录列表（用于追加 manifest）
    """
    src_imgs = list_images(src_class_dir)
    existing = len(src_imgs)
    need = max(target - existing, 0)
    if need <= 0:
        print(f'  [SKIP] {class_name} 已有 {existing} >= {target}')
        return []
    if not src_imgs:
        print(f'  [WARN] {class_name} 目录无图片')
        return []

    random.seed(seed)
    np.random.seed(seed)

    records = []
    idx = 0
    fail = 0
    max_fail = need * 5
    while idx < need and fail < max_fail:
        src_path = random.choice(src_imgs)
        try:
            with Image.open(src_path) as img:
                img = img.convert('RGB')
                aug = apply_pipeline(img, size=img_size)
                out_name = f'aug__s{seed}__{idx:06d}.jpg'
                out_path = src_class_dir / out_name
                aug.save(out_path, format='JPEG', quality=90)
                records.append({
                    'source_dataset': 'augment',
                    'split': 'train',
                    'class': class_name,
                    'source': str(src_path),
                    'destination': str(out_path),
                })
                idx += 1
        except Exception as e:
            print(f'    增强失败 {src_path.name}: {e}')
            fail += 1
            continue

    done = idx
    print(f'  [AUG]  {class_name}: {existing} -> {existing + done} (新增 {done}/{need})')
    return records


def append_manifest(root_dir: Path, new_records: list[dict]):
    """把增强记录追加到主 manifest.csv"""
    manifest_path = root_dir / 'manifest.csv'
    if not manifest_path.exists():
        print(f'  [WARN] 主 manifest.csv 不存在，跳过追加')
        return
    with open(manifest_path, 'a', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['source_dataset', 'split', 'class', 'source', 'destination'])
        writer.writerows(new_records)
    print(f'  ✓ manifest.csv 追加 {len(new_records)} 条')


def update_processing_report(root_dir: Path, augment_records: list[dict], target_per_class: int):
    """更新 processing_report.json 的 class_counts[split][class] 和 split_totals.train"""
    report_path = root_dir / 'processing_report.json'
    if not report_path.exists():
        print(f'  [WARN] processing_report.json 不存在，跳过更新')
        return
    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)
    from collections import Counter
    delta = Counter(r['class'] for r in augment_records)
    for cls, add in delta.items():
        if cls not in report['class_counts']:
            report['class_counts'][cls] = {'train': 0, 'validation': 0, 'test': 0}
        report['class_counts'][cls]['train'] += add
        report['split_totals']['train'] += add
    report['total_images'] = (
        report['split_totals']['train']
        + report['split_totals']['validation']
        + report['split_totals']['test']
    )
    report.setdefault('augmentation_info', {})
    report['augmentation_info'].update({
        'applied': True,
        'target_per_class_train': target_per_class,
        'newly_added_train_images': len(augment_records),
        'augmentation_pipeline': 'RandomResizedCrop(224,scale=0.8-1.0)+HFlip(p=0.5)+VFlip(p=0.3)+Rotate(±15)+Brightness/Contrast(±0.2)+GaussianBlur(p=0.3)',
        'notes': '只对 train 样本数 < target 的类别进行线下增强；val/test 保持不动；不修改 saturation/hue',
    })
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f'  ✓ processing_report.json 已更新 (新增 {len(augment_records)} 张 train)')


def main():
    parser = argparse.ArgumentParser(description='离线少样本增强（写入源 train 目录并更新关联文件）')
    parser.add_argument('--dataset_root', type=str,
                        default=str(default_dataset_root()),
                        help='合并数据集根目录（内含 train/val/test + manifest.csv + processing_report.json）。默认: 相对 ../data/合并数据集')
    parser.add_argument('--target_per_class', type=int, default=2000,
                        help='train 每类目标样本数（超过则不增强）')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    root = Path(args.dataset_root)
    train_root = root / 'train'
    assert root.is_dir() and train_root.is_dir(), f'目录结构不对: {root}'

    class_dirs = sorted([p for p in train_root.iterdir() if p.is_dir()])
    print(f'数据集根目录: {root.resolve()}')
    print(f'类别数: {len(class_dirs)}')
    print(f'train 每类目标样本数: {args.target_per_class}')
    print('=' * 70)

    all_records = []
    for cls_dir in class_dirs:
        all_records.extend(
            augment_one_class(
                src_class_dir=cls_dir,
                target=args.target_per_class,
                class_name=cls_dir.name,
                img_size=args.img_size,
                seed=args.seed,
            )
        )

    print('-' * 70)
    if all_records:
        aug_manifest = root / 'manifest_augmented.csv'
        with open(aug_manifest, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['source_dataset', 'split', 'class', 'source', 'destination'])
            writer.writeheader()
            writer.writerows(all_records)
        print(f'✓ manifest_augmented.csv: {aug_manifest} ({len(all_records)} 条)')
        append_manifest(root, all_records)
        update_processing_report(root, all_records, args.target_per_class)
    else:
        print('所有类别已达标，未生成增强图。')


if __name__ == '__main__':
    main()
