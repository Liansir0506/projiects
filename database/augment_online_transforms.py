# -*- coding: utf-8 -*-
"""
在线数据增强配置（供 3 号 ResNet / 4 号 ConvNeXt 训练时使用）
使用方式：
    from augment_online_transforms import get_train_transform, get_val_test_transform
    train_ds = ImageFolder(train_dir, transform=get_train_transform(img_size=224))
    val_ds   = ImageFolder(val_dir,   transform=get_val_test_transform(img_size=224))

说明：
  1. 只在 train 集使用强增强，val/test 只做 Resize + CenterCrop + Normalize
  2. 故意不调 saturation/hue：小麦锈病颜色是关键特征
  3. 旋转/裁剪强度保守，不破坏病斑结构
作者：2 号（数据处理）
"""
from torchvision import transforms

# ImageNet 预训练权重的统计量，ResNet / ConvNeXt 都通用
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transform(img_size: int = 224):
    """
    训练集在线增强变换
    - 轻微随机裁剪缩放（模拟不同拍摄距离）
    - 水平/垂直翻转（叶片翻转语义不变）
    - 小角度旋转（模拟不同拍摄角度）
    - 仅亮度+对比度扰动（不碰饱和度/色调，避免改变病斑颜色）
    - 轻微高斯模糊（抗过拟合）
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(
            img_size,
            scale=(0.80, 1.0),   # 下限 0.8 避免病斑被裁掉
            ratio=(0.9, 1.1),    # 宽高比基本保持 1:1
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),   # ±15°，不超 30°
        transforms.ColorJitter(
            brightness=0.2,   # 亮度 ±20%
            contrast=0.2,     # 对比度 ±20%
            saturation=0.0,   # ⚠️ 饱和度不动
            hue=0.0,          # ⚠️ 色调不动
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
            p=0.3,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_test_transform(img_size: int = 224):
    """
    验证集 / 测试集变换（无随机性，保证评估一致）
    Resize(短边到 img_size+32) -> CenterCrop(img_size) -> Normalize
    """
    resize_size = int(img_size * 1.14)  # ImageNet 标准 224 -> 256
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_predict_transform(img_size: int = 224):
    """
    线上推理时的变换（同 val/test，保证训练/推理分布一致）
    供 1 号接口在收到用户上传图片后预处理使用。
    """
    return get_val_test_transform(img_size)


if __name__ == '__main__':
    # 打印配置，方便确认
    print('Train transform:')
    print(get_train_transform(224))
    print('\nVal/Test transform:')
    print(get_val_test_transform(224))
