import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class ThyroidDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, is_train=True):
        """
        甲状腺超声图像数据集类
        
        参数:
            image_paths: 图像路径列表
            mask_paths: 掩码路径列表
            transform: 数据增强变换
            is_train: 是否为训练集(决定是否应用数据增强)
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.is_train = is_train
        
        # 探头标记去除参数
        self.kernel = np.ones((5, 5), np.uint8)
        
        # CLAHE增强参数
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        参数:
            idx: 样本索引
            
        返回:
            image: 预处理后的图像张量
            mask: 预处理后的掩码张量
        """
        # 读取图像和掩码
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # 检查图像是否成功加载
        if image is None or mask is None:
            raise ValueError(f"无法加载图像或掩码: {self.image_paths[idx]} 或 {self.mask_paths[idx]}")
        
        # 探头标记去除(仅在训练集应用)
        if self.is_train:
            image = cv2.morphologyEx(image, cv2.MORPH_OPEN, self.kernel)
        
        # CLAHE增强(仅在训练集应用)
        image = self.clahe.apply(image)
        
        # 归一化到[0,1]范围
        image = image / 255.0
        
        # 数据增强(仅在训练集应用)
        if self.transform and self.is_train:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        
        # 转换为PyTorch张量并调整维度
        # 从HWC格式转换为CHW格式，并增加batch维度
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)     # [1, H, W]
        
        return image, mask

def visualize_sample(dataset, idx=0):
    """
    可视化数据集中的样本
    
    参数:
        dataset: 数据集对象
        idx: 要可视化的样本索引
    """
    image, mask = dataset[idx]
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(121)
    plt.axis("off")
    plt.imshow(image.squeeze().numpy(), cmap='gray')
    plt.title("Input Image")
    
    plt.subplot(122)
    plt.axis("off")
    plt.imshow(mask.squeeze().numpy(), cmap='gray')
    plt.title("Ground Truth Mask")
    
    plt.show()

def create_dataloaders():
    """
    创建训练集和验证集的数据加载器
    
    返回:
        train_loader: 训练集数据加载器
        val_loader: 验证集数据加载器
    """
    # 获取所有图像和掩码路径
    image_paths = sorted(glob.glob("data/images/*.png"))
    mask_paths = sorted(glob.glob("data/masks/*.png"))
    
    # 检查路径数量是否匹配
    if len(image_paths) != len(mask_paths):
        raise ValueError("图像和掩码的数量不匹配!")
    
    # 划分训练集和验证集(80%训练, 20%验证)
    train_img, val_img, train_msk, val_msk = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    # 定义训练集的数据增强变换
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),  # 随机水平翻转
        A.RandomRotate90(p=0.5),  # 随机旋转90度
        A.RandomBrightnessContrast(p=0.2),  # 随机亮度对比度调整
        ToTensorV2()  # 转换为PyTorch张量
    ])
    
    # 创建训练集和验证集
    train_dataset = ThyroidDataset(
        train_img, train_msk, transform=train_transform, is_train=True
    )
    val_dataset = ThyroidDataset(
        val_img, val_msk, transform=None, is_train=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False, num_workers=4
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # 创建数据加载器
    train_loader, val_loader = create_dataloaders()
    
    # 可视化训练集中的第一个样本
    train_dataset = ThyroidDataset(
        sorted(glob.glob("data/images/*.png"))[:8],
        sorted(glob.glob("data/masks/*.png"))[:8],
        transform=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            ToTensorV2()
        ]),
        is_train=True
    )
    visualize_sample(train_dataset)