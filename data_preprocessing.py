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
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.is_train = is_train
        self.kernel = np.ones((5,5), np.uint8)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        if self.is_train:
            image = cv2.morphologyEx(image, cv2.MORPH_OPEN, self.kernel)
            image = self.clahe.apply(image)
        
        image = image / 255.0
        
        if self.transform and self.is_train:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        
        return image, mask

def create_dataloaders(data_dir="data"):
    image_paths = sorted(glob.glob(os.path.join(data_dir, "images", "*.png")))
    mask_paths = sorted(glob.glob(os.path.join(data_dir, "masks", "*.png")))
    
    if len(image_paths) != len(mask_paths):
        raise ValueError("图像和掩码数量不匹配")
    
    train_img, val_img, train_msk, val_msk = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        ToTensorV2()
    ])
    
    train_dataset = ThyroidDataset(train_img, train_msk, train_transform, True)
    val_dataset = ThyroidDataset(val_img, val_msk, None, False)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = create_dataloaders()
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")