import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp
import numpy as np

class ThyroidMultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = smp.Unet(encoder_name="resnet34", 
                               encoder_weights="imagenet").encoder
        self.segmentation_head = smp.Unet.decoder['resnet34'](
            encoder_channels=[64, 128, 256, 512, 1024],
            decoder_channels=[256, 128, 64, 32, 16]
        )
        self.seg_final = nn.Conv2d(16, 1, kernel_size=1)
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 4)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        seg_mask = torch.sigmoid(self.seg_final(self.segmentation_head(features)))
        cls_logits = self.classification_head(features.mean(dim=[2,3]))
        return seg_mask, cls_logits

def dice_loss(pred, target):
    smooth = 1e-6
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (2 * intersection + smooth) / (union + smooth)

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ThyroidMultiTaskModel().to(device)
    train_loader, val_loader = create_dataloaders()
    
    seg_loss_fn = dice_loss
    cls_loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    
    best_dice = 0.0
    
    for epoch in range(20):
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/20 [Train]")
        for images, masks in train_pbar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            seg_pred, cls_pred = model(images)
            
            loss_seg = seg_loss_fn(seg_pred, masks)
            cls_target = (masks.squeeze() > 0.5).long()
            loss_cls = cls_loss_fn(cls_pred, cls_target)
            loss = loss_seg + 0.5 * loss_cls
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({"loss": loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        val_loss, val_dice, val_acc = validate(model, val_loader, device, seg_loss_fn, cls_loss_fn)
        scheduler.step(val_dice)
        
        print(f"Epoch {epoch+1}/20 - "
              f"Train Loss: {avg_train_loss:.4f} - "
              f"Val Loss: {val_loss:.4f} - "
              f"Val Dice: {val_dice:.4f} - "
              f"Val Acc: {val_acc:.4f}")
        
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best model saved with Dice score: {best_dice:.4f}")
    
    print("Training completed!")

def validate(model, loader, device, seg_loss_fn, cls_loss_fn):
    model.eval()
    total_loss = 0.0
    dice_scores = []
    cls_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        val_pbar = tqdm(loader, desc="Validating")
        for images, masks in val_pbar:
            images, masks = images.to(device), masks.to(device)
            
            seg_pred, cls_pred = model(images)
            
            loss_seg = seg_loss_fn(seg_pred, masks)
            cls_target = (masks.squeeze() > 0.5).long()
            loss_cls = cls_loss_fn(cls_pred, cls_target)
            loss = loss_seg + 0.5 * loss_cls
            total_loss += loss.item()
            
            dice = 2 * (seg_pred * masks).sum() / (seg_pred.sum() + masks.sum() + 1e-6)
            dice_scores.append(dice.item())
            
            _, predicted = torch.max(cls_pred, 1)
            cls_correct += (predicted == cls_target).sum().item()
            total_samples += cls_target.size(0)
            
            val_pbar.set_postfix({
                "loss": loss.item(),
                "dice": np.mean(dice_scores[-10:]) if dice_scores else 0
            })
    
    avg_loss = total_loss / len(loader)
    avg_dice = np.mean(dice_scores)
    avg_acc = cls_correct / total_samples
    
    return avg_loss, avg_dice, avg_acc

if __name__ == "__main__":
    train_model()