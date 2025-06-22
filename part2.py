import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import torch.optim as optim
from tqdm import tqdm

class ThyroidMultiTaskModel(nn.Module):
    def __init__(self):
        """
        甲状腺结节分割与分类多任务学习模型
        
        模型架构:
        - 共享编码器(ResNet34)
        - 分割分支(UNet解码器)
        - 分类分支(全局平均池化+全连接层)
        """
        super().__init__()
        
        # 共享编码器
        self.encoder = smp.Unet(encoder_name="resnet34", 
                               encoder_weights="imagenet").encoder
        
        # 分割分支(使用UNet解码器)
        self.segmentation_head = smp.Unet.decoder['resnet34'](
            encoder_channels=[64, 128, 256, 512, 1024],
            decoder_channels=[256, 128, 64, 32, 16]
        )
        self.seg_final = nn.Conv2d(16, 1, kernel_size=1)  # 输出1通道二值掩码
        
        # 分类分支
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Flatten(),             # 展平
            nn.Linear(16, 64),        # 全连接层
            nn.ReLU(),                # ReLU激活
            nn.Dropout(0.2),          # Dropout
            nn.Linear(64, 4)          # 输出4类(TI-RADS 1-4级)
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入图像张量
            
        返回:
            seg_mask: 分割掩码预测
            cls_logits: 分类logits
        """
        # 共享特征提取
        features = self.encoder(x)
        
        # 分割任务
        seg_features = self.segmentation_head(features)
        seg_output = torch.sigmoid(self.seg_final(seg_features))
        
        # 分类任务
        cls_features = F.adaptive_avg_pool2d(features['decoder_out'], 1)
        cls_output = self.classification_head(cls_features.squeeze(-1).squeeze(-1))
        
        return seg_output, cls_output

def train_model():
    """
    训练多任务学习模型
    """
    # 初始化模型
    model = ThyroidMultiTaskModel().cuda()
    print(model)
    
    # 定义损失函数
    seg_loss_fn = smp.losses.DiceLoss(mode='binary') + nn.BCEWithLogitsLoss()
    cls_loss_fn = nn.CrossEntropyLoss()  # 可以替换为FocalLoss
    
    # 定义优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # 定义学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5, verbose=True
    )
    
    # 训练参数
    num_epochs = 20
    best_dice = 0.0
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        # 使用tqdm显示进度条
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, masks in train_pbar:
            # 将数据移动到GPU
            images, masks = images.cuda(), masks.cuda()
            
            # 前向传播
            seg_pred, cls_pred = model(images)
            
            # 计算分割损失
            loss_seg = seg_loss_fn(seg_pred, masks)
            
            # 计算分类损失
            # 注意: 这里简化了分类标签的生成，实际应用中需要更精确的标签
            cls_target = (masks.squeeze() > 0.5).long()  # 简单二分类
            loss_cls = cls_loss_fn(cls_pred, cls_target)
            
            # 总损失
            loss = loss_seg + 0.5 * loss_cls
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 累计损失
            train_loss += loss.item()
            
            # 更新进度条
            train_pbar.set_postfix({"loss": loss.item()})
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        val_loss, val_dice, val_acc = validate_model(model, val_loader, seg_loss_fn, cls_loss_fn)
        
        # 更新学习率
        scheduler.step(val_dice)
        
        # 打印epoch结果
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.4f} - "
              f"Val Loss: {val_loss:.4f} - "
              f"Val Dice: {val_dice:.4f} - "
              f"Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best model saved with Dice score: {best_dice:.4f}")
    
    print("Training completed!")

def validate_model(model, loader, seg_loss_fn, cls_loss_fn):
    """
    验证模型性能
    
    参数:
        model: 要验证的模型
        loader: 验证数据加载器
        seg_loss_fn: 分割损失函数
        cls_loss_fn: 分类损失函数
        
    返回:
        avg_loss: 平均损失
        avg_dice: 平均Dice系数
        avg_acc: 平均分类准确率
    """
    model.eval()
    total_loss = 0.0
    dice_scores = []
    cls_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        val_pbar = tqdm(loader, desc="Validating")
        for images, masks in val_pbar:
            # 将数据移动到GPU
            images, masks = images.cuda(), masks.cuda()
            
            # 前向传播
            seg_pred, cls_pred = model(images)
            
            # 计算分割损失
            loss_seg = seg_loss_fn(seg_pred, masks)
            
            # 计算分类损失
            cls_target = (masks.squeeze() > 0.5).long()  # 简化分类标签
            loss_cls = cls_loss_fn(cls_pred, cls_target)
            
            # 总损失
            loss = loss_seg + 0.5 * loss_cls
            total_loss += loss.item()
            
            # 计算Dice系数
            dice = 2 * (seg_pred * masks).sum() / (seg_pred.sum() + masks.sum() + 1e-6)
            dice_scores.append(dice.item())
            
            # 计算分类准确率
            _, predicted = torch.max(cls_pred, 1)
            cls_correct += (predicted == cls_target).sum().item()
            total_samples += cls_target.size(0)
            
            # 更新进度条
            val_pbar.set_postfix({
                "loss": loss.item(),
                "dice": np.mean(dice_scores[-10:]) if dice_scores else 0
            })
    
    # 计算平均指标
    avg_loss = total_loss / len(loader)
    avg_dice = np.mean(dice_scores)
    avg_acc = cls_correct / total_samples
    
    return avg_loss, avg_dice, avg_acc

if __name__ == "__main__":
    # 创建数据加载器
    train_loader, val_loader = create_dataloaders()
    
    # 训练模型
    train_model()