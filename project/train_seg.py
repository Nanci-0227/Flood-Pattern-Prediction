"""
训练积水区域分割模型（U-Net + ResNet 编码器）。

数据目录结构（默认 data/floodnet）：
    images/   待分割图片 (jpg/png/bmp)
    masks/    二值掩码 (png)，非零像素视为积水（与水面对应同文件名）

用法示例：
    python train_seg.py --image_dir data/floodnet/images --mask_dir data/floodnet/masks
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import albumentations as A

import config
from config import get_device
from models.segmentor import Segmentor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FloodDataset(Dataset):
    """图片 + 二值掩码 的分割数据集。"""

    def __init__(self, image_dir, mask_dir, size=config.SEG_INPUT_SIZE, augment=False):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.size = size
        self.augment = augment
        self.images = sorted(p for p in self.image_dir.glob("*")
                             if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        if not self.images:
            raise FileNotFoundError(f"{image_dir} 下未找到图片")

    def __len__(self):
        return len(self.images)

    def _find_mask(self, stem: str) -> Path:
        for ext in (".png", ".jpg", ".jpeg", ".bmp"):
            p = self.mask_dir / (stem + ext)
            if p.exists():
                return p
        # 也支持 mask 目录下同名目录结构
        raise FileNotFoundError(f"缺少掩码：{self.mask_dir / (stem + '.png')}")

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self._find_mask(img_path.stem)), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"无法读取掩码：{img_path.stem}")

        img = cv2.resize(img, self.size)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)

        if self.augment:
            aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            ])
            out = aug(image=img, mask=mask)
            img, mask = out["image"], out["mask"]

        img = img.astype(np.float32) / 255.0
        img = (img - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
        img = torch.from_numpy(img).permute(2, 0, 1).float()   # (3,H,W)

        mask = (np.asarray(mask) > 0).astype(np.int64)
        mask = torch.from_numpy(mask).long()                    # (H,W)
        return img, mask


class FocalDiceLoss(nn.Module):
    """Focal + Dice 组合损失，缓解类别不平衡并优化分割边界。"""

    def __init__(self, alpha=0.25, gamma=2.0, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)      # (B,2,H,W)
        water = probs[:, 1]                        # 积水通道概率
        t = target.float()

        pt = torch.where(t > 0.5, water, 1 - water)
        focal = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt + 1e-8)
        focal = focal.mean()

        inter = (water * t).sum(dim=(1, 2))
        union = water.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        dice_loss = (1 - dice).mean()

        return focal + dice_loss


@torch.no_grad()
def evaluate(model, loader, device):
    """验证集平均 Dice。"""
    model.eval()
    dices = []
    for imgs, masks in loader:
        imgs = imgs.to(device)
        probs = torch.softmax(model(imgs), dim=1)[:, 1]
        preds = (probs > 0.5).float()
        inter = (preds * masks.to(device)).sum(dim=(1, 2))
        union = preds.sum(dim=(1, 2)) + masks.to(device).sum(dim=(1, 2))
        dices.append(((2 * inter + 1) / (union + 1)).cpu().numpy())
    return float(np.concatenate(dices).mean()) if dices else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", default=str(config.FLOODNET_DIR / "images"))
    ap.add_argument("--mask_dir", default=str(config.FLOODNET_DIR / "masks"))
    ap.add_argument("--epochs", type=int, default=config.SEG_EPOCHS)
    ap.add_argument("--batch_size", type=int, default=config.SEG_BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.SEG_LR)
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = get_device()
    print(f"[设备] {device}")

    train_base = FloodDataset(args.image_dir, args.mask_dir, augment=True)
    val_base = FloodDataset(args.image_dir, args.mask_dir, augment=False)
    if len(train_base) < 2:
        raise ValueError("至少需要 2 张图像，才能划分训练集和验证集")
    n_val = max(1, int(len(train_base) * args.val_split))
    n_val = min(n_val, len(train_base) - 1)
    indices = torch.randperm(len(train_base), generator=torch.Generator().manual_seed(42)).tolist()
    train_ds = Subset(train_base, indices[n_val:])
    val_ds = Subset(val_base, indices[:n_val])
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin_memory)

    model = Segmentor(encoder_name=config.SEG_ENCODER,
                      encoder_weights=config.SEG_ENCODER_WEIGHTS,
                      classes=config.SEG_CLASSES).to(device)
    criterion = FocalDiceLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=config.SEG_WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = config.SEG_USE_AMP and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    best_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                logits = model(imgs)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        val_dice = evaluate(model, val_loader, device)
        scheduler.step()
        print(f"Epoch {epoch} | train_loss={total_loss / max(n, 1):.4f} | val_dice={val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            config.SEG_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), config.SEG_CHECKPOINT)
            print(f"  -> 保存最优模型 Dice={best_dice:.4f}")

    print(f"训练完成，最优 Dice={best_dice:.4f}，权重：{config.SEG_CHECKPOINT}")


if __name__ == "__main__":
    main()
