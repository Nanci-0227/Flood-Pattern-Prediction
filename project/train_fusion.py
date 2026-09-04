"""
训练多模态融合模型（小 MLP 风险分类）。

输入：特征 CSV，列顺序可自定义，默认特征列：
    ratio(当前面积比), change_rate(变化率), pred_ratio(LSTM预测),
    rain(降雨mm/h), drain_rate(历史排水速率，可选)
若 CSV 无 label 列，则用 config.rule_risk_index 生成伪标签，便于快速跑通。

用法示例：
    python train_fusion.py --csv data/series/fusion_features.csv
"""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import Adam
from tqdm import tqdm

import config
from config import get_device, rule_risk_index
from models.fusion import RiskFusionNet

FEATURE_COLS = ["ratio", "change_rate", "pred_ratio", "rain", "drain_rate"]


def load_features(csv_path: str):
    df = pd.read_csv(csv_path)
    cols = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(["ratio", "rain"]) - set(cols)
    if missing:
        raise SystemExit(f"CSV 缺少必要特征列：{missing}")
    X = df[cols].to_numpy(dtype=np.float32)

    if "label" in df.columns:
        y = df["label"].to_numpy(dtype=np.int64)
    else:
        print("[提示] 未发现 label 列，使用 config.rule_risk_index 生成伪标签")
        y = np.array([rule_risk_index(r[0], r[2] if "pred_ratio" in cols else r[0],
                                      r[cols.index("rain")]) for r in X],
                     dtype=np.int64)
    return X, y, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--epochs", type=int, default=config.FUSION_EPOCHS)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=config.FUSION_LR)
    ap.add_argument("--val_split", type=float, default=0.2)
    args = ap.parse_args()

    X, y, cols = load_features(args.csv)
    input_dim = X.shape[1]
    print(f"[数据] 特征列：{cols}，样本数：{len(y)}")

    device = get_device()
    ds = TensorDataset(torch.from_numpy(X).float().to(device),
                       torch.from_numpy(y).long().to(device))
    n_val = max(1, int(len(ds) * args.val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = RiskFusionNet(input_dim=input_dim,
                          hidden=config.FUSION_HIDDEN,
                          num_classes=config.FUSION_CLASSES,
                          dropout=config.FUSION_DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, n = 0.0, 0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n += 1

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb).argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        acc = correct / max(total, 1)
        print(f"Epoch {epoch} | train_loss={train_loss / n:.4f} | val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            config.FUSION_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), config.FUSION_CHECKPOINT)
            print(f"  -> 保存最优模型 acc={best_acc:.4f}")

    print(f"训练完成，最优 val_acc={best_acc:.4f}，权重：{config.FUSION_CHECKPOINT}")


if __name__ == "__main__":
    main()