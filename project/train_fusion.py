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
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from tqdm import tqdm

import config
from config import get_device, rule_risk_index
from models.fusion import RiskFusionNet
from utils.metrics import classification_metrics

FEATURE_COLS = ["ratio", "change_rate", "pred_ratio", "rain", "drain_rate"]
REQUIRED_FEATURE_COLS = ["ratio", "change_rate", "pred_ratio", "rain"]


def load_features(csv_path: str):
    df = pd.read_csv(csv_path)
    missing = set(REQUIRED_FEATURE_COLS) - set(df.columns)
    if missing:
        raise SystemExit(f"CSV 缺少必要特征列：{missing}")
    # 在线推理固定输入五维特征；训练阶段缺少排水速率时以 0 填补，
    # 避免训练权重与 infer_video.py 的输入维度不一致。
    if "drain_rate" not in df.columns:
        df["drain_rate"] = 0.0
    cols = FEATURE_COLS
    X = df[cols].to_numpy(dtype=np.float32)

    if "label" in df.columns:
        y = df["label"].to_numpy(dtype=np.int64)
    else:
        print("[提示] 未发现 label 列，使用 config.rule_risk_index 生成伪标签")
        y = np.array([rule_risk_index(r[0], r[2], r[3]) for r in X],
                     dtype=np.int64)
    if np.any((y < 0) | (y >= config.FUSION_CLASSES)):
        raise SystemExit(f"label 必须在 0..{config.FUSION_CLASSES - 1} 范围内")
    return X, y, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="按时间升序排列的特征 CSV")
    ap.add_argument("--epochs", type=int, default=config.FUSION_EPOCHS)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=config.FUSION_LR)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = ap.parse_args()
    if not 0.0 < args.val_split < 1.0:
        raise SystemExit("val_split 必须位于 0 与 1 之间")

    X, y, cols = load_features(args.csv)
    input_dim = X.shape[1]
    print(f"[数据] 特征列：{cols}，样本数：{len(y)}")

    device = get_device(args.device)
    if len(y) < 2:
        raise SystemExit("至少需要 2 条特征记录，才能划分训练集和验证集")
    n_val = min(max(1, int(len(y) * args.val_split)), len(y) - 1)
    split_point = len(y) - n_val
    # CSV 按时间升序排列时，尾部样本作为未来验证集，避免随机切分造成时序泄漏。
    train_ds = TensorDataset(torch.from_numpy(X[:split_point]).float(),
                             torch.from_numpy(y[:split_point]).long())
    val_ds = TensorDataset(torch.from_numpy(X[split_point:]).float(),
                           torch.from_numpy(y[split_point:]).long())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = RiskFusionNet(input_dim=input_dim,
                          hidden=config.FUSION_HIDDEN,
                          num_classes=config.FUSION_CLASSES,
                          dropout=config.FUSION_DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, n = 0.0, 0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n += 1

        model.eval()
        all_pred, all_target = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=1)
                all_pred.append(pred.cpu().numpy())
                all_target.append(yb.cpu().numpy())
        report = classification_metrics(
            np.concatenate(all_pred), np.concatenate(all_target), config.FUSION_CLASSES
        )
        recalls = ", ".join(
            f"{config.risk_label(i)}:{score:.3f}"
            for i, score in enumerate(report["per_class_recall"])
        )
        print(
            f"Epoch {epoch} | train_loss={train_loss / max(n, 1):.4f} "
            f"| val_acc={report['accuracy']:.4f} | macro_f1={report['macro_f1']:.4f} "
            f"| 各等级召回率[{recalls}]"
        )

        if report["macro_f1"] > best_f1:
            best_f1 = report["macro_f1"]
            config.FUSION_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), config.FUSION_CHECKPOINT)
            print(f"  -> 保存最优模型 macro_f1={best_f1:.4f}")

    print(f"训练完成，最优 macro_f1={best_f1:.4f}，权重：{config.FUSION_CHECKPOINT}")


if __name__ == "__main__":
    main()
