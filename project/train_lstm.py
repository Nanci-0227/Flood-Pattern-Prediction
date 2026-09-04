"""
训练 LSTM 积水趋势预测模型。

输入：CSV 时序文件（至少包含 `ratio` 列，单位归一化面积比）。
可由分割模型处理视频后生成，或 `utils/video_processing.series_from_ratios` 生成。

用法示例：
    python train_lstm.py --csv data/series/ratio.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import Adam
from tqdm import tqdm

import config
from config import get_device
from models.predictor import LSTMPredictor
from utils.video_processing import build_lstm_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="包含 ratio 列的 CSV 文件")
    ap.add_argument("--seq_len", type=int, default=config.SEQ_LEN)
    ap.add_argument("--future_steps", type=int, default=config.FUTURE_STEPS)
    ap.add_argument("--epochs", type=int, default=config.LSTM_EPOCHS)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=config.LSTM_LR)
    ap.add_argument("--val_split", type=float, default=0.2)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "ratio" not in df.columns:
        raise SystemExit("CSV 缺少 `ratio` 列")
    ratios = df["ratio"].to_numpy(dtype=np.float32)

    X, y = build_lstm_dataset(ratios, args.seq_len, args.future_steps)
    print(f"[数据] 序列样本数：{X.shape}，标签：{y.shape}")

    device = get_device()
    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y).to(device)

    ds = TensorDataset(X_t, y_t)
    n_val = max(1, int(len(ds) * args.val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = LSTMPredictor(input_dim=config.LSTM_INPUT_DIM,
                          hidden_size=config.LSTM_HIDDEN,
                          num_layers=config.LSTM_LAYERS,
                          dropout=config.LSTM_DROPOUT,
                          future_steps=args.future_steps).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, n = 0.0, 0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n += 1

        model.eval()
        val_loss, m = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb)
                val_loss += criterion(out, yb).item()
                m += 1
        val_loss = val_loss / max(m, 1)
        print(f"Epoch {epoch} | train_loss={train_loss / n:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            config.LSTM_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), config.LSTM_CHECKPOINT)
            print(f"  -> 保存最优模型 val_loss={best_loss:.6f}")

    print(f"训练完成，最优 val_loss={best_loss:.6f}，权重：{config.LSTM_CHECKPOINT}")


if __name__ == "__main__":
    main()