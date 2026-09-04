"""
视频 / 帧处理工具，以及 LSTM 训练序列的切窗生成。
"""
import math
from typing import Iterator, Tuple

import cv2
import numpy as np
import pandas as pd


def read_video_info(path: str):
    """返回 (fps, 总帧数) 的元组。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频：{path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total


def iter_frames(path: str, step: int = 1, resize=None) -> Iterator[np.ndarray]:
    """
    按帧步长迭代读取视频帧（BGR）。
    step=1 表示逐帧；resize=(w, h) 时同步缩放。
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频：{path}")
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            if resize is not None:
                frame = cv2.resize(frame, resize)
            yield frame
        idx += 1
    cap.release()


def series_from_ratios(ratios: np.ndarray, sample_interval: float = 2.0) -> pd.DataFrame:
    """
    由积水面积比序列生成时序特征 DataFrame。
    列：time, ratio, change_rate
    """
    ratios = np.asarray(ratios, dtype=np.float32)
    n = len(ratios)
    t = np.arange(n, dtype=np.float32) * sample_interval
    change = np.zeros_like(ratios)
    if n > 1:
        change[1:] = np.diff(ratios) / max(sample_interval, 1e-6)
    return pd.DataFrame({
        "time": t,
        "ratio": ratios,
        "change_rate": change,
    })


def build_lstm_dataset(ratios: np.ndarray, seq_len: int = 10,
                       future_steps: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    切窗生成 LSTM 训练样本。

    特征 X: (M, seq_len, 2)，两维分别为 [面积比, 变化率]
    标签 y: (M, future_steps)，为未来 future_steps 个时刻的面积比
    """
    ratios = np.asarray(ratios, dtype=np.float32)
    n_frames = len(ratios) - future_steps - seq_len + 1
    if n_frames <= 0:
        raise ValueError(
            f"序列长度不足：需要至少 {seq_len + future_steps} 个点，当前 {len(ratios)}")

    xs, ys = [], []
    for i in range(n_frames):
        window = ratios[i:i + seq_len]
        change = np.zeros_like(window)
        change[1:] = np.diff(window)
        feat = np.stack([window, change], axis=1)          # (seq_len, 2)
        label = ratios[i + seq_len:i + seq_len + future_steps]  # (future_steps,)
        xs.append(feat)
        ys.append(label)

    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def pad_to_seq(values: np.ndarray, seq_len: int) -> np.ndarray:
    """在线推理时，长度不足 seq_len 则用首值复制补齐。"""
    values = np.asarray(values, dtype=np.float32)
    n = len(values)
    if n >= seq_len:
        return values[-seq_len:]
    pad = np.full(seq_len - n, values[0])
    return np.concatenate([pad, values], axis=0)