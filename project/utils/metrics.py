"""
评估指标：用于分割（Dice / IoU / mIoU）与回归（MAE / RMSE）。
"""
import numpy as np


def dice_coef(pred_mask, gt_mask, smooth: float = 1e-6) -> float:
    """二值掩码 Dice 系数，输入为 numpy 0/1 数组。"""
    pred = np.asarray(pred_mask).astype(bool)
    gt = np.asarray(gt_mask).astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return (2.0 * inter + smooth) / (pred.sum() + gt.sum() + smooth)


def iou(pred_mask, gt_mask, smooth: float = 1e-6) -> float:
    """二值掩码 IoU（交并比）。"""
    pred = np.asarray(pred_mask).astype(bool)
    gt = np.asarray(gt_mask).astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return (inter + smooth) / (union + smooth)


def mIoU(pred_masks, gt_masks) -> float:
    """多张掩码的平均 IoU。"""
    scores = [iou(p, g) for p, g in zip(pred_masks, gt_masks)]
    return float(np.mean(scores)) if scores else 0.0


def mae(pred, target) -> float:
    """平均绝对误差。"""
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    return float(np.mean(np.abs(pred - target)))


def rmse(pred, target) -> float:
    """均方根误差。"""
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    return float(np.sqrt(np.mean((pred - target) ** 2)))