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


def classification_metrics(pred, target, num_classes: int) -> dict:
    """计算多分类混淆矩阵、准确率及宏平均 Precision/Recall/F1。"""
    pred = np.asarray(pred, dtype=np.int64).reshape(-1)
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    if pred.shape != target.shape:
        raise ValueError("pred 与 target 的形状必须一致")
    if len(target) == 0:
        raise ValueError("分类评估至少需要一个样本")
    if np.any((pred < 0) | (pred >= num_classes)) or np.any(
            (target < 0) | (target >= num_classes)):
        raise ValueError("分类标签超出有效范围")

    matrix = np.bincount(
        num_classes * target + pred,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    tp = np.diag(matrix).astype(np.float64)
    precision = tp / np.maximum(matrix.sum(axis=0), 1)
    recall = tp / np.maximum(matrix.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "confusion_matrix": matrix,
        "accuracy": float(tp.sum() / matrix.sum()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
    }
