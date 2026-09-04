"""
传统 HSV 颜色阈值兜底（基线方案）
在深度学习分割模型尚未训练/加载完成时，可先用此方法跑通全流程演示。
阈值可根据实际场景在 hsv_water_mask 内调整。
"""
import cv2
import numpy as np


def hsv_water_mask(bgr: np.ndarray, min_area: int = 200) -> np.ndarray:
    """
    使用 HSV 颜色空间检测积水区域，返回 0/1 二值掩码（uint8）。

    积水/水渍在城市路面通常呈现：低饱和度 + 中等亮度（灰蓝/反光）。
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 条件一：低饱和、中等亮度（灰白反光水面）
    gray_water = (s < 80) & (v > 40) & (v < 220)
    # 条件二：略带蓝色且有一定亮度（较深积水）
    blue_water = (h > 90) & (h < 130) & (s > 35) & (v > 50)

    mask = ((gray_water | blue_water) * 255).astype(np.uint8)

    # 形态学去噪：先开后闭，去除孤立噪点、连通水面
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 过滤过小的连通域
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(clean, [c], -1, 255, -1)

    return (clean > 0).astype(np.uint8)