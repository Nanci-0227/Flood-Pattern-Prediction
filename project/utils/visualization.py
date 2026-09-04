"""
可视化工具：掩码叠加、风险面板绘制、趋势曲线导出。
"""
import cv2
import numpy as np

import config


def overlay_mask(frame: np.ndarray, mask: np.ndarray, color=(0, 0, 255),
                 alpha: float = 0.45) -> np.ndarray:
    """将二值掩码以半透明颜色叠加到帧上。mask 为 0/1 (H, W)。"""
    overlay = frame.copy()
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    colored = np.zeros_like(frame)
    colored[mask > 0] = color
    return cv2.addWeighted(overlay, 1.0, colored, alpha, 0.0)


def draw_risk_panel(frame: np.ndarray, info: dict) -> np.ndarray:
    """
    在帧左上角绘制风险信息面板。
    info 需包含：area_ratio, change_rate, risk_idx, pred_ratio(可选), rain
    """
    out = frame.copy()
    h = out.shape[0]

    risk_idx = int(info.get("risk_idx", 0))
    label, desc, color = config.RISK_LEVELS[risk_idx]
    # 顶部风险色条
    cv2.rectangle(out, (0, 0), (out.shape[1], 46), color, -1)
    cv2.putText(out, f"风险等级: {label} {desc}", (14, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    lines = [
        f"积水面积比: {info.get('area_ratio', 0.0):.3f}",
        f"面积变化率: {info.get('change_rate', 0.0):.4f}",
    ]
    if info.get("pred_ratio") is not None:
        lines.append(f"未来预测比: {info['pred_ratio']:.3f}")
    if info.get("rain") is not None:
        lines.append(f"降雨强度: {info['rain']:.1f} mm/h")

    y = 70
    for line in lines:
        cv2.putText(out, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26

    # 底部公安预案提示条
    plan = config.EMERGENCY_PLAN[risk_idx]
    bar_y = h - 40
    cv2.rectangle(out, (0, bar_y), (out.shape[1], h), (30, 30, 30), -1)
    cv2.putText(out, plan, (14, h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 1, cv2.LINE_AA)

    return out


def draw_trend_curve(history: list, out_path: str):
    """将面积比历史曲线导出为 PNG（需 matplotlib）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.plot(history, marker="o", color="#1f77b4")
    plt.title("积水面积比随时间变化")
    plt.xlabel("采样点")
    plt.ylabel("积水面积比")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()