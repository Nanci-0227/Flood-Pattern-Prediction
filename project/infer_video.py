"""
端到端推理与演示脚本
链路：视频 → 深度分割(或HSV兜底) → 积水面积比/变化率 → LSTM预测 → 融合 → 风险等级 → 输出叠加视频。

用法示例：
    python infer_video.py --video data/video/demo.mp4 --backend unet --rain 30
    python infer_video.py --video data/video/demo.mp4 --backend hsv --rain 30

说明：
    - --backend unet 且未找到分割权重时，自动回退到 hsv；
    - 未找到 LSTM / 融合权重时，分别用线性外推 / 规则映射兜底，
      保证全流程在任何数据准备阶段都能跑通。
"""
import argparse

import cv2
import numpy as np
import torch

import config
from config import get_device, rule_risk_index
from models.segmentor import Segmentor
from models.predictor import LSTMPredictor
from models.fusion import RiskFusionNet
from utils.baseline import hsv_water_mask
from utils.visualization import overlay_mask, draw_risk_panel, draw_trend_curve
from utils.video_processing import read_video_info, iter_frames

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_segmentor(device):
    if not config.SEG_CHECKPOINT.exists():
        return None
    model = Segmentor(encoder_name=config.SEG_ENCODER, encoder_weights=None,
                      classes=config.SEG_CLASSES).to(device)
    model.load_state_dict(torch.load(config.SEG_CHECKPOINT, map_location=device))
    model.eval()
    return model


def load_lstm(device):
    if not config.LSTM_CHECKPOINT.exists():
        return None
    model = LSTMPredictor(input_dim=config.LSTM_INPUT_DIM,
                          hidden_size=config.LSTM_HIDDEN,
                          num_layers=config.LSTM_LAYERS,
                          dropout=config.LSTM_DROPOUT,
                          future_steps=config.FUTURE_STEPS).to(device)
    model.load_state_dict(torch.load(config.LSTM_CHECKPOINT, map_location=device))
    model.eval()
    return model


def load_fusion(device):
    if not config.FUSION_CHECKPOINT.exists():
        return None
    model = RiskFusionNet(input_dim=config.FUSION_INPUT_DIM,
                          hidden=config.FUSION_HIDDEN,
                          num_classes=config.FUSION_CLASSES,
                          dropout=config.FUSION_DROPOUT).to(device)
    model.load_state_dict(torch.load(config.FUSION_CHECKPOINT, map_location=device))
    model.eval()
    return model


def to_tensor(frame: np.ndarray) -> torch.Tensor:
    """BGR帧 -> (1,3,512,512) 归一化张量。"""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, config.SEG_INPUT_SIZE)
    img = img.astype(np.float32) / 255.0
    img = (img - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()


def segment_frame(frame, seg_model, device, backend):
    mask_512 = None
    if backend == "unet" and seg_model is not None:
        x = to_tensor(frame).to(device)
        mask_512 = seg_model.predict_mask(x)[0].cpu().numpy()  # (512,512) 0/1
        mask = cv2.resize(mask_512.astype(np.uint8),
                          (frame.shape[1], frame.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        return mask > 0
    return hsv_water_mask(frame).astype(bool)


def linear_extrapolate(history, future_steps):
    """无 LSTM 时的简单线性外推兜底。"""
    if not history:
        return 0.0
    if len(history) < 2:
        return history[-1]
    k = min(5, len(history))
    ys = np.array(history[-k:], dtype=np.float32)
    slope = np.polyfit(np.arange(k), ys, 1)[0]
    return float(np.clip(history[-1] + slope * future_steps, 0.0, 1.0))


def lstm_predict(lstm, device, history):
    seq = np.asarray(history[-config.SEQ_LEN:], dtype=np.float32)
    change = np.zeros_like(seq)
    change[1:] = np.diff(seq)
    feat = np.stack([seq, change], axis=1)               # (seq_len, 2)
    x = torch.from_numpy(feat).unsqueeze(0).to(device)   # (1, seq_len, 2)
    with torch.no_grad():
        pred = lstm(x)[0].cpu().numpy()                  # (future_steps,)
    return float(np.clip(pred[-1], 0.0, 1.0))


def fusion_predict(fusion, device, ratio, change, pred_ratio, rain):
    feat = torch.tensor([[ratio, change, pred_ratio, rain, 0.0]],
                        dtype=torch.float32).to(device)
    with torch.no_grad():
        return int(fusion(feat).argmax(dim=1).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="输入视频路径")
    ap.add_argument("--backend", choices=["unet", "hsv"], default="unet")
    ap.add_argument("--rain", type=float, default=25.0, help="降雨强度 mm/h")
    ap.add_argument("--output", default=None, help="输出视频路径（默认 result/output.mp4）")
    args = ap.parse_args()

    device = get_device()
    print(f"[设备] {device}")

    seg_model = load_segmentor(device)
    backend = args.backend
    if backend == "unet" and seg_model is None:
        print("[提示] 未找到分割权重，回退到 HSV 兜底。")
        backend = "hsv"
    else:
        print(f"[分割] 使用 {'U-Net' if backend == 'unet' else 'HSV 传统方法'}")

    lstm = load_lstm(device)
    fusion = load_fusion(device)
    print(f"[预测] LSTM：{'已加载' if lstm else '未加载(线性外推兜底)'}")
    print(f"[融合] MLP：{'已加载' if fusion else '未加载(规则映射兜底)'}")

    fps, _ = read_video_info(args.video)
    sample_step = max(1, int(round(fps * config.SAMPLE_INTERVAL)))

    out_path = args.output or str(config.RESULT_DIR / "output.mp4")
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)

    writer = None
    history = []
    frame_idx = 0

    for frame in iter_frames(args.video, step=1):
        mask = segment_frame(frame, seg_model, device, backend)
        ratio = float(mask.mean())

        if frame_idx % sample_step == 0:
            history.append(ratio)

        change = (history[-1] - history[-2]) if len(history) >= 2 else 0.0

        if len(history) >= config.SEQ_LEN:
            pred_ratio = (lstm_predict(lstm, device, history) if lstm
                          else linear_extrapolate(history, config.FUTURE_STEPS))
        else:
            pred_ratio = ratio

        if fusion is not None:
            risk_idx = fusion_predict(fusion, device, ratio, change, pred_ratio, args.rain)
        else:
            risk_idx = rule_risk_index(ratio, pred_ratio, args.rain)

        _, _, color = config.RISK_LEVELS[risk_idx]
        vis = overlay_mask(frame, mask, color=color, alpha=config.OVERLAY_ALPHA)
        vis = draw_risk_panel(vis, {
            "area_ratio": ratio,
            "change_rate": change,
            "risk_idx": risk_idx,
            "pred_ratio": pred_ratio,
            "rain": args.rain,
        })

        if writer is None:
            h, w = vis.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (w, h))
        writer.write(vis)
        frame_idx += 1

    if writer is not None:
        writer.release()

    if history:
        draw_trend_curve(history, str(config.RESULT_DIR / "trend.png"))
        print(f"[完成] 共 {frame_idx} 帧，采样点 {len(history)} 个")
    print(f"输出视频：{out_path}")


if __name__ == "__main__":
    main()