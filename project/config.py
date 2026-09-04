"""
全局配置文件
集中管理项目路径、模型超参数、风险等级映射与公安预案规则。
所有模块通过 `import config` 共享同一套配置，避免散落魔法数字。
"""
from pathlib import Path

# ===================== 路径 =====================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FLOODNET_DIR = DATA_DIR / "floodnet"          # 公开分割训练数据
VIDEO_DIR = DATA_DIR / "video"                 # 自采积水验证视频
SERIES_DIR = DATA_DIR / "series"               # LSTM 训练用时间序列 CSV
RESULT_DIR = ROOT / "result"                   # 输出视频与曲线
CHECKPOINT_DIR = ROOT / "checkpoints"          # 模型权重

for _d in (FLOODNET_DIR, VIDEO_DIR, SERIES_DIR, RESULT_DIR, CHECKPOINT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ===================== 分割模型（U-Net） =====================
SEG_ENCODER = "resnet50"
SEG_ENCODER_WEIGHTS = "imagenet"   # 也可用 None 从头训练
SEG_CLASSES = 2                     # 背景 / 积水
SEG_INPUT_SIZE = (512, 512)
SEG_BATCH_SIZE = 8                  # 训练
SEG_BATCH_SIZE_INFER = 16           # 推理
SEG_LR = 1e-4
SEG_WEIGHT_DECAY = 1e-5
SEG_EPOCHS = 30
SEG_USE_AMP = True                  # 混合精度 fp16
SEG_CHECKPOINT = CHECKPOINT_DIR / "seg_best.pth"

# ===================== LSTM 时序预测 =====================
SEQ_LEN = 10                        # 输入序列长度（帧/采样点）
FUTURE_STEPS = 10                   # 预测未来步数
LSTM_INPUT_DIM = 2                  # [积水面积比, 面积比变化率]
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.2
LSTM_LR = 1e-3
LSTM_EPOCHS = 100
LSTM_CHECKPOINT = CHECKPOINT_DIR / "lstm_best.pth"

# ===================== 多模态融合 =====================
# 输入特征：当前面积比、变化率、LSTM未来预测、降雨强度、(可选)历史排水速率
FUSION_INPUT_DIM = 5
FUSION_HIDDEN = [16, 8]
FUSION_CLASSES = 4                  # 蓝/黄/橙/红
FUSION_DROPOUT = 0.1
FUSION_LR = 1e-3
FUSION_EPOCHS = 100
FUSION_CHECKPOINT = CHECKPOINT_DIR / "fusion_best.pth"

# ===================== 推理/演示 =====================
SAMPLE_INTERVAL = 2.0               # 秒，采样间隔
INFER_DEVICE = "auto"               # auto / cuda / cpu
OVERLAY_ALPHA = 0.45                # 分割掩码叠加透明度

# ===================== 风险等级（BGR 颜色） =====================
# 键 = 类别索引，元组 = (名称, 描述, BGR颜色)
RISK_LEVELS = {
    0: ("蓝", "低风险", (255, 144, 30)),       # opencv 下为橙色/蓝色，见 visualization
    1: ("黄", "中风险", (0, 255, 255)),
    2: ("橙", "高风险", (0, 140, 255)),
    3: ("红", "极高风险", (0, 0, 255)),
}

# ===================== 降雨强度分级 =====================
# 按小时降雨量(mm/h)粗略分级
RAIN_LEVELS = [
    (0.0,  "无降雨/微量"),
    (10.0, "小雨"),
    (25.0, "中雨"),
    (50.0, "大雨"),
    (999.0, "暴雨及以上"),
]

# ===================== 公安应急处置预案 =====================
EMERGENCY_PLAN = {
    0: "蓝色预警：加强视频巡查，持续监测积水变化",
    1: "黄色预警：交警到场疏导交通，设置警示标志",
    2: "橙色预警：封闭低洼路段，调配移动抽水设备",
    3: "红色预警：启动应急预案，组织周边疏散并联动多部门抢险",
}


def rain_level_index(rain_mm_per_h: float) -> int:
    """将降雨强度(mm/h)映射为等级索引 0..4"""
    for i, (thr, _) in enumerate(RAIN_LEVELS):
        if rain_mm_per_h < thr:
            return i
    return len(RAIN_LEVELS) - 1


def risk_label(idx: int) -> str:
    return RISK_LEVELS[idx][0]


def get_device(device_str: str = None):
    """解析运行设备：auto 时优先使用 GPU（cuda）。"""
    import torch
    s = device_str or INFER_DEVICE
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def rule_risk_index(ratio: float, pred_ratio: float, rain_mmh: float) -> int:
    """
    规则兜底：当融合模型不可用时，用阈值映射给出风险等级 0..3。
    ratio=当前积水面积比, pred_ratio=预测面积比, rain_mmh=降雨强度。
    """
    score = 0
    if ratio > 0.15:
        score += 1
    if ratio > 0.30:
        score += 1
    if pred_ratio > 0.35:
        score += 1
    if rain_level_index(rain_mmh) >= 2:
        score += 1
    return min(3, score)