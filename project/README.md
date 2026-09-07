# 视频积水识别与洪涝风险预警系统

基于深度学习的城市道路积水检测与风险分级预警系统。通过对监控视频逐帧分析，自动分割积水区域、预测积水蔓延趋势，并结合降雨强度融合多模态信息，输出四级风险预警与公安应急处置预案。

## 功能特性

- **积水区域分割（任务 A）**：基于 U-Net + ResNet 编码器的语义分割模型，识别图像中的积水区域。
- **积水趋势预测（任务 B）**：基于双层 LSTM，根据历史积水面积比序列，回归预测未来面积比。
- **多模态风险融合（任务 C）**：轻量 MLP 融合视觉特征、时序特征与气象特征，输出四级风险等级。
- **全链路兜底机制**：模型权重缺失时自动回退，保证任意数据准备阶段都能跑通完整流程。
- **端到端推理演示**：输入视频 → 输出叠加了分割掩码、风险面板与预案提示的可视化视频。

## 系统流程

```
视频帧 → 积水区域分割(U-Net / HSV兜底) → 面积比/变化率
        → LSTM 未来预测(线性外推兜底) → 多模态融合(规则映射兜底)
        → 风险等级 → 叠加可视化输出视频 + 趋势曲线
```

## 风险分级

| 等级 | 名称 | 应急响应 |
| :---: | :---: | :--- |
| 蓝 | 低风险 | 加强视频巡查，持续监测积水变化 |
| 黄 | 中风险 | 交警到场疏导交通，设置警示标志 |
| 橙 | 高风险 | 封闭低洼路段，调配移动抽水设备 |
| 红 | 极高风险 | 启动应急预案，组织疏散并联动多部门抢险 |

## 目录结构

```
project/
├── config.py               # 全局配置：路径、超参数、风险等级、预案规则
├── models/
│   ├── segmentor.py        # U-Net 分割模型（任务A）
│   ├── predictor.py        # LSTM 趋势预测模型（任务B）
│   └── fusion.py           # MLP 风险融合模型（任务C）
├── utils/
│   ├── baseline.py         # HSV 传统方法兜底
│   ├── video_processing.py # 视频读取、序列切窗生成
│   ├── metrics.py          # 评估指标（Dice/IoU/MAE/RMSE）
│   └── visualization.py    # 掩码叠加、风险面板、趋势曲线
├── train_seg.py            # 训练分割模型
├── train_lstm.py           # 训练 LSTM 预测模型
├── train_fusion.py         # 训练风险融合模型
├── infer_video.py          # 端到端推理演示
└── requirements.txt        # 依赖清单
```

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：PyTorch、torchvision、OpenCV、segmentation-models-pytorch、albumentations、pandas、matplotlib。

## 数据准备

首次运行时，`config.py` 会自动创建以下数据目录：

```
data/
├── floodnet/        # 分割训练数据（images/ 图片 + masks/ 二值掩码）
├── video/           # 自采积水验证视频
└── series/          # LSTM/融合训练的时序 CSV
```

## 使用方法

### 1. 训练分割模型

```bash
python train_seg.py --image_dir data/floodnet/images --mask_dir data/floodnet/masks
```

### 2. 训练 LSTM 预测模型

CSV 需包含 `ratio` 列（归一化面积比）：

```bash
python train_lstm.py --csv data/series/ratio.csv
```

### 3. 训练风险融合模型

特征列：`ratio`、`change_rate`、`pred_ratio`、`rain`、`drain_rate`（可选），无 `label` 列时自动用规则生成伪标签：

```bash
python train_fusion.py --csv data/series/fusion_features.csv
```

### 4. 端到端推理演示

```bash
# 使用 U-Net 分割（无权重时自动回退 HSV）
python infer_video.py --video data/video/demo.mp4 --backend unet --rain 30

# 使用 HSV 传统方法
python infer_video.py --video data/video/demo.mp4 --backend hsv --rain 30
```

参数说明：

| 参数 | 说明 |
| :--- | :--- |
| `--video` | 输入视频路径 |
| `--backend` | 分割后端：`unet` / `hsv`，默认 `unet` |
| `--rain` | 降雨强度（mm/h），默认 25 |
| `--output` | 输出视频路径，默认 `result/output.mp4` |
| `--event_log` | 风险事件 CSV，默认 `result/risk_events.csv` |
| `--risk_window` | 风险平滑窗口，默认 5 个采样点 |
| `--downgrade_hold` | 允许降级前需连续确认的低风险采样点数，默认 3 |

推理会将每个采样点的面积比、预测值、原始风险、稳定风险、模型置信度（如有）及处置建议写入 CSV。风险升级即时生效；风险降级需要连续低风险确认，以减少光照、反光和短暂遮挡导致的告警跳变。

## 兜底机制

| 缺失组件 | 兜底方案 |
| :--- | :--- |
| 分割权重 | 回退 HSV 颜色阈值分割 |
| LSTM 权重 | 线性外推预测 |
| 融合权重 | 规则阈值映射 |

## 环境要求

- Python 3.8+
- 建议使用 CUDA GPU（推理/训练均支持 CPU 回退）
