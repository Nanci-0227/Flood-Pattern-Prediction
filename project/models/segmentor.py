"""
积水区域分割模型（任务A）
基于 segmentation_models_pytorch 封装的 U-Net + ResNet 编码器。
"""
import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
except ImportError as e:  # pragma: no cover
    raise ImportError("请先安装 segmentation-models-pytorch：pip install segmentation-models-pytorch") from e


class Segmentor(nn.Module):
    """U-Net 语义分割网络，输出 2 类 softmax 概率图（背景 / 积水）。"""

    def __init__(self, encoder_name: str = "resnet50",
                 encoder_weights: str = "imagenet",
                 in_channels: int = 3,
                 classes: int = 2):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation="softmax",   # 输出 (B, C, H, W) 概率图
        )

    def forward(self, x):
        return self.model(x)

    @torch.no_grad()
    def predict_mask(self, x):
        """推理：返回二值掩码 (B, H, W)，1 表示积水区域。"""
        self.eval()
        probs = self.forward(x)          # (B, 2, H, W)
        return probs.argmax(dim=1)       # 类别 1 为积水


def extract_water_ratio(mask: torch.Tensor) -> torch.Tensor:
    """由二值掩码 (B, H, W) 计算积水面积占比 (B,)。"""
    return mask.float().mean(dim=(1, 2))