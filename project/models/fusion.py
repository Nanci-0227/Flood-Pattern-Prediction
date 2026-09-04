"""
多模态信息融合模型（任务C）
轻量 MLP：融合视觉特征(面积比)、时序特征(LSTM预测)、气象特征(降雨)，
输出 4 类风险等级 logits。
"""
import torch
import torch.nn as nn


class RiskFusionNet(nn.Module):
    """小规模 MLP 融合网络，隐藏层 config 由参数指定。"""

    def __init__(self, input_dim: int = 5, hidden=(16, 8),
                 num_classes: int = 4, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)            # (B, num_classes) 未归一化 logits

    @torch.no_grad()
    def predict(self, x):
        """返回风险等级索引 (B,)。"""
        self.eval()
        return self.forward(x).argmax(dim=1)