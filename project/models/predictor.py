"""
洪涝风险趋势预测模型（任务B）
基于双层 LSTM，输入积水面积比时间序列，回归预测未来面积比。
"""
import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    """双层 LSTM，输入 (B, seq_len, input_dim)，输出未来 future_steps 步预测。"""

    def __init__(self, input_dim: int = 2, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2,
                 future_steps: int = 1):
        super().__init__()
        self.future_steps = future_steps
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, future_steps)

    def forward(self, x):
        # x: (B, seq_len, input_dim)
        out, _ = self.lstm(x)
        last = out[:, -1, :]          # 取最后一个时间步的隐状态
        return self.fc(last)          # (B, future_steps)