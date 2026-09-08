"""风险预警的稳定化与审计记录工具。"""
import csv
from collections import deque
from pathlib import Path

import numpy as np


class RiskStateSmoother:
    """高风险立即升级，降级需连续确认，避免监控画面导致告警频繁跳变。"""

    def __init__(self, window_size: int = 5, downgrade_hold: int = 3):
        if window_size < 1 or downgrade_hold < 1:
            raise ValueError("window_size 和 downgrade_hold 必须大于 0")
        self.history = deque(maxlen=window_size)
        self.current = 0
        self.lower_count = 0
        self.downgrade_hold = downgrade_hold

    def update(self, raw_risk: int) -> int:
        if not 0 <= raw_risk <= 3:
            raise ValueError("raw_risk 必须在 0..3 范围内")
        self.history.append(raw_risk)
        median_risk = int(np.median(self.history))

        # 真正的升级不等待窗口，避免延迟高风险预警。
        if raw_risk > self.current:
            self.current = raw_risk
            self.lower_count = 0
        elif median_risk < self.current:
            self.lower_count += 1
            if self.lower_count >= self.downgrade_hold:
                self.current = median_risk
                self.lower_count = 0
        else:
            self.lower_count = 0
        return self.current


class RiskEventWriter:
    """以 CSV 留存每个采样点的决策输入和输出，便于复盘。"""

    FIELDNAMES = [
        "time_sec", "area_ratio", "change_rate", "pred_ratio", "rain_mmh",
        "raw_risk", "stable_risk", "source", "confidence", "emergency_plan",
    ]

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="", encoding="utf-8-sig")
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDNAMES)
        self.writer.writeheader()

    def write(self, **event) -> None:
        self.writer.writerow({name: event.get(name, "") for name in self.FIELDNAMES})
        self.file.flush()

    def close(self) -> None:
        self.file.close()
