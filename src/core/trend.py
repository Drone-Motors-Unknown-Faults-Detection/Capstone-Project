"""變化點分析（實驗三核心）：EWMA 異常比例 + 單邊 CUSUM。

- EWMA 異常比例：把逐樣本「開集分數 > 1」的旗標做指數加權平均，
  作為異常密度的平滑估計；比例越過 alarm_frac 觸發警報。
- CUSUM（Page, 1954）：累積 (score − 1) 的正偏移，呈現偏移能量（展示用統計量）。
- 漸進 / 突發判別：警報當下回看最近 window 筆，計算異常比例停留在
  「中間帶」[onset_frac, alarm_frac) 的時間 transition——
  漸進退化會在中間帶徘徊很久，突發跳變只是快速穿過。
  transition ≤ sudden_max 判為「突發」，否則「漸進」。
  （以停留時間而非「起點到警報」計，可免疫健康期零星誤報與比例短暫回落。）
"""

from __future__ import annotations

from collections import deque


class TrendMonitor:
    def __init__(
        self,
        alpha: float = 0.08,
        onset_frac: float = 0.2,
        alarm_frac: float = 0.5,
        sudden_max: int = 12,
        window: int = 120,
        warmup: int = 10,
    ) -> None:
        self.alpha = alpha
        self.onset_frac = onset_frac
        self.alarm_frac = alarm_frac
        self.sudden_max = sudden_max
        self.window = window
        self.warmup = warmup
        self.t = 0
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.ewma = 0.0
        self.cusum = 0.0
        self.alarm_t: int | None = None
        self.kind: str | None = None
        self.transition: int | None = None
        self._history: deque[float] = deque(maxlen=self.window)
        self._armed_at = 0

    def rearm(self) -> None:
        """處理完一次警報（或切換注入來源）後重新武裝；保留時間軸。"""
        self.ewma = 0.0
        self.cusum = 0.0
        self.alarm_t = None
        self.kind = None
        self.transition = None
        self._history.clear()
        self._armed_at = self.t

    def update(self, score: float) -> dict:
        self.t += 1
        flag = 1.0 if score > 1.0 else 0.0
        self.ewma += self.alpha * (flag - self.ewma)
        self.cusum = max(0.0, self.cusum + (score - 1.0))

        alarm_now = False
        if self.alarm_t is None and (self.t - self._armed_at) > self.warmup:
            if self.ewma >= self.alarm_frac:
                self.alarm_t = self.t
                self.transition = sum(
                    1 for e in self._history if self.onset_frac <= e < self.alarm_frac
                )
                self.kind = "sudden" if self.transition <= self.sudden_max else "gradual"
                alarm_now = True
        self._history.append(self.ewma)

        return {
            "ewma": round(self.ewma, 4),
            "cusum": round(self.cusum, 2),
            "alarmed": self.alarm_t is not None,
            "alarm_now": alarm_now,
            "kind": self.kind,
            "transition": self.transition,
        }
