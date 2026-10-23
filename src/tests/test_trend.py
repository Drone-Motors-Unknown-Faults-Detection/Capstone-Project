"""core.trend.TrendMonitor：EWMA 異常比例、CUSUM、warmup、突發／漸進判別與 rearm（#276 K05）。"""

from __future__ import annotations

import unittest

from core.trend import TrendMonitor


def _feed(monitor: TrendMonitor, scores) -> list[dict]:
    return [monitor.update(score) for score in scores]


class TrendMonitorTests(unittest.TestCase):
    def test_defaults_match_agent_md_table(self):
        monitor = TrendMonitor()
        self.assertEqual((monitor.alpha, monitor.onset_frac, monitor.alarm_frac), (0.08, 0.2, 0.5))
        self.assertEqual((monitor.sudden_max, monitor.window, monitor.warmup), (12, 120, 10))
        self.assertEqual(monitor.t, 0)

    def test_update_returns_rounded_fields(self):
        out = TrendMonitor().update(3.0)
        self.assertEqual(set(out), {"ewma", "cusum", "alarmed", "alarm_now", "kind", "transition"})
        self.assertEqual(out["ewma"], 0.08)  # 0 + 0.08 × (1 − 0)
        self.assertEqual(out["cusum"], 2.0)  # max(0, 0 + (3 − 1))
        self.assertFalse(out["alarmed"])
        self.assertIsNone(out["kind"])

    def test_healthy_scores_keep_ewma_and_cusum_at_zero(self):
        monitor = TrendMonitor()
        outputs = _feed(monitor, [0.3, 0.9, 1.0] * 20)  # 1.0 不算超線（嚴格大於 1）
        self.assertTrue(all(o["ewma"] == 0.0 and o["cusum"] == 0.0 for o in outputs))
        self.assertFalse(monitor.alarm_t)

    def test_cusum_accumulates_positive_shift_and_floors_at_zero(self):
        monitor = TrendMonitor()
        monitor.update(3.0)
        self.assertEqual(monitor.cusum, 2.0)
        monitor.update(0.5)  # 2.0 + (0.5 − 1) = 1.5
        self.assertEqual(monitor.cusum, 1.5)
        _feed(monitor, [0.0] * 5)
        self.assertEqual(monitor.cusum, 0.0)

    def test_no_alarm_inside_warmup_even_when_every_sample_is_abnormal(self):
        monitor = TrendMonitor(warmup=10)
        outputs = _feed(monitor, [5.0] * 10)
        self.assertTrue(all(not o["alarmed"] for o in outputs))
        self.assertGreater(monitor.ewma, 0.5)  # 比例早已過線，只是 warmup 擋住
        self.assertTrue(monitor.update(5.0)["alarm_now"])
        self.assertEqual(monitor.alarm_t, 11)

    def test_sudden_jump_alarms_once_with_short_transition(self):
        monitor = TrendMonitor()
        _feed(monitor, [0.5] * 30)
        outputs = _feed(monitor, [10.0] * 30)
        alarms = [i for i, o in enumerate(outputs) if o["alarm_now"]]
        self.assertEqual(len(alarms), 1)
        self.assertEqual(monitor.kind, "sudden")
        self.assertLessEqual(monitor.transition, monitor.sudden_max)
        self.assertTrue(all(o["alarmed"] for o in outputs[alarms[0]:]))
        self.assertTrue(all(o["kind"] == "sudden" for o in outputs[alarms[0]:]))
        # 警報後 ewma 繼續更新但 alarm_t 不再改變
        alarm_t = monitor.alarm_t
        monitor.update(10.0)
        self.assertEqual(monitor.alarm_t, alarm_t)

    def test_gradual_drift_stays_in_middle_band_long_enough(self):
        # 異常比例約 1/3 停在 [0.2, 0.5) 很久，再全數超線：中間帶停留 > sudden_max → 漸進
        monitor = TrendMonitor()
        _feed(monitor, [1.5, 0.5, 0.5] * 40)
        self.assertIsNone(monitor.alarm_t)
        self.assertTrue(0.2 <= monitor.ewma < 0.5)
        _feed(monitor, [2.0] * 30)
        self.assertEqual(monitor.kind, "gradual")
        self.assertGreater(monitor.transition, monitor.sudden_max)

    def test_transition_counts_only_middle_band_samples_inside_window(self):
        # window=5：警報當下只回看最近 5 筆 ewma，長時間徘徊也最多算 5
        monitor = TrendMonitor(window=5, warmup=0)
        _feed(monitor, [1.5, 0.5, 0.5] * 40)
        _feed(monitor, [2.0] * 30)
        self.assertEqual(monitor.kind, "sudden")
        self.assertLessEqual(monitor.transition, 5)

    def test_rearm_keeps_time_axis_and_reapplies_warmup(self):
        monitor = TrendMonitor(warmup=3, alpha=0.9)  # alpha 0.9：一筆超線 ewma 就到 0.9
        _feed(monitor, [10.0] * 10)
        self.assertEqual(monitor.alarm_t, 4)
        t_before = monitor.t
        monitor.rearm()
        self.assertEqual(monitor.t, t_before)
        self.assertEqual((monitor.ewma, monitor.cusum, monitor.alarm_t, monitor.kind, monitor.transition),
                         (0.0, 0.0, None, None, None))
        outputs = _feed(monitor, [10.0] * 3)
        self.assertTrue(all(not o["alarmed"] for o in outputs))
        self.assertTrue(monitor.update(10.0)["alarm_now"])
        self.assertEqual(monitor.alarm_t, t_before + 4)

    def test_reset_restarts_the_time_axis(self):
        monitor = TrendMonitor()
        _feed(monitor, [10.0] * 20)
        monitor.reset()
        self.assertEqual((monitor.t, monitor.ewma, monitor.cusum, monitor.alarm_t), (0, 0.0, 0.0, None))
        self.assertEqual(monitor.update(0.0), {"ewma": 0.0, "cusum": 0.0, "alarmed": False,
                                               "alarm_now": False, "kind": None, "transition": None})

    def test_same_scores_give_same_outputs(self):
        scores = [0.4, 1.2, 0.8, 3.0, 0.2] * 30
        self.assertEqual(_feed(TrendMonitor(), scores), _feed(TrendMonitor(), scores))


if __name__ == "__main__":
    unittest.main()
