"""實驗頁「邊跑邊畫」目前／最近一次播放的狀態，給 AI 問答讀取。

server.py 的 StreamHandler 每送出一個 frame 就呼叫 ``PlaybackTracker.update()``；本模組只彙整，
不做任何判定，數字全部取自各實驗 iter_run() 的既有輸出。

狀態分三層：
    current    最近一筆的系統判定（分數、判定、EWMA、健康指數…），不含真值
    events     重點事件：候選群、自學、變化點警報、嚴重度改變（最多 MAX_EVENTS 筆）
    reference  事後對照：由注入配置（真值）算出的評估結果，只能當評估說明，不是系統當下的判定

真值欄位（注入配置、expected、is_fault、候選群 majority／purity、onset）只進 reference
（AGENT.md 鐵則 6：真實身分只在確認／評估時揭示）。
"""

from __future__ import annotations

import copy
import time

from experiments.exp3_trend import KIND_DISPLAY
from web.experiments import CATALOG

MAX_EVENTS = 12
MAX_REFERENCE = 40
_SPEC = {e["id"]: e for e in CATALOG}

# 實驗八逐窗輸出中屬於系統判定的欄位（is_fault 是真值，不在此列）
WINDOW_FIELDS = ("window_index", "health_index", "severity_stage", "trend", "degradation_rate",
                 "openset_score", "is_unknown_fault", "fault_type", "prediction_confidence",
                 "uncertainty", "alarm_state", "alarm_reason", "change_point_state", "data_quality")


def _round(v, nd=3):
    return round(v, nd) if isinstance(v, float) else v


def _now() -> str:
    return time.strftime("%H:%M:%S")


class PlaybackTracker:
    def __init__(self):
        self.state: dict | None = None

    # -- 生命週期 ----------------------------------------------------------------

    def start(self, exp_id: str, params: dict) -> None:
        spec = _SPEC.get(exp_id, {})
        self.state = {
            "exp_id": exp_id,
            "no": spec.get("no", exp_id),
            "sub": spec.get("sub"),
            "title": spec.get("title", ""),
            "params": {},
            "status": "running",
            "started_at": _now(),
            "updated_at": _now(),
            "samples": 0,
            "known": None,
            "current": None,
            "events": [],
            "reference": [],
            "result": None,
            "error": None,
        }
        self._set_params(params)

    def finish(self, status: str, error: str | None = None) -> None:
        """串流結束時呼叫；已經 done／error 的不覆蓋（被瀏覽器中斷才記 stopped）。"""
        st = self.state
        if st is None or st["status"] != "running":
            return
        st["status"] = status
        st["error"] = error
        st["updated_at"] = _now()

    def snapshot(self) -> dict | None:
        return copy.deepcopy(self.state)

    # -- frame 彙整 --------------------------------------------------------------

    def update(self, frame: dict) -> None:
        st = self.state
        if st is None:
            return
        st["updated_at"] = _now()
        kind = frame.get("event")
        handler = getattr(self, f"_on_{kind}", None)
        if handler is not None:
            handler(st, frame)

    def _event(self, st: dict, text: str) -> None:
        st["events"].append(text)
        del st["events"][:-MAX_EVENTS]

    def _reference(self, st: dict, item: dict) -> None:
        st["reference"].append(item)
        del st["reference"][:-MAX_REFERENCE]

    def _set_params(self, params: dict) -> None:
        params = dict(params or {})
        source = params.pop("config", None)   # 實驗八選的注入配置 = 真值
        self.state["params"] = params
        if source is not None:
            self.state["source_config"] = source

    def _on_start(self, st, f):
        self._set_params(f.get("params") or {})

    def _on_fitted(self, st, f):
        if f.get("known") is not None:
            st["known"] = list(f["known"])
        self._event(st, "模型擬合完成，開始播放")

    def _on_scores(self, st, f):   # 實驗一：一段配置的分數
        scores = f.get("scores") or []
        st["samples"] += len(scores)
        if f.get("offset", 0) == 0:
            st["segment"] = st.get("segment", 0) + 1
        st["current"] = {
            "segment": st.get("segment", 1),
            "segment_kind": "健康 holdout" if f.get("kind") == "healthy-holdout" else "未知配置",
            "last_scores": [_round(s) for s in scores[-5:]],
            "flagged_in_chunk": sum(1 for s in scores if s > 1),
            "chunk_size": len(scores),
        }

    def _on_row(self, st, f):     # 實驗一：一段配置播完的偵測率（依注入配置對照）
        r = f.get("row") or {}
        self._reference(st, {k: _round(r.get(k)) for k in
                             ("display", "kind", "n", "flagged", "detect_rate", "auroc", "median_score")})

    def _on_tick(self, st, f):
        st["samples"] += 1
        if "quarantine" in f:      # 實驗二
            st["current"] = {"t": f.get("t"), "score": _round(f.get("score")), "verdict": f.get("verdict"),
                             "predicted": f.get("label"), "quarantine": f.get("quarantine"),
                             "n_known": f.get("known")}
        else:                      # 實驗三
            st["current"] = {"scenario": f.get("scenario"), "trial": f.get("trial"), "t": f.get("t"),
                             "score": _round(f.get("score")), "ewma": _round(f.get("ewma")),
                             "cusum": _round(f.get("cusum")), "alarm_kind": f.get("kind")}
            if f.get("alarm_now"):
                self._event(st, f"劇本 {f.get('scenario')} 第 {int(f.get('trial', 0)) + 1} 次 t={f.get('t')}："
                                f"變化點警報，判別為「{KIND_DISPLAY.get(f.get('kind'), f.get('kind'))}」")

    def _on_trial(self, st, f):   # 實驗三：一次重複的判別結果（expected 是真值）
        t = f.get("trial") or {}
        self._reference(st, {k: _round(t.get(k)) for k in
                             ("scenario", "trial", "expected", "verdict", "correct", "alarm_t", "latency", "transition")})

    def _on_candidate(self, st, f):   # 實驗二：隔離區聚出候選群
        c = f.get("candidate") or {}
        rng = c.get("t_range") or ["?", "?"]
        self._event(st, f"隔離區聚出候選群：{c.get('size')} 筆，蒐集區間 t={rng[0]}~{rng[-1]}，"
                        f"HDBSCAN 參數 {c.get('cluster_params')}")
        self._reference(st, {"candidate_size": c.get("size"), "majority": c.get("majority_display"),
                             "purity": c.get("purity")})

    def _on_adopted(self, st, f):     # 實驗二：只用候選群自學成新類
        o = f.get("outcome") or {}
        if f.get("known") is not None:
            st["known"] = list(f["known"])
        self._event(st, f"已自學為「{o.get('display')}」（只用這群 {o.get('size')} 筆）")
        self._reference(st, {"learned_as": o.get("display"), "matched": o.get("matched_display"),
                             "purity": o.get("purity")})

    def _on_window(self, st, f):      # 實驗八 8-3：逐窗健康監測
        w = f.get("window") or {}
        st["samples"] += 1
        prev = (st.get("current") or {}).get("severity_stage")
        st["current"] = {k: _round(w.get(k)) for k in WINDOW_FIELDS if k in w}
        if prev is not None and w.get("severity_stage") != prev:
            self._event(st, f"第 {w.get('window_index')} 窗嚴重度 {prev} → {w.get('severity_stage')}"
                            f"（{w.get('alarm_reason') or '—'}）")
        if "is_fault" in w:          # 真值：這段資料是否來自故障配置
            st["source_is_fault"] = bool(w["is_fault"])

    def _on_part_start(self, st, f):  # 實驗四：子實驗開始
        st["current"] = {"part": f.get("name"), "state": "計算中"}
        self._event(st, f"開始子實驗 {f.get('name')}")

    def _on_part(self, st, f):
        st["current"] = {"part": f.get("name"), "state": "完成"}
        self._event(st, f"子實驗 {f.get('name')} 完成")

    def _on_done(self, st, f):
        payload = f.get("payload") or {}
        st["status"] = "done"
        result = payload.get("result")
        summary = {"seconds": payload.get("seconds"), "saved": payload.get("saved")}
        if isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, (int, float, str, bool)) and key not in ("seed",):
                    summary[key] = _round(value)
            if isinstance(result.get("final_known"), list):
                summary["final_known"] = result["final_known"]
            # 以下由真值比對算出，放事後對照
            for name, row in (result.get("summary") or {}).items():   # 實驗三：每個劇本的警報率／正確率
                if isinstance(row, dict):
                    self._reference(st, {"scenario_summary": name, **{
                        k: _round(row.get(k)) for k in ("name", "expected", "trials", "alarm_rate",
                                                        "correct_rate", "mean_latency", "mean_transition")}})
            for stage in result.get("stages") or []:                    # 實驗二：每個階段的發現結果
                if isinstance(stage, dict):
                    self._reference(st, {"stage_injected": stage.get("display"), **{
                        k: _round(stage.get(k)) for k in ("discovered", "learned_display", "matched_display",
                                                        "samples_to_candidate", "cluster_size", "cluster_purity",
                                                        "own_holdout_acc", "healthy_holdout_acc")}})
        elif isinstance(result, list):
            summary["windows"] = len(result)
        st["result"] = summary
        self._event(st, f"播放完成（{payload.get('seconds')} 秒）")

    def _on_stopped(self, st, f):
        st["status"] = "stopped"
        self._event(st, "播放被停止")

    def _on_error(self, st, f):
        st["status"] = "error"
        st["error"] = f.get("error")
        self._event(st, f"播放失敗：{f.get('error')}")
