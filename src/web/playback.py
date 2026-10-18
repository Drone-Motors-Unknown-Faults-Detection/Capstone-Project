"""實驗頁的執行紀錄（「⏵ 邊跑邊畫」與「▶ 執行」），給 AI 問答讀取。

每個實驗各留最近一次紀錄（records[exp_id]），latest 指向最近動過的實驗：
    邊跑邊畫  StreamHandler 每送出一個 frame 呼叫 ``update()``，播完的結果一併摘要
    ▶ 執行    RunHandler 完成後呼叫 ``record_run(exp_id, payload)``
    存檔      記憶體沒有紀錄（例：伺服器重啟）時，``load_saved()`` 讀 results_root 下最新的
              ``*/experiments/{exp_id}_2*.json``（ExperimentRunner._save 寫的 payload；
              `_2` 對上時間戳開頭，exp6 不會誤抓 exp6_formal_* 檔）
本模組只彙整，不做任何判定，數字全部取自各實驗既有輸出。

邊跑邊畫的狀態分三層：
    current    最近一筆的系統判定（分數、判定、EWMA、健康指數…），不含真值
    events     重點事件：候選群、自學、變化點警報、嚴重度改變（最多 MAX_EVENTS 筆）
    reference  事後對照：由注入配置（真值）算出的評估結果，只能當評估說明，不是系統當下的判定

真值欄位（注入配置、expected、is_fault、候選群 majority／purity、onset）只進 reference
（AGENT.md 鐵則 6：真實身分只在確認／評估時揭示）。
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import re
from pathlib import Path

from experiments.exp3_trend import KIND_DISPLAY
from web.presentation import build_exp12, table_text
from web.experiments import CATALOG
from web.run_timing import timing_fields, utc_now

MAX_EVENTS = 12
MAX_REFERENCE = 40
RESULT_BUDGET = 6000          # 結果摘要字數上限（模型 context 32k，另有手冊與對話）
ANSWER_SCHEMA = "exp12_standard_answer_v2"
RESULTS_ROOT = Path("output/web_server")
_SPEC = {e["id"]: e for e in CATALOG}
_SEQ = itertools.count(1)     # 紀錄先後順序：同一編號有多張卡（6-1／6-2）時取最近的
_SAVED_NAME = re.compile(r"(.+)_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})(?:_(\d+))?\.json")

# 實驗八逐窗輸出中屬於系統判定的欄位（is_fault 是真值，不在此列）
WINDOW_FIELDS = ("window_index", "health_index", "severity_stage", "trend", "degradation_rate",
                 "openset_score", "is_unknown_fault", "fault_type", "prediction_confidence",
                 "uncertainty", "alarm_state", "alarm_reason", "change_point_state", "data_quality")


def _round(v, nd=3):
    return round(v, nd) if isinstance(v, float) else v


def _now() -> str:
    return utc_now()


def _saved_order(path: Path) -> tuple:
    match = _SAVED_NAME.fullmatch(path.name)
    return (match[2], int(match[3] or 0), str(path)) if match else ("", 0, str(path))


# -- 結果摘要 ----------------------------------------------------------------------

SKIP_KEYS = {"model", "resolved_config", "dataset_fingerprint", "fingerprint_version", "dataset_root",
             "checkpoint_identifier"}


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return "—" if v is None else str(v)


def _scalar(v) -> bool:
    return v is None or isinstance(v, (int, float, str, bool))


def _render(key: str, v, depth: int, out: list[str]) -> None:
    pad = "  " * depth
    label = f"{pad}{key}：" if key else pad
    if isinstance(v, dict) and isinstance(v.get("labels"), list) and isinstance(v.get("matrix"), list):
        labels = [str(x) for x in v["labels"]]   # 混淆矩陣：列 = 真值、欄 = 預測
        out.append(f"{label}（列 = 真值、欄 = 預測）")
        out.append(table_text(["真值＼預測", *labels],
                              [[name, *(_fmt(x) for x in row)] for name, row in zip(labels, v["matrix"])]))
        return
    if isinstance(v, dict):
        if depth >= 4:
            out.append(f"{label}（{len(v)} 項，略）")
            return
        if key:
            out.append(label)
        for k, x in v.items():
            if str(k).startswith("_") or k in SKIP_KEYS:   # _ 開頭是畫圖用座標等內部資料
                continue
            if _scalar(x):
                out.append(f"{pad}{'  ' if key else ''}{k}：{_fmt(x)}")
            else:
                _render(str(k), x, depth + (1 if key else 0), out)
        return
    if isinstance(v, list):
        if not v:
            out.append(f"{label}（空）")
        elif all(_scalar(x) for x in v):
            nums = [x for x in v if isinstance(x, (int, float)) and not isinstance(x, bool)]
            if len(v) <= 20:
                out.append(f"{label}" + "、".join(_fmt(x) for x in v))
            elif nums:
                out.append(f"{label}{len(v)} 個數值，最小 {_fmt(min(nums))}、最大 {_fmt(max(nums))}、"
                           f"平均 {_fmt(sum(nums) / len(nums))}")
            else:
                out.append(f"{label}{len(v)} 項")
        elif all(isinstance(x, dict) for x in v):
            cols = []
            for row in v:
                cols += [k for k, x in row.items() if _scalar(x) and k not in cols and not str(k).startswith("_")]
            cols = cols[:14]
            out.append(f"{label}（{len(v)} 列）")
            if cols:
                out.append(table_text(cols, [[_fmt(row.get(c)) for c in cols] for row in v[:30]]))
            if len(v) > 30:
                out.append(f"{pad}…另有 {len(v) - 30} 列")
        else:
            out.append(f"{label}{len(v)} 項（巢狀，略）")
        return
    out.append(f"{label}{_fmt(v)}")


def _exp12_answer(result, budget: int, source: dict | None) -> dict:
    return build_exp12(result, budget, source, ANSWER_SCHEMA)



def summarize_result(result, budget: int = RESULT_BUDGET, *, exp_id: str | None = None,
                     source: dict | None = None) -> str:
    """實驗結果轉成模型讀得完的文字：去掉畫圖座標與模型細節，矩陣轉表格，長清單截斷。"""
    if exp_id == "exp12":
        return _exp12_answer(result, budget, source)["text"]
    out: list[str] = []
    _render("", result, 0, out)
    text = "\n".join(line for line in out if line.strip())
    if len(text) > budget:
        text = text[:budget] + "\n…（以下略，完整結果見存檔）"
    return text


def _base(exp_id: str, mode: str) -> dict:
    spec = _SPEC.get(exp_id, {})
    return {
        "exp_id": exp_id, "no": spec.get("no", exp_id), "sub": spec.get("sub"), "title": spec.get("title", ""),
        "mode": mode, "params": {}, "status": "running", "started_at": None, "finished_at": None, "updated_at": _now(),
        "samples": 0, "known": None, "current": None, "events": [], "reference": [],
        "result": None, "result_text": None, "error": None, "seq": next(_SEQ),
    }


def _split_params(st: dict, params: dict) -> None:
    params = dict(params or {})
    source = params.pop("config", None)   # 實驗八選的注入配置 = 真值
    st["params"] = params
    if source is not None:
        st["source_config"] = source


def _fill_result(st: dict, payload: dict) -> None:
    st["status"] = "done"
    timing = timing_fields(payload)
    st.update(timing)
    st["result"] = {**timing, **{k: payload[k] for k in ("saved", "saved_sha256") if payload.get(k) is not None}}
    source = {"exp_id": st["exp_id"], "mode": st["mode"], "status": st["status"],
              "started_at": st["started_at"], "params": st["params"], **st["result"]}
    if isinstance(payload.get("streamed"), bool):
        source["streamed"] = payload["streamed"]
    if st["exp_id"] == "exp12":
        st["matrix_answer"] = _exp12_answer(payload.get("result"), RESULT_BUDGET, source)
        st["result_text"] = st["matrix_answer"]["text"]
        st["fact_text"] = st["result_text"]
    else:
        st["result_text"] = summarize_result(payload.get("result"))


class PlaybackTracker:
    def __init__(self, results_root: Path | str = RESULTS_ROOT):
        self.results_root = Path(results_root)
        self.records: dict[str, dict] = {}
        self.latest: str | None = None

    @property
    def state(self) -> dict | None:
        """最近動過的那筆紀錄（邊跑邊畫的 update 寫進這裡）。"""
        return self.records.get(self.latest) if self.latest else None

    # -- 生命週期 ----------------------------------------------------------------

    def start(self, exp_id: str, params: dict) -> None:
        st = _base(exp_id, "stream")
        _split_params(st, params)
        self.records[exp_id] = st
        self.latest = exp_id

    def record_run(self, exp_id: str, payload: dict) -> None:
        """「▶ 執行」完成：整份結果摘要後存成該實驗最近一次紀錄。"""
        st = _base(exp_id, "run")
        _split_params(st, payload.get("params") or {})
        _fill_result(st, payload)
        self._event(st, f"執行完成（{payload.get('seconds')} 秒）")
        self.records[exp_id] = st
        self.latest = exp_id

    def record_run_error(self, exp_id: str, params: dict, error: str) -> None:
        st = _base(exp_id, "run")
        _split_params(st, params)
        st.update(status="error", error=error)
        self.records[exp_id] = st
        self.latest = exp_id

    def load_saved(self, exp_id: str | None = None) -> dict | None:
        """記憶體沒有紀錄時讀存檔；exp_id=None 取所有實驗中最新的一份。"""
        pattern = f"*/experiments/{exp_id}_2*.json" if exp_id else "*/experiments/*_2*.json"
        files = sorted(self.results_root.glob(pattern), key=_saved_order)
        for path in reversed(files):
            rec = self._read_saved(path, exp_id)
            if rec is not None:
                return rec
        return None

    def _read_saved(self, path: Path, exp_id: str | None) -> dict | None:
        try:
            relative = path.absolute().relative_to(self.results_root.absolute())
            if len(relative.parts) != 3 or relative.parts[1] != "experiments":
                return None
            for parent in (path, *path.parents):
                if parent == self.results_root.absolute() or parent == self.results_root:
                    break
                if parent.is_symlink():
                    return None
            path.resolve().relative_to(self.results_root.resolve())
            match = _SAVED_NAME.fullmatch(path.name)
            if not match or (exp_id and match[1] != exp_id):
                return None
            raw = path.read_bytes()
            payload = json.loads(raw)
            if not isinstance(payload, dict) or not isinstance(payload.get("params", {}), dict):
                return None
            exp = match[1]
            if payload.get("experiment", exp) != exp:
                return None
        except (OSError, ValueError):
            return None
        st = _base(exp, "saved")
        _split_params(st, payload.get("params") or {})
        st["updated_at"] = payload.get("finished_at") if isinstance(payload.get("finished_at"), str) else None
        _fill_result(st, {**payload, "saved": str(path), "saved_sha256": hashlib.sha256(raw).hexdigest()})
        return st

    def load_saved_record(self, reference: str, exp_id: str = "exp12") -> dict:
        """只定位指定原檔；不得以較新的另一筆取代，也不讀允許根目錄外的檔案。"""
        if not isinstance(reference, str) or not reference or len(reference) > 600:
            raise ValueError("存檔識別必須為有效字串")
        ref = reference.replace("\\", "/")
        if any(c in ref for c in ("\x00", "%", "?", "#")) or ".." in ref.split("/"):
            raise ValueError("拒絕不合法的存檔路徑")
        if ref.startswith("output/web_server/"):
            ref = ref[len("output/web_server/"):]
        candidate = Path(ref)
        if len(candidate.parts) == 1:
            match = _SAVED_NAME.fullmatch(candidate.name)
            if not match or match[1] != exp_id:
                raise ValueError("存檔名稱與實驗不一致")
            candidates = list(self.results_root.glob(f"*/experiments/{candidate.name}"))
            if len(candidates) != 1:
                raise ValueError("指定存檔不存在或同名多筆；請提供完整保存路徑")
            candidate = candidates[0]
        elif not candidate.is_absolute():
            if re.match(r"^[A-Za-z]:", ref):
                raise ValueError("拒絕非本機允許根目錄的路徑")
            if self.results_root.is_absolute() or not candidate.is_relative_to(self.results_root):
                candidate = self.results_root / candidate
        rec = self._read_saved(candidate, exp_id)
        if rec is None:
            raise ValueError("無法核驗指定存檔的路徑、內容或實驗身分；未改取最新紀錄")
        return rec

    def finish(self, status: str, error: str | None = None) -> None:
        """串流結束時呼叫；已經 done／error 的不覆蓋（被瀏覽器中斷才記 stopped）。"""
        st = self.state
        if st is None or st["status"] != "running":
            return
        st["status"] = status
        st["error"] = error
        st["updated_at"] = _now()

    def snapshot(self) -> dict | None:
        """提問當下凍結全部紀錄；沒有任何紀錄回 None。"""
        if not self.records:
            return None
        return {"latest": self.latest, "records": copy.deepcopy(self.records)}

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

    def _on_start(self, st, f):
        _split_params(st, f.get("params") or {})
        st.update(timing_fields(f))

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
        _fill_result(st, payload)
        result = payload.get("result")
        if isinstance(result, dict):
            if isinstance(result.get("final_known"), list):
                st["known"] = list(result["final_known"])
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
        self._event(st, f"播放完成（{payload.get('seconds')} 秒）")

    def _on_stopped(self, st, f):
        st["status"] = "stopped"
        self._event(st, "播放被停止")

    def _on_error(self, st, f):
        st["status"] = "error"
        st["error"] = f.get("error")
        self._event(st, f"播放失敗：{f.get('error')}")
