"""Web執行邊界的UTC起訖與單調耗時，不推算歷史資料時間。"""

from datetime import datetime, timezone
import math
import re
import time


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class RunTimer:
    def __init__(self, scope: str):
        self.counter = time.perf_counter()
        self.started_at = utc_now()
        self.scope = scope

    def start_fields(self) -> dict:
        return {"started_at": self.started_at, "timing_version": "web_run_timing_v1",
                "timing_scope": self.scope}

    def finish(self) -> dict:
        finished_at = utc_now()
        return timing_fields({**self.start_fields(), "finished_at": finished_at,
                              "seconds": round(time.perf_counter() - self.counter, 2)})


def timing_fields(payload: dict) -> dict:
    """唯讀解讀保存欄位；不由時差、路徑或讀取時間補出歷史起點。"""
    notes = []

    def stamp(key, label):
        value = payload.get(key)
        if value is None or value == "":
            return None, None, "原檔未記錄"
        if isinstance(value, str):
            try:
                if not re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", value):
                    raise ValueError
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                pass
            else:
                return value, parsed, value + ("（原檔未提供時區）" if parsed.utcoffset() is None else "")
        notes.append(f"{label}格式無效，原值為{value!r}")
        return value if isinstance(value, str) else None, None, "格式無效，未推算"

    start, begin, start_text = stamp("started_at", "開始時間")
    finish, end, finish_text = stamp("finished_at", "完成時間")
    seconds = payload.get("seconds")
    try:
        valid_seconds = isinstance(seconds, (int, float)) and not isinstance(seconds, bool) and math.isfinite(seconds) and seconds >= 0
    except OverflowError:
        valid_seconds = False
    if not valid_seconds:
        if seconds is not None:
            notes.append(f"耗時無效，原值為{seconds!r}")
        seconds = None
    status = "INCOMPLETE"
    if begin is not None and end is not None:
        if (begin.utcoffset() is None) != (end.utcoffset() is None):
            notes.append("起訖時區資訊不足，不能比較時間差")
        else:
            elapsed = (end - begin).total_seconds()
            # 舊秒級字串的兩端截斷誤差小於1秒；新毫秒與兩位耗時誤差小於0.011秒。
            tolerance = 0.011 if all(re.search(r":\d{2}\.\d{3,}", s) for s in (start, finish)) else 1.01
            if elapsed < 0 or (seconds is not None and abs(elapsed - seconds) > tolerance):
                notes.append("時間紀錄不一致：起訖順序或時間差與紀錄耗時不符；保留原值，未推算開始")
                status = "INCONSISTENT"
            elif seconds is not None and begin.utcoffset() is not None:
                status = "VERIFIED"
    scope = payload.get("timing_scope")
    scope = scope if scope in ("batch_execution", "stream_elapsed_including_playback") else None
    text = (f"開始時間：{start_text}；完成時間：{finish_text}；"
            + (f"耗時：{seconds} 秒。" if seconds is not None else "耗時：原檔未記錄或數值無效。"))
    if scope == "stream_elapsed_including_playback":
        text += "串流經過時間包含播放等待，不是純模型計算耗時。"
    elif scope == "batch_execution":
        text += "批次耗時包含載入、運算與結果轉換，排除存檔及HTTP傳輸。"
    elif seconds is not None:
        text += "原檔未提供耗時邊界。"
    if notes:
        text += "；".join(notes) + "。"
    return {"started_at": start, "finished_at": finish, "seconds": seconds,
            "timing_version": payload.get("timing_version") if payload.get("timing_version") == "web_run_timing_v1" else None,
            "timing_scope": scope, "timing_status": status, "timing_warnings": notes, "timing_text": text}
