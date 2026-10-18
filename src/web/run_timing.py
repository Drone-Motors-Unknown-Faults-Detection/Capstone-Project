"""Web執行邊界的UTC起訖與單調耗時，不推算歷史資料時間。"""

from datetime import datetime, timezone
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
        return {**self.start_fields(), "finished_at": finished_at,
                "seconds": round(time.perf_counter() - self.counter, 2)}
