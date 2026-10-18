"""匯出去敏呈現驗收摘要；原始問答、路徑、來源metadata與容器資料不對外發布。"""

import argparse
import json
from pathlib import Path
import re

from core.logger import setup_run

VIEWS = {"default", "time", "lw_classes", "full", "tsne", "method"}


def test_counts(report):
    counts = {key: report[key] for key in ("tests", "failures", "errors", "skipped", "returncode")}
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in counts.values()):
        raise ValueError("測試計數無效，拒絕匯出")
    status = report.get("status")
    if status not in ("PASS", "FAILED", "INCOMPLETE"):
        raise ValueError("測試狀態無效")
    return {**counts, "status": status}


def public_summary(presentation, related, windows):
    if presentation.get("status") != "VERIFIED_HTTP_NODE" or presentation.get("browser") != "NOT_VERIFIED":
        raise ValueError("只匯出已完整核對的HTTP／Node結果，瀏覽器仍保留未驗收")
    sha = presentation.get("deployment_sha", "")
    if not re.fullmatch(r"[a-f0-9]{40}", sha):
        raise ValueError("公開程式版本格式無效")
    cases = []
    for c in presentation.get("cases", []):
        status = "COMPLETED_LIMITED_PROBE" if c.get("view") == "method" else "VERIFIED_HTTP_NODE"
        if c.get("view") not in VIEWS or c.get("status") != status:
            raise ValueError("問答類型或驗收狀態無效，不匯出原始文字")
        cases.append({"view": c["view"], "status": status})
    if len(cases) != 7:
        raise ValueError("本輪須有六題程式問答與一次方法抽驗")
    return {"schema_version": "assistant_presentation_public_v1", "status": "VERIFIED_HTTP_NODE",
            "deployed_public_commit": sha, "browser": "NOT_VERIFIED", "cases": cases,
            "related_tests": test_counts(related), "windows_full_tests": test_counts(windows),
            "source_bytes_unchanged": bool(presentation.get("sources_before")) and presentation["sources_before"] == presentation.get("sources_after"),
            "runtime_unchanged_during_acceptance": bool(presentation.get("server_before")) and presentation["server_before"] == presentation.get("server_after"),
            "scope": "只核對既有結果的呈現；未重跑研究，不代表準確率提高或模型可靠。",
            "privacy": "不含原始問答、檔案路徑、來源metadata、容器識別或原始來源SHA。"}


def read_report(path, program):
    resolved = Path(path).resolve()
    resolved.relative_to((Path.cwd() / "output" / program).resolve())
    if resolved.name != "summary.json":
        raise ValueError("只接受指定程式的summary.json")
    return json.loads(resolved.read_text(encoding="utf-8"))


def run(presentation_report, related_tests, windows_tests):
    summary = public_summary(read_report(presentation_report, "assistant_presentation_evidence"),
                             read_report(related_tests, "pytest_evidence"), read_report(windows_tests, "pytest_evidence"))
    log, paths = setup_run("assistant_presentation_public_evidence", unique=True)
    target = paths.output_dir / "public_summary.json"
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info("去敏摘要已產生；原始問答與來源資料未複製：{}", target)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("presentation-report", "related-tests", "windows-tests"):
        parser.add_argument("--" + flag, required=True)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
