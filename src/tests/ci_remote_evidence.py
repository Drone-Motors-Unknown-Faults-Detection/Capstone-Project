"""保存已完成 Actions 的公開 run／job 與白名單摘要；不收集私人 logs。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

from core.logger import setup_run


def gh_json(*arguments):
    output = subprocess.check_output(["gh", *arguments], text=True, encoding="utf-8")
    return json.loads(output)


def run(run_ids: list[int]) -> dict:
    log, paths = setup_run("ci_remote_evidence")
    runs = []
    for run_id in run_ids:
        record = gh_json("run", "view", str(run_id), "--json", "databaseId,headSha,headBranch,status,conclusion,url,jobs,event")
        if record["status"] != "completed":
            raise ValueError(f"Actions run {run_id} 尚未完成，不能封存為完成證據")
        artifacts = gh_json("api", f"repos/Drone-Motors-Unknown-Faults-Detection/Lineage/actions/runs/{run_id}/artifacts")
        record["artifacts"] = [{key: row[key] for key in ("id", "name", "size_in_bytes", "expired")}
                               for row in artifacts["artifacts"]]
        summaries = []
        for artifact in record["artifacts"]:
            if not artifact["name"].startswith("去敏契約結果-") or artifact["expired"]:
                continue
            with tempfile.TemporaryDirectory() as directory:
                subprocess.run(["gh", "run", "download", str(run_id), "--name", artifact["name"], "--dir", directory], check=True)
                for path in sorted(Path(directory).rglob("public_summary.json")):
                    summary = json.loads(path.read_text(encoding="utf-8"))
                    if summary.get("schema_version") != "ci_evidence_v1":
                        raise ValueError("遠端去敏摘要 schema 不符")
                    summaries.append({"artifact_id": artifact["id"], "name": artifact["name"], "summary": summary})
        record["public_summaries"] = summaries
        runs.append(record)
    result = {"schema_version": "ci_remote_evidence_v1", "runs": runs,
              "scope": "真實 GitHub Actions；沒有 artifact 的歷史成功run不補造逐項計數"}
    (paths.output_dir / "remote_runs.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info("封存 {} 次真實 Actions 執行", len(runs))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, action="append", required=True)
    run(parser.parse_args().run_id)


if __name__ == "__main__":
    main()
