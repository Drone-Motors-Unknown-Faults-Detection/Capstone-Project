"""8787起訖驗收：歷史檔唯讀、最多一次實驗一，不呼叫共享模型。"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import urllib.request

from core.logger import setup_run
from tests.assistant_http_evidence import parse_sse, validate_url, verify_result


def validate_reference(saved):
    if not re.fullmatch(r"output/web_server/[A-Za-z0-9_-]+/experiments/exp\d+(?:_[A-Za-z]+)?_\d{4}(?:-\d{2}){5}(?:_\d+)?\.json", saved):
        raise ValueError("只允許8787標準保存路徑")


def remote_record(saved):
    validate_reference(saved)
    script = ("import pathlib,json,hashlib; "
              "root=pathlib.Path('/home/andy/capstone-test-runtime-20261018'); "
              f"p=root/{saved!r}; "
              "p.resolve().relative_to(root.resolve()); b=p.read_bytes(); d=json.loads(b); "
              "print(json.dumps({'sha256':hashlib.sha256(b).hexdigest(),'keys':list(d),"
              "'timing':{k:d.get(k) for k in ('started_at','finished_at','seconds','timing_text','timing_status')},"
              "'result_sha256':hashlib.sha256(json.dumps(d.get('result'),sort_keys=True).encode()).hexdigest()}))")
    key = Path.home() / ".ssh/lineage_aero01_andy_ed25519"
    process = subprocess.run(["ssh", "-T", "-i", str(key), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
                              "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=8", "-o", "ConnectionAttempts=1",
                              "andy@140.134.138.33", "python3 -c " + shlex.quote(script)],
                             capture_output=True, text=True, encoding="utf-8", timeout=20)
    if process.returncode:
        raise RuntimeError("SSH唯讀來源驗證失敗，未改讀另一份來源")
    return json.loads(process.stdout)


def verify_time(frames, expected, *, old=False, sha=None):
    assert frames[0].get("mode") == "record_timing", "時間問答未走紀錄模式"
    assert frames[-1].get("event") == "done", "時間問答未完成"
    deltas = [f for f in frames if f.get("event") == "delta"]
    assert deltas and all(f.get("origin") == "record" for f in deltas), "混入非紀錄正文"
    text = "".join(f["text"] for f in deltas)
    identity = json.loads(text.split("\n來源識別：", 1)[1])
    for field in ("started_at", "finished_at", "seconds"):
        assert identity.get(field) == expected.get(field), "各層時間不一致：" + field
    if sha:
        assert identity.get("saved_sha256") == sha, "讀回來源SHA不符"
    if old:
        assert "開始時間：原檔未記錄" in text and "原檔未提供時區" in text, "舊檔缺值說明不足"
    else:
        assert identity["timing_status"] == "VERIFIED", "新起訖未通過一致性檢查"
        assert expected["timing_text"] in text
    return identity


def run(base_url, old_saved, old_sha, deployment_sha, *, new_saved=None):
    validate_url(base_url)
    validate_reference(old_saved)
    if not re.fullmatch(r"[a-f0-9]{64}", old_sha) or not re.fullmatch(r"[a-f0-9]{40}", deployment_sha):
        raise ValueError("來源SHA與部署SHA必須完整")
    log, paths = setup_run("execution_timing_evidence")
    result = {"schema_version": "execution_timing_evidence_v1", "base_url": base_url,
              "expected_deployment_sha": deployment_sha, "browser": "NOT_VERIFIED", "cases": []}

    def request(path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(base_url.rstrip("/") + path, data=data,
                                     headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=60).read().decode("utf-8")

    def ask(saved, experiment, question):
        context = {"experiment": experiment, "mode": "result"}
        if saved:
            context["saved"] = saved
        raw = request("/api/assistant/ask", {"question": question, "context": context, "history": []})
        frames = parse_sse(raw)
        result["cases"].append({"context": context, "question": question, "sse": raw, "frames": frames})
        return frames

    try:
        result["old_before"] = old = remote_record(old_saved)
        assert old["sha256"] == old_sha, "指定舊檔SHA不符"
        assert "started_at" not in old["keys"], "這份指定舊檔與重現條件不同"
        old_frames = ask(old_saved, "exp12", "這次實驗何時開始、何時完成、花多久？")
        verify_time(old_frames, old["timing"], old=True, sha=old_sha)
        matrix = ask(old_saved, "exp12", "總結實驗十二結果與限制。")
        assert not verify_result(matrix, old_sha), "歷史矩陣回歸不一致"
        assert "合計17980筆" in "".join(f.get("text", "") for f in matrix)
        if new_saved:
            validate_reference(new_saved)
            payload = None
        else:
            catalog = json.loads(request("/api/experiments"))
            assert catalog.get("running") is None, "已有實驗執行中，未開始新run"
            assert "T1/8000rpm" in catalog["datasets"], "指定低成本工況不存在，未自行更換"
            payload = json.loads(request("/api/experiments/exp1/run", {
                "dataset": "T1/8000rpm", "openset_method": "mahalanobis", "seed": 42}))
            new_saved = payload["saved"]
        result["new_saved"] = new_saved
        result["new_source"] = new = remote_record(new_saved)
        result["new_http_payload"] = payload
        if payload:
            for field in ("started_at", "finished_at", "seconds", "timing_text"):
                assert new["timing"][field] == payload[field], "HTTP與原始JSON時間不一致"
            verify_time(ask(None, "exp1", "這次實驗何時開始、何時完成、花多久？"), new["timing"])
        saved_frames = ask(new_saved, "exp1", "這次實驗何時開始、何時完成、花多久？")
        verify_time(saved_frames, new["timing"], sha=new["sha256"])
        result["old_after"] = remote_record(old_saved)
        assert result["old_before"] == result["old_after"], "歷史檔案被改動"
        root = Path(__file__).resolve().parents[2]
        node = shutil.which("node")
        assert node, "缺Node，不能驗證前端資料鏈"
        assets = {}
        html = request("/")
        for name in ("ai_chat.js", "experiments.js"):
            url = re.search(r"/static/" + re.escape(name) + r'\?[^"\s<>]+', html)
            assert url, "缺少HTML前端資產引用"
            actual = request(url.group()).replace("\r\n", "\n")
            local = (root / "src/web/static" / name).read_text(encoding="utf-8").replace("\r\n", "\n")
            assert actual == local, "線上前端資產不符：" + name
            assets[name] = hashlib.sha256(actual.encode()).hexdigest()
        result["assets_sha256_lf"] = assets
        for frames in (old_frames, saved_frames, matrix):
            proc = subprocess.run([node, str(root / "src/tests/ai_chat_frontend.cjs"),
                                   str(root / "src/web/static/markdown.js"), str(root / "src/web/static/ai_chat.js"), "--frames"],
                                  input=json.dumps(frames), capture_output=True, text=True, encoding="utf-8", timeout=15)
            assert proc.returncode == 0, "實際時間SSE至前端JS失敗"
        result.update(status="VERIFIED_HTTP", frontend_node="VERIFIED")
    except Exception as exc:
        result.update(status="FAILED", reason=f"{type(exc).__name__}：{exc}")
    (paths.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    log.info("時間驗收={}；新run={}；瀏覽器未驗收", result["status"], result.get("new_saved", "未產生"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--old-saved", required=True)
    parser.add_argument("--old-sha", required=True)
    parser.add_argument("--deployment-sha", required=True)
    parser.add_argument("--new-saved", help="重試驗證已產生的新run，不再fit")
    return 0 if run(**vars(parser.parse_args()))["status"] == "VERIFIED_HTTP" else 1


if __name__ == "__main__":
    raise SystemExit(main())
