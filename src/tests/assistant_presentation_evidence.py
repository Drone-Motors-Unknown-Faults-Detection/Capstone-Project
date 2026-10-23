"""8787呈現驗收：唯讀既有JSON、六次程式問答與最多一次有限模型問答。"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
import urllib.request

from core.logger import setup_run
from experiments.exp12_confusion_tsne import summarize_matrices
from tests.assistant_http_evidence import json_headers, open_url, parse_sse, validate_url
from tests.execution_timing_evidence import validate_reference, verify_time
from web.presentation import markdown, rate

PROTECTED = ("lineage-web-1", "lineage-vllm-1", "capstone-test-20261018-ai-proxy-1")
WEB = "capstone-test-20261018-web-1"
FILES = ("src/core/confusion_summary.py", "src/web/playback.py", "src/web/presentation.py", "src/web/assistant.py",
         "src/web/run_timing.py", "src/web/static/markdown.js", "src/web/static/ai_chat.js",
         "src/web/static/assistant_presentation.js", "src/web/static/experiments.js", "src/web/static/index.html")


def ssh_python(code):
    key = Path.home() / ".ssh/lineage_aero01_andy_ed25519"
    process = subprocess.run(["ssh", "-T", "-i", str(key), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=8", "-o", "ConnectionAttempts=1",
        "andy@140.134.138.33", "python3 -c " + shlex.quote(code)], capture_output=True, text=True, encoding="utf-8", timeout=25)
    if process.returncode:
        raise RuntimeError("SSH唯讀核對失敗，未變更伺服器設定")
    return json.loads(process.stdout)


def source_receipt(saved):
    validate_reference(saved)
    code = f"""
import pathlib,json,hashlib
root=pathlib.Path('/home/andy/capstone-test-runtime-20261018')
p=root/{saved!r}
p.resolve().relative_to(root.resolve())
raw=p.read_bytes(); d=json.loads(raw); r=d['result']; vis={{}}
for k,v in r.get('_tsne_by_dataset',{{}}).items():
    assert len(v['config_labels']) == len(v['verdicts']) == len(v['embedding'])
    vis[k]=len(v['config_labels'])
print(json.dumps({{'saved':{saved!r},'sha256':hashlib.sha256(raw).hexdigest(),
 'metadata':{{k:v for k,v in d.items() if k in ('experiment','params','started_at','finished_at','seconds','timing_text','timing_status','timing_scope')}},
 'result':{{k:r[k] for k in ('seed','confidence','perplexity','n_datasets','binary_confusion','openset_confusion') if k in r}},'visualization':vis}}))
"""
    return ssh_python(code)


def server_receipt():
    inner = "import hashlib,json,pathlib; print(json.dumps({p:hashlib.sha256(pathlib.Path('/app',p).read_bytes()).hexdigest() for p in " + repr(FILES) + "}))"
    return ssh_python(f"""
import subprocess,json
names={(*PROTECTED, WEB)!r}
data=json.loads(subprocess.check_output(['docker','inspect',*names]))
rows=[]
for d in data:
 rows.append({{'name':d['Name'].lstrip('/'),'id':d['Id'],'image':d['Image'],'started_at':d['State']['StartedAt'],
  'restart_count':d['RestartCount'],'status':d['State']['Status'],'health':d['State'].get('Health',{{}}).get('Status'),
  'revision':d['Config'].get('Labels',{{}}).get('org.opencontainers.image.revision')}})
source=json.loads(subprocess.check_output(['docker','exec',{WEB!r},'python','-c',{inner!r}]))
print(json.dumps({{'containers':rows,'source_sha256':source}}))
""")


def verify_presentation(frames, receipt, view="default"):
    assert frames and frames[-1].get("event") == "done", "SSE未完整結束"
    assert not any(f.get("event") in ("error", "facts") for f in frames), "錯誤或重複核算框"
    mode = frames[0]
    p = next(f["presentation"] for f in frames if f.get("event") == "presentation")
    assert p["schema_version"] == "assistant_presentation_v1" and p["mode"] == mode["mode"] == "matrix_summary"
    assert p["origin"] == "matrix" and p["source_id"] == mode["source_id"]
    assert p["source"] == mode["presentation_source"]
    assert p["source"]["saved"] == mode["context"]["saved"] == receipt["saved"]
    assert p["source"]["saved_sha256"] == receipt["sha256"]
    deltas = [f for f in frames if f.get("event") == "delta"]
    assert deltas and all(f["origin"] == "matrix" for f in deltas)
    assert "".join(f["text"] for f in deltas) == markdown(p)
    sections = {s["id"]: s for s in p["sections"]}
    assert not sections["limits"]["detail"] and all(not s["detail"] for s in p["sections"] if s["kind"] == "warning")
    facts = summarize_matrices(receipt["result"])
    if view in ("default", "full"):
        b = facts["binary"]["binary"]
        assert sections["binary"]["rows"][3][1] == rate(b["healthy_false_positive_rate"])
        methods = list(facts["openset"].values())
        assert sections["methods"]["rows"][1][1:] == [rate(f["openset"]["known_accuracy"]) for f in methods]
        assert len(sections["summary"]["items"]) <= 3
    if view in ("default", "full", "lw_classes"):
        f = facts["openset"]["ledoit_wolf"]
        assert [int(row[3]) for row in sections["classes-ledoit_wolf"]["rows"]] == [f["per_class"][k]["rejected"] for k in f["openset"]["known_labels"]]
    if view in ("default", "full", "tsne"):
        assert sections["tsne"]["rows"] == [[k, n] for k, n in receipt["visualization"].items()] + [["合計", sum(receipt["visualization"].values())]]
    if view == "lw_classes":
        assert not sections["classes-ledoit_wolf"]["detail"] and "methods" not in sections
    if view == "tsne":
        assert not sections["tsne"]["detail"] and "methods" not in sections
    if view == "full":
        assert all(s["open"] for s in p["sections"])
        for key, entry in [("binary", receipt["result"]["binary_confusion"]), *receipt["result"]["openset_confusion"].items()]:
            assert sections["matrix-" + key]["rows"] == [[label, *row] for label, row in zip(entry["labels"], entry["matrix"])]
    return p


def run(base_url, new_saved, new_sha, old_saved, old_sha, deployment_sha, *, method_probe=False):
    validate_url(base_url)
    for saved in (new_saved, old_saved):
        validate_reference(saved)
    if not all(re.fullmatch(r"[a-f0-9]{64}", s) for s in (new_sha, old_sha)) or not re.fullmatch(r"[a-f0-9]{40}", deployment_sha):
        raise ValueError("必須提供完整來源與部署SHA")
    log, paths = setup_run("assistant_presentation_evidence", unique=True)
    result = {"schema_version": "assistant_presentation_evidence_v1", "base_url": base_url,
              "deployment_sha": deployment_sha, "browser": "NOT_VERIFIED", "cases": [],
              "research_rerun": False, "method_probe_limit": 1 if method_probe else 0}
    def request(path, body=None):
        req = urllib.request.Request(base_url.rstrip("/") + path, data=json.dumps(body).encode() if body else None,
                                     headers=json_headers(base_url))
        return open_url(req, 100 if body else 10)

    try:
        before = result["server_before"] = server_receipt()
        assert next(c for c in before["containers"] if c["name"] == WEB)["revision"] == deployment_sha
        root = Path(__file__).resolve().parents[2]
        assert before["source_sha256"] == {name: hashlib.sha256((root / name).read_text(encoding="utf-8").replace("\r\n", "\n").encode()).hexdigest() for name in FILES}
        sources = result["sources_before"] = [source_receipt(p) for p in (new_saved, old_saved)]
        assert [r["sha256"] for r in sources] == [new_sha, old_sha]
        html = request("/")
        result["assets_sha256"] = assets = {}
        for name in ("markdown.js", "ai_chat.js", "assistant_presentation.js", "experiments.js"):
            url = re.search(r"/static/" + re.escape(name) + r'\?[^"\s<>]+', html)
            assert url, "HTML缺少版本化資產：" + name
            asset = request(url.group()).replace("\r\n", "\n")
            assert asset == (root / "src/web/static" / name).read_text(encoding="utf-8").replace("\r\n", "\n")
            assets[name] = hashlib.sha256(asset.encode()).hexdigest()
        questions = [("最新實驗十二結果與限制", None, "default", sources[0]),
            ("這次實驗何時開始與完成，花多久？", new_saved, "time", sources[0]),
            ("總結這筆實驗十二舊檔結果", old_saved, "default", sources[1]),
            ("只看 LW 逐類結果", new_saved, "lw_classes", sources[0]),
            ("展開全部矩陣", new_saved, "full", sources[0]),
            ("t-SNE實際抽樣與圖例", new_saved, "tsne", sources[0])]
        if method_probe:
            questions.append(("說明實驗一健康冷啟動的方法與資料切分，請查工具並指出未知資訊。", None, "method", None))
        for question, saved, view, receipt in questions:
            body = {"question": question, "history": []}
            if view != "method":
                body["context"] = {"experiment": "exp12", "mode": "result", **({"saved": saved} if saved else {})}
            started = time.monotonic()
            raw = request("/api/assistant/ask", body)
            frames = parse_sse(raw)
            case = {"question": question, "view": view, "request": body, "seconds": round(time.monotonic() - started, 3), "frames": frames}
            result["cases"].append(case)
            if view == "time":
                verify_time(frames, receipt["metadata"], sha=receipt["sha256"])
            elif view == "method":
                assert frames[-1].get("event") == "done" and any(f.get("event") == "tool" for f in frames)
                assert not any(f.get("mode") == "matrix_summary" or f.get("event") in ("error", "presentation") for f in frames)
                case["scope"] = "一次有限方法查詢；不把模型自由文字當研究證據"
            else:
                verify_presentation(frames, receipt, view)
            if view != "method":
                node = shutil.which("node")
                assert node, "缺Node，無法驗證實際SSE到JS"
                process = subprocess.run([node, str(root / "src/tests/ai_chat_frontend.cjs"),
                    str(root / "src/web/static/markdown.js"), str(root / "src/web/static/ai_chat.js"), "--frames"],
                    input=json.dumps(frames), capture_output=True, text=True, encoding="utf-8", timeout=30)
                case["node"] = {"returncode": process.returncode, "stdout": process.stdout, "stderr": process.stderr}
                assert not process.returncode, "SSE→JS契約失敗"
            case["status"] = "VERIFIED_HTTP_NODE" if view != "method" else "COMPLETED_LIMITED_PROBE"
            log.info("已核對{}：{}", view, case["status"])
        result["sources_after"] = [source_receipt(p) for p in (new_saved, old_saved)]
        assert result["sources_before"] == result["sources_after"], "原始來源遭變更"
        result["server_after"] = after = server_receipt()
        assert before == after, "驗收期間容器或程式來源遭變更"
        result["status"] = "VERIFIED_HTTP_NODE"
    except Exception as exc:
        result.update(status="FAILED", reason=f"{type(exc).__name__}：{exc}")
        log.error("呈現驗收未完成：{}", result["reason"])
    target = paths.output_dir / "summary.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info("呈現驗收={}，瀏覽器未驗收；{}", result["status"], target)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-url", "new-saved", "new-sha", "old-saved", "old-sha", "deployment-sha"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--method-probe", action="store_true", help="額外執行一次有界一般方法問答；預設不呼叫模型")
    result = run(**vars(parser.parse_args()))
    return 0 if result["status"] == "VERIFIED_HTTP_NODE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
