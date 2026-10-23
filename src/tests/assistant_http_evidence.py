"""對8787測試站做有限次問答並保存SSE；不呼叫實驗執行或正式網站。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

from core.logger import setup_run

QUESTIONS = (
    "總結最新實驗十二，包含數字與限制。",
    "legacy全部拒絕，是否代表太保守或完全失效？",
    "93.12%是只有故障的正確率嗎？",
    "t-SNE總共多少筆？三角形是否只是真實未知配置？",
    "這些結果不是模型判斷嗎？",
    "健康池313筆是不是全部用來訓練？",
)
METHODS = (
    "說明實驗一健康冷啟動的方法與資料切分，請查工具並指出未知資訊。",
    "請比較實驗一與實驗二的方法、資料角色、輸出與限制，查核執行狀態、事件、手冊與知識庫後整理；沒有資料就明說。",
)


def validate_url(url: str) -> None:
    if url.rstrip("/") not in ("http://140.134.138.33:8787", "http://127.0.0.1:8787"):
        raise ValueError("本驗證只允許指定8787測試站，不能對正式服務執行")


def parse_sse(raw: str) -> list[dict]:
    return [json.loads(part[6:]) for part in raw.replace("\r\n", "\n").split("\n\n") if part.startswith("data: ")]


def verify_result(frames: list[dict], source_sha: str) -> list[str]:
    errors = []
    mode = next((f for f in frames if f.get("event") == "mode"), {})
    deltas = [f for f in frames if f.get("event") == "delta"]
    text = "".join(f.get("text", "") for f in deltas)
    if mode.get("mode") != "matrix_summary" or mode.get("schema") != "exp12_standard_answer_v2":
        errors.append("結果未走新版程式核算模式")
    if not deltas or any(f.get("origin") != "matrix" for f in deltas):
        errors.append("結果含非矩陣正文或沒有正文")
    for fragment in (source_sha, "2736／2938＝93.12%", "202／2938＝6.88%", "45／572＝7.87%",
                     "不是純故障正確率", "三角形表示模型判為unknown", "並非全部工況合計2000",
                     "train擬合187／cal校準62／holdout測試64", "模型產生預測，再以事後真值計算"):
        if fragment not in text:
            errors.append("正文缺少必要核算或語意：" + fragment)
    if any(f.get("event") == "error" for f in frames) or not frames or frames[-1].get("event") != "done":
        errors.append("問答沒有正常完成")
    if any(f.get("event") == "facts" for f in frames):
        errors.append("程式正文與facts重複")
    return errors


def json_headers(base_url: str) -> dict:
    """POST 的標頭：伺服器拒絕沒有 Origin／Referer／Sec-Fetch-Site 的狀態端點請求（#304 F5-02），腳本帶同站 Origin。"""
    parts = urllib.parse.urlparse(base_url)
    return {"Content-Type": "application/json", "Origin": f"{parts.scheme}://{parts.netloc}"}


class HostNotAllowedError(RuntimeError):
    """測試站以 Host 允許清單拒絕請求（#304）；訊息直接寫出 8787 該怎麼設定。"""


def open_url(request, timeout: float) -> str:
    """urlopen 並解碼；403 且伺服器說 Host 不在名單時改丟帶設定提示的 HostNotAllowedError。"""
    try:
        return urllib.request.urlopen(request, timeout=timeout).read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code == 403 and "Host" in exc.read().decode("utf-8", "replace"):
            url = request.full_url if isinstance(request, urllib.request.Request) else request
            host = urllib.parse.urlparse(url).hostname
            raise HostNotAllowedError(
                f"測試站回 403：Host {host} 不在伺服器的允許清單；容器要在 docker/.env 設 "
                f"ALLOW_HOST={host}（傳給 --allow-host）並重建後再驗收") from exc
        raise


def run(base_url: str, saved: str, source_sha: str, deployment_sha: str) -> dict:
    validate_url(base_url)
    if not re.fullmatch(r"[a-f0-9]{64}", source_sha) or not re.fullmatch(r"[a-f0-9]{40}", deployment_sha):
        raise ValueError("必須提供完整來源SHA256與測試部署commit")
    log, paths = setup_run("assistant_http_evidence")
    base_url = base_url.rstrip("/")
    failures, cases = [], []

    def get(path):
        return urllib.request.urlopen(base_url + path, timeout=10).read().decode("utf-8")

    status = json.loads(get("/api/assistant/status"))
    if status.get("app", {}).get("answer_schema") != "exp12_standard_answer_v2":
        failures.append("後端摘要schema不符")
    html = get("/")
    script = re.search(r'(?P<url>/static/ai_chat\.js\?[^"\s<>]+)', html)
    asset_sha = None
    if script:
        asset = get(script["url"]).replace("\r\n", "\n")
        asset_sha = hashlib.sha256(asset.encode()).hexdigest()
        local = (Path(__file__).resolve().parents[2] / "src/web/static/ai_chat.js").read_text(encoding="utf-8")
        if asset_sha != hashlib.sha256(local.replace("\r\n", "\n").encode()).hexdigest():
            failures.append("實際HTML引用的前端資產與本輪原始碼不同")
    else:
        failures.append("HTML未提供可核對的聊天資產")
    context = {"experiment": "exp12", "mode": "result", "saved": saved}
    for index, question in enumerate((*QUESTIONS, *METHODS)):
        started = time.monotonic()
        body = {"question": question, "history": []}
        if index < len(QUESTIONS):
            body.update(context=context, history=[{"role": "assistant", "content": "legacy過於保守，93.12%只有故障，三角只表示真實未知配置。"}])
        request = urllib.request.Request(base_url + "/api/assistant/ask", data=json.dumps(body).encode(), headers=json_headers(base_url))
        try:
            raw = open_url(request, 100)
            frames = parse_sse(raw)
            errors = verify_result(frames, source_sha) if index < len(QUESTIONS) else []
            if index >= len(QUESTIONS) and (not any(f.get("event") == "tool" for f in frames)
                    or any(f.get("event") == "error" for f in frames) or not frames or frames[-1].get("event") != "done"):
                errors.append("一般方法問答未取得工具資料並完成／降級")
            cases.append({"question": question, "request": body, "seconds": round(time.monotonic()-started, 3),
                          "status": "FAILED" if errors else "VERIFIED", "errors": errors, "sse": raw, "frames": frames})
            failures.extend(errors)
        except Exception as exc:
            reason = str(exc) if isinstance(exc, HostNotAllowedError) else type(exc).__name__
            cases.append({"question": question, "status": "FAILED", "reason": reason})
            failures.append(f"第{index+1}題無法完成：{reason}")
    node = shutil.which("node")
    frontend = "UNAVAILABLE"
    if node and cases and cases[0].get("frames"):
        root = Path(__file__).resolve().parents[2]
        process = subprocess.run([node, str(root / "src/tests/ai_chat_frontend.cjs"),
                                  str(root / "src/web/static/markdown.js"), str(root / "src/web/static/ai_chat.js"), "--frames"],
                                 input=json.dumps(cases[0]["frames"]), capture_output=True, text=True, encoding="utf-8", timeout=15)
        frontend = "VERIFIED" if process.returncode == 0 else "FAILED"
        if process.returncode:
            failures.append("實際測試站SSE至前端JS驗證失敗")
    result = {"schema_version": "assistant_http_evidence_v1", "status": "FAILED" if failures else "VERIFIED_HTTP",
              "base_url": base_url, "expected_deployment_sha": deployment_sha,
              "deployment_sha_verification": "需另以容器OCI標籤及映像來源核對；不由HTTP輸入參數假定",
              "source_sha256": source_sha, "app_status": status, "asset_sha256_lf": asset_sha,
              "cases": cases, "failures": failures, "frontend_node": frontend, "browser": "NOT_VERIFIED",
              "scope": "六題結果與兩題一般方法的有限HTTP抽驗；模型自由文字仍須逐段審閱"}
    (paths.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    log.info("8787驗證={}；題數={}；問題數={}；瀏覽器未驗收", result["status"], len(cases), len(failures))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--saved", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--deployment-sha", required=True)
    args = parser.parse_args()
    return 0 if run(args.base_url, args.saved, args.source_sha, args.deployment_sha)["status"] == "VERIFIED_HTTP" else 1


if __name__ == "__main__":
    raise SystemExit(main())
