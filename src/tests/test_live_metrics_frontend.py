"""以 Node 執行首頁 onMetrics()：健康來源擴張前判未知率是誤報率，不套「越高越好」的上色（#305）。"""
import shutil
import subprocess
from pathlib import Path


def test_healthy_pre_rate_is_not_coloured_as_detection():
    node = shutil.which("node")
    assert node, "前端規範測試需要 Node.js；缺少時不能視為通過"
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([node, str(root / "src/tests/live_metrics_frontend.cjs"),
                             str(root / "src/web/static/index.html")],
                            capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
