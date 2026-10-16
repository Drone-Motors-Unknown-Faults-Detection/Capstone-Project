"""以 Node 執行真正的 guide.js，驗證 snapshot 與選單到達順序。"""
import shutil
import subprocess
from pathlib import Path


def test_guide_frontend_state_contract():
    node = shutil.which("node")
    assert node, "前端契約測試需要 Node.js；缺少時不能視為通過"
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([node, str(root / "tests/guide_frontend.cjs"),
                             str(root / "web/static/guide.js")], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
