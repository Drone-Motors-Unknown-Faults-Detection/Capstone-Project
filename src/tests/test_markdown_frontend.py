"""以 Node 執行真正的 markdown.js，驗證 AI 回答的 Markdown 轉換與 HTML 跳脫。"""
import shutil
import subprocess
from pathlib import Path


def test_markdown_frontend_contract():
    node = shutil.which("node")
    assert node, "前端契約測試需要 Node.js；缺少時不能視為通過"
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([node, str(root / "src/tests/markdown_frontend.cjs"),
                             str(root / "src/web/static/markdown.js")], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
