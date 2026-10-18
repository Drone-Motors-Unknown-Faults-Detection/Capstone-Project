"""以 Node 執行真正的 markdown.js + ai_chat.js：狀態燈、拖曳改大小、Enter／選字、串流 Markdown 與依據。"""
import shutil
import subprocess
from pathlib import Path


def test_ai_chat_frontend_contract():
    node = shutil.which("node")
    assert node, "前端規範測試需要 Node.js；缺少時不能視為通過"
    root = Path(__file__).resolve().parents[2]
    static = root / "src/web/static"
    result = subprocess.run([node, str(root / "src/tests/ai_chat_frontend.cjs"),
                             str(static / "markdown.js"), str(static / "ai_chat.js")],
                            capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
