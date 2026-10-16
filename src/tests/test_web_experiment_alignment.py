"""目的、資料角色及真實前端呈現契約；不讀正式data或重跑研究。"""
from pathlib import Path
import shutil
import subprocess

from tests.test_experiment_handbooks import selected_documents


def test_each_handbook_explains_denominators_and_presentation():
    for doc in selected_documents():
        text = doc.read_text(encoding="utf-8")
        for phrase in ("資料角色用量表", "比例與分母", "發表與理解檢查", "口頭", "組員"):
            assert phrase in text, (doc.name, phrase)


def test_actual_experiments_javascript_contract():
    node = shutil.which("node")
    assert node, "前端回歸需安裝Node.js，不能把缺執行器當通過"
    result = subprocess.run([node, str(Path(__file__).with_name("experiments_frontend.cjs"))],
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
