"""測試在各自的暫存目錄執行，setup_run 的相對路徑 logs/、output/ 不會寫進 repo。"""
import pytest


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
