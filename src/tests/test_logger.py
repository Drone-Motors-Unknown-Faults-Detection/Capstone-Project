"""setup_run 以相對路徑寫 logs/、output/；conftest 讓測試期間落在暫存目錄。"""
from pathlib import Path

from core.logger import setup_run

REPO = Path(__file__).resolve().parents[2]


def test_setup_run_inside_tests_never_writes_repo_logs_or_output(tmp_path):
    _, paths = setup_run("conftest_isolation_probe", unique=True)
    assert paths.log_file.resolve().is_relative_to(tmp_path.resolve())
    assert paths.output_dir.resolve().is_relative_to(tmp_path.resolve())
    assert not (REPO / "logs" / "conftest_isolation_probe").exists()
    assert not (REPO / "output" / "conftest_isolation_probe").exists()
