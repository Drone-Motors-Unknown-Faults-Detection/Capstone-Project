"""實驗六廣度基準（exp6_osr_benchmark）：空資料根要明確失敗，不產生 NaN 摘要（#285 R2-04）。"""
import tempfile
import unittest
from pathlib import Path

from experiments.exp6_osr_benchmark import run


class OsrBenchmarkInputTests(unittest.TestCase):
    def test_empty_data_root_is_rejected_before_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                run(Path(directory), seed=1)


if __name__ == "__main__":
    unittest.main()
