"""shell 腳本可攜性：macOS 內建 bash 3.2 會把緊接在 $VAR 後的非 ASCII 位元組當成變數名的一部分。

例如 `$CURRENT，` 被讀成變數 `CURRENT\\xef`，展開為空，只剩半個全形逗號，輸出變成非法 UTF-8（#223）。
緊接中文或全形標點的變數一律寫成 ${VAR}。
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UNBRACED_BEFORE_NON_ASCII = re.compile(rb"\$[A-Za-z_][A-Za-z0-9_]*[\x80-\xff]")


class ShellScriptPortabilityTests(unittest.TestCase):
    def test_vars_before_non_ascii_are_braced(self):
        scripts = sorted(ROOT.glob("scripts/**/*.sh")) + sorted(ROOT.glob("docker/**/*.sh"))
        self.assertTrue(scripts)
        bad = [
            f"{path.relative_to(ROOT)}:{n}"
            for path in scripts
            for n, line in enumerate(path.read_bytes().splitlines(), 1)
            if UNBRACED_BEFORE_NON_ASCII.search(line)
        ]
        self.assertEqual(bad, [])

    def test_pattern_catches_the_223_case(self):
        self.assertTrue(UNBRACED_BEFORE_NON_ASCII.search("=$CURRENT，".encode()))
        self.assertIsNone(UNBRACED_BEFORE_NON_ASCII.search("=${CURRENT}，".encode()))


if __name__ == "__main__":
    unittest.main()
