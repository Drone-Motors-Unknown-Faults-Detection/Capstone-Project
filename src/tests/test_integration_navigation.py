"""主線文件與獨立導覽的整合邊界回歸。"""
from pathlib import Path
import re
import unittest

from web.experiments import CATALOG
from web.guide import GuideHub

ROOT = Path(__file__).resolve().parents[2]


class IntegrationNavigationTests(unittest.TestCase):
    def test_main_catalog_keeps_all_original_entries(self):
        self.assertEqual([entry["id"] for entry in CATALOG],
                         ["exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp6_formal", "exp7", "exp8", "exp8_monitor"])

    def test_navigation_local_links_resolve_and_names_follow_main(self):
        for path in (ROOT / "docs/navigation").iterdir():
            self.assertTrue(path.name.startswith("exp24_"), path.name)
            if path.suffix != ".md":
                continue
            for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
                if re.match(r"https?://|#", target):
                    continue
                self.assertTrue((path.parent / target.split("#")[0]).exists(), (path, target))

    def test_empty_guide_exposes_limit_without_fit(self):
        hub = GuideHub([])
        try:
            state = hub.full_state()
            self.assertIsNone(hub.demo)
            self.assertEqual(state["step"], "缺少資料")
            self.assertIn("UNKNOWN", state["evidence"])
            self.assertIn("INCOMPLETE", state["evidence"])
        finally:
            hub.executor.shutdown()

    def test_modes_do_not_send_model_commands(self):
        text = (ROOT / "src/web/static/guide.js").read_text(encoding="utf-8")
        body = re.search(r"function mode\(\)\{([^\n]+)\}", text).group(1)
        self.assertNotIn("send(", body)
        self.assertIn("research=!research", body)

    def test_source_references_are_fixed_not_moving_research_branch(self):
        text = (ROOT / "docs/navigation/exp24_來源索引.md").read_text(encoding="utf-8")
        self.assertIn("b3d68f145cfa187e208e38cae74bf33f68d7c367", text)
        self.assertNotIn("/blob/research-improvements-20260920/", text)
