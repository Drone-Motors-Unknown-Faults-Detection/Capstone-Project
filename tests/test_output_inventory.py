"""清冊核對：不以副檔名、同目錄或 writer 存在冒充來源證明。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.output_inventory_evidence import (
    inspect, main, run, verify_document_relocations, verify_external_relocations,
)


class OutputInventoryTests(unittest.TestCase):
    def test_absent_manual_report_does_not_disable_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("tests.output_inventory_evidence.inspect", return_value={"files": []}), \
                    patch("tests.output_inventory_evidence.verify_cleanup", side_effect=ValueError("來源損壞")) as verify:
                with self.assertRaisesRegex(ValueError, "來源損壞"):
                    run(root)
            verify.assert_called_once_with(root)

    def test_invalid_relocation_fails_cli_without_manual_report(self):
        verification = {"other_tracked_output_unchanged": True, "manifest_targets_exist": True,
                        "broken_links": [], "moves": [],
                        "external_relocations": [{"same_bytes": False, "old_absent": True}]}
        with patch("tests.output_inventory_evidence.run", return_value={"cleanup_verification": verification}), \
                patch("sys.argv", ["output_inventory_evidence"]):
            self.assertEqual(main(), 1)

    def test_writer_unknown_and_screenshot_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tests").mkdir()
            (root / "tests/example.py").write_text('def run():\n    target.write_text("results.json")\n', encoding="utf-8")
            for name in ("example/run/results.json", "example/run/manual.md", "integration_browser/guide_trend_a.png",
                         "integration_browser/run/image.png", ".gitkeep"):
                path = root / "output" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            with patch("tests.output_inventory_evidence.subprocess.check_output", side_effect=["", "fixture-head"]):
                result = inspect(root)
            by_name = {row["path"]: row for row in result["files"]}
            self.assertEqual(by_name["output/example/run/results.json"]["category"], "WRITER_MATCH_HISTORY_UNCONFIRMED")
            self.assertEqual(by_name["output/example/run/manual.md"]["category"], "UNKNOWN_HISTORY")
            self.assertEqual(by_name["output/integration_browser/guide_trend_a.png"]["category"], "DOCUMENTED_EXTERNAL_SCREENSHOT")
            self.assertEqual(by_name["output/integration_browser/run/image.png"]["category"], "UNKNOWN_HISTORY")
            self.assertIsNone(by_name["output/.gitkeep"]["log"])
            self.assertEqual(by_name["output/.gitkeep"]["category"], "UNKNOWN_HISTORY")
            self.assertTrue(all(row["action"] == "KEEP" for row in result["files"]))
            self.assertNotIn(str(root), json.dumps(result))

    def test_repeat_inventory_is_stable_and_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "output/history/run").mkdir(parents=True)
            path = root / "output/history/run/evidence.json"
            path.write_bytes(b"historical\r\n")
            with patch("tests.output_inventory_evidence.subprocess.check_output", side_effect=["", "head", "", "head"]):
                first, second = inspect(root), inspect(root)
            self.assertEqual(first, second)
            self.assertEqual(path.read_bytes(), b"historical\r\n")

    def test_relocated_bytes_and_no_output_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "docs/image.png"
            target.parent.mkdir()
            target.write_bytes(b"original\r\n")
            with patch("tests.output_inventory_evidence.subprocess.check_output", return_value=b"original\r\n"):
                rows = verify_external_relocations(root, "fixed", {"output/image.png": "docs/image.png"})
            self.assertTrue(rows[0]["same_bytes"])
            self.assertTrue(rows[0]["old_absent"])
            (root / "output").mkdir()
            (root / "output/image.png").write_bytes(b"original\r\n")
            target.write_bytes(b"changed\n")
            with patch("tests.output_inventory_evidence.subprocess.check_output", return_value=b"original\r\n"):
                rows = verify_external_relocations(root, "fixed", {"output/image.png": "docs/image.png"})
            self.assertFalse(rows[0]["same_bytes"])
            self.assertFalse(rows[0]["old_absent"])

    def test_missing_destination_is_not_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("tests.output_inventory_evidence.subprocess.check_output", return_value=b"original"):
                rows = verify_external_relocations(Path(directory), "fixed", {"output/image.png": "docs/image.png"})
            self.assertFalse(rows[0]["same_bytes"])
            self.assertIsNone(rows[0]["target_sha256"])

    def test_command_summary_relocation_checks_content_and_old_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new = "output/baseline/run/test_summary.json", "reports/baseline/test_summary.json"
            target = root / new
            target.parent.mkdir(parents=True)
            target.write_bytes(b'{"tests": 111}\r\n')
            with patch("tests.output_inventory_evidence.subprocess.check_output",
                       side_effect=["same-blob\n", "same-blob\n"]) as git:
                row = verify_document_relocations(root, "fixed-source", {old: new})[0]
            self.assertTrue(row["same_git_content"])
            self.assertTrue(row["old_absent"])
            self.assertTrue(row["new_exists"])
            self.assertEqual(git.call_args_list[0].args[0], ["git", "rev-parse", f"fixed-source:{old}"])
            self.assertEqual(git.call_args_list[1].args[0], ["git", "hash-object", "--path", new, new])
            (root / old).parent.mkdir(parents=True)
            (root / old).write_bytes(target.read_bytes())
            with patch("tests.output_inventory_evidence.subprocess.check_output",
                       side_effect=["original\n", "tampered\n"]):
                row = verify_document_relocations(root, "fixed-source", {old: new})[0]
            self.assertFalse(row["same_git_content"])
            self.assertFalse(row["old_absent"])

    def test_missing_command_summary_target_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("tests.output_inventory_evidence.subprocess.check_output",
                       side_effect=["original\n", subprocess_error := FileNotFoundError("缺目的檔")]):
                with self.assertRaises(FileNotFoundError) as error:
                    verify_document_relocations(Path(directory), "fixed", {"output/summary.json": "reports/summary.json"})
            self.assertIs(error.exception, subprocess_error)
