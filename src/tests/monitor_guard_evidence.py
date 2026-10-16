"""未擬合防護的紅燈／綠燈與固定版本配對證據。"""
import io
import json
import platform
import unittest

from core.logger import setup_run
from tests.test_monitor_guard import paired_baseline


def run():
    log, paths = setup_run("monitor_guard_evidence")
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_monitor_guard")
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (paths.output_dir / "tests.txt").write_text(stream.getvalue(), encoding="utf-8")
    evidence = {"python": platform.python_version(), "tests_run": result.testsRun,
                "failure_entries": len(result.failures), "error_entries": len(result.errors),
                "successful": result.wasSuccessful()}
    if result.wasSuccessful():
        evidence["paired_baseline"] = paired_baseline()
    (paths.output_dir / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("{}項；{} failure entries／{} error entries；successful={}",
             result.testsRun, len(result.failures), len(result.errors), result.wasSuccessful())
    return evidence


def main():
    raise SystemExit(0 if run()["successful"] else 1)


if __name__ == "__main__":
    main()
