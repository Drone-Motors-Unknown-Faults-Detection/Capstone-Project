"""pytest 摘要狀態測試；skip、非零結束與零收集均不冒充通過。"""
import unittest
from tests.pytest_evidence import summarize, validate_targets


class PytestEvidenceTests(unittest.TestCase):
    def test_pass_skip_failure_and_empty(self):
        for tests, failures, errors, skipped, code, status in (
                (3, 0, 0, 0, 0, "PASS"), (3, 0, 0, 1, 0, "INCOMPLETE"),
                (3, 1, 0, 0, 1, "FAILED"), (3, 0, 1, 0, 1, "FAILED"),
                (0, 0, 0, 0, 0, "FAILED"), (3, 0, 0, 0, 2, "FAILED")):
            xml = f'<testsuites><testsuite tests="{tests}" failures="{failures}" errors="{errors}" skipped="{skipped}"/></testsuites>'.encode()
            self.assertEqual(summarize(xml, code)["status"], status)

    def test_trace_and_paths_are_not_serialized(self):
        validate_targets(["src/tests/", "src/tests/test_provenance.py"])
        for targets in ([], ["/private/tests"], ["src/tests/../private"], ["D:/private/tests"]):
            with self.assertRaises(ValueError):
                validate_targets(targets)
        xml = b'<testsuites><testsuite tests="1" failures="1"><testcase name="private-token"><failure>private-path</failure></testcase></testsuite></testsuites>'
        result = str(summarize(xml, 1))
        self.assertNotIn("private", result)
