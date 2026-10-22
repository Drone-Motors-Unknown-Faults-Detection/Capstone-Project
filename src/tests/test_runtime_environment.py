"""環境紀錄與鎖版規範的 fixture；完全不讀正式 data。"""
from __future__ import annotations

from contextlib import contextmanager
from importlib import metadata
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from loguru import logger
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

from core.logger import setup_run
from core import runtime_environment as runtime

ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def working_directory(root):
    before = Path.cwd()
    try:
        os.chdir(root)
        yield
    finally:
        logger.remove()
        os.chdir(before)


class RuntimeEnvironmentTests(unittest.TestCase):
    def test_missing_git_is_unknown(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runtime, "_git", return_value=None):
            result = runtime.collect_environment(Path(directory))
        self.assertEqual(result["git"], {"head": None, "tracked_dirty": None, "status": "UNKNOWN"})
        self.assertIsNone(result["constraints_sha256"])

    def test_git_failure_has_no_private_stderr(self):
        with patch.object(runtime.subprocess, "run", return_value=Mock(returncode=1, stderr="private token")):
            self.assertIsNone(runtime._git(ROOT, "status"))

    def test_git_missing_executable_and_timeout(self):
        for error in (OSError("private path"), subprocess.TimeoutExpired("git", 3)):
            with self.subTest(error=type(error).__name__), patch.object(runtime.subprocess, "run", side_effect=error):
                self.assertIsNone(runtime._git(ROOT, "status"))

    def test_verified_dirty_and_constraints_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b'[[package]]\nname = "numpy"\nversion = "2.2.6"\n'
            (root / "uv.lock").write_bytes(payload)
            with patch.object(runtime, "_git", side_effect=[directory, "a" * 40, " M core/logger.py"]):
                result = runtime.collect_environment(root)
        self.assertTrue(result["git"]["tracked_dirty"])
        self.assertEqual(result["git"]["head"], "a" * 40)
        self.assertEqual(result["constraints_sha256"], hashlib.sha256(payload).hexdigest())

    def test_clean_git_and_missing_status_are_distinct(self):
        for status, expected in (("", False), (None, None)):
            with self.subTest(status=status), patch.object(runtime, "_git", side_effect=[str(ROOT), "b" * 40, status]):
                result = runtime.collect_environment(ROOT)
            self.assertEqual(result["git"]["tracked_dirty"], expected)
            self.assertEqual(result["git"]["status"], "UNKNOWN" if status is None else "VERIFIED")

    def test_package_names_are_normalized_and_sorted(self):
        distributions = [Mock(metadata={"Name": "Z_Name"}, version="1"),
                         Mock(metadata={"Name": "a"}, version="2"), Mock(metadata={}, version="3")]
        with patch.object(runtime.metadata, "distributions", return_value=distributions):
            result = runtime.collect_environment(ROOT)
        self.assertEqual(list(result["packages"]), ["a", "z-name"])
        self.assertEqual(result["packages"]["z-name"], "1")

    def test_public_metadata_excludes_paths_and_environment(self):
        with patch.dict(os.environ, {"LINEAGE_TEST_SECRET": "private-token-value"}):
            result = runtime.collect_environment(ROOT)
        serialized = json.dumps(result)
        self.assertNotIn(str(ROOT), serialized)
        self.assertNotIn("private-token-value", serialized)
        self.assertNotIn("executable", result)
        self.assertNotIn("environ", result)

    def test_records_blas_and_host(self):
        with tempfile.TemporaryDirectory() as directory:
            result = runtime.collect_environment(Path(directory))
        self.assertIn("host", result["os"])
        self.assertLessEqual({"blas", "blas_version", "lapack"}, set(result["numeric"]))
        self.assertIsNotNone(result["numeric"]["blas"])

    def test_blas_unavailable_is_unknown(self):
        with patch("numpy.show_config", side_effect=RuntimeError("no config")):
            info = runtime.blas_info()
        self.assertEqual(info["status"], "UNKNOWN")
        self.assertIsNone(info["blas"])

    def test_data_root_from_argv(self):
        self.assertEqual(runtime._argv_data_root(["x", "--seed", "1"]), "data")
        self.assertEqual(runtime._argv_data_root(["x", "--data-root", "d2"]), "d2")
        self.assertEqual(runtime._argv_data_root(["x", "--data-root=d3"]), "d3")

    def test_data_info_unknown_without_data(self):
        with tempfile.TemporaryDirectory() as directory:
            info = runtime.data_info(Path(directory) / "missing")
        self.assertEqual(info["status"], "UNKNOWN")
        self.assertIsNone(info["dataset_fingerprint"])

    def test_data_info_fingerprints_feature_csvs(self):
        with tempfile.TemporaryDirectory() as directory:
            csv = Path(directory) / "Step-1/myfeature/T1/8000rpm/8screws/T1_Group_feature_data_clean.csv"
            csv.parent.mkdir(parents=True)
            csv.write_text("a,b\n1,2\n", encoding="utf-8")
            first = runtime.data_info(directory)
            csv.write_text("a,b\n1,3\n", encoding="utf-8")
            second = runtime.data_info(directory)
        self.assertEqual(first["status"], "VERIFIED")
        self.assertEqual(first["n_files"], 1)
        self.assertNotEqual(first["dataset_fingerprint"], second["dataset_fingerprint"])
        self.assertNotIn(directory, json.dumps(first))

    def test_setup_run_writes_json(self):
        with tempfile.TemporaryDirectory() as directory, working_directory(directory):
            _, paths = setup_run("fixture_environment")
            result = json.loads((paths.output_dir / "environment.json").read_text(encoding="utf-8"))
            self.assertEqual(result["schema_version"], "runtime_environment_v1")
            self.assertIn("python", result)
            self.assertTrue(paths.log_file.is_file())

    def test_no_output_records_environment_in_log(self):
        with tempfile.TemporaryDirectory() as directory, working_directory(directory):
            _, paths = setup_run("fixture_log_only", make_output=False)
            self.assertIsNone(paths.output_dir)
            self.assertIn("runtime_environment_v1", paths.log_file.read_text(encoding="utf-8"))
            self.assertFalse(Path("output").exists())

    def test_constraints_are_exact_and_unique(self):
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib
        packages = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))["package"]
        names = [item["name"].lower() for item in packages]
        self.assertEqual(len(names), len(set(names)))
        for item in packages:
            with self.subTest(package=item["name"]):
                self.assertIn("version", item)
                self.assertNotIn("*", item["version"])
        self.assertIn("cloudpickle", names)
        self.assertTrue({"pytest", "ruff"}.issubset(names))

    def test_active_dependency_closure_is_pinned(self):
        locked = set(runtime.lock_versions(ROOT))
        installed = {dist.metadata["Name"].lower().replace("_", "-")
                     for dist in metadata.distributions() if dist.metadata.get("Name")}
        for parent in sorted(locked & installed):
            for entry in metadata.requires(parent) or []:
                child = Requirement(entry)
                if child.marker is not None and not child.marker.evaluate({"extra": ""}):
                    continue
                with self.subTest(parent=parent, child=child.name):
                    self.assertIn(child.name.lower().replace("_", "-"), locked)

    def test_python_policy_is_minor_not_patch(self):
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        value = next(line.split('"')[1] for line in text.splitlines() if line.startswith("requires-python"))
        policy = SpecifierSet(value)
        self.assertIn("3.10.19", policy)
        self.assertIn("3.10.20", policy)
        self.assertNotIn("3.9.0", policy)
        self.assertNotIn("3.14.0", policy)

    def test_build_scripts_do_not_delete_existing_environments(self):
        quantity = ("run_pytest.sh", "run_ruff.sh")
        opt = ("build_uv.sh", "restart_docker.sh", "run_llm.sh", "run_web.sh")
        self.assertEqual(sorted(path.name for path in (ROOT / "scripts/quantity").glob("*.sh")), sorted(quantity))
        self.assertEqual(sorted(path.name for path in (ROOT / "scripts/opt").glob("*.sh")), sorted(opt))
        for name in (*quantity, *opt, "build_uv.ps1", "build_uv_mac.sh"):
            self.assertFalse((ROOT / name).exists(), name)
        self.assertFalse((ROOT / "requirements-legacy-mac.txt").exists())
        build = (ROOT / "scripts/opt/build_uv.sh").read_text(encoding="utf-8")
        self.assertNotIn("rm -rf", build)
        self.assertNotIn("Remove-Item", build)
        self.assertNotIn("legacy-mac", build)
        self.assertIn("--locked", build)
        self.assertIn('VENV_DIR="venv"', build)
        self.assertIn('cd -- "$(dirname -- "$0")/../.."', build)
        for name in (*quantity, "run_web.sh"):
            folder = "quantity" if name in quantity else "opt"
            text = (ROOT / "scripts" / folder / name).read_text(encoding="utf-8")
            self.assertIn('cd "$(dirname "$0")/../.."', text)
        ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("venv/", ignored)
        self.assertIn(".venv310/", ignored)

    def test_run_and_main_preserve_environment_contract(self):
        with tempfile.TemporaryDirectory() as directory, working_directory(directory):
            result = runtime.run("fixture_api")
            self.assertEqual(result["schema_version"], "runtime_environment_v1")
            with patch("sys.argv", ["runtime_environment", "--program", "runtime_environment"]):
                runtime.main()
            self.assertTrue(any(Path("output/runtime_environment").glob("*/environment.json")))


if __name__ == "__main__":
    unittest.main()
