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


def fixture_repository(root, identity, *, commit=True):
    """只建立暫存程式樹；Git 作者設定限這次 commit，不改主機設定。"""
    module = root / "src/core/runtime_environment.py"
    module.parent.mkdir(parents=True)
    module.write_text("# 測試用程式位置\n", encoding="utf-8")
    (root / "pyproject.toml").write_text('[project]\nname = "fixture"\n', encoding="utf-8")
    (root / "uv.lock").write_text(f'identity = "{identity}"\n', encoding="utf-8")
    if commit:
        subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(root), "add", "."], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(root), "-c", "user.name=Fixture",
                        "-c", "user.email=fixture@example.invalid", "commit", "-qm", identity],
                       check=True, capture_output=True)
    return module


class RuntimeEnvironmentTests(unittest.TestCase):
    def test_default_program_identity_does_not_follow_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            program, caller, outside = (base / name for name in ("program", "caller", "outside"))
            module = fixture_repository(program, "program")
            fixture_repository(caller, "caller")
            outside.mkdir()
            (program / "uv.lock").write_text('identity = "program-dirty"\n', encoding="utf-8")
            expected_head = runtime._git(program, "rev-parse", "HEAD")
            expected_lock = hashlib.sha256((program / "uv.lock").read_bytes()).hexdigest()
            for cwd in (program, program / "src", outside, caller):
                with self.subTest(cwd=cwd.name), working_directory(cwd), \
                        patch.object(runtime, "__file__", str(module)):
                    result = runtime.collect_environment(data=runtime.source_data_info(None))
                    self.assertEqual(result["git"]["head"], expected_head)
                    self.assertIs(result["git"]["tracked_dirty"], True)
                    self.assertEqual(result["constraints_sha256"], expected_lock)
                    self.assertNotIn(str(base), json.dumps(result))

    def test_export_inside_another_repo_does_not_borrow_parent_git(self):
        from experiments import exp6_formal_benchmark as formal, exp6_matrix as matrix, aggregate_exp6
        from experiments import health_index_benchmark as health
        from tests.exp6_fixture import write_matrix
        from tests.test_exp6_benchmark import _fixture

        with tempfile.TemporaryDirectory() as directory:
            caller = Path(directory) / "caller"
            fixture_repository(caller, "caller")
            exported = caller / "exported"
            module = fixture_repository(exported, "exported", commit=False)
            data = _fixture(Path(directory) / "features")
            manifest = write_matrix(Path(directory) / "parents")
            with working_directory(caller), patch.object(runtime, "__file__", str(module)):
                result = runtime.collect_environment(data=runtime.source_data_info(None))
                aggregate_env = aggregate_exp6.aggregate(manifest)["provenance"]["environment"]
                formal_result = formal.run(data, openset_method="knn", require_nine=False)
                health_result = health.run(data, openset_method="knn", require_nine=False)
                keys = [{"run_id": matrix.run_key("T1", "8000rpm", 17, "knn"),
                         "motor": "T1", "rpm": "8000rpm", "seed": 17, "method": "knn"}]
                with self.assertRaisesRegex(ValueError, "可信 Git"):
                    matrix.matrix_contract(data, keys, ("knn",))
            self.assertIsNone(result["git"]["head"])
            self.assertIsNone(result["git"]["tracked_dirty"])
            self.assertEqual(result["git"]["status"], "UNKNOWN")
            self.assertEqual(result["constraints_sha256"], hashlib.sha256((exported / "uv.lock").read_bytes()).hexdigest())
            self.assertEqual(result["program_source"]["git_reason"], "NOT_PROGRAM_CHECKOUT")
            self.assertEqual(aggregate_env["git"], result["git"])
            for run in (formal_result, health_result):
                self.assertEqual(run["commit_sha"], "unknown")

    def test_explicit_root_keeps_caller_selected_repo_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            program, caller = base / "program", base / "caller"
            module = fixture_repository(program, "program")
            fixture_repository(caller, "caller")
            with working_directory(program), patch.object(runtime, "__file__", str(module)):
                result = runtime.collect_environment(caller / "src", data=runtime.source_data_info(None))
            self.assertEqual(result["git"]["head"], runtime._git(caller, "rev-parse", "HEAD"))
            self.assertEqual(result["constraints_sha256"], hashlib.sha256((caller / "uv.lock").read_bytes()).hexdigest())
            self.assertEqual(result["program_source"]["selection"], "EXPLICIT_ROOT")

    def test_installed_layout_never_falls_back_to_callers_repo(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            caller = base / "caller"
            fixture_repository(caller, "caller")
            installed = base / "site-packages/core/runtime_environment.py"
            installed.parent.mkdir(parents=True)
            installed.touch()
            with working_directory(caller), patch.object(runtime, "__file__", str(installed)):
                result = runtime.collect_environment(data=runtime.source_data_info(None))
            self.assertEqual(result["git"], {"head": None, "tracked_dirty": None, "status": "UNKNOWN"})
            self.assertIsNone(result["constraints_sha256"])
            self.assertEqual(result["program_source"]["git_reason"], "PROGRAM_ROOT_UNAVAILABLE")
            self.assertEqual(result["program_source"]["lock_status"], "UNKNOWN")

    def test_export_without_git_keeps_only_its_own_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "exported"
            module = fixture_repository(root, "exported", commit=False)
            with patch.object(runtime, "__file__", str(module)):
                result = runtime.collect_environment(data=runtime.source_data_info(None))
            self.assertIsNone(result["git"]["head"])
            self.assertIsNone(result["git"]["tracked_dirty"])
            self.assertEqual(result["program_source"]["lock_status"], "VERIFIED")
            self.assertEqual(result["constraints_sha256"], hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest())

    def test_missing_or_unreadable_lock_does_not_hide_verified_git(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "program"
            module = fixture_repository(root, "program")
            for reason in ("LOCK_MISSING", "LOCK_UNREADABLE"):
                error = FileNotFoundError("私人路徑") if reason == "LOCK_MISSING" else PermissionError("私人路徑")
                with self.subTest(reason=reason), patch.object(runtime, "__file__", str(module)), \
                        patch.object(runtime, "sha256_file", side_effect=error):
                    result = runtime.collect_environment(data=runtime.source_data_info(None))
                self.assertEqual(result["git"]["status"], "VERIFIED")
                self.assertIsNone(result["constraints_sha256"])
                self.assertEqual(result["program_source"]["lock_reason"], reason)
                self.assertNotIn("私人路徑", json.dumps(result))

    def test_default_git_failure_keeps_unknown_dirty_and_safe_reason(self):
        for error in (OSError("私人 Git 路徑"), subprocess.TimeoutExpired("git", 3)):
            with self.subTest(error=type(error).__name__), \
                    patch.object(runtime.subprocess, "run", side_effect=error):
                result = runtime.collect_environment(data=runtime.source_data_info(None))
            self.assertEqual(result["git"], {"head": None, "tracked_dirty": None, "status": "UNKNOWN"})
            self.assertEqual(result["program_source"]["git_reason"], "GIT_UNAVAILABLE_OR_INCOMPLETE")
            self.assertNotIn("私人", json.dumps(result))

    def test_common_entries_keep_program_git_data_and_output_roles_separate(self):
        import argparse
        import numpy as np
        from core.data import discover_datasets
        from core.provenance import load_pool_source
        from core.runner import execution_context, run_dataset, run_root_sources
        from experiments import exp6_formal_benchmark as formal, exp6_matrix as matrix, aggregate_exp6
        from tests.exp6_fixture import write_matrix
        from tests.test_exp6_benchmark import _fixture

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            caller = base / "caller"
            fixture_repository(caller, "caller")
            data = _fixture(base / "features")
            manifest = write_matrix(base / "parents")
            expected = runtime.collect_environment(ROOT, data=runtime.source_data_info(None))
            with working_directory(caller):
                _, paths = setup_run("program_fixture", unique=True, data_root=data)
                saved = json.loads((paths.output_dir / "environment.json").read_text())
                args = argparse.Namespace(data_root=data, motor="T1", rpm="8000rpm")
                _, pool = run_dataset(args, paths, lambda pools, source_context:
                                      execution_context(pools, source_context))
                memory = execution_context({"8screws": np.zeros((3, 105))})
                def compute(data_root, *, source_run):
                    source_run.load_all()
                    return {}
                root = run_root_sources(compute, data, output_dir=paths.output_dir)["provenance"]
                formal_result = formal.run(data, openset_method="knn", require_nine=False)
                formal_env = formal_result["environment"]
                self.assertEqual(formal_result["commit_sha"], expected["git"]["head"])
                # 只核對環境來源的單工況契約，不執行或放寬正式九工況 runner。
                keys = [{"run_id": matrix.run_key("T1", "8000rpm", 17, method),
                         "motor": "T1", "rpm": "8000rpm", "seed": 17, "method": method}
                        for method in ("mahalanobis", "knn")]
                _, matrix_env = matrix.matrix_contract(data, keys, ("mahalanobis", "knn"))
                aggregate_env = aggregate_exp6.aggregate(manifest)["provenance"]["environment"]
                pools, context = load_pool_source(data, discover_datasets(data)[0]["path"])
                source_env = execution_context(pools, context)["environment"]
                self.assertEqual(paths.output_dir.resolve().parents[2], caller)
            for name, environment in (("setup", saved), ("pool", pool["environment"]),
                    ("root", root["environment"]), ("formal", formal_env), ("matrix", matrix_env),
                    ("aggregate", aggregate_env), ("memory", memory["environment"]), ("source", source_env)):
                with self.subTest(entry=name):
                    self.assertEqual(environment["git"], expected["git"])
                    self.assertEqual(environment["constraints_sha256"], expected["constraints_sha256"])
                    self.assertNotIn(str(base), json.dumps(environment))
            self.assertEqual(pool["source"]["dataset_fingerprint"], root["source"]["dataset_fingerprint"])
            self.assertEqual(saved["data"]["dataset_fingerprint"], pool["source"]["dataset_fingerprint"])
            self.assertEqual(memory["environment"]["data"]["reason"], "UNVERIFIED_MEMORY")

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

    def test_no_argv_sniffing(self):
        """資料根只從呼叫端傳入；命令列的 --data-root 不再被偷看（#269）。"""
        self.assertFalse(hasattr(runtime, "_argv_data_root"))
        with patch.object(runtime, "data_info", side_effect=lambda r: {"root": str(r)}):
            with patch("sys.argv", ["x", "--data-root", "from_argv"]):
                self.assertEqual(runtime.collect_environment(ROOT)["data"], {"root": "data"})

    def test_data_info_failed_keeps_error_type_but_not_message(self):
        """manifest 與內容不符是 FAILED，不是「沒有資料」的 UNKNOWN（#269 C24）。"""
        from core.provenance import ProvenanceError

        with patch("core.provenance.capture_source", side_effect=ProvenanceError("private-path-secret")):
            info = runtime.data_info("data")
        self.assertEqual(info, {"content_version": None, "dataset_fingerprint": None, "n_files": 0,
                                "status": "FAILED", "error_type": "ProvenanceError"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv = root / "Step-1/myfeature/T1/8000rpm/8screws/T1_Group_feature_data_clean.csv"
            csv.parent.mkdir(parents=True)
            csv.write_text("a,b\n1,2\n", encoding="utf-8")
            (root / "formal_materialization_manifest.json").write_text("{not json", encoding="utf-8")
            info = runtime.data_info(root)
        self.assertEqual((info["status"], info["error_type"]), ("FAILED", "ProvenanceError"))

    def test_git_head_tolerates_missing_git(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(runtime.git_head(Path(directory)))
        with patch.object(runtime.subprocess, "run", side_effect=OSError("no git")):
            self.assertIsNone(runtime.git_head(ROOT))
        self.assertRegex(runtime.git_head(ROOT) or "", r"^[0-9a-f]{40}$")

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

    def test_explicit_data_root_and_precomputed_data(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runtime, "data_info", side_effect=lambda r: {"root": str(r)}) as info:
            self.assertEqual(runtime.collect_environment(Path(directory), data_root="explicit")["data"], {"root": "explicit"})
            precomputed = {"content_version": "v", "dataset_fingerprint": "f", "n_files": 1, "status": "VERIFIED"}
            self.assertEqual(runtime.collect_environment(Path(directory), data=precomputed)["data"], precomputed)
            self.assertEqual(info.call_count, 1)

    def test_source_data_info_matches_capture_source(self):
        with tempfile.TemporaryDirectory() as directory:
            csv = Path(directory) / "Step-1/myfeature/T1/8000rpm/8screws/T1_Group_feature_data_clean.csv"
            csv.parent.mkdir(parents=True)
            csv.write_text("a,b\n1,2\n", encoding="utf-8")
            from core.provenance import capture_source

            self.assertEqual(runtime.source_data_info(capture_source(directory)[0]), runtime.data_info(directory))

    def test_memory_environment_does_not_inspect_default_data(self):
        with patch.object(runtime, "data_info", side_effect=AssertionError("不得讀預設資料根")):
            result = runtime.collect_environment(ROOT, data=runtime.source_data_info(None))
        self.assertEqual(result["data"]["status"], "UNKNOWN")
        self.assertIsNone(result["data"]["dataset_fingerprint"])
        self.assertEqual(result["data"]["reason"], "UNVERIFIED_MEMORY")

    def test_setup_run_writes_json(self):
        with tempfile.TemporaryDirectory() as directory, working_directory(directory):
            _, paths = setup_run("fixture_environment")
            result = json.loads((paths.output_dir / "environment.json").read_text(encoding="utf-8"))
            self.assertEqual(result["schema_version"], "runtime_environment_v1")
            self.assertIn("python", result)
            self.assertTrue(paths.log_file.is_file())

    def test_setup_run_can_skip_unrelated_raw_data(self):
        with tempfile.TemporaryDirectory() as directory, working_directory(directory), \
                patch.object(runtime, "data_info", side_effect=AssertionError("不得掃描無關data")):
            _, paths = setup_run("artifact_only", include_data=False)
            result = json.loads((paths.output_dir / "environment.json").read_text())
            self.assertEqual(result["data"]["status"], "UNKNOWN")
            self.assertIsNone(result["data"]["dataset_fingerprint"])

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
