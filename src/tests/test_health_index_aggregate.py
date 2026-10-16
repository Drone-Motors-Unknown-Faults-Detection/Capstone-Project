"""實驗八重算 fixtures：不讀忽略的 data，不擬合模型。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from experiments import health_index_aggregate as aggregate


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _stats(values):
    mean = sum(values) / len(values)
    return mean, math.sqrt(sum((value-mean)**2 for value in values)/(len(values)-1))


def _fixture(base: Path):
    root = base / "archive"
    root.mkdir()
    contract = json.loads(aggregate.DEFAULT_CONTRACT.read_text(encoding="utf-8"))
    contract["dataset_fingerprint"] = "a" * 64
    entries, all_rows = [], []
    conditions = sorted((motor, rpm) for motor in contract["motors"] for rpm in contract["rpms"])
    for seed_index, seed in enumerate(contract["seeds"]):
        folder = root / f"seed_{seed}"
        folder.mkdir()
        for method in contract["methods"]:
            rows = []
            for index, (motor, rpm) in enumerate(conditions):
                mean = 0.2*(seed_index+1)+0.01*index+(0.05 if method == "knn" else 0)
                row = {"dataset": f"{motor}/{rpm[:-3]}", "motor": motor, "rpm": rpm, "seed": seed,
                       "method": method, "mahalanobis_method": "ledoit_wolf" if method == "mahalanobis" else None,
                       "knn_neighbors": 5 if method == "knn" else None,
                       "n_train": 30+index, "n_calibration": 10+index, "n_known_test": 10+index,
                       "n_unknown_test": 100+index*10, "n_unknown_classes": 9,
                       "known_health_mean": mean, "known_health_std": 0.2, "known_health_median": mean,
                       "known_healthy_fraction": 0.3, "unknown_health_mean": 0.0, "unknown_health_std": 0.0,
                       "unknown_health_median": 0.0, "unknown_critical_fraction": 1.0,
                       "health_gap_known_minus_unknown": mean, "health_effect_size": 2.0,
                       "known_score_median": 0.8, "unknown_score_median": 8.0,
                       "open_set_accuracy": 0.9, "auroc": 1.0, "aupr_unknown_positive": 1.0,
                       "unknown_recall": 1.0, "unknown_f1": 0.9,
                       "calibration_healthy_anchor": 0.7, "calibration_critical_anchor": 1.0,
                       "rul_available": False, "status": "completed"}
                rows.append(row)
            value = {"schema_version": 1, "status": "completed", "experiment": "health_index_benchmark",
                     "method": method, "seed": seed, "confidence": 0.95, "formal_condition_count": 9,
                     "dataset_root": "fixture", "dataset_fingerprint": "a"*64,
                     "score_direction": "higher_is_worse", "health_direction": "higher_is_better",
                     "health_interpretation": "工程測試資料，不作研究結果", "unknown_calibration_leakage": False,
                     "rul_available": False, "commit_sha": "b"*40, "python": "3.10.19", "rows": rows}
            _write_json(folder / f"{method}.json", value)
            _write_csv(folder / f"{method}.csv", rows)
            entries.append({"run_id": f"seed_{seed}_{method}", "seed": seed, "method": method,
                            "status": "completed", "path": f"seed_{seed}\\{method}.json"})
            all_rows.extend(rows)
    _write_json(root / "matrix_manifest.json", {"schema_version": 1, "experiment": "health_index_benchmark",
                "status": "completed", "data_root": "fixture", "seeds": contract["seeds"],
                "methods": contract["methods"], "expected_runs": 6, "expected_conditions_per_run": 9,
                "completed_runs": 6, "failed_runs": 0, "runs": entries})
    # 獨立建立 fixture 舊彙總，平均與 ddof=1 用明寫公式，不呼叫受測彙總函數。
    summary = {"schema_version": 1, "experiment": "health_index_benchmark", "seeds": contract["seeds"],
               "methods": sorted(contract["methods"]), "runs": 6, "rows": 54,
               "dataset_fingerprint": "a"*64, "overall": {}, "difference_knn_minus_mahalanobis": {}}
    table = []
    for method in summary["methods"]:
        subset = [row for row in all_rows if row["method"] == method]
        summary["overall"][method] = {}
        for metric in aggregate.METRICS:
            mean, std = _stats([row[metric] for row in subset])
            summary["overall"][method][metric] = {"mean": round(mean, 6), "std": round(std, 6)}
        for dataset in sorted({row["dataset"] for row in subset}):
            record = {"index": len(table), "method": method, "dataset": dataset}
            for metric in aggregate.METRICS:
                mean, std = _stats([row[metric] for row in subset if row["dataset"] == dataset])
                record[f"{metric}_mean"] = mean
                record[f"{metric}_std"] = std
            table.append(record)
    for metric in aggregate.METRICS:
        left = _stats([row[metric] for row in all_rows if row["method"] == "knn"])[0]
        right = _stats([row[metric] for row in all_rows if row["method"] == "mahalanobis"])[0]
        summary["difference_knn_minus_mahalanobis"][metric] = round(left-right, 6)
    _write_json(root / "aggregate_summary.json", summary)
    _write_csv(root / "aggregate_by_condition.csv", table)
    (root / "README.md").write_text("工程測試封存，不含 data 依賴。", encoding="utf-8")
    contract_path = base / "contract.json"
    _refresh(root, contract_path, contract)
    return root, contract_path


def _refresh(root, path, contract=None):
    if contract is None:
        contract = json.loads(path.read_text(encoding="utf-8"))
    contract["source_sha256"] = {file.relative_to(root).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
                                 for file in root.rglob("*") if file.is_file()}
    contract["source_git_lf_sha256"] = dict(contract["source_sha256"])
    _write_json(path, contract)


class HealthAggregateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root, self.contract = _fixture(self.base)
        self.addCleanup(self.temp.cleanup)

    def load(self):
        return aggregate.load_matrix(self.root, self.contract)

    def mutate(self, relative, change, sync_csv=False):
        path = self.root / relative
        value = json.loads(path.read_text(encoding="utf-8"))
        change(value)
        _write_json(path, value)
        if sync_csv:
            _write_csv(path.with_suffix(".csv"), value["rows"])
        _refresh(self.root, self.contract)

    def test_full_matrix_equal_weights_ddof_and_comparison(self):
        contract, rows, evidence = self.load()
        summary, table = aggregate.aggregate_rows(rows, contract)
        self.assertEqual(len(rows), 54)
        self.assertEqual(len(table), 18)
        self.assertEqual(summary["overall"]["mahalanobis"]["known_health_mean"]["mean"], 0.44)
        self.assertEqual(summary["difference_knn_minus_mahalanobis"]["known_health_mean"], 0.05)
        self.assertAlmostEqual(table[0]["known_health_mean_std"], 0.2)
        self.assertTrue(all(item["matches"] for item in aggregate.compare(evidence["snapshot"], summary, table, contract)))
        self.assertEqual(len(evidence["inventory"]["files"]), 16)

    def test_missing_run_file_fails(self):
        (self.root / "seed_42/knn.json").unlink()
        with self.assertRaisesRegex(ValueError, "來源檔案集合"):
            self.load()

    def test_missing_manifest_run_fails(self):
        self.mutate("matrix_manifest.json", lambda value: value["runs"].pop())
        with self.assertRaisesRegex(ValueError, "run 數"):
            self.load()

    def test_duplicate_run_fails(self):
        self.mutate("matrix_manifest.json", lambda value: value["runs"].__setitem__(1, value["runs"][0]))
        with self.assertRaisesRegex(ValueError, "run 重複"):
            self.load()

    def test_wrong_seed_fails(self):
        self.mutate("seed_42/knn.json", lambda value: value.update(seed=123))
        with self.assertRaisesRegex(ValueError, "seed"):
            self.load()

    def test_wrong_method_fails(self):
        self.mutate("seed_42/knn.json", lambda value: value.update(method="mahalanobis"))
        with self.assertRaisesRegex(ValueError, "method"):
            self.load()

    def test_manifest_fingerprint_fails(self):
        self.mutate("matrix_manifest.json", lambda value: value.update(dataset_fingerprint="c"*64))
        with self.assertRaisesRegex(ValueError, "manifest 指紋"):
            self.load()

    def test_run_fingerprint_fails(self):
        self.mutate("seed_42/knn.json", lambda value: value.update(dataset_fingerprint="c"*64))
        with self.assertRaisesRegex(ValueError, "dataset_fingerprint"):
            self.load()

    def test_source_bytes_tamper_fails_without_repin(self):
        path = self.root / "seed_42/knn.json"
        path.write_text(path.read_text(encoding="utf-8")+" ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "來源 SHA"):
            self.load()

    def test_required_metric_missing_fails(self):
        self.mutate("seed_42/knn.json", lambda value: value["rows"][0].pop("auroc"))
        with self.assertRaisesRegex(ValueError, "列欄位"):
            self.load()

    def test_nan_fails(self):
        self.mutate("seed_42/knn.json", lambda value: value["rows"][0].update(auroc=float("nan")))
        with self.assertRaisesRegex(ValueError, "JSON 非有限"):
            self.load()

    def test_null_ranking_metric_fails(self):
        self.mutate("seed_42/knn.json", lambda value: value["rows"][0].update(auroc=None))
        with self.assertRaisesRegex(ValueError, "非有限"):
            self.load()

    def test_zero_denominator_fails(self):
        self.mutate("seed_42/knn.json", lambda value: value["rows"][0].update(n_unknown_test=0))
        with self.assertRaisesRegex(ValueError, "整數錯誤"):
            self.load()

    def test_duplicate_condition_fails(self):
        self.mutate("seed_42/knn.json", lambda value: value["rows"].__setitem__(1, value["rows"][0]))
        with self.assertRaisesRegex(ValueError, "重複或錯誤工況"):
            self.load()

    def test_incomplete_condition_matrix_fails(self):
        self.mutate("seed_42/knn.json", lambda value: value["rows"].pop())
        with self.assertRaisesRegex(ValueError, "工況數不足"):
            self.load()

    def test_configuration_count_fails(self):
        self.mutate("seed_42/knn.json", lambda value: value["rows"][0].update(n_unknown_classes=8))
        with self.assertRaisesRegex(ValueError, "未知配置數"):
            self.load()

    def test_dataset_motor_rpm_mismatch_fails(self):
        self.mutate("seed_42/knn.json", lambda value: value["rows"][0].update(dataset="T2/6000"))
        with self.assertRaisesRegex(ValueError, "dataset 與"):
            self.load()

    def test_unknown_calibration_flag_fails(self):
        self.mutate("seed_42/knn.json", lambda value: value.update(unknown_calibration_leakage=True))
        with self.assertRaisesRegex(ValueError, "unknown_calibration_leakage"):
            self.load()

    def test_wrong_score_direction_fails(self):
        self.mutate("seed_42/knn.json", lambda value: value.update(score_direction="lower_is_worse"))
        with self.assertRaisesRegex(ValueError, "score_direction"):
            self.load()

    def test_attached_csv_mismatch_fails(self):
        path = self.root / "seed_42/knn.csv"
        text = path.read_text(encoding="utf-8").replace(",knn,", ",wrong,")
        path.write_text(text, encoding="utf-8")
        _refresh(self.root, self.contract)
        with self.assertRaisesRegex(ValueError, "附帶 CSV"):
            self.load()

    def test_stable_sort_and_repeat(self):
        contract, rows, _ = self.load()
        first = aggregate.aggregate_rows(rows, contract)
        self.assertEqual(first, aggregate.aggregate_rows(list(reversed(rows)), contract))
        self.mutate("matrix_manifest.json", lambda value: value["runs"].reverse())
        _, reordered, _ = self.load()
        self.assertEqual(rows, reordered)

    def test_hist_summary_difference_is_recorded_not_hidden(self):
        self.mutate("aggregate_summary.json", lambda value: value["overall"]["knn"]["auroc"].update(mean=0.7))
        contract, rows, evidence = self.load()
        summary, table = aggregate.aggregate_rows(rows, contract)
        mismatch = [item for item in aggregate.compare(evidence["snapshot"], summary, table, contract) if not item["matches"]]
        self.assertEqual(len(mismatch), 1)
        self.assertAlmostEqual(mismatch[0]["difference"], 0.3)

    def test_old_extra_field_is_recorded(self):
        self.mutate("aggregate_summary.json", lambda value: value.update(extra="不可略過"))
        contract, rows, evidence = self.load()
        summary, table = aggregate.aggregate_rows(rows, contract)
        self.assertFalse(next(item for item in aggregate.compare(evidence["snapshot"], summary, table, contract)
                              if item["key"] == "summary/extra")["matches"])

    def test_extra_source_file_fails(self):
        (self.root / "extra.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "來源檔案集合"):
            self.load()

    def test_contract_weighting_change_fails(self):
        value = json.loads(self.contract.read_text(encoding="utf-8"))
        value["weighting"] = "sample_weighted"
        _write_json(self.contract, value)
        with self.assertRaisesRegex(ValueError, "契約規則"):
            self.load()

    def test_run_writes_new_outputs_without_changing_archive(self):
        output = self.base / "output"
        output.mkdir()
        paths = SimpleNamespace(output_dir=output, log_file=self.base / "run.log")
        before = {p.relative_to(self.root).as_posix(): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        with patch.object(aggregate, "setup_run", return_value=(Mock(), paths)):
            result = aggregate.run(self.root, self.contract)
        self.assertEqual(result["comparison_status"], "MATCH")
        self.assertFalse(result["model_fit_performed"])
        self.assertEqual(result["historical_formula_status"], "UNKNOWN")
        self.assertTrue((output / "field_differences.csv").is_file())
        self.assertEqual(before, {p.relative_to(self.root).as_posix(): p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_run_failure_saves_failure_audit(self):
        output = self.base / "output"
        output.mkdir()
        paths = SimpleNamespace(output_dir=output, log_file=self.base / "run.log")
        (self.root / "seed_42/knn.json").unlink()
        with patch.object(aggregate, "setup_run", return_value=(Mock(), paths)), self.assertRaises(ValueError):
            aggregate.run(self.root, self.contract)
        self.assertEqual(json.loads((output / "failure.json").read_text(encoding="utf-8"))["status"], "FAILED")
        self.assertFalse((output / "aggregate_summary.json").exists())

    def test_invalid_knn_parameter_type_fails(self):
        self.mutate("seed_42/knn.json", lambda value: value["rows"][0].update(knn_neighbors=5.0))
        with self.assertRaisesRegex(ValueError, "鄰居數"):
            self.load()

    def test_legacy_covariance_fails(self):
        self.mutate("seed_42/mahalanobis.json", lambda value: value["rows"][0].update(mahalanobis_method="legacy"))
        with self.assertRaisesRegex(ValueError, "共變異數"):
            self.load()

    def test_count_pairing_mismatch_fails(self):
        self.mutate("seed_42/knn.json", lambda value: value["rows"][0].update(n_train=99))
        with self.assertRaisesRegex(ValueError, "樣本數"):
            self.load()

    def test_run_required_field_missing_fails(self):
        self.mutate("seed_42/knn.json", lambda value: value.pop("python"))
        with self.assertRaisesRegex(ValueError, "必備欄位"):
            self.load()

    def test_manifest_path_escape_fails(self):
        self.mutate("matrix_manifest.json", lambda value: value["runs"][0].update(path="../outside.json"))
        with self.assertRaisesRegex(ValueError, "run 路徑"):
            self.load()

    def test_csv_integer_ids_are_exact(self):
        path = self.root / "seed_42/knn.csv"
        path.write_text(path.read_text(encoding="utf-8").replace(",42,", ",42.0,"), encoding="utf-8")
        _refresh(self.root, self.contract)
        with self.assertRaisesRegex(ValueError, "整數或 ID"):
            self.load()

    def test_effect_size_zero_denominator_is_nullable(self):
        self.mutate("seed_42/mahalanobis.json", lambda value: value["rows"][0].update(
            known_health_std=0.0, unknown_health_std=0.0, health_effect_size=None), sync_csv=True)
        self.assertEqual(len(self.load()[1]), 54)

    def test_effect_size_missing_with_nonzero_denominator_fails(self):
        self.mutate("seed_42/mahalanobis.json", lambda value: value["rows"][0].update(health_effect_size=None))
        with self.assertRaisesRegex(ValueError, "效果量"):
            self.load()

    def test_empty_extra_object_is_not_hidden(self):
        self.mutate("aggregate_summary.json", lambda value: value.update(extra={}))
        contract, rows, evidence = self.load()
        summary, table = aggregate.aggregate_rows(rows, contract)
        self.assertFalse(next(item for item in aggregate.compare(evidence["snapshot"], summary, table, contract)
                              if item["key"] == "summary/extra")["matches"])

    def test_cli_reports_difference_exit_three(self):
        with patch.object(aggregate, "run", return_value={"comparison_status": "DIFFERENT"}), patch("sys.stdout", new=__import__("io").StringIO()):
            self.assertEqual(aggregate.main(["--source-root", str(self.root), "--contract", str(self.contract)]), 3)

    def test_only_explicit_known_line_endings_are_accepted(self):
        # 工程 fixture 的兩種完整位元組 SHA 都先登錄，任意空白仍不能通過。
        contract = json.loads(self.contract.read_text(encoding="utf-8"))
        path = self.root / "seed_42/knn.csv"
        lf = path.read_bytes().replace(b"\r\n", b"\n")
        contract["source_git_lf_sha256"]["seed_42/knn.csv"] = hashlib.sha256(lf).hexdigest()
        _write_json(self.contract, contract)
        path.write_bytes(lf)
        self.assertEqual(len(self.load()[1]), 54)
        path.write_bytes(lf+b" ")
        with self.assertRaisesRegex(ValueError, "來源 SHA"):
            self.load()


if __name__ == "__main__":
    unittest.main()
