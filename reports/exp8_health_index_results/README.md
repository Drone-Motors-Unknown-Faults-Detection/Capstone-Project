# Experiment 8 (exp8): Formal health-index benchmark

> 由 `experiments/health_index_matrix.py` 寫入（逐次呼叫 `experiments/health_index_benchmark.py` 的 `run()`）。技術報告 `docs/experiments/exp8_health.md`。2026-10-03 由 `reports/health_index_results/` 改名。

## Protocol

- Data root: the existing `formal_local` materialisation; no raw data was
  changed.
- Conditions: 9 (`T1/T2/T3 × 6000/8000/11000 rpm`).
- Seeds: 42, 123, 2026.
- Methods: Mahalanobis with Ledoit–Wolf covariance and class-wise k-NN.
- Each condition uses the existing 60/20/20 healthy split. `8screws` train and
  calibration fit the scaler, detector, threshold and relative health mapping.
  All other configurations are held out and are used only after scoring.
- The Open Set threshold remains `score > 1`; larger score means more unknown,
  while larger health index means healthier.

The six JSON files contain full per-condition provenance and the CSV files are
compact tables for analysis. `matrix_manifest.json` confirms all six runs and
54 condition rows completed. The dataset fingerprint is
`81c9192476ecd1e23a17b3ed5a343dcd50cdc2e5b7e0cc5466e9162941898228`.

## Aggregate result (mean over 9 conditions × 3 seeds)

| method | relative health gap (known − unknown) | Open Set accuracy | AUROC | unknown recall | unknown F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mahalanobis + Ledoit–Wolf | 0.583144 ± 0.073301 | 0.998849 ± 0.000887 | 1.000000 | 1.000000 | 0.999412 ± 0.000453 |
| k-NN | 0.601657 ± 0.071637 | 0.998842 ± 0.000937 | 1.000000 | 1.000000 | 0.999409 ± 0.000478 |

The k-NN relative health gap is higher by `0.018513` (about 1.85 percentage
points on the relative 0–1 scale). Mahalanobis has a negligibly higher mean
Open Set accuracy by `0.000007`; AUROC and unknown recall are tied. These are
relative separation results, not proof that either method measures physical
damage severity.

## Interpretation limits

All held-out unknown samples map to health index 0.0 under the current
calibration anchors. This means the present formal data separates unknown from
healthy very strongly but does not provide enough labelled progression to rank
unknown samples by physical severity. There are no motor IDs, session IDs,
timestamps, failure endpoints or cause labels, so event delay, true trend,
ordinal severity and RUL remain unavailable. See
`docs/experiments/exp8_health.md` and the removed audits
[health_monitoring_data_capability.md](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/80bdf54fdf43d466ea19549fb7c0b4e391394799/reports/health_monitoring_data_capability.md) and
[fault_type_data_requirements.md](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/80bdf54fdf43d466ea19549fb7c0b4e391394799/reports/fault_type_data_requirements.md) (commit `80bdf54`) before making those claims.

