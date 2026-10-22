# exp6 formal matrix aggregate

- status: completed
- expected runs: 54
- completed runs: 54
- methods: mahalanobis, knn
- seeds: 42, 123, 2026

## Macro summary

| method | AUROC mean ± std | open-set accuracy mean ± std | unknown F1 mean ± std |
|---|---:|---:|---:|
| mahalanobis | 1.000000 ± 0.000000 | 0.998860 ± 0.000873 | 0.999418 ± 0.000446 |
| knn | 1.000000 ± 0.000000 | 0.998867 ± 0.000932 | 0.999421 ± 0.000476 |

## Paired differences

Positive values mean kNN is higher than Mahalanobis for that metric.

| metric | mean difference | std difference | pairs |
|---|---:|---:|---:|
| known_accuracy | 0.000012 | 0.042671 | 27 |
| open_set_accuracy | 0.000006 | 0.000947 | 27 |
| auroc | 0.000000 | 0.000000 | 27 |
| aupr_unknown_positive | 0.000000 | 0.000000 | 27 |
| fpr_at_tpr95 | 0.000000 | 0.000000 | 27 |
| unknown_precision | 0.000007 | 0.000966 | 27 |
| unknown_recall | 0.000000 | 0.000000 | 27 |
| unknown_f1 | 0.000004 | 0.000484 | 27 |
