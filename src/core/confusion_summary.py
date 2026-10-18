"""按明確任務核算原始計數；不擬合模型，也不猜測標籤的物理意義。"""

from __future__ import annotations

from numbers import Integral


def _ratio(numerator: int | None, denominator: int | None) -> dict:
    value = numerator / denominator if numerator is not None and denominator else None
    return {"numerator": numerator, "denominator": denominator,
            "ratio": value, "percent": value * 100 if value is not None else None}


def summarize_confusion(labels, matrix, *, task: str, value_kind: str,
                        healthy_label: str = "healthy", fault_label: str = "fault",
                        unknown_label: str = "unknown") -> dict:
    """回傳支持數與完整精度比率；counts 契約拒絕浮點值及正規化比例。"""
    if value_kind != "counts":
        raise ValueError("只接受原始 counts；正規化比例不能當樣本計數")
    if task not in ("binary", "openset"):
        raise ValueError("必須明示 binary 或 openset 任務")
    if not isinstance(labels, (list, tuple)) or not labels or any(
            not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("labels 必須是非空字串清單")
    if len(set(labels)) != len(labels):
        raise ValueError("labels 不可重複")
    n = len(labels)
    if not isinstance(matrix, (list, tuple)) or len(matrix) != n:
        raise ValueError("matrix 必須與 labels 等長且為方陣")
    counts = []
    for row in matrix:
        if not isinstance(row, (list, tuple)) or len(row) != n:
            raise ValueError("matrix 必須與 labels 等長且為方陣")
        if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 0 for v in row):
            raise ValueError("matrix 只接受非負整數計數；不接受布林、浮點或比例")
        counts.append([int(v) for v in row])
    support = [sum(row) for row in counts]
    result = {"schema_version": "confusion_counts_v1", "task": task, "value_kind": value_kind,
              "labels": list(labels), "total": sum(support), "per_class": {
                  label: {"support": support[i], "correct": counts[i][i],
                          "recall": _ratio(counts[i][i], support[i])}
                  for i, label in enumerate(labels)}}
    if task == "binary":
        if healthy_label == fault_label or set(labels) != {healthy_label, fault_label}:
            raise ValueError("binary 必須恰有指定的 healthy 與 fault 標籤")
        h, f = labels.index(healthy_label), labels.index(fault_label)
        result["binary"] = {
            "healthy_label": healthy_label, "fault_label": fault_label,
            "healthy_total": support[h], "healthy_false_alarms": counts[h][f],
            "healthy_false_positive_rate": _ratio(counts[h][f], support[h]),
            "fault_total": support[f], "fault_missed": counts[f][h],
            "fault_recall": _ratio(counts[f][f], support[f]),
        }
        return result
    if not isinstance(unknown_label, str) or not unknown_label.strip():
        raise ValueError("openset 必須指定未知標籤語意")
    u = labels.index(unknown_label) if unknown_label in labels else None
    known = [i for i in range(n) if i != u]
    known_total = sum(support[i] for i in known)
    correct = sum(counts[i][i] for i in known)
    rejected = sum(counts[i][u] for i in known) if u is not None else None
    wrong = sum(counts[i][j] for i in known for j in known if i != j)
    unknown_total = support[u] if u is not None else None
    unknown_correct = counts[u][u] if u is not None else None
    for i in known:
        rejected_i = counts[i][u] if u is not None else None
        result["per_class"][labels[i]].update(
            rejected=rejected_i,
            misclassified=sum(counts[i][j] for j in known if j != i),
            rejection_rate=_ratio(rejected_i, support[i]),
        )
    result["openset"] = {
        "unknown_label": unknown_label, "unknown_present": u is not None,
        "known_labels": [labels[i] for i in known], "known_total": known_total,
        "known_correct": correct, "known_rejected": rejected, "known_misclassified": wrong,
        "known_accuracy": _ratio(correct, known_total),
        "known_rejection_rate": _ratio(rejected, known_total),
        "unknown_total": unknown_total, "unknown_correct": unknown_correct,
        "unknown_accepted": sum(counts[u][j] for j in known) if u is not None else None,
        "unknown_recall": _ratio(unknown_correct, unknown_total),
    }
    if known_total != correct + (rejected or 0) + wrong:
        raise ValueError("已知計數分解不一致")
    return result
