"""計數語意、標籤交換與分母邊界；不依賴正式資料或正在執行的模型。"""

from copy import deepcopy

import pytest

from core.confusion_summary import summarize_confusion
from experiments.exp12_confusion_tsne import summarize_matrices
from tests.exp12_counts_fixture import historical_result


def calculate(labels, matrix, task="openset", **kwargs):
    return summarize_confusion(labels, matrix, task=task, value_kind="counts", **kwargs)


def test_historical_binary_counts_and_denominators():
    fact = summarize_matrices(historical_result())["binary"]
    b = fact["binary"]
    assert (b["healthy_total"], b["fault_total"], fact["total"]) == (572, 26070, 26642)
    assert (b["healthy_false_alarms"], b["fault_missed"]) == (45, 0)
    assert b["healthy_false_positive_rate"] == {
        "numerator": 45, "denominator": 572, "ratio": 45 / 572, "percent": 45 / 572 * 100}
    assert b["fault_recall"]["ratio"] == 1


def test_historical_lw_counts_per_class_and_invariant():
    fact = summarize_matrices(historical_result())["openset"]["ledoit_wolf"]
    o = fact["openset"]
    assert [v["support"] for v in fact["per_class"].values()] == [572, 601, 605, 591, 569, 14381]
    assert [v["correct"] for v in fact["per_class"].values()] == [525, 562, 559, 561, 529, 14381]
    assert (o["known_total"], o["known_correct"], o["known_rejected"], o["known_misclassified"]) == (2938, 2736, 202, 0)
    assert o["known_total"] == o["known_correct"] + o["known_rejected"] + o["known_misclassified"]
    assert o["known_accuracy"]["ratio"] == 2736 / 2938
    assert o["known_rejection_rate"]["ratio"] == 202 / 2938
    assert (o["unknown_total"], o["unknown_correct"], o["unknown_accepted"]) == (14381, 14381, 0)
    assert fact["total"] == 17319


def test_legacy_has_correct_unknown_without_guessing_interface_reason():
    fact = summarize_matrices(historical_result())["openset"]["legacy"]
    o = fact["openset"]
    assert (o["known_correct"], o["known_rejected"], o["unknown_correct"]) == (0, 2938, 14381)
    assert "legacy" not in str(fact)
    assert "interface" not in str(fact)


def test_nonzero_known_confusion_unknown_acceptance_and_custom_label():
    o = calculate(["a", "unseen", "b"], [[4, 2, 3], [1, 5, 2], [2, 1, 6]],
                  unknown_label="unseen")["openset"]
    assert (o["known_total"], o["known_correct"], o["known_rejected"], o["known_misclassified"]) == (18, 10, 3, 5)
    assert (o["unknown_total"], o["unknown_correct"], o["unknown_accepted"]) == (8, 5, 3)


@pytest.mark.parametrize("task,labels,matrix", [
    ("binary", ["healthy", "fault"], [[7, 2], [3, 9]]),
    ("openset", ["a", "unknown", "b"], [[4, 2, 3], [1, 5, 2], [2, 1, 6]]),
])
def test_permuting_labels_rows_and_columns_preserves_semantics(task, labels, matrix):
    original = calculate(labels, matrix, task)
    order = list(reversed(range(len(labels))))
    changed = calculate([labels[i] for i in order], [[matrix[i][j] for j in order] for i in order], task)
    if task == "binary":
        assert changed[task] == original[task]
    else:
        assert {**changed[task], "known_labels": sorted(changed[task]["known_labels"])} == {
            **original[task], "known_labels": sorted(original[task]["known_labels"])}
    assert changed["total"] == original["total"]
    assert changed["per_class"] == original["per_class"]


def test_changing_a_cell_changes_facts_without_mutating_input():
    source = historical_result()
    saved = deepcopy(source)
    original = summarize_matrices(source)
    assert source == saved
    source["binary_confusion"]["matrix"][0][1] += 7
    changed = summarize_matrices(source)
    assert changed["binary"]["total"] == original["binary"]["total"] + 7
    assert changed["binary"]["binary"]["healthy_false_positive_rate"]["ratio"] == 52 / 579


def test_zero_support_and_unknown_only_are_not_perfect_known_accuracy():
    b = calculate(["healthy", "fault"], [[0, 0], [0, 0]], "binary")["binary"]
    assert b["fault_recall"]["ratio"] is None
    assert b["healthy_false_positive_rate"]["percent"] is None
    o = calculate(["unknown"], [[4]])["openset"]
    assert o["known_total"] == 0 and o["known_accuracy"]["ratio"] is None
    assert o["unknown_correct"] == 4


def test_missing_unknown_metrics_are_explicitly_not_applicable():
    o = calculate(["a", "b"], [[2, 1], [3, 4]])["openset"]
    assert o["known_misclassified"] == 4
    assert not o["unknown_present"]
    assert o["known_rejected"] is None
    assert o["unknown_total"] is None
    assert o["unknown_recall"]["ratio"] is None


@pytest.mark.parametrize("labels,matrix", [
    ([], []), (["a", "a"], [[1, 0], [0, 1]]), ([""], [[1]]),
    (["a"], [[1, 2]]), (["a", "b"], [[1, 2]]), (["a"], [1]),
    (["a"], [[-1]]), (["a"], [[True]]), (["a"], [[float("nan")]]),
    (["a"], [[float("inf")]]), (["a"], [[1.5]]), (["a"], [[1.0]]),
    (["a", "b"], [[0.5, 0.5], [0.1, 0.9]]), (["a"], [["1"]]),
])
def test_invalid_matrix_is_rejected(labels, matrix):
    with pytest.raises(ValueError):
        calculate(labels, matrix)


def test_normalized_kind_and_unspecified_semantics_fail_even_with_integer_values():
    with pytest.raises(ValueError, match="counts"):
        summarize_confusion(["a"], [[1]], task="openset", value_kind="normalized")
    with pytest.raises(ValueError, match="任務"):
        calculate(["a"], [[1]], task="unspecified")
    with pytest.raises(ValueError, match="binary"):
        calculate(["a"], [[1]], task="binary")


@pytest.mark.parametrize("result", [None, {}, {"binary_confusion": None},
                                    {"openset_confusion": []}, {"openset_confusion": {}},
                                    {"openset_confusion": {"legacy": None}}])
def test_exp12_wrapper_does_not_guess_unknown_payloads(result):
    with pytest.raises(ValueError):
        summarize_matrices(result)


def test_saved_derived_summary_cannot_override_raw_counts():
    result = historical_result()
    expected = summarize_matrices(result)
    result["matrix_facts"] = {"binary": {"total": 1}}
    assert summarize_matrices(result) == expected
