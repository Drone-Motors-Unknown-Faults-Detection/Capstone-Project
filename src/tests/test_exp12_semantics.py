"""資料角色、逐工況抽樣及模型標記語意，不重跑昂貴嵌入。"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from experiments.exp12_confusion_tsne import _tsne_one_dataset, data_role_counts, visualization_counts
from tests.test_assistant_results import OfflineAssistant, collect, result_snapshot


@pytest.mark.parametrize("n,expected", [(313, (187, 62, 64)), (500, (300, 100, 100)), (11, (6, 2, 3)), (0, (0, 0, 0))])
def test_roles_follow_actual_splitter(n, expected):
    counts = data_role_counts(n)
    assert tuple(counts[k] for k in ("train", "cal", "holdout")) == expected
    assert sum(counts[k] for k in ("train", "cal", "holdout")) == n


def test_per_dataset_counts_are_actual_not_limit_or_matrix_totals():
    def entry(n):
        return {"embedding": [[0, 0]] * n, "config_labels": ["8screws"] * n, "verdicts": ["unknown"] * n}

    counts = visualization_counts({"_tsne_by_dataset": {"T1/6000": entry(2), "T2/8000": entry(5)}})
    assert counts["datasets"] == {"T1/6000": 2, "T2/8000": 5}
    assert counts["total"] == 7
    assert visualization_counts({})["total"] is None
    broken = entry(2)
    broken["verdicts"].pop()
    assert visualization_counts({"_tsne_by_dataset": {"a": broken}})["status"] == "INVALID"


def test_shapes_follow_predictions_including_known_rejection_and_unknown_acceptance():
    pools = {"8screws": np.arange(3).reshape(-1, 1), "5screws": np.arange(3, 6).reshape(-1, 1)}
    predictions = {0: "8screws", 1: None, 2: "1screw", 3: None, 4: "8screws", 5: None}
    monitor = SimpleNamespace(known={"8screws"}, scaler=SimpleNamespace(transform=lambda x: x),
                              classify=lambda x: [predictions[int(row[0])] for row in x])
    fake_tsne = SimpleNamespace(fit_transform=lambda x: x.repeat(2, axis=1))
    with patch("experiments.exp12_confusion_tsne.build_ancestor_monitor", return_value=monitor), patch(
            "sklearn.manifold.TSNE", return_value=fake_tsne):
        result = _tsne_one_dataset(pools, 42, 0.95, 5, 200)
    rows = {int(x[0]): (label, verdict) for x, label, verdict in zip(result["embedding"], result["config_labels"], result["verdicts"])}
    assert rows[1] == ("8screws", "unknown")
    assert rows[3] == ("5screws", "unknown")
    assert rows[4] == ("5screws", "known_wrong")
    assert rows[0] == ("8screws", "known_correct")


@pytest.mark.parametrize("question", ["legacy全部拒絕是否太保守", "93.12%只有故障嗎", "t-SNE總共多少筆，三角形只表示未知配置嗎", "這些結果不是模型判斷嗎", "健康池313筆是不是全部訓練"])
def test_visible_primary_answer_contains_semantic_contract_without_llm(question):
    frames = collect(OfflineAssistant("unused"), question, snap=result_snapshot())
    text = "".join(f.get("text", "") for f in frames)
    assert "不是純故障正確率" in text
    assert "並非全部工況合計2000" in text
    assert "實際數量未知" in text
    assert "三角形表示模型判為unknown" in text
    assert "train擬合187／cal校準62／holdout測試64" in text
    assert "模型產生預測，再以事後真值計算" in text
    assert "legacy過於保守" not in text
