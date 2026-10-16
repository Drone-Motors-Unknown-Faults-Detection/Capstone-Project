"""實驗二：量尺擴張（Scale Growth）— 未知偵測 → 隔離累積 → HDBSCAN 分群 → 操作員確認 → 擴張重訓。

ScaleGrowthSession 是核心狀態機（Web 即時展示直接驅動它）：
    1. 每筆樣本以 OpenSetMonitor 打分；判為未知者進隔離區
    2. 隔離區累積達 min_cluster_size 後以 HDBSCAN 找穩定叢集 → 產生「候選新故障」
    3. confirm() 模擬操作員確認：以叢集多數真實配置為標註（此時才揭示 ground truth），
       納入已知類別並重新擬合 → 量尺擴張
    偵測與分群過程從未使用標籤；標籤只在確認步驟扮演「操作員」。

兩段閾值設計：
    - 偵測線 1.0：正規化分數 > 1 即判「未知」（逐樣本判定與警報用）
    - 隔離線 quarantine_margin（預設 2.0）：分數要明顯超線才進隔離區參與新類發現。
      校準閾值天生讓 ~5–11% 已知樣本些微超線（實測健康誤報 1.01–1.08），
      而真實故障遠在其上（實測九種配置最低分 >= 10）——隔離線落在空隙中，
      擋掉邊界誤報、不漏任何故障，避免誤報累積自聚成假候選。

批次執行（產出論文數據：逐配置的發現延遲、叢集純度、擴張後準確率）：
    venv/bin/python -m experiments.exp2_scale_growth --motor T1 --rpm 8000rpm
"""

from __future__ import annotations

import argparse

import hdbscan
import numpy as np

from core.data import HEALTHY, CycleSampler, config_sort_key, display_name
from core.geometry import PolarMap
from core.logger import setup_run
from core.monitor import OpenSetMonitor
from core.runner import add_dataset_args, add_openset_args, resolve_dataset, save_json


class ScaleGrowthSession:
    """從健康基準起步、隨確認逐步擴張的開放集監測 session。"""

    def __init__(
        self,
        pools: dict[str, np.ndarray],
        seed: int = 42,
        confidence: float = 0.95,
        method: str = "ledoit_wolf",
        openset_method: str = "mahalanobis",
        knn_neighbors: int = 5,
        min_cluster_size: int = 25,
        min_samples: int = 3,
        recluster_every: int = 10,
        quarantine_margin: float = 2.0,
    ) -> None:
        self.pools = pools
        self.monitor = OpenSetMonitor(
            pools,
            seed=seed,
            confidence=confidence,
            method=method,
            openset_method=openset_method,
            knn_neighbors=knn_neighbors,
        )
        self.monitor.fit_initial()
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.recluster_every = recluster_every
        self.quarantine_margin = quarantine_margin
        self.quarantine_X: list[np.ndarray] = []
        self.quarantine_truth: list[str] = []
        self.quarantine_t: list[int] = []
        self.candidate: dict | None = None
        self._since_cluster = 0
        self.n_seen = 0
        self.cluster_attempts = 0  # 連續分群失敗次數（成功即歸零；供 UI 呈現進度）
        self.polar = PolarMap(self.monitor)  # 極座標幾何（實驗四），每次擴張後重建

    # -- 串流 ------------------------------------------------------------------

    def process(self, x_raw: np.ndarray, truth: str) -> dict:
        """處理一筆樣本。truth 僅存入隔離區供之後的「操作員確認」揭示。"""
        self.n_seen += 1
        score = float(self.monitor.score(x_raw)[0])
        label = self.monitor.classify(x_raw)[0]
        # 極座標判讀（方向熟悉度為連續參考指標；不影響既有的已知/未知判定）
        direction = self.polar.analyze(x_raw)[0]
        candidate_new = False
        # 隔離線：分數明顯超線才參與新類發現，邊界誤報（如健康的 1.0x）僅警報不進隔離區
        if label is None and score > self.quarantine_margin:
            self.quarantine_X.append(np.asarray(x_raw, dtype=float).ravel())
            self.quarantine_truth.append(truth)
            self.quarantine_t.append(self.n_seen)
            self._since_cluster += 1
            if (
                self.candidate is None
                and len(self.quarantine_X) >= self.min_cluster_size
                and self._since_cluster >= self.recluster_every
            ):
                candidate_new = self._try_cluster()
        return {
            "score": score,
            "label": label,
            "quarantine": len(self.quarantine_X),
            "candidate_new": candidate_new,
            "cluster_attempts": self.cluster_attempts,
            "direction": direction,
        }

    def _cluster_params(self, n: int) -> list[tuple[int, int]]:
        """密度參數階梯：先嚴格，隔離區越大越放寬。

        實測（T1/8000rpm，嚴重鬆動類如 2screws 在特徵空間較發散）：
        mcs=25 要累積 225 筆才成叢，mcs=15 需 150 筆，mcs=10/min_samples=2
        只需 100 筆就聚出 73 筆的大叢。候選門檻（叢 >= min_cluster_size）不變，
        放寬的只是 HDBSCAN 的密度要求，純度不受影響（mcs 越小切得越細）。
        """
        ladder = [(self.min_cluster_size, self.min_samples)]
        if n >= 3 * self.min_cluster_size:
            ladder.append((max(15, self.min_cluster_size // 2), self.min_samples))
        if n >= 4 * self.min_cluster_size:
            ladder.append((max(10, self.min_cluster_size // 3), 2))
        return ladder

    def _try_cluster(self) -> bool:
        X = self.monitor.scaler.transform(np.vstack(self.quarantine_X))
        self._since_cluster = 0
        best, best_size, used = None, 0, None
        for mcs, ms in self._cluster_params(len(X)):
            labels = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms).fit_predict(X)
            for cl in set(labels) - {-1}:
                size = int((labels == cl).sum())
                if size > best_size:
                    best, best_size, used = cl, size, (mcs, ms)
                    best_labels = labels
            if best is not None and best_size >= self.min_cluster_size:
                break
        if best is None or best_size < self.min_cluster_size:
            self.cluster_attempts += 1
            return False
        self.cluster_attempts = 0
        labels = best_labels
        idx = np.flatnonzero(labels == best)
        truths = [self.quarantine_truth[i] for i in idx]
        ts = [self.quarantine_t[i] for i in idx]
        majority = max(set(truths), key=truths.count)
        self.candidate = {
            "indices": idx.tolist(),
            "size": best_size,
            "majority": majority,
            "majority_display": display_name(majority),
            "purity": round(truths.count(majority) / len(truths), 4),
            # 叢集樣本的蒐集時間區間——區分「舊帳」（如健康期累積的誤報）與現行串流
            "t_range": [int(min(ts)), int(max(ts))],
            "is_known": majority in self.monitor.known,
            "cluster_params": list(used),
        }
        return True

    # -- 擴張 ------------------------------------------------------------------

    def confirm(self) -> dict:
        """操作員確認候選叢集（此步驟才揭示真實標籤）。兩種結果：

        - ``learned``：叢集多數是尚未學過的故障 → 納入已知並重新擬合（量尺擴張）。
        - ``rejected_known``：叢集多數其實是**已知類別的邊界樣本**（例如健康誤報
          ——95% 校準閾值天生會讓 ~5–11% 的已知樣本被判未知，累積夠多就會自己
          聚成一叢）→ 操作員退回：清除該叢，不擴張量尺。
        """
        if self.candidate is None:
            raise RuntimeError("目前沒有待確認的候選叢集")
        config = self.candidate["majority"]
        result = {
            "config": config,
            "display": display_name(config),
            "size": self.candidate["size"],
            "purity": self.candidate["purity"],
        }
        if config in self.monitor.known:
            cluster = set(self.candidate["indices"])
            keep = [i for i in range(len(self.quarantine_truth)) if i not in cluster]
            result["action"] = "rejected_known"
        else:
            result["action"] = "learned"
            result["label"] = self.monitor.add_class(config)
            self.polar = PolarMap(self.monitor)  # 量尺擴張 → 重建極座標幾何
            keep = [i for i, t in enumerate(self.quarantine_truth) if t != config]
        self.quarantine_X = [self.quarantine_X[i] for i in keep]
        self.quarantine_truth = [self.quarantine_truth[i] for i in keep]
        self.quarantine_t = [self.quarantine_t[i] for i in keep]
        self.candidate = None
        # 殘餘的未知樣本（可能屬於另一種故障）在下一筆未知樣本進來時即可再分群
        self._since_cluster = self.recluster_every
        return result

    def state(self) -> dict:
        return {
            "known": [
                {"config": c, "display": display_name(c), "label": label}
                for c, label in self.monitor.known.items()
            ],
            "quarantine": len(self.quarantine_X),
            "candidate": self.candidate
            and {k: v for k, v in self.candidate.items() if k != "indices"},
        }


# -- 批次重播（論文數據） -----------------------------------------------------


def run(
    pools: dict[str, np.ndarray],
    sequence: list[str] | None = None,
    seed: int = 42,
    confidence: float = 0.95,
    method: str = "ledoit_wolf",
    openset_method: str = "mahalanobis",
    knn_neighbors: int = 5,
    max_ticks_per_stage: int = 600,
) -> dict:
    """依序注入各未知配置：量測發現延遲 → 自動確認 → 驗證擴張後的分類能力。"""
    if sequence is None:
        sequence = sorted((c for c in pools if c != HEALTHY), key=config_sort_key)
    session = ScaleGrowthSession(
        pools,
        seed=seed,
        confidence=confidence,
        method=method,
        openset_method=openset_method,
        knn_neighbors=knn_neighbors,
    )
    rng = np.random.default_rng(seed + 1)

    stages = []
    for config in sequence:
        sampler = CycleSampler(pools[config], np.arange(len(pools[config])), rng)
        res, found_at, rejected = None, None, 0
        for tick in range(1, max_ticks_per_stage + 1):
            session.process(sampler.draw(), config)
            if session.candidate is not None:
                outcome = session.confirm()
                if outcome["action"] == "rejected_known":
                    rejected += 1  # 已知類別邊界樣本聚成的叢，操作員退回後續跑
                    continue
                res, found_at = outcome, tick
                break
        row = {
            "config": config,
            "display": display_name(config),
            "discovered": res is not None,
            "rejected_known_clusters": rejected,
        }
        if res is None:
            stages.append(row)
            continue
        own = session.monitor.classify(session.monitor.holdout(res["config"]))
        healthy = session.monitor.classify(session.monitor.holdout(HEALTHY))
        row.update(
            {
                "learned": res["config"],
                "samples_to_candidate": found_at,
                "cluster_size": res["size"],
                "cluster_purity": res["purity"],
                "own_holdout_acc": float(np.mean([label == res["config"] for label in own])),
                "healthy_holdout_acc": float(np.mean([label == HEALTHY for label in healthy])),
                "known_after": len(session.monitor.known),
            }
        )
        stages.append(row)

    return {
        "stages": stages,
        "final_known": list(session.monitor.known),
        "model": session.monitor.summary(),
        "seed": seed,
        "confidence": confidence,
        "openset_method": openset_method,
        "method": method,
        "score_type": session.monitor.summary()["score_type"],
        "threshold": 1.0,
        "threshold_strategy": session.monitor.summary()["threshold_strategy"],
        "calibration_source": "known-only 20% calibration split",
        "split": {"train": 0.6, "calibration": 0.2, "holdout": 0.2},
        "knn_neighbors": knn_neighbors if openset_method == "knn" else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(parser)
    add_openset_args(parser)
    parser.add_argument(
        "--sequence", nargs="*", default=None,
        help="注入順序（預設：所有未知配置由輕到重）",
    )
    args = parser.parse_args()

    ds, pools = resolve_dataset(args)
    log, paths = setup_run("exp2_scale_growth")
    log.info(f"實驗二：量尺擴張重播 — {ds['motor']}/{ds['rpm']}"
             f"（openset={args.openset_method}, method={args.method}, "
             f"confidence={args.confidence}, seed={args.seed}）")

    result = run(
        pools, sequence=args.sequence, seed=args.seed,
        confidence=args.confidence, method=args.method,
        openset_method=args.openset_method, knn_neighbors=args.knn_neighbors,
    )
    result["dataset"] = {"motor": ds["motor"], "rpm": ds["rpm"]}

    import pandas as pd

    pd.DataFrame(result["stages"]).to_csv(paths.output_dir / "stages.csv", index=False)
    save_json(paths.output_dir / "summary.json", result)

    log.info(f"{'注入配置':<14}{'發現(筆)':>9}{'叢集':>6}{'純度':>8}{'自類準確':>10}{'健康保持':>10}{'已知數':>8}")
    for s in result["stages"]:
        if not s["discovered"]:
            log.info(f"{s['config']:<14}{'未發現':>9}")
            continue
        log.info(
            f"{s['config']:<14}{s['samples_to_candidate']:>9}{s['cluster_size']:>6}"
            f"{s['cluster_purity']*100:>7.1f}%{s['own_holdout_acc']*100:>9.1f}%"
            f"{s['healthy_holdout_acc']*100:>9.1f}%{s['known_after']:>8}"
        )
    log.info(f"最終已知類別：{' → '.join(result['final_known'])}")
    log.info(f"輸出：{paths.output_dir}")


if __name__ == "__main__":
    main()
