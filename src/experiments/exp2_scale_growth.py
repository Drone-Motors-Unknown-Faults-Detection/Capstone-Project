"""實驗二：量尺擴張（Scale Growth）— 未知偵測 → 隔離累積 → HDBSCAN 分群 → 自行納入。

ScaleGrowthSession 是核心狀態機（首頁即時展示由操作員呼叫 confirm()）：
    1. 每筆樣本以 OpenSetMonitor 打分；判為未知且分數 > quarantine_margin 者進隔離區
    2. 隔離區累積達 min_cluster_size 後以 HDBSCAN 找穩定叢集 → 產生候選
    3. adopt() 不請操作員指出故障程度：群沒有比已知校準資料更遠就退回；否則只用這一群
       已到達樣本切分並重擬合，類名是 discovered-N，再把隔離區中已被新模型接受的殘餘移出。
       真實配置只留在事後對照，不進模型。
    批次重播先從每個未知配置池切出 20% 外部評估集，重播只抽其餘列。

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
from core.runner import (add_dataset_args, add_openset_args, run_dataset, save_json,
                         execution_context, finish_execution)


class ScaleGrowthSession:
    """從健康基準起步、每納入一個候選叢集（adopt 或 confirm）就擴張的開放集監測 session。"""

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
        adaptive_density: bool = True,
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
        self.adaptive_density = adaptive_density
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
        """處理一筆樣本。truth 只存入隔離區，供 confirm() 揭示或 adopt() 事後對照，不參與判定。"""
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
        if not self.adaptive_density:  # 消融對照：只用最嚴格的一階
            return ladder
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

    def _drop_quarantine(self, drop: set[int]) -> None:
        keep = [i for i in range(len(self.quarantine_X)) if i not in drop]
        self.quarantine_X = [self.quarantine_X[i] for i in keep]
        self.quarantine_truth = [self.quarantine_truth[i] for i in keep]
        self.quarantine_t = [self.quarantine_t[i] for i in keep]

    def known_cal_max(self) -> float:
        """目前所有已知類校準列的最大正規化分數：已知資料最遠能到哪裡。"""
        cal = np.vstack([self.monitor.calibration(c) for c in self.monitor.known])
        return float(self.monitor.score(cal).max())

    def _prune_explained(self) -> int:
        """重擬合後，隔離區中已被任一已知類接受的列移出，不再參與分群。不讀 truth。"""
        if not self.quarantine_X:
            return 0
        labels = self.monitor.classify(np.vstack(self.quarantine_X))
        explained = {i for i, label in enumerate(labels) if label is not None}
        self._drop_quarantine(explained)
        return len(explained)

    def adopt(self) -> dict:
        """候選一成形就用該群樣本自學。不讀完整配置池，也不等操作員命名。

        三種結果，都不讀 truth（matched／purity 只放事後對照）：
        - ``rejected_boundary``：群的中位分數沒有超過已知類校準最大值，與已知邊界無從區分 → 清掉該群。
        - ``insufficient``：群不足 5 筆，切不出 train／cal。
        - ``learned``：註冊成 discovered-N 並重擬合，再把隔離區中已被新模型接受的殘餘移出。
        """
        if self.candidate is None:
            raise RuntimeError("目前沒有可納入的候選叢集")
        indices = list(self.candidate["indices"])
        matched = self.candidate["majority"]
        serial = sum(name.startswith("discovered-") for name in self.monitor.known) + 1
        name = f"discovered-{serial}"
        result = {
            "config": name,
            "display": f"模型新類 {serial}",
            "matched": matched,
            "matched_display": self.candidate["majority_display"],
            "size": self.candidate["size"],
            "purity": self.candidate["purity"],
            "source": "cluster_only",
        }
        samples = np.vstack([self.quarantine_X[i] for i in indices])
        cal_max = self.known_cal_max()
        median_score = float(np.median(self.monitor.score(samples)))
        result.update({"median_score": median_score, "known_cal_max": cal_max})
        self.candidate = None
        self._since_cluster = self.recluster_every
        if median_score <= cal_max:
            result["action"] = "rejected_boundary"
            self._drop_quarantine(set(indices))
            return result
        if len(indices) < 5:
            result["action"] = "insufficient"
            return result
        result["action"] = "learned"
        result["label"] = self.monitor.add_observed(name, samples)
        self.polar = PolarMap(self.monitor)
        self._drop_quarantine(set(indices))
        result["pruned"] = self._prune_explained()
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


def _verdict(label: str | None) -> str:
    if label is None:
        return "unknown"
    if label == HEALTHY:
        return "healthy"
    return "known"


EVAL_FRACTION = 0.2


def split_stream_eval(pools: dict[str, np.ndarray], configs: list[str], seed: int) -> dict:
    """每個未知配置池切出外部評估集（20%，向上取整），重播只抽其餘列，兩者互斥。"""
    rng = np.random.default_rng(seed + 2)
    out = {}
    for config in configs:
        order = rng.permutation(len(pools[config]))
        n_eval = int(np.ceil(EVAL_FRACTION * len(order)))
        out[config] = {"stream": np.sort(order[n_eval:]), "eval": np.sort(order[:n_eval])}
    return out


def _eval_breakdown(labels: list[str | None]) -> dict:
    """外部評估集被判成什麼：未知、健康、某個新類（比例），及最多被判成哪一類。"""
    n = len(labels)
    named = [label for label in labels if label not in (None, HEALTHY)]
    return {
        "n": n,
        "unknown_rate": sum(label is None for label in labels) / n,
        "healthy_rate": sum(label == HEALTHY for label in labels) / n,
        "known_rate": len(named) / n,
        "top_class": max(set(named), key=named.count) if named else None,
    }


def iter_playback(
    pools: dict[str, np.ndarray],
    sequence: list[str] | None = None,
    seed: int = 42,
    confidence: float = 0.95,
    method: str = "ledoit_wolf",
    openset_method: str = "mahalanobis",
    knn_neighbors: int = 5,
    max_ticks_per_stage: int = 600,
    adaptive_density: bool = True,
    source_context=None,
):
    """依序注入未知配置。候選出現時立刻用該群樣本自學，不等操作員確認。"""
    if sequence is None:
        sequence = sorted((c for c in pools if c != HEALTHY), key=config_sort_key)
    provenance = execution_context(pools, source_context, seed=seed, confidence=confidence,
        method=method, openset_method=openset_method, knn_neighbors=knn_neighbors,
        sequence=list(sequence), max_ticks_per_stage=max_ticks_per_stage,
        adaptive_density=adaptive_density)
    session = ScaleGrowthSession(
        pools,
        seed=seed,
        confidence=confidence,
        method=method,
        openset_method=openset_method,
        knn_neighbors=knn_neighbors,
        adaptive_density=adaptive_density,
    )
    rng = np.random.default_rng(seed + 1)
    yield {
        "event": "fitted",
        "model": session.monitor.summary(),
        "known": list(session.monitor.known),
    }

    parts = split_stream_eval(pools, sequence, seed)
    # 事後對照：某配置第一次成為新類多數的時點（從該配置開始注入起算，可跨段）
    discovery = {c: {"discovered": False} for c in sequence}
    started: dict[str, int] = {}
    global_tick = 0
    rejections = []  # 未學成的候選（rejected_boundary／insufficient）

    stages = []
    for config in sequence:
        sampler = CycleSampler(pools[config], parts[config]["stream"], rng)
        started[config] = global_tick
        res, found_at = None, None
        for tick in range(1, max_ticks_per_stage + 1):
            global_tick += 1
            out = session.process(sampler.draw(), config)
            yield {
                "event": "tick",
                "config": config,
                "display": display_name(config),
                "t": tick,
                "score": out["score"],
                "label": out["label"],
                "verdict": _verdict(out["label"]),
                "quarantine": out["quarantine"],
                "known": len(session.monitor.known),
            }
            if session.candidate is None:
                continue
            yield {
                "event": "candidate",
                "config": config,
                "candidate": session.state()["candidate"],
            }
            outcome = session.adopt()
            yield {
                "event": "adopted",
                "config": config,
                "outcome": outcome,
                "known": list(session.monitor.known),
            }
            if outcome["action"] != "learned":
                rejections.append({"config": config, "t": tick, "action": outcome["action"],
                                   "size": outcome["size"], "matched": outcome["matched"],
                                   "median_score": outcome.get("median_score"),
                                   "known_cal_max": outcome.get("known_cal_max")})
                continue
            m = outcome["matched"]
            if m in discovery and not discovery[m]["discovered"]:
                discovery[m] = {"discovered": True, "learned": outcome["config"],
                                "delay": global_tick - started[m]}
            res, found_at = outcome, tick
            break
        row = {
            "config": config,
            "display": display_name(config),
            "discovered": res is not None,
        }
        if res is not None:
            own = session.monitor.classify(session.monitor.holdout(res["config"]))
            healthy = session.monitor.classify(session.monitor.holdout(HEALTHY))
            ev = _eval_breakdown(session.monitor.classify(pools[config][parts[config]["eval"]]))
            row.update(
                {
                    "learned": res["config"],
                    "learned_display": res["display"],
                    "matched": res["matched"],
                    "matched_display": res["matched_display"],
                    "matched_is_config": res["matched"] == config,
                    "samples_to_candidate": found_at,
                    "cluster_size": res["size"],
                    "cluster_purity": res["purity"],
                    "pruned": res["pruned"],
                    "own_holdout_acc": float(np.mean([label == res["config"] for label in own])),
                    "own_holdout_n": len(own),
                    "eval_known_rate": ev["known_rate"],
                    "eval_n": ev["n"],
                    "healthy_holdout_acc": float(np.mean([label == HEALTHY for label in healthy])),
                    "known_after": len(session.monitor.known),
                }
            )
        stages.append(row)

    final_eval = {
        c: _eval_breakdown(session.monitor.classify(pools[c][parts[c]["eval"]])) for c in sequence
    }

    model = session.monitor.summary()
    yield {"event": "result", "result": {
        "stages": stages,
        "discovery": discovery,
        "rejections": rejections,
        "final_eval": final_eval,
        "eval_fraction": EVAL_FRACTION,
        "known_cal_max": session.known_cal_max(),
        "final_known": list(session.monitor.known),
        "model": model,
        "seed": seed,
        "confidence": confidence,
        "openset_method": openset_method,
        "method": method,
        "score_type": model["score_type"],
        "threshold": 1.0,
        "threshold_strategy": model["threshold_strategy"],
        "calibration_source": "known-only 20% calibration split",
        "split": {"train": 0.6, "calibration": 0.2, "holdout": 0.2},
        "knn_neighbors": knn_neighbors if openset_method == "knn" else None,
        "adaptive_density": adaptive_density,
        "provenance": finish_execution(provenance, pools, source_context),
    }}


def run(
    pools: dict[str, np.ndarray],
    sequence: list[str] | None = None,
    seed: int = 42,
    confidence: float = 0.95,
    method: str = "ledoit_wolf",
    openset_method: str = "mahalanobis",
    knn_neighbors: int = 5,
    max_ticks_per_stage: int = 600,
    adaptive_density: bool = True,
    source_context=None,
) -> dict:
    """依序注入各未知配置：量測發現延遲 → 用候選群自學 → 檢查新類與健康是否還認得。"""
    for event in iter_playback(
        pools, sequence=sequence, seed=seed, confidence=confidence, method=method,
        openset_method=openset_method, knn_neighbors=knn_neighbors,
        max_ticks_per_stage=max_ticks_per_stage, adaptive_density=adaptive_density,
        source_context=source_context,
    ):
        if event["event"] == "result":
            return event["result"]
    raise RuntimeError("量尺擴張沒有產生結果")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(parser)
    add_openset_args(parser)
    parser.add_argument(
        "--sequence", nargs="*", default=None,
        help="注入順序（預設：所有未知配置由輕到重）",
    )
    parser.add_argument(
        "--fixed-density", action="store_true",
        help="關掉 HDBSCAN 密度階梯，只用 (min_cluster_size, min_samples)；消融對照用",
    )
    args = parser.parse_args()

    log, paths = setup_run("exp2_scale_growth", unique=True)
    ds, result = run_dataset(
        args, paths, run, sequence=args.sequence, seed=args.seed,
        confidence=args.confidence, method=args.method,
        openset_method=args.openset_method, knn_neighbors=args.knn_neighbors,
        adaptive_density=not args.fixed_density,
    )
    log.info("實驗二：量尺擴張重播 — {}/{}", ds["motor"], ds["rpm"])
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
