"""實驗二：量尺擴張（Scale Growth）— 未知偵測 → 隔離累積 → HDBSCAN 分群 → 自行納入。

ScaleGrowthSession 是核心狀態機，批次、實驗頁、首頁即時展示與導覽都走同一條路：
    1. 每筆樣本以 OpenSetMonitor 打分；判為未知且分數 > quarantine_margin 者進隔離區
    2. 隔離區累積達 min_cluster_size 後以 HDBSCAN 找穩定叢集 → 產生候選
    3. 候選一成形，呼叫端立刻 adopt()。不請操作員指出故障程度：群沒有比已知校準資料更遠就退回；
       否則只用這一群已到達樣本切分並重擬合，類名是 discovered-N，再把隔離區中已被新模型
       接受的殘餘移出。真實配置只留在事後對照，不進模型。
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
from collections import Counter

import hdbscan
import numpy as np

from core.data import HEALTHY, CycleSampler, config_sort_key, display_name
from core.feature_schema import CONFIG_ALIASES
from core.geometry import PolarMap
from core.logger import setup_run
from core.monitor import OpenSetMonitor
from core.runner import (add_dataset_args, add_openset_args, run_dataset, save_json,
                         execution_context, finish_execution)


def _majority(items: list[str]) -> str:
    """多數決；平手取字串排序最小者，不依賴雜湊順序（鐵則 7）。"""
    counts = Counter(items)
    return min(counts, key=lambda name: (-counts[name], name))


class ScaleGrowthSession:
    """從健康基準起步、每 adopt() 一個候選叢集就擴張的開放集監測 session。

    min_cluster_size 至少 5：候選群至少這麼多筆，adopt() 才切得出 train／cal（add_observed 的下限）。
    """

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
        if min_cluster_size < 5:
            raise ValueError(f"min_cluster_size 至少 5（候選群要切得出 train／cal）：{min_cluster_size}")
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
        self.quarantine_row: list[int | None] = []  # 來源池列號（呼叫端給；只供事後對照與串流排除擬合列）
        self.quarantine_t: list[int] = []
        self.learned_rows: dict[str, dict] = {}  # discovered-N → 候選群列來源與其中拿去擬合（train／cal）的列
        self.quarantine_duplicates = 0  # 與隔離區既有列完全相同、未再存入的筆數
        self.candidate: dict | None = None
        self._since_cluster = 0
        self.n_seen = 0
        self.cluster_attempts = 0  # 連續分群失敗次數（成功即歸零；供 UI 呈現進度）
        self.polar = PolarMap(self.monitor)  # 極座標幾何（實驗四），每次擴張後重建

    # -- 串流 ------------------------------------------------------------------

    def _row_id(self, i: int) -> int | None:
        """隔離區第 i 列的來源池列號；呼叫端沒給（或測試直接塞列）時是 None。"""
        return self.quarantine_row[i] if i < len(self.quarantine_row) else None

    def _is_duplicate(self, row: np.ndarray) -> bool:
        """隔離區已有完全相同的原始特徵向量（循環重播會重抽同一列）；只比特徵，不讀 truth。"""
        return any(np.array_equal(row, kept) for kept in self.quarantine_X)

    def process(self, x_raw: np.ndarray, truth: str, row_id: int | None = None) -> dict:
        """處理一筆樣本。truth 與 row_id（來源池列號）只存入隔離區，供 adopt() 事後對照與列來源紀錄，不參與判定。

        回傳 candidate_new=True 時呼叫端在同一拍就 adopt()，沒有待確認狀態。
        """
        self.n_seen += 1
        scores, labels = self.monitor.score_and_classify(x_raw)
        score, label = float(scores[0]), labels[0]
        # 極座標判讀（方向熟悉度為連續參考指標；不影響既有的已知/未知判定）
        direction = self.polar.analyze(x_raw)[0]
        candidate_new = False
        # 隔離線：分數明顯超線才參與新類發現，邊界誤報（如健康的 1.0x）僅警報不進隔離區
        if label is None and score > self.quarantine_margin:
            row = np.asarray(x_raw, dtype=float).ravel()
            if self._is_duplicate(row):
                self.quarantine_duplicates += 1
            else:
                self.quarantine_X.append(row)
                self.quarantine_truth.append(truth)
                self.quarantine_row.append(None if row_id is None else int(row_id))
                self.quarantine_t.append(self.n_seen)
                self._since_cluster += 1
                if (
                    len(self.quarantine_X) >= self.min_cluster_size
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
        majority = _majority(truths)
        self.candidate = {
            "indices": idx.tolist(),
            "size": best_size,
            "majority": majority,
            "majority_display": display_name(majority),
            "purity": round(truths.count(majority) / len(truths), 4),
            # 叢集樣本的蒐集時間區間——區分「舊帳」（如健康期累積的誤報）與現行串流
            "t_range": [int(min(ts)), int(max(ts))],
            "cluster_params": list(used),
            # 列來源 [truth, 池列號]，與候選群矩陣同序；只供事後對照／串流排除，不進模型
            "rows": [[self.quarantine_truth[i], self._row_id(i)] for i in idx],
        }
        return True

    # -- 擴張 ------------------------------------------------------------------

    def _drop_quarantine(self, drop: set[int]) -> None:
        keep = [i for i in range(len(self.quarantine_X)) if i not in drop]
        self.quarantine_X = [self.quarantine_X[i] for i in keep]
        self.quarantine_truth = [self.quarantine_truth[i] for i in keep]
        self.quarantine_row = [self._row_id(i) for i in keep]
        self.quarantine_t = [self.quarantine_t[i] for i in keep]

    def known_cal_max(self) -> float:
        """目前所有已知類校準列的最大正規化分數：已知資料最遠能到哪裡。"""
        cal = np.vstack([self.monitor.calibration(c) for c in self.monitor.known])
        return float(self.monitor.score(cal).max())

    def _prune_explained(self) -> int:
        """重擬合後對隔離區再套一次進入規則：被任一已知類接受、或分數不再超過隔離線的列移出。不讀 truth。"""
        if not self.quarantine_X:
            return 0
        scores, labels = self.monitor.score_and_classify(np.vstack(self.quarantine_X))
        explained = {
            i for i, (score, label) in enumerate(zip(scores, labels))
            if label is not None or score <= self.quarantine_margin
        }
        self._drop_quarantine(explained)
        return len(explained)

    def adopt(self) -> dict:
        """候選一成形就用該群樣本自學。不讀完整配置池，也不等操作員命名。

        三種結果，都不讀 truth（matched／purity 只放事後對照）：
        - ``rejected_boundary``：群的中位分數沒有超過已知類校準最大值，與已知邊界無從區分 → 清掉該群。
        - ``insufficient``：群不足 5 筆，切不出 train／cal（min_cluster_size >= 5 時 _try_cluster 不會產生這種群，
          只擋直接塞進 candidate 的小群）。
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
            self._drop_quarantine(set(indices))  # 不清掉會在下一筆隔離樣本又分出同一群（#236）
            return result
        result["action"] = "learned"
        result["label"] = self.monitor.add_observed(name, samples)
        # 候選群矩陣的列來源，以及其中切成 train／cal 的列：串流端據此不再重抽擬合列（#296 R3-02）
        rows = [[self.quarantine_truth[i], self._row_id(i)] for i in indices]
        split = self.monitor.splits[name]
        fitted = [rows[i] for i in np.concatenate([split.train, split.cal])]
        self.learned_rows[name] = {"rows": rows, "fitted": fitted}
        result.update({"rows": rows, "fitted_rows": fitted})
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


def _eval_breakdown(labels: list[str | None], config: str, class_matched: dict[str, str]) -> dict:
    """外部評估集被判成什麼：未知、健康、某個新類（比例），及最多被判成哪一類。

    known_rate 只要求被任一非健康類接受，是上限；matched_rate 要求判進事後多數等於該配置的新類（#243）。
    """
    n = len(labels)
    named = [label for label in labels if label not in (None, HEALTHY)]
    return {
        "n": n,
        "unknown_rate": sum(label is None for label in labels) / n,
        "healthy_rate": sum(label == HEALTHY for label in labels) / n,
        "known_rate": len(named) / n,
        "matched_rate": sum(class_matched.get(label) == config for label in named) / n,
        "top_class": _majority(named) if named else None,
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
    healthy_mix: float = 0.0,
    source_context=None,
):
    """依序注入未知配置。候選出現時立刻用該群樣本自學，不等操作員確認。"""
    if not 0.0 <= healthy_mix <= 1.0:
        raise ValueError(f"healthy_mix 必須在 0～1：{healthy_mix}")
    if sequence is None:
        sequence = sorted((c for c in pools if c != HEALTHY), key=config_sort_key)
    else:
        # 池 key 是正規名；使用者的 --sequence 先正規化，不在池裡的直接拒絕（#285 R2-03）
        sequence = [CONFIG_ALIASES.get(c, c) for c in sequence]
        missing = [c for c in sequence if c not in pools or c == HEALTHY]
        if missing:
            raise ValueError(f"sequence 含不可注入的配置：{missing}")
    provenance = execution_context(pools, source_context, seed=seed, confidence=confidence,
        method=method, openset_method=openset_method, knn_neighbors=knn_neighbors,
        sequence=list(sequence), max_ticks_per_stage=max_ticks_per_stage,
        adaptive_density=adaptive_density, healthy_mix=healthy_mix)
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
    class_matched: dict[str, str] = {}  # 事後對照：discovered-N → 學成當下的真值多數
    # 健康混入（#240）：從健康 holdout 抽，另用一支 rng；healthy_mix=0 時不抽亂數，結果與不混入相同
    mix_rng = np.random.default_rng(seed + 3)
    healthy_sampler = CycleSampler(pools[HEALTHY], session.monitor.splits[HEALTHY].holdout, mix_rng)

    stages = []
    for config in sequence:
        sampler = CycleSampler(pools[config], parts[config]["stream"], rng)
        started[config] = global_tick
        res, found_at = None, None
        healthy_seen = healthy_quarantined = 0
        for tick in range(1, max_ticks_per_stage + 1):
            global_tick += 1
            if healthy_mix > 0 and mix_rng.random() < healthy_mix:
                x, truth = healthy_sampler.draw(), HEALTHY
                row_id = healthy_sampler.last_index
            else:
                x, truth = sampler.draw(), config
                row_id = sampler.last_index
            before = len(session.quarantine_X)
            out = session.process(x, truth, row_id)
            if truth == HEALTHY:
                healthy_seen += 1
                healthy_quarantined += len(session.quarantine_X) > before
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
            class_matched[outcome["config"]] = m
            if m in discovery and not discovery[m]["discovered"]:
                discovery[m] = {"discovered": True, "learned": outcome["config"],
                                "delay": global_tick - started[m]}
            res, found_at = outcome, tick
            break
        # discovered 只表示這段形成了新類；是否對上本段注入配置看 matched_is_config（#286）
        row = {
            "config": config,
            "display": display_name(config),
            "discovered": res is not None,
            "matched_is_config": res is not None and res["matched"] == config,
            "healthy_seen": healthy_seen,
            "healthy_quarantined": int(healthy_quarantined),
        }
        if res is not None:
            own = session.monitor.classify(session.monitor.holdout(res["config"]))
            healthy = session.monitor.classify(session.monitor.holdout(HEALTHY))
            ev = _eval_breakdown(session.monitor.classify(pools[config][parts[config]["eval"]]), config, class_matched)
            row.update(
                {
                    "learned": res["config"],
                    "learned_display": res["display"],
                    "matched": res["matched"],
                    "matched_display": res["matched_display"],
                    "samples_to_candidate": found_at,
                    "cluster_size": res["size"],
                    "cluster_purity": res["purity"],
                    "pruned": res["pruned"],
                    "own_holdout_acc": float(np.mean([label == res["config"] for label in own])),
                    "own_holdout_n": len(own),
                    "eval_known_rate": ev["known_rate"],
                    "eval_matched_rate": ev["matched_rate"],
                    "eval_n": ev["n"],
                    "healthy_holdout_acc": float(np.mean([label == HEALTHY for label in healthy])),
                    "known_after": len(session.monitor.known),
                }
            )
        stages.append(row)

    final_eval = {
        c: _eval_breakdown(session.monitor.classify(pools[c][parts[c]["eval"]]), c, class_matched) for c in sequence
    }

    model = session.monitor.summary()
    yield {"event": "result", "result": {
        "stages": stages,
        "n_stages": len(stages),
        "n_learned": sum(s["discovered"] for s in stages),
        "n_matched_config": sum(s["matched_is_config"] for s in stages),
        "discovery": discovery,
        "rejections": rejections,
        "final_eval": final_eval,
        "eval_fraction": EVAL_FRACTION,
        "known_cal_max": session.known_cal_max(),
        "quarantine_duplicates": session.quarantine_duplicates,
        "final_known": list(session.monitor.known),
        "model": model,
        "seed": seed,
        "confidence": confidence,
        "openset_method": model["openset_method"],
        "method": model["method"],
        "score_type": model["score_type"],
        "threshold": 1.0,
        "threshold_strategy": model["threshold_strategy"],
        "calibration_source": model["calibration_source"],
        "split": model["split"],
        "knn_neighbors": model["knn_neighbors"],
        "adaptive_density": adaptive_density,
        "provenance": finish_execution(provenance, pools, source_context),
        "healthy_mix": healthy_mix,
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
    healthy_mix: float = 0.0,
    source_context=None,
) -> dict:
    """依序注入各未知配置：量測發現延遲 → 用候選群自學 → 檢查新類與健康是否還認得。"""
    for event in iter_playback(
        pools, sequence=sequence, seed=seed, confidence=confidence, method=method,
        openset_method=openset_method, knn_neighbors=knn_neighbors,
        max_ticks_per_stage=max_ticks_per_stage, adaptive_density=adaptive_density,
        source_context=source_context,
        healthy_mix=healthy_mix,
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
    parser.add_argument(
        "--healthy-mix", type=float, default=0.0,
        help="每筆以此機率改送一筆健康 holdout 窗（0～1，預設 0 不混入）",
    )
    args = parser.parse_args()

    log, paths = setup_run("exp2_scale_growth", unique=True, data_root=args.data_root)
    ds, result = run_dataset(
        args, paths, run, sequence=args.sequence, seed=args.seed,
        confidence=args.confidence, method=args.method,
        openset_method=args.openset_method, knn_neighbors=args.knn_neighbors,
        adaptive_density=not args.fixed_density, healthy_mix=args.healthy_mix,
    )
    log.info("實驗二：量尺擴張重播 — {}/{}（healthy_mix={}）",
             ds["motor"], ds["rpm"], args.healthy_mix)
    result["dataset"] = {"motor": ds["motor"], "rpm": ds["rpm"]}

    import pandas as pd

    frame = pd.DataFrame(result["stages"])
    # 未學成段缺值時 pandas 會把整數欄轉成 float（215.0）；改用可空整數寫出（#256）
    for column in ("samples_to_candidate", "cluster_size", "pruned", "own_holdout_n", "eval_n", "known_after"):
        if column in frame:
            frame[column] = frame[column].astype("Int64")
    frame.to_csv(paths.output_dir / "stages.csv", index=False)
    save_json(paths.output_dir / "summary.json", result)

    log.info(f"{'注入配置':<14}{'學成(筆)':>9}{'叢集':>6}{'純度':>8}{'事後多數':<14}{'對上':>4}"
             f"{'自類準確':>10}{'健康保持':>10}{'已知數':>8}")
    for s in result["stages"]:
        if not s["discovered"]:
            log.info(f"{s['config']:<14}{'未學成':>9}")
            continue
        log.info(
            f"{s['config']:<14}{s['samples_to_candidate']:>9}{s['cluster_size']:>6}"
            f"{s['cluster_purity']*100:>7.1f}%{s['matched']:<14}{'✓' if s['matched_is_config'] else '✗':>4}"
            f"{s['own_holdout_acc']*100:>9.1f}%{s['healthy_holdout_acc']*100:>9.1f}%{s['known_after']:>8}"
        )
    log.info(f"形成新類 {result['n_learned']}／{result['n_stages']} 段；"
             f"對上本段配置 {result['n_matched_config']}／{result['n_stages']} 段")
    log.info(f"最終已知類別：{' → '.join(result['final_known'])}")
    log.info(f"輸出：{paths.output_dir}")


if __name__ == "__main__":
    main()
