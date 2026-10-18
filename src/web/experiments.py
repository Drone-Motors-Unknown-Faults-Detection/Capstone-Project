"""Web 實驗頁的目錄與執行層。

每個實驗頁對應一支 src/experiments/ 模組：本模組只負責把前端送來的參數驗證後，
轉成該模組 run() 的引數、把回傳值轉成可 JSON 化的 dict，並把每次執行的結果
存到 output/web_server/{ts}/experiments/。判定、統計與分群邏輯仍全部在
src/experiments/ 與 src/core/（AGENT.md 鐵則 2）。

已提交的矩陣結果（實驗六正式矩陣、實驗八 54 列）只讀取、不從網頁重跑，
避免覆蓋 output/ 與 reports/ 裡進版的證據。
"""

from __future__ import annotations

import csv
import importlib
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from core.data import HEALTHY, config_sort_key, display_name, load_pools
from core.detectors import ALL_DETECTORS
from core.openset import SUPPORTED_OPENSET_METHODS

ROOT = Path(__file__).resolve().parents[2]

_OPENSET = {
    "name": "openset_method", "label": "Open Set 方法", "type": "select",
    "options": list(SUPPORTED_OPENSET_METHODS), "default": "mahalanobis",
}
_SEED = {"name": "seed", "label": "seed", "type": "int", "default": 42, "min": 0, "max": 100000}
_DATASET = {"name": "dataset", "label": "資料集", "type": "dataset"}

CATALOG: list[dict] = [
    {
        "id": "exp1", "no": "實驗一", "title": "冷啟動偵測能力",
        "question": "只用 8screws 建健康基準，九種故障抓不抓得到？",
        "code": "src/experiments/exp1_cold_start.py",
        "doc": "docs/experiments/exp1_cold_start.md",
        "fits": "8screws 的 RobustScaler 與開集偵測器（Mahalanobis 為逐類 Ledoit–Wolf 共變異數、k-NN 為參考樣本近鄰索引），再用校準集定閾值",
        "seconds": 0.3,
        "stream": True,
        "params": [_DATASET, _OPENSET, _SEED],
    },
    {
        "id": "exp2", "no": "實驗二", "title": "量尺擴張",
        "question": "陌生樣本形成群之後，模型能不能只用這一群自己學會新類？",
        "code": "src/experiments/exp2_scale_growth.py",
        "doc": "docs/experiments/exp2_scale_growth.md",
        "fits": "8screws 的 RobustScaler 與開集偵測器；候選群出現後只用該群已到達樣本切分並重擬合，類名是模型新類，不使用完整配置池",
        "seconds": 11,
        "stream": True,
        "params": [_DATASET, _OPENSET, _SEED],
    },
    {
        "id": "exp3", "no": "實驗三", "title": "漸進 vs 突發判別",
        "question": "漸進鬆脫與突發切換能否從分數序列分開？",
        "code": "src/experiments/exp3_trend.py",
        "doc": "docs/experiments/exp3_trend.md",
        "fits": "8screws 的 RobustScaler 與開集偵測器（Mahalanobis 為逐類 Ledoit–Wolf 共變異數、k-NN 為參考樣本近鄰索引），再用校準集定閾值",
        "seconds": 0.6,
        "stream": True,
        "params": [_DATASET, _OPENSET, _SEED,
                   {"name": "n_trials", "label": "每劇本次數", "type": "int",
                    "default": 20, "min": 1, "max": 100}],
    },
    {
        "id": "exp4", "no": "實驗四", "title": "極座標健康地圖",
        "question": "故障方向與半徑能不能對上螺絲配置？",
        "code": "src/experiments/exp4_polar_map.py",
        "doc": "docs/experiments/exp4_polar_map.md",
        "fits": "監測器（同實驗一）與極座標健康地圖 PolarMap，並逐步加入故障射線重擬合",
        "seconds": 10,
        "stream": True,
        "params": [_DATASET, _SEED],
    },
    {
        "id": "exp5", "no": "實驗五", "title": "跨工況冷啟動泛化",
        "question": "某一工況的健康基準能否搬到別的工況？範圍以實際掃描工況為準。",
        "code": "src/experiments/exp5_cross_condition.py",
        "doc": "docs/experiments/exp5_cross_condition.md",
        "fits": "各可用工況擬合健康監測器，再交叉測試；策略比較的三種基準都只用外層 train+cal 擬合",
        "seconds": 7,
        "params": [_SEED],
    },
    {
        "id": "exp6", "no": "實驗六", "sub": "6-1", "title": f"{len(ALL_DETECTORS)} 種偵測器廣度比較",
        "question": "相同健康切分下，各方法的排序與誤報有何不同？legacy 門檻來源不同，須分開解讀。",
        "code": "src/experiments/exp6_osr_benchmark.py",
        "doc": "docs/experiments/exp6_osr_benchmark.md",
        "fits": f"各可用工況 × {len(ALL_DETECTORS)} 種偵測器：Ledoit–Wolf 與 legacy 馬氏距離、One-Class SVM、Isolation Forest、LOF、k-NN 距離、PCA 重建、MLP AutoEncoder",
        "seconds": 4,
        "params": [_SEED],
    },
    {
        "id": "exp6_formal", "no": "實驗六", "sub": "6-2", "title": "正式版 Mahalanobis vs k-NN（單次）",
        "question": "同一健康切分規則下，跑一個方法與 seed；Web 允許部分工況，須看實際覆蓋。",
        "code": "src/experiments/exp6_formal_benchmark.py",
        "doc": "docs/experiments/exp6_formal_benchmark.md",
        "fits": "各可用工況的健康 train 擬合 RobustScaler 與所選開集偵測器，健康 cal 定閾值",
        "seconds": 2,
        "params": [_OPENSET, _SEED],
        "committed": "exp6_formal_matrix",
        "committed_sub": "6-3",
        "committed_title": "正式矩陣與彙總（9 工況 × 3 seed × 2 方法）",
        "committed_code": "src/experiments/exp6_matrix.py → experiments/aggregate_exp6.py",
        "committed_doc": "docs/experiments/exp6_matrix.md、docs/experiments/exp6_aggregate.md",
    },
    {
        "id": "exp7", "no": "實驗七", "title": "單工況 Mahalanobis vs k-NN",
        "question": "同一工況、同一切分下，換成 k-NN 距離差多少？",
        "code": "src/experiments/compare_openset.py",
        "doc": "docs/experiments/exp7_compare_openset.md",
        "fits": "同一切分下 Mahalanobis 與 k-NN 兩個監測器各擬合一次",
        "seconds": 0.6,
        "params": [_DATASET, _SEED],
    },
    {
        "id": "exp8", "no": "實驗八", "sub": "8-1", "title": "相對健康指數（單次比較）",
        "question": "開集分數映成 0～1 相對指數後，健康與未知分得多開？不是損壞百分比。",
        "code": "src/experiments/health_index_benchmark.py",
        "doc": "docs/experiments/exp8_health.md",
        "fits": "各可用工況的健康 train 建開集參考；同一份健康 cal 定門檻與第 10／95 百分位刻度",
        "seconds": 2,
        "params": [_OPENSET, _SEED],
        "committed": "exp8_health_index_results",
        "committed_sub": "8-2",
        "committed_title": "正式矩陣（9 工況 × 3 seed × 2 方法 = 54 列）",
        "committed_code": "src/experiments/health_index_matrix.py",
        "committed_doc": "docs/experiments/exp8_health.md",
    },
    {
        "id": "exp8_monitor", "no": "實驗八", "sub": "8-3", "title": "健康監測逐窗輸出",
        "question": "健康 holdout 或故障全池的前段列逐筆重播，相對指數與告警怎麼走？原始時間未確認。",
        "code": "src/experiments/health_monitor.py",
        "doc": "docs/experiments/exp8_health.md",
        "fits": "單一工況的開集偵測器與健康指數刻度；逐窗趨勢狀態只存在這次執行",
        "seconds": 0.2,
        "stream": True,
        "params": [_DATASET, _OPENSET, _SEED,
                   {"name": "config", "label": "注入配置", "type": "config", "default": HEALTHY},
                   {"name": "max_windows", "label": "最多窗口數", "type": "int",
                    "default": 120, "min": 5, "max": 1000}],
    },
    {
        "id": "exp9", "no": "實驗九", "title": "非線性重建誤差",
        "question": "AutoEncoder 重建誤差比線性 PCA 重建更分得開嗎？",
        "code": "src/core/detectors.py",
        "doc": "docs/experiments/exp9_autoencoder.md",
        "fits": "沒有獨立執行器。權重只在實驗六的 mlp_autoencoder 裡，由健康 train 建立",
        "seconds": 0,
        "executable": False,
        "params": [],
    },
    {
        "id": "exp10", "no": "實驗十", "title": "少量目標健康資料的跨工況遷移",
        "question": "目標端少量健康資料時，中心平移比從零冷啟動划算嗎？",
        "code": "src/experiments/exp10_transfer.py",
        "doc": "docs/experiments/exp10_transfer.md",
        "fits": "每個來源工況的健康 train 建 Ledoit–Wolf 基準；adapted 只平移目標健康中位數，scratch 只用 N 筆目標健康從零擬合",
        "seconds": 20,
        "params": [_SEED],
    },
    {
        "id": "exp11", "no": "實驗十一", "title": "原始論文協定 Accuracy／F1 對照",
        "question": "同一批配置角色下，legacy 與 Ledoit–Wolf 的分類分數差多少？",
        "code": "src/experiments/exp11_ancestor_comparison.py",
        "doc": "docs/experiments/exp11_ancestor_comparison.md",
        "fits": "協定 A 把 8／1／2／3／4 螺絲當已知並逐類擬合；協定 B 只擬合健康。網頁使用畫面上的單一 seed，兩種共變異數都跑",
        "seconds": 25,
        "params": [_SEED],
    },
    {
        "id": "exp12", "no": "實驗十二", "title": "混淆矩陣與 t-SNE",
        "question": "錯分落在哪一格，特徵投影看不看得出配置？",
        "code": "src/experiments/exp12_confusion_tsne.py",
        "doc": "docs/experiments/exp12_confusion_tsne.md",
        "fits": "二元矩陣只用健康基準；多類矩陣沿實驗十一的已知配置擬合 legacy 與 Ledoit–Wolf；t-SNE 只對已縮放特徵嵌入，不更新偵測器",
        "seconds": 30,
        "params": [_SEED],
    },
]

_EXPLANATIONS = {
    "exp1": ("健康 train 建基準、cal 定線；健康 holdout 與故障全池只評估。", "只學健康，不是九故障分類器；列級切分不證明錄製獨立。"),
    "exp2": ("初始只 fit 健康；未知全池循環重播。候選群出現後只用該群樣本自學。", "真實配置只做事後對照，不進類名，也不用完整配置池重擬合。首頁即時展示仍是另一個手動確認 session。"),
    "exp3": ("一份健康基準供所有 trial；健康從 holdout、故障從全池循環抽樣。", "劇本順序不是實際退化；延遲單位為筆，不能當秒或 RUL。"),
    "exp4": ("已確認配置 train 建射線；fault holdout 另校準方向，排序圖含錨點。", "固定 LW 幾何；方向、半徑沒有物理損傷標定，也非獨立方向測試。"),
    "exp5": ("source 健康建立基準；跨 target 用完整健康及 fault 池。", "策略 b 三種基準共用同一份外層健康 holdout，擬合不含它；T1/T2/T3 是不同個體，不是生命週期。"),
    "exp6": ("同工況所有方法共用健康 train／cal／holdout；fault 全池只測試。", "包含 AutoEncoder，內部驗證另切；legacy 用 train 定線，不是相同校準保證。"),
    "exp6_formal": ("每工況健康 train／cal／holdout；其他配置全池為 unknown。", "known 只有健康，accuracy 是二元識別；部分工況不等於完整九工況。"),
    "exp7": ("兩方法在同一資料、seed 與健康切分下比較；unknown 不進 fit／cal。", "只支持同條件健康／未知比較，不是跨馬達 fault-type 準確率。"),
    "exp8": ("健康 train 建模型；同一 cal 設距離線及健康錨點；其餘只評估。", "指數可能飽和；0～1 不等於損壞百分比、故障機率或剩餘壽命。"),
    "exp8_monitor": ("健康 train／cal 建基準；healthy 用 holdout，fault 用全池前段。", "window_index 是重播索引，時間戳缺失；告警政策與實體事件真值不同。"),
    "exp9": ("健康 train 建 RobustScaler 與 MLP；內部 validation 只用於提早停止。", "沒有獨立執行。比較結果在實驗六的 mlp_autoencoder 列，權重不保存。"),
    "exp10": ("來源健康建基準；目標健康抽 support 做平移或從零擬合，其餘目標健康與故障全池只評估。", "direct 用完整目標健康池，與 adapted／scratch 的測試列不完全相同。沒有逐筆播放。"),
    "exp11": ("協定 A 的已知類用各自 train／cal；未知配置全池只分類。協定 B 只註冊健康。", "網頁只跑所選 seed。legacy 的分類標籤固定，分差不能全歸因於共變異數。"),
    "exp12": ("混淆矩陣用既有基準的預測計數；t-SNE 從各配置抽樣嵌入。", "t-SNE 含擬合用列，只供診斷。沒有逐筆播放，嵌入距離不是嚴重度。"),
}
for _entry in CATALOG:
    _entry["data_roles"], _entry["limitations"] = _EXPLANATIONS[_entry["id"]]
_BY_ID = {e["id"]: e for e in CATALOG}
SCORE_CHUNK = 4  # 實驗一每個 frame 送幾筆分數


def jsonable(value: Any) -> Any:
    """把 numpy 型別、NaN/inf 轉成瀏覽器 JSON.parse 吃得下的值。"""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


class ExperimentRunner:
    def __init__(self, datasets: list[dict], data_root: str, out_dir: Path | None = None,
                 detector_options: dict | None = None):
        self.datasets = datasets
        self.data_root = data_root
        self.out_dir = Path(out_dir) / "experiments" if out_dir else None
        self.detector_options = detector_options or {}
        self._pools_cache: dict[str, dict] = {}

    # -- 目錄 ------------------------------------------------------------------

    def catalog(self) -> dict:
        configs = []
        if self.datasets:
            pools = self._pools(self._dataset_key(self.datasets[0]))
            configs = [{"config": c, "display": display_name(c)}
                       for c in sorted(pools, key=config_sort_key)]
        return {
            "experiments": CATALOG,
            "datasets": [self._dataset_key(d) for d in self.datasets],
            "configs": configs,
            "data_root": self.data_root,
        }

    # -- 參數 ------------------------------------------------------------------

    @staticmethod
    def _dataset_key(ds: dict) -> str:
        return f"{ds['motor']}/{ds['rpm']}"

    def _find_dataset(self, key: str) -> dict:
        for ds in self.datasets:
            if self._dataset_key(ds) == key:
                return ds
        raise ValueError(f"未知資料集：{key}")

    def _pools(self, key: str) -> dict:
        if key not in self._pools_cache:
            self._pools_cache[key] = load_pools(self._find_dataset(key)["path"])
        return self._pools_cache[key]

    def _clean(self, spec: dict, raw: dict) -> dict:
        out: dict[str, Any] = {}
        for p in spec["params"]:
            name, value = p["name"], raw.get(p["name"])
            if p["type"] == "dataset":
                value = value or self._dataset_key(self.datasets[0])
                self._find_dataset(value)
            elif p["type"] == "select":
                value = value or p["default"]
                if value not in p["options"]:
                    raise ValueError(f"{p['label']} 不接受 {value!r}")
            elif p["type"] == "config":
                value = value or p["default"]
            elif p["type"] == "int":
                value = int(p["default"] if value in (None, "") else value)
                if not p["min"] <= value <= p["max"]:
                    raise ValueError(f"{p['label']} 必須在 {p['min']}～{p['max']}")
            out[name] = value
        return out

    # -- 執行 ------------------------------------------------------------------

    def run(self, exp_id: str, raw_params: dict) -> dict:
        spec = _BY_ID.get(exp_id)
        if spec is None:
            raise ValueError(f"未知實驗：{exp_id}")
        params = self._clean(spec, raw_params or {})
        t0 = time.time()
        result = jsonable(self._dispatch(exp_id, params))
        payload = {
            "experiment": exp_id,
            "no": spec["no"],
            "code": spec["code"],
            "params": params,
            "seconds": round(time.time() - t0, 2),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "result": result,
        }
        payload["saved"] = self._save(exp_id, payload)
        return payload

    def _dispatch(self, exp_id: str, p: dict, entry: str = "run", *,
                  scenario: str | None = None) -> Any:
        """呼叫實驗模組的 run()；entry="iter_run" 時改呼叫逐步產生事件的版本。"""
        opts = dict(self.detector_options)
        mahal = opts.get("mahalanobis_method", "ledoit_wolf")
        conf = opts.get("confidence", 0.95)
        k = opts.get("knn_neighbors", 5)
        seed = p["seed"]
        method = p.get("openset_method", "mahalanobis")
        if exp_id == "exp1":
            run = getattr(importlib.import_module("experiments.exp1_cold_start"), entry)
            return run(self._pools(p["dataset"]), seed=seed, confidence=conf, method=mahal,
                       openset_method=method, knn_neighbors=k)
        if exp_id == "exp2":
            module = importlib.import_module("experiments.exp2_scale_growth")
            if entry == "iter_run":
                return module.iter_playback(
                    self._pools(p["dataset"]), seed=seed, confidence=conf, method=mahal,
                    openset_method=method, knn_neighbors=k,
                )
            return module.run(self._pools(p["dataset"]), seed=seed, confidence=conf, method=mahal,
                              openset_method=method, knn_neighbors=k)
        if exp_id == "exp3":
            run = getattr(importlib.import_module("experiments.exp3_trend"), entry)
            kwargs = {}
            if entry == "iter_run" and scenario:
                kwargs["scenarios"] = (scenario,)
            return run(self._pools(p["dataset"]), n_trials=p["n_trials"], seed=seed,
                       confidence=conf, method=mahal, openset_method=method, knn_neighbors=k,
                       **kwargs)
        if exp_id == "exp4":
            run = getattr(importlib.import_module("experiments.exp4_polar_map"), entry)
            return run(self._pools(p["dataset"]), seed=seed)
        if exp_id == "exp5":
            run = getattr(importlib.import_module("experiments.exp5_cross_condition"), entry)
            return run(self.data_root, seed=seed, confidence=conf, method=mahal)
        if exp_id == "exp6":
            run = getattr(importlib.import_module("experiments.exp6_osr_benchmark"), entry)
            return run(self.data_root, seed=seed, confidence=conf)
        if exp_id == "exp6_formal":
            run = getattr(importlib.import_module("experiments.exp6_formal_benchmark"), entry)
            return run(self.data_root, seed=seed, confidence=conf, openset_method=method,
                       mahalanobis_method="ledoit_wolf" if mahal == "legacy" else mahal,
                       knn_neighbors=k, require_nine=False)
        if exp_id == "exp7":
            run = getattr(importlib.import_module("experiments.compare_openset"), entry)
            return run(self._pools(p["dataset"]), seed=seed, confidence=conf,
                       mahalanobis_method=mahal, knn_neighbors=k)
        if exp_id == "exp8":
            run = getattr(importlib.import_module("experiments.health_index_benchmark"), entry)
            return run(self.data_root, seed=seed, openset_method=method,
                       mahalanobis_method=mahal, confidence=conf, knn_neighbors=k,
                       require_nine=False)
        if exp_id == "exp8_monitor":
            run = getattr(importlib.import_module("experiments.health_monitor"), entry)
            ds = self._find_dataset(p["dataset"])
            if p["config"] not in self._pools(p["dataset"]):
                raise ValueError(f"資料集 {p['dataset']} 沒有配置 {p['config']}")
            return run(self.data_root, ds["motor"], ds["rpm"], p["config"], seed=seed,
                       openset_method=method, mahalanobis_method=mahal, knn_neighbors=k,
                       motor_id="web-demo", session_id=f"{p['dataset']}-{p['config']}",
                       max_windows=p["max_windows"], require_identity=True)
        if exp_id == "exp9":
            raise ValueError("實驗九沒有獨立執行器，請改執行實驗六")
        if exp_id == "exp10":
            run = getattr(importlib.import_module("experiments.exp10_transfer"), entry)
            return run(self.data_root, seed=seed, confidence=conf, method=mahal)
        if exp_id == "exp11":
            run = getattr(importlib.import_module("experiments.exp11_ancestor_comparison"), entry)
            return run(self.data_root, seeds=(seed,), confidence=conf)
        if exp_id == "exp12":
            run = getattr(importlib.import_module("experiments.exp12_confusion_tsne"), entry)
            return run(self.data_root, seed=seed, confidence=conf)
        raise ValueError(f"未知實驗：{exp_id}")

    def stream(self, exp_id: str, raw_params: dict):
        """邊跑邊畫：逐步產生 (frame, units)。

        frame 直接送給前端；units 是這個 frame 含的樣本數，伺服器依此控制播放速度
        （units=0 的 frame 不等待）。計算走實驗模組的 iter_run()（實驗二為 iter_playback()），數字與 run() 相同。
        最後一個 frame 是 done，內容與 run() 的回傳值同格式，並已存檔。
        """
        spec = _BY_ID.get(exp_id)
        if spec is None or not spec.get("stream"):
            raise ValueError(f"{exp_id} 不支援邊跑邊畫")
        raw_params = raw_params or {}
        scenario = raw_params.get("scenario") if exp_id == "exp3" else None
        if exp_id == "exp3" and scenario not in (None, "A", "B"):
            raise ValueError(f"scenario: expected A or B, got {scenario!r}")
        params = self._clean(spec, raw_params)
        if scenario:
            params = {**params, "scenario": scenario}
        t0 = time.time()
        yield {"event": "start", "params": params}, 0
        result: Any = None
        windows: list[dict] = []
        for event in self._dispatch(exp_id, params, entry="iter_run", scenario=scenario):
            event = jsonable(event)
            kind = event["event"]
            if kind == "result":
                result = event["result"]
            elif kind == "scores":
                scores = event.pop("scores")
                for i in range(0, len(scores), SCORE_CHUNK):
                    chunk = scores[i:i + SCORE_CHUNK]
                    yield {**event, "offset": i, "scores": chunk}, len(chunk)
            elif kind == "tick":
                # 每個劇本只播第一次重複；其餘重複照算，只送 trial 結果。
                # 實驗二的 tick 沒有 trial，全部播放。
                if event.get("trial", 0) == 0:
                    yield event, 1
            elif kind == "window":
                windows.append(event["window"])
                yield event, 1
            else:
                yield event, 0
        if exp_id == "exp8_monitor":
            result = windows  # 與 health_monitor.run() 的回傳同格式
        payload = {
            "experiment": exp_id,
            "no": spec["no"],
            "code": spec["code"],
            "params": params,
            "seconds": round(time.time() - t0, 2),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "result": result,
            "streamed": True,
        }
        payload["saved"] = self._save(exp_id, payload)
        yield {"event": "done", "payload": payload}, 0

    def _save(self, exp_id: str, payload: dict) -> str | None:
        if self.out_dir is None:
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = self.out_dir / f"{exp_id}_{stamp}.json"
        n = 1
        while path.exists():
            path = self.out_dir / f"{exp_id}_{stamp}_{n}.json"
            n += 1
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            return str(path.resolve().relative_to(ROOT))
        except ValueError:
            return str(path)

    # -- 已提交結果 ------------------------------------------------------------

    def committed(self, name: str) -> dict:
        loaders: dict[str, Callable[[], dict]] = {
            "exp6_formal_matrix": _load_exp6_matrix,
            "exp8_health_index_results": _load_exp8_results,
        }
        if name not in loaders:
            raise ValueError(f"沒有已提交結果：{name}")
        return jsonable(loaders[name]())


def _load_exp6_matrix() -> dict:
    path = ROOT / "output/exp6_formal_matrix/aggregate/aggregate.json"
    if not path.is_file():
        return {"available": False, "path": str(path.relative_to(ROOT))}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "available": True,
        "path": str(path.relative_to(ROOT)),
        "producer": "src/experiments/exp6_matrix.py → src/experiments/aggregate_exp6.py",
        "completed_runs": data.get("completed_runs"),
        "expected_runs": data.get("expected_runs"),
        "seeds": data.get("seeds"),
        "dataset_root": data.get("dataset_root"),
        "macro_summary": data.get("macro_summary", []),
        "condition_summary": data.get("condition_summary", []),
        "paired_differences": data.get("paired_differences", []),
    }


def _load_exp8_results() -> dict:
    base = ROOT / "reports/exp8_health_index_results"
    summary = base / "aggregate_summary.json"
    if not summary.is_file():
        return {"available": False, "path": str(summary.relative_to(ROOT))}
    data = json.loads(summary.read_text(encoding="utf-8"))
    by_condition = []
    csv_path = base / "aggregate_by_condition.csv"
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                row.pop("index", None)
                by_condition.append({k: (v if k in ("method", "dataset") else float(v))
                                     for k, v in row.items()})
    return {
        "available": True,
        "path": str(base.relative_to(ROOT)),
        "producer": "src/experiments/health_index_matrix.py",
        "seeds": data.get("seeds"),
        "rows": data.get("rows"),
        "overall": data.get("overall", {}),
        "difference_knn_minus_mahalanobis": data.get("difference_knn_minus_mahalanobis", {}),
        "by_condition": by_condition,
    }
