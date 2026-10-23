"""可信結果的文字與區段格式；不擬合模型、不重算統計公式。"""

import copy
from datetime import datetime
import hashlib
import json
import math
from zoneinfo import ZoneInfo

from experiments.exp12_confusion_tsne import data_role_counts, summarize_matrices, visualization_counts
from web.run_timing import timing_fields

VERSION = "assistant_presentation_v1"

EXP2_METRICS = {
    "own_holdout_acc": ("群內保留列正確辨識模型新類的比例", "own_holdout_n"),
    "eval_known_rate": ("本段配置外部評估列被任一新類接受的比例", "eval_n"),
    "eval_matched_rate": ("本段配置外部評估列被判進事後多數等於該配置的新類的比例（事後對照）", "eval_n"),
    "healthy_holdout_acc": ("健康保留列仍被判健康的比例", None),
    "cluster_purity": ("候選群中真值多數配置的占比（事後對照）", "cluster_size"),
}
# 重播結束後各配置外部評估列（result["final_eval"][config]）的比例欄位
FINAL_EVAL_METRICS = {
    "known_rate": ("重播結束後該配置外部評估列被任一新類接受的比例", "n"),
    "matched_rate": ("重播結束後該配置外部評估列被判進事後多數等於該配置的新類的比例（事後對照）", "n"),
    "unknown_rate": ("重播結束後仍被拒絕為未知；不是學會前未知召回", "n"),
}


def _metric_row(container: dict, key: str, meaning: str, denominator_key, **where) -> dict:
    value = container[key]
    valid = (isinstance(value, (int, float)) and not isinstance(value, bool)
             and math.isfinite(value) and 0 <= value <= 1)
    denominator = container.get(denominator_key) if denominator_key else None
    # 沒有分母欄位的比例（healthy_holdout_acc）只看值本身；有分母欄位才要求正整數
    count_valid = denominator_key is None or (isinstance(denominator, int) and not isinstance(denominator, bool)
                                              and denominator > 0)
    status = "VERIFIED_SAVED_FIELDS" if valid and count_valid else "INCOMPLETE"
    if denominator is not None and not count_valid:
        status, valid = "INVALID_DENOMINATOR", False
    return {**where, "metric": key, "meaning": meaning, "value": value if valid and count_valid else None,
            "denominator": denominator if count_valid else None, "denominator_required": denominator_key is not None,
            "status": status}


def exp2_metric_contract(result, budget=6000):
    """只標示既有比例的資料角色與分母欄位；不重算統計或補推未知指標。"""
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ValueError("摘要預算必須為正整數")
    contract = {"version": "exp2_metric_semantics_v2", "status": "complete", "rows": [], "tables": [],
                "unknown_auroc": "NOT_PROVIDED", "unknown_recall": "NOT_PROVIDED"}
    lines = ["【實驗二指標契約：保存結果的事後評估，不是即時配置辨識】",
             "比例原值介於 0 與 1；分母只讀保存欄位，缺值不由示例或其他實驗補入。",
             "own_holdout_acc 不等於未知召回；eval_known_rate 不要求判對自身配置；純度不參與線上病因判定。",
             "本份彙總未提供可核算的未知 AUROC／未知召回，不能從群內準確率推估。"]
    def saved_rows(item, definitions, timepoint, index, config):
        selected = {}
        for key, (meaning, denominator_key) in definitions.items():
            if key not in item:
                continue
            row = _metric_row(item, key, meaning, denominator_key,
                              stage_index=index if timepoint == "stage" else "final_eval",
                              config=config, configuration=config, timepoint=timepoint)
            if row["status"] != "VERIFIED_SAVED_FIELDS":
                contract["status"] = "incomplete"
            contract["rows"].append(row)
            selected[key] = row
        return [(f"{selected[key]['value']}（n={selected[key]['denominator']}）" if selected[key]["denominator_required"] else
                 f"{selected[key]['value']}（此欄未定義分母）")
                if key in selected and selected[key]["status"] == "VERIFIED_SAVED_FIELDS" else
                selected[key]["status"] if key in selected else "未保存" for key in definitions]

    stages = result.get("stages") if isinstance(result, dict) else None
    stage_table = {"id": "stages", "title": "各段學會當下的保存欄位",
                   "headers": ["階段（零起算）", "注入配置（事後）", "候選群多數（事後）", "對上本段配置（事後）",
                               *EXP2_METRICS], "rows": []}
    if not isinstance(stages, list) or not stages:
        contract["status"] = "incomplete"
        lines.append("缺少有效 stages，未列推測數字。")
    else:
        for index, stage in enumerate(stages):
            if not isinstance(stage, dict):
                contract["status"] = "incomplete"
                continue
            config = stage.get("config") if isinstance(stage.get("config"), str) else None
            values = saved_rows(stage, EXP2_METRICS, "stage", index, config)
            if any(key in stage for key in EXP2_METRICS):
                matched = stage.get("matched")
                is_config = stage.get("matched_is_config")
                stage_table["rows"].append([index, config or "未保存配置",
                    matched if isinstance(matched, str) else "未保存",
                    ("是" if is_config else "否") if isinstance(is_config, bool) else "未保存", *values])
    final_table = {"id": "final", "title": "全部階段結束後的外部評估",
                   "headers": ["事後配置", *FINAL_EVAL_METRICS], "rows": []}
    final = result.get("final_eval") if isinstance(result, dict) else None
    if isinstance(final, dict):
        for config, item in final.items():
            if not isinstance(item, dict):
                contract["status"] = "incomplete"
                continue
            values = saved_rows(item, FINAL_EVAL_METRICS, "final", None, config)
            if any(key in item for key in FINAL_EVAL_METRICS):
                final_table["rows"].append([f"final_eval：{config}", *values])
    else:
        contract["status"] = "incomplete"
        lines.append("未保存有效final_eval，不由stage資料補出全部階段結束後的結果。")
    totals = {k: result.get(k) for k in ("n_stages", "n_learned", "n_matched_config")} if isinstance(result, dict) else {}
    if totals and all(isinstance(v, int) and not isinstance(v, bool) for v in totals.values()):
        # #286：形成新類（discovered）不等於辨識出該段配置；兩個總數由實驗層算好，這裡只轉述
        contract["totals"] = totals
        lines.append(f"形成新類 {totals['n_learned']}／{totals['n_stages']} 段；對上本段配置 "
                     f"{totals['n_matched_config']}／{totals['n_stages']} 段（matched_is_config，事後對照）。"
                     "形成新類只表示該段學到一個 discovered-N，不表示辨識出該段配置。")
    if not contract["rows"]:
        contract["status"] = "incomplete"
    contract["tables"] = [table for table in (stage_table, final_table) if table["rows"]]
    for table in contract["tables"]:
        lines += [table["title"], table_text(table["headers"], table["rows"])]
    lines.append(table_text(["欄位", "意思"], [[key, value[0]] for key, value in
        [*EXP2_METRICS.items(), *FINAL_EVAL_METRICS.items()]]))
    lines += ["VERIFIED_SAVED_FIELDS 只表示比例有效且符合該欄分母規格；未定義分母的欄位不補造分母，不表示採集獨立或模型可靠。",
              "stage與final_eval的時點分開；known_rate不要求判對自身配置，matched_rate才要求對應，但屬事後對照，不是線上病因辨識。",
              "matched_rate 才要求判進事後多數等於該配置的新類；這仍是事後對照。",
              "缺值、缺必要分母或無效欄位不顯示數值；未從另一欄、另一時點或rounded比例反推成功筆數。",
              "confidence=0.95 是校準距離分位數，不是 95% 正確機率；187／62／64 只屬手冊的指定示例。",
              "同池 20% 外部列未參與重播，仍不等於新採集或 fresh final；相對量尺不是損壞百分比或 RUL。"]
    text = "\n\n".join(lines)
    if len(text) > budget:
        contract["status"] = "incomplete"
        contract["budget_exceeded"] = True
        warning = "【摘要不完整】實驗二來源、指標及分母超出摘要預算；請讀指定原檔，不截出數字推估未知召回或 AUROC。"
        text = warning if len(warning) <= budget else "不完整" if budget >= 3 else "略"
    contract["text"] = text
    return contract


def string(value):
    if value is None:
        return "未提供"
    if isinstance(value, bool) or isinstance(value, float) and not math.isfinite(value):
        return "值無效"
    return str(value)


def escape(value):
    text = string(value)
    for char in ("\\", "|", "`", "*", "[", "]", "(", ")", "~", "#", ">"):
        text = text.replace(char, "\\" + char)
    return text.replace("\r", "\\r").replace("\n", "\\n")


def table_text(headers, rows):
    def line(values):
        return "| " + " | ".join(escape(v) for v in values) + " |"
    return "\n".join([line(headers), line(["---"] * len(headers)), *(line(row) for row in rows)])


def rate(value):
    if value["percent"] is None:
        return f"不適用（分子={string(value['numerator'])}，分母={string(value['denominator'])}）"
    return f"{value['numerator']}／{value['denominator']}＝{value['percent']:.2f}%"


def section(key, title, kind, *, detail=False, **content):
    return {"id": key, "title": title, "detail": detail, "open": False, "kind": kind, **content}


def source_id(source):
    # 僅作呈現來源綁定；不冒充原始檔案SHA，後者只讀saved_sha256。
    return hashlib.sha256(json.dumps(source, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def display_time(value):
    if not value:
        return "原檔未記錄"
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.utcoffset() is None:
            return value + "（原檔未提供時區）"
        return dt.astimezone(ZoneInfo("Asia/Taipei")).isoformat(sep=" ", timespec="milliseconds")[:-6]
    except (ValueError, TypeError, AttributeError):
        return string(value) + "（格式無效，未推算）"


def source_sections(source):
    timing = timing_fields(source)
    warnings = list(dict.fromkeys([*timing["timing_warnings"], *(
        v for v in source.get("timing_warnings", []) if isinstance(v, str))]))
    if timing["started_at"] is None:
        warnings.append("開始時間：原檔未記錄；沒有由完成時間或檔名推算。")
    if "原檔未提供時區" in timing["timing_text"]:
        warnings.append("原檔未提供時區；保留原值，未擅自轉成台灣時間。")
    scope = timing["timing_scope"]
    mode = source.get("mode")
    method = "串流播放" if scope == "stream_elapsed_including_playback" or source.get("streamed") is True else (
        "批次執行" if scope == "batch_execution" or mode == "run" else "原檔未記錄")
    quality = {"VERIFIED": "起訖與量測耗時一致（僅時間欄位檢查）", "INCONSISTENT": "時間紀錄不一致",
               "INCOMPLETE": "時間資訊不足，無法完整檢查"}[timing["timing_status"]]
    scope_text = {"batch_execution": "包含載入、運算與結果轉換；不含存檔及HTTP傳輸。",
                  "stream_elapsed_including_playback": "包含播放等待的串流經過時間，不是純模型計算耗時。"}.get(scope, "原檔未提供耗時邊界。")
    rows = [["實驗", "實驗十二｜混淆矩陣與 t-SNE" if source.get("exp_id") == "exp12" else string(source.get("exp_id"))],
            ["狀態", {"done": "已完成", "running": "執行中", "error": "失敗", "stopped": "已停止", "incomplete": "未完整完成"}.get(source.get("status"), "未提供")],
            ["執行方式", method], ["紀錄來源", "讀取既有存檔" if mode == "saved" else "本次執行紀錄" if mode in ("run", "stream") else "未提供"],
            ["開始時間", display_time(timing["started_at"])], ["完成時間", display_time(timing["finished_at"])],
            ["顯示時區", "明確時區者使用Asia/Taipei（UTC+08:00）；無時區者保留原值"],
            ["耗時", f"{timing['seconds']} 秒" if timing["seconds"] is not None else "原檔未記錄或數值無效"],
            ["時間檢查", quality], ["耗時範圍", scope_text]]
    source_detail = {"存檔位置": source.get("saved") or "未提供", "原檔SHA256": source.get("saved_sha256") or "未提供",
                     "安全來源物件": source}
    return [section("source", "來源與執行資訊", "fields", rows=rows),
            section("source_details", "存檔位置與技術詳情", "code", detail=True,
                    text=json.dumps(source_detail, ensure_ascii=False, indent=2, allow_nan=False),
                    copies=[["存檔位置", source.get("saved") or "未提供"], ["原始來源", json.dumps(source, ensure_ascii=False, indent=2, allow_nan=False)]])], warnings


def markdown(presentation):
    out = ["# " + escape(presentation["title"]), presentation["subtitle"]]
    for s in presentation["sections"]:
        out += ["", "## " + escape(s["title"]), ""]
        if s["kind"] in ("list", "warning"):
            out += ["- " + escape(v) for v in s["items"]]
        elif s["kind"] == "fields":
            out += ["- " + escape(k) + "：" + escape(v) for k, v in s["rows"]]
        elif s["kind"] == "table":
            out.append(table_text(s["headers"], s["rows"]))
        elif s["kind"] == "code":
            out += ["```json", s["text"], "```"]
    return "\n".join(out)


def new_presentation(source, mode, title, subtitle, sections, status="complete"):
    return {"schema_version": VERSION, "mode": mode, "origin": "matrix" if mode == "matrix_summary" else "record",
            "source_id": source_id(source), "source": source, "title": title, "subtitle": subtitle,
            "status": status, "sections": sections}


def data_usage_contract(result):
    """只讀保存的模型角色筆數，不由比例、示例或其他實驗推算。"""
    model = result.get("model") if isinstance(result, dict) else None
    classes = model.get("classes") if isinstance(model, dict) else None
    rows = []
    if isinstance(classes, list):
        for item in classes:
            if not isinstance(item, dict) or not isinstance(item.get("config"), str):
                continue
            counts = [item.get(key) for key in ("n_train", "n_cal", "n_holdout")]
            if all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in counts):
                rows.append([item["config"], *counts])
    return {"version": "saved_model_usage_v1", "rows": rows,
            "status": "complete" if rows and len(rows) == len(classes) else "incomplete"}


def data_usage_presentation(contract, source, sections):
    """方法規則、文件示例與當次保存筆數分欄，不交給模型再改寫。"""
    references = [s for s in sections if s["source"] == "methods.md"]
    roles = [s for s in references if "60/20/20" in s["body"]]
    confidence = [s for s in references if "不是每筆 95% 正確機率" in s["body"]]
    rules = []
    if roles:
        rules += ["已知資料的共通切分約定是60/20/20（train／calibration／holdout）；整數筆數以make_split為準，不要求整數分配後比例恰好相等。",
                  "187／62／64只屬T1／8000rpm健康池313列的指定示例，不是本次或所有工況的固定筆數。",
                  "初始健康冷啟動的未知配置不參與擬合或校準；候選新類使用通過規則的群，不能套用完整配置池的示例筆數。",
                  "實驗二每配置預留的20%外部評估，與健康資料20%校準的母群不同，不可互換。"]
    if confidence:
        rules.append("confidence=0.95是已知校準距離分位數，不是分類準確率，也不是每筆95%正確機率或跨馬達固定5%誤報保證；legacy的訓練距離門檻另按該方法定義。")
    if not roles or not confidence:
        rules.append("方法文件未提供完整可核對的切分／校準說明，未用一般常識或其他實驗補值。")
    body = [section("rules", "規則與示例", "list", items=rules)]
    rows = contract.get("rows", []) if source.get("status") == "done" else []
    if rows:
        body.append(section("usage", "本次已保存資料（模型摘要時點）", "table",
                            headers=["模型類別", "train", "calibration", "holdout"], rows=rows))
        if contract["status"] != "complete":
            body.append(section("usage_partial", "保存筆數不足", "warning",
                                items=["部分模型類別的角色筆數缺失或無效；只列欄位齊全的類別，不補值。"]))
    else:
        body.append(section("usage_missing", "本次已保存資料", "warning",
                            items=["本次未保存可核對的模型切分筆數，或該次紀錄尚未完成；無法回答當次實際用量，未以手冊示例代入。"]))
    body.append(section("limits", "限制", "list", items=[
        "筆數只讀此來源的model.classes保存欄位；不從比例反推，也不從exp12示例、歷史回答或另一份run補入。",
        "列級切分互斥不證明錄製、raw視窗或跨馬達獨立，也不證明可靠部署。"] ))
    source_body, _ = source_sections(source)
    body.extend(source_body)
    return new_presentation(source, "data_usage", "資料切分與校準說明", "由程式讀取方法文件及同一筆保存欄位；不是模型自由生成。",
                            body, "complete" if rows and contract["status"] == "complete" and roles and confidence else "incomplete")


def exp2_metric_presentation(contract, source):
    """共用已驗證的欄位字典與表格；不重算接受率或推估未知指標。"""
    body = [section("definitions", "指標意思與母群", "table", headers=["欄位", "意思"], rows=[
        [key, value[0]] for key, value in [*EXP2_METRICS.items(), *FINAL_EVAL_METRICS.items()]])]
    if contract.get("budget_exceeded"):
        body.append(section("budget", "摘要不足", "warning", items=[contract["text"]]))
    else:
        body += [section(table["id"], table["title"], "table", headers=table["headers"], rows=table["rows"])
                 for table in contract["tables"]]
    body.append(section("limits", "限制", "warning", items=[
        "保存比例介於0與1；n只讀同欄保存分母。未定義分母的合法比例保留，缺必要分母或無效欄位不提供數值。",
        "被任一新類接受不等於判進對應配置。matched／純度只屬事後對照，不是線上病因辨識。",
        "final_eval.unknown_rate是學習後仍拒絕，不能改名為學會前未知召回；本份彙總未提供可核算的未知召回或AUROC。",
        "資料或分母缺失不由其他比例、時點或示例補入；同池外部保留不證明新採集或可靠部署。"] ))
    source_body, _ = source_sections(source)
    return new_presentation(source, "exp2_metrics", "實驗二保存指標解讀", "由程式讀取同一份保存欄位；不是模型自由生成。",
                            [*body, *source_body], contract["status"])


def timing_presentation(source):
    sections, warnings = source_sections(source)
    if warnings:
        sections.insert(1, section("warnings", "時間注意事項", "warning", items=warnings))
    return new_presentation(source, "record_timing", "實驗執行時間", "由程式讀取同一筆執行紀錄；缺值不推算。", sections)


def build_exp12(result, budget, source, schema):
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ValueError("摘要預算必須為正整數")
    source = source or {"exp_id": "exp12"}
    sections, warnings = source_sections(source)
    params = source.get("params") or {}
    status = "complete"
    try:
        if isinstance(result, dict) and any(k in params and k in result and params[k] != result[k]
                                            for k in ("seed", "confidence", "perplexity")):
            raise ValueError("執行參數與結果設定不一致，需核對來源")
        facts = summarize_matrices(result)
    except ValueError as exc:
        status, facts = "invalid", None
        warnings.insert(0, f"矩陣核算失敗：{exc}；不提供推測數字。")
    settings = []
    for key, label in (("seed", "隨機種子"), ("confidence", "校準信心水準"), ("n_datasets", "工況數"), ("perplexity", "t-SNE perplexity")):
        value = result.get(key) if isinstance(result, dict) else None
        if value is None:
            value = params.get(key)
        valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        if key == "confidence":
            valid = valid and 0 < value < 1
        display = f"{value:.0%}（不是分類準確率）" if key == "confidence" and valid else string(value) if valid else "未提供或值無效"
        if isinstance(result, dict) and key in params and key in result and params[key] != result[key]:
            display = f"執行參數{string(params[key])}；結果設定{string(result[key])}（衝突）"
        settings.append([label, display])
    sections.insert(1, section("settings", "執行設定", "fields", rows=settings))
    if facts is not None:
        summary = []
        if "binary" not in facts:
            warnings.append("資料不完整：缺少二元矩陣，無法核算健康誤報與故障漏判。")
            status = "incomplete"
        else:
            f, b = facts["binary"], facts["binary"]["binary"]
            summary.append(f"健康誤報{rate(b['healthy_false_positive_rate'])}；故障漏判為健康{b['fault_missed']}／{b['fault_total']}。")
            rows = [["健康樣本", string(b["healthy_total"]), "健康測試支持數"], ["故障樣本", string(b["fault_total"]), "未知故障測試支持數"],
                    ["合計", string(f["total"]), "本次二元評估總量"], ["健康誤報", rate(b["healthy_false_positive_rate"]), "健康被判成故障"],
                    ["故障漏判為健康", rate(b["fault_missed_rate"]), "故障被接受為健康"], ["故障召回", rate(b["fault_recall"]), "故障被拒絕為未知"]]
            sections.append(section("binary", "二元健康冷啟動結果", "table", headers=["項目", "筆數／比例", "白話意思"], rows=rows))
        if not facts["openset"]:
            warnings.append("資料不完整：缺少多類開集矩陣，無法核算已知分類與未知拒絕。")
            status = "incomplete"
        else:
            methods = facts["openset"]
            roles = {(tuple(sorted(f["openset"]["known_labels"])), f["openset"]["known_total"], f["openset"]["unknown_total"])
                     for f in methods.values()}
            if len(roles) > 1:
                warnings.append("兩方法的已知標籤或支持數不同；僅並列原值，不可當相同資料角色的配對比較。")
            names = {"ledoit_wolf": "Ledoit–Wolf（LW）", "legacy": "legacy"}
            metrics = [("已知樣本", lambda o: string(o["known_total"])), ("已知判對", lambda o: rate(o["known_accuracy"])),
                       ("已知誤拒為未知", lambda o: rate(o["known_rejection_rate"])), ("已知類互相錯分", lambda o: string(o["known_misclassified"])),
                       ("未知正確拒絕", lambda o: rate(o["unknown_recall"])), ("未知誤接受為已知", lambda o: rate(o["unknown_acceptance_rate"]))]
            sections.append(section("methods", "多類開集方法比較", "table", headers=["指標", *(names.get(k, k) for k in methods)],
                                    rows=[[label, *(func(f["openset"]) for f in methods.values())] for label, func in metrics]))
            if "ledoit_wolf" in methods:
                o = methods["ledoit_wolf"]["openset"]
                summary.append(f"LW已知類判對{rate(o['known_accuracy'])}，誤拒{o['known_rejected']}筆；此已知集合包含健康。")
            if "legacy" in methods:
                o = methods["legacy"]["openset"]
                summary.append(f"legacy未知正確拒絕{rate(o['unknown_recall'])}、已知判對{rate(o['known_accuracy'])}、已知誤拒{rate(o['known_rejection_rate'])}；不能只看未知召回判斷好壞，也不能僅此確認邊界過嚴。")
            for method, f in methods.items():
                o = f["openset"]
                if not o["unknown_present"]:
                    warnings.append(f"{method}缺unknown標籤；未知與誤拒指標不適用，不能推定未知完全判對。")
                rows = [[label + ("（健康）" if label == "8screws" else ""), string(v["support"]), string(v["correct"]),
                         string(v.get("rejected")), string(v.get("misclassified")), rate(v["recall"])]
                        for label, v in f["per_class"].items() if label != o["unknown_label"]]
                sections.append(section("classes-" + method, names.get(method, method) + "逐類分類結果", "table", detail=True,
                                        headers=["配置", "樣本數", "判對", "誤拒", "錯分其他已知", "判對率"], rows=rows))
        sections.insert(2, section("summary", "重點結論", "list", items=summary[:3]))
        vis = visualization_counts(result)
        if vis["status"] == "VERIFIED":
            sections.append(section("tsne", "t-SNE實際抽樣", "table", detail=True, headers=["工況", "實際筆數"],
                                    rows=[*[list(pair) for pair in vis["datasets"].items()], ["合計", vis["total"]]]))
        else:
            warnings.append("t-SNE實際數量未知／無法核對，未用2000或矩陣樣本數代替。")
        sections.append(section("legend", "t-SNE抽樣與圖例", "list", detail=True, items=[
            "現行預設每工況每配置最多抽200筆，完整十配置時每工況最多2000筆，並非全部工況合計2000筆；上限不等於實際量。",
            "顏色＝真實配置。三角形表示模型判為unknown，包含已知誤拒與未知正確拒絕。", "圓點＝已知判對；叉＝錯誤接受，包含未知被判為已知。",
            "各工況分別fit，t-SNE可能含train／cal資料，只供事後觀察；群距無物理損傷意義。"] ))
        roles = data_role_counts(313)
        sections.append(section("roles", "資料使用與長版限制", "list", detail=True, items=[
            "二元只學健康；多類A協定的目標是健康＋四種已知故障配置，實際已知標籤以逐類表為準。缺類不代表完整coverage。",
            f"資料角色說明示例（T1／8000rpm，非當次全部工況統計）：健康池{roles['pool']}筆，train擬合{roles['train']}／cal校準{roles['cal']}／holdout測試{roles['holdout']}，不是整池皆用來訓練。",
            "本次為批次實驗：模型產生預測，再以事後真值計算指標；不是沒有模型判斷。",
            "故障漏判為健康、健康誤報、已知誤拒與未知未拒是不同指標；未知全拒不代表整個系統沒有誤報。",
            "不使用同一test調門檻；legacy的-1標籤介面限制須核對來源程式版本，本紀錄未提供可核驗版本，不能僅憑全unknown確認原因；不支持原方法完全失效。"] ))
        for key, label in (("binary_confusion", "二元"), ("openset_confusion", "多類")):
            entries = {"binary": result[key]} if key == "binary_confusion" and key in result else result.get(key, {})
            for name, entry in entries.items():
                sections.append(section("matrix-" + name, f"原始混淆矩陣：{label} {name}（橫列＝真值；直欄＝預測）", "table", detail=True,
                                        headers=["真值＼預測", *entry["labels"]], rows=[[label, *row] for label, row in zip(entry["labels"], entry["matrix"])]))
    if warnings:
        sections.insert(2, section("warnings", "注意事項", "warning", items=list(dict.fromkeys(warnings))))
    sections.insert(3, section("limits", "必要限制", "list", items=[
        "已知判對率包含健康，不是純故障正確率；二元與多類分母不能混加。",
        "unknown→unknown是正確拒絕，沒有辨識具體未知配置身分。",
        "這是本批資料的事後評估，不證明換到未見馬達仍可靠。"] ))
    # 詳情最後呈現；同一份區段也產生文字fallback，沒有第二套核算。
    order = {key: i for i, key in enumerate(("source", "settings", "summary", "binary", "methods", "warnings", "limits"))}
    sections = sorted((s for s in sections if not s["detail"]), key=lambda s: order.get(s["id"], 99)) + [
        s for s in sections if s["detail"] and s["id"] != "source_details"] + [s for s in sections if s["id"] == "source_details"]
    p = new_presentation(source, "matrix_summary", "實驗十二結果摘要", "由程式依矩陣核算；事後評估。", sections, status)
    text = markdown(p)
    if len(text) > budget:
        fallback = "【摘要不完整】來源、分母與限制無法完整容納；請讀指定原存檔，不據此推算。"
        return {"schema_version": schema, "status": "incomplete", "text": fallback if len(fallback) <= budget else "不完整" if budget >= 3 else "略"}
    return {"schema_version": schema, "status": status, "text": text, "presentation": p}


def select_view(presentation, view):
    p = copy.deepcopy(presentation)
    if view == "lw_classes":
        ids = {"source", "source_details", "warnings", "limits", "classes-ledoit_wolf"}
        p["sections"] = [s for s in p["sections"] if s["id"] in ids]
        p["title"] = "Ledoit–Wolf逐類結果"
        if not any(s["id"] == "classes-ledoit_wolf" for s in p["sections"]):
            if p["status"] == "complete":
                p["status"] = "incomplete"
            p["sections"].insert(1, section("missing-lw", "資料不完整", "warning", items=["本紀錄缺少LW逐類結果，未改取另一筆來源。"]))
        for s in p["sections"]:
            if s["id"] == "classes-ledoit_wolf":
                s.update(detail=False, open=True)
    elif view == "tsne":
        p["sections"] = [s for s in p["sections"] if s["id"] in {"source", "source_details", "warnings", "limits", "tsne", "legend", "roles"}]
        p["title"] = "t-SNE抽樣與圖例"
        for s in p["sections"]:
            if s["id"] in ("tsne", "legend"):
                s.update(detail=False, open=True)
    elif view == "full":
        for s in p["sections"]:
            s["open"] = True
    return p
