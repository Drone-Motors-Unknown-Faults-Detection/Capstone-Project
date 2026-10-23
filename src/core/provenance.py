"""建立 data-root 下 Step-*/…/*_Group_feature_data_clean.csv 的 SHA256 內容來源清冊，並與 formal_materialization_manifest.json 比對；不載入模型。"""
from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import tempfile
import uuid

import numpy as np

CONTENT_VERSION = "source_content_v2"
MATERIALIZATION_VERSION = "formal_materialization_v3"


class ProvenanceError(ValueError):
    """可公開的來源錯誤，不含任意OS例外payload。"""


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """分塊讀檔算 SHA256；讀不到就丟原本的 OSError。專案內所有檔案雜湊都走這裡（#276 K03）。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def file_sha(path: Path) -> str:
    """sha256_file 的來源驗證版：OSError 包成可公開的 ProvenanceError，不帶路徑。"""
    try:
        return sha256_file(path)
    except OSError as exc:
        raise ProvenanceError(f"來源無法讀取：{type(exc).__name__}") from exc


def _manifest_id(value: str, *, legacy: bool) -> str:
    if not isinstance(value, str) or not value:
        raise ProvenanceError("manifest缺少output ID")
    windows = PureWindowsPath(value)
    posix = PurePosixPath(value)
    absolute = windows.is_absolute() or posix.is_absolute() or bool(windows.drive)
    if legacy and absolute:
        parts = windows.parts if "\\" in value or windows.drive else posix.parts
        starts = [i for i, part in enumerate(parts) if part.startswith("Step-")]
        if len(starts) != 1:
            raise ProvenanceError("legacy output無唯一Step相對ID")
        value = "/".join(parts[starts[0]:])
    elif absolute or "\\" in value:
        raise ProvenanceError("output ID必須為portable相對路徑")
    parts = value.split("/")
    if any(part in {"", ".", ".."} or ":" in part for part in parts):
        raise ProvenanceError("output ID含不合法路徑節")
    if len(parts) != 6 or not parts[0].startswith("Step-") or parts[1] != "myfeature":
        raise ProvenanceError("output ID不符合正式檔案層級")
    return "/".join(parts)


def capture_source(data_root: Path | str) -> tuple[dict, dict]:
    """回傳(public, private)；副本辨識不構成採集獨立證明。"""
    root = Path(data_root).expanduser().resolve()
    files = [{"source_id": path.relative_to(root).as_posix(), "sha256": file_sha(path)}
             for path in sorted(root.glob("Step-*/myfeature/*/*/*/*_Group_feature_data_clean.csv"))]
    if not files:
        raise FileNotFoundError("找不到正式特徵CSV；不能對空清冊宣稱來源已驗證")
    manifest_path = root / "formal_materialization_manifest.json"
    manifest_schema, manifest_sha, validation = None, None, "ABSENT"
    if manifest_path.is_file():
        try:
            payload = manifest_path.read_bytes()
            manifest = json.loads(payload)
        except (OSError, ValueError) as exc:
            raise ProvenanceError(f"manifest損壞或無法讀取：{type(exc).__name__}") from exc
        if not isinstance(manifest, dict):
            raise ProvenanceError("manifest必須為object")
        manifest_schema = manifest.get("schema_version")
        if (not isinstance(manifest_schema, (type(None), int, str)) or isinstance(manifest_schema, bool)
                or manifest_schema not in {None, 1, "formal_materialization_v1", "formal_materialization_v2", MATERIALIZATION_VERSION}):
            raise ProvenanceError("未知materialization schema")
        if manifest_schema is None:
            if not all(key in manifest for key in ("archive_sha256", "formal_contract", "files")):
                raise ProvenanceError("未版本化manifest缺少歷史識別欄位")
            manifest_schema = "legacy_unversioned_materialization"
        records = manifest.get("files")
        if not isinstance(records, list) or not records:
            raise ProvenanceError("manifest缺少非空files清冊")
        declared = {}
        for record in records:
            if not isinstance(record, dict):
                raise ProvenanceError("manifest file不是object")
            identity = _manifest_id(record.get("output"), legacy=manifest_schema != MATERIALIZATION_VERSION)
            if identity in declared:
                raise ProvenanceError("manifest重複output ID")
            checksum = record.get("output_sha256") if manifest_schema == MATERIALIZATION_VERSION else record.get("source_sha256")
            if not isinstance(checksum, str) or len(checksum) != 64:
                raise ProvenanceError("manifest缺少已宣告的output SHA")
            declared[identity] = checksum
        actual = {record["source_id"]: record["sha256"] for record in files}
        if declared != actual:
            raise ProvenanceError("manifest output清冊／內容SHA不符")
        validation = "VERIFIED_OUTPUT_BYTES_ONLY"
        manifest_sha = hashlib.sha256(payload).hexdigest()
    aliases = defaultdict(list)
    for record in files:
        aliases[record["sha256"]].append(record["source_id"])
    public = {"schema_version": CONTENT_VERSION, "dataset_fingerprint": canonical_sha(files),
              "data_version": f"{CONTENT_VERSION}:{canonical_sha(files)}",
              "files": files, "manifest_schema": manifest_schema, "manifest_sha256": manifest_sha,
              "manifest_validation": validation,
              "content_aliases": [ids for _, ids in sorted(aliases.items()) if len(ids) > 1],
              "sample_source": "relative_file_id_plus_original_row_index",
              "raw_session_independence": "UNKNOWN", "fresh_test": "UNKNOWN"}
    private = {"schema_version": "private_source_context_v1", "resolved_data_root": str(root),
               "resolved_manifest": str(manifest_path) if manifest_path.is_file() else None,
               "dataset_fingerprint": public["dataset_fingerprint"], "fingerprint_version": CONTENT_VERSION}
    return public, private


def write_private_context(context: dict, *, directory: Path | str = ".lineage_private") -> str:
    """原子寫入本機明文sidecar；只回傳不含路徑的opaque ID。"""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    identity = uuid.uuid4().hex
    destination = root / f"{identity}.json"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(context, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return identity


def pools_sha(pools: dict) -> str:
    """僅用於綁定已載入陣列；不當成原始錄製或檔案 SHA。"""
    digest = hashlib.sha256()
    for name, values in pools.items():
        array = np.asarray(values, dtype="<f8", order="C")
        digest.update(canonical_sha([name, list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class PoolSource:
    """同一次載入的公開清冊與本機驗證資訊；不得直接序列化整個物件。"""

    public: dict
    context_sha256: str
    pools_sha256: str
    root: Path = field(repr=False)


def load_pool_source(data_root, dataset_path, *, cache=None) -> tuple[dict, PoolSource]:
    """以 parser 實際讀到的 bytes SHA 綁定池與 CSV 原列；載入變動時拒絕。"""
    from core.data import CONFIG_ALIASES, load_pools

    root = Path(data_root).resolve()
    dataset = Path(dataset_path).resolve()
    try:
        prefix = dataset.relative_to(root).as_posix()
    except ValueError:
        raise ProvenanceError("資料集不在所宣告來源根內") from None
    before, private = capture_source(root)
    audit = {}
    try:
        pools = (cache.load(dataset, audit=audit) if cache is not None
                 else load_pools(dataset, audit=audit))
    except (OSError, ValueError) as exc:
        raise ProvenanceError(f"資料載入失敗：{type(exc).__name__}") from None
    after, _ = capture_source(root)
    if before != after:
        raise ProvenanceError("資料載入期間來源變動")
    files = {row["source_id"]: row["sha256"] for row in before["files"]}
    mapping, offsets = [], defaultdict(int)
    for row in audit["files"]:
        source_id = f"{prefix}/{row['file_id']}"
        if files.get(source_id) != row["sha256"]:
            raise ProvenanceError("parser／快取實際載入內容與來源 SHA 不符")
        physical_config = row["file_id"].split("/")[0]
        config = CONFIG_ALIASES.get(physical_config, physical_config)
        start = offsets[config]
        offsets[config] += row["rows"]
        mapping.append({"configuration": config, "source_id": source_id,
                        "sha256": row["sha256"], "pool_start": start,
                        "pool_stop": offsets[config], "csv_row_start": 0,
                        "csv_row_stop": row["rows"], "schema_version": row["schema_version"]})
    if dict(offsets) != {key: len(value) for key, value in pools.items()}:
        raise ProvenanceError("來源列映射與資料池長度不符")
    private["resolved_dataset"] = str(dataset)
    try:
        identity = write_private_context(private)
    except OSError as exc:
        raise ProvenanceError(f"私人來源紀錄無法保存：{type(exc).__name__}") from None
    public = {**before, "pool_source_schema": "pool_source_v1", "status": "VERIFIED",
              "dataset_id": prefix, "private_context_id": identity,
              "row_mapping": mapping,
              "row_semantics": "zero_based_clean_csv_data_rows_not_raw_time_windows"}
    return pools, PoolSource(public, canonical_sha(public), pools_sha(pools), root)


def source_for_pools(pools, source: PoolSource | None = None) -> dict:
    """記憶體 API 明示 UNKNOWN；有來源則核對清冊、陣列與目前磁碟。"""
    if source is None:
        return {"pool_source_schema": "pool_source_v1", "status": "UNVERIFIED_MEMORY",
                "data_version": "UNKNOWN", "files": [], "row_mapping": [],
                "raw_session_independence": "UNKNOWN", "fresh_test": "UNKNOWN"}
    if not isinstance(source, PoolSource):
        raise ProvenanceError("來源必須由 load_pool_source 取得")
    if canonical_sha(source.public) != source.context_sha256 or pools_sha(pools) != source.pools_sha256:
        raise ProvenanceError("來源清冊或已載入陣列被修改")
    current, _ = capture_source(source.root)
    if any(current[key] != source.public[key] for key in current):
        raise ProvenanceError("來源內容已變動，拒絕替舊資料池背書")
    return copy.deepcopy(source.public)


def run(data_root: Path | str) -> dict:
    from core.logger import setup_run

    log, paths = setup_run("source_provenance", unique=True, data_root=data_root)
    try:
        public, private = capture_source(data_root)
        public["private_context_id"] = write_private_context(private)
    except (ProvenanceError, FileNotFoundError, OSError) as exc:
        public = {"schema_version": CONTENT_VERSION, "status": "FAILED", "error_type": type(exc).__name__,
                  "raw_session_independence": "UNKNOWN"}
        (paths.output_dir / "source.json").write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    public["status"] = "VERIFIED"
    (paths.output_dir / "source.json").write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("內容來源{}檔；raw/session仍UNKNOWN", len(public["files"]))
    return public


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    run(parser.parse_args().data_root)


if __name__ == "__main__":
    main()
