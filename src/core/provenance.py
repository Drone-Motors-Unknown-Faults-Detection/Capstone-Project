"""內容來源清冊；環境蒐集沿用runtime_environment，不碰模型。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import tempfile
import uuid

CONTENT_VERSION = "source_content_v2"
MATERIALIZATION_VERSION = "formal_materialization_v3"


class ProvenanceError(ValueError):
    """可公開的來源錯誤，不含任意OS例外payload。"""


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise ProvenanceError(f"來源無法讀取：{type(exc).__name__}") from exc
    return digest.hexdigest()


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


def run(data_root: Path | str) -> dict:
    from core.logger import setup_run

    log, paths = setup_run("source_provenance", unique=True)
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
