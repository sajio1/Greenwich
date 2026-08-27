from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import shutil
import time

from bodydata_decode import materialize
from bodydata_index import CACHE_ROOT, DATA_ROOT, connect


def _safe_part(value: str, fallback: str) -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(value or "")).strip(" .")
    return value[:120] or fallback


def _new_destination(requested: str) -> Path:
    if not requested.strip():
        raise ValueError("Export destination is required")
    base = Path(requested).expanduser().resolve()
    if base == Path(base.anchor):
        raise ValueError("Choose a named export folder, not the drive root")
    if base in {DATA_ROOT.resolve(), CACHE_ROOT.resolve()}:
        raise ValueError("Export destination must not be the raw data or preview-cache root")
    candidate = base
    suffix = 2
    while candidate.exists():
        candidate = base.with_name(f"{base.name}_{suffix}")
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def export_assets(asset_ids: list[str], destination: str, context: dict | None = None) -> dict:
    unique_ids = list(dict.fromkeys(str(value) for value in asset_ids if value))
    if not unique_ids:
        raise ValueError("Select at least one asset to export")
    db = connect()
    placeholders = ",".join("?" for _ in unique_ids)
    rows = db.execute(f"SELECT * FROM assets WHERE id IN ({placeholders})", unique_ids).fetchall()
    db.close()
    assets = {row["id"]: dict(row) for row in rows}
    target_root = _new_destination(destination)
    manifest, failures = [], []
    for identity in unique_ids:
        asset = assets.get(identity)
        if asset is None:
            failures.append({"asset_id": identity, "error": "Asset not found in index"})
            continue
        try:
            source_path = materialize(asset)
            source_dir = target_root / _safe_part(asset["source"], "source")
            folder = str(asset.get("folder") or "").replace("\\", "/")
            for part in (piece for piece in folder.split("/") if piece not in {"", ".", ".."}):
                source_dir /= _safe_part(part, "folder")
            source_dir.mkdir(parents=True, exist_ok=True)
            filename = _safe_part(Path(asset.get("inner_path") or source_path.name).name, f"{identity}.{asset['format']}")
            output = source_dir / filename
            if output.exists():
                output = source_dir / f"{identity}_{filename}"
            shutil.copy2(source_path, output)
            manifest.append({
                "asset_id": identity,
                "title": asset["title"],
                "source": asset["source"],
                "folder": asset["folder"],
                "kind": asset["kind"],
                "format": asset["format"],
                "locator_type": asset["locator_type"],
                "original_container": asset["container"],
                "original_inner_path": asset["inner_path"],
                "export_relative_path": str(output.relative_to(target_root)),
                "file_size": output.stat().st_size,
            })
        except Exception as exc:
            failures.append({"asset_id": identity, "title": asset.get("title", ""), "error": f"{type(exc).__name__}: {exc}"})
    payload = {
        "created_at": time.time(),
        "source_root": str(DATA_ROOT),
        "destination": str(target_root),
        "context": context or {},
        "exported": len(manifest),
        "failed": len(failures),
        "members": manifest,
        "failures": failures,
    }
    json_path = target_root / "manifest.json"
    csv_path = target_root / "manifest.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["asset_id", "title", "source", "folder", "kind", "format", "locator_type", "original_container", "original_inner_path", "export_relative_path", "file_size"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)
    payload["manifest_json"] = str(json_path)
    payload["manifest_csv"] = str(csv_path)
    return payload
