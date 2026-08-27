from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sqlite3
import subprocess
import tarfile
import time
import zipfile

import numpy as np


from bodydata_config import CACHE_ROOT, DATA_ROOT, DB_PATH, SEVEN_ZIP


def asset_id(source: str, locator: str) -> str:
    return hashlib.sha1(f"{source}|{locator}".encode("utf-8")).hexdigest()[:24]


def connect() -> sqlite3.Connection:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=60)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS assets (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            folder TEXT NOT NULL DEFAULT '',
            kind TEXT NOT NULL,
            format TEXT NOT NULL,
            locator_type TEXT NOT NULL,
            container TEXT NOT NULL,
            inner_path TEXT NOT NULL DEFAULT '',
            aux_json TEXT NOT NULL DEFAULT '{}',
            animated INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'ready',
            size INTEGER NOT NULL DEFAULT 0,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_assets_source_title ON assets(source, title);
        CREATE INDEX IF NOT EXISTS idx_assets_status ON assets(status);
        CREATE TABLE IF NOT EXISTS state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS asset_recipes (
            asset_id TEXT PRIMARY KEY REFERENCES assets(id) ON DELETE CASCADE,
            recipe_json TEXT NOT NULL DEFAULT '{}',
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS folder_recipes (
            source TEXT NOT NULL,
            folder TEXT NOT NULL,
            recipe_json TEXT NOT NULL DEFAULT '{}',
            updated_at REAL NOT NULL,
            PRIMARY KEY(source, folder)
        );
        CREATE TABLE IF NOT EXISTS collections (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE COLLATE NOCASE,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS collection_assets (
            collection_id TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
            asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
            position INTEGER NOT NULL DEFAULT 0,
            added_at REAL NOT NULL,
            PRIMARY KEY(collection_id, asset_id)
        );
        CREATE INDEX IF NOT EXISTS idx_collection_assets_position
            ON collection_assets(collection_id, position, added_at);
        CREATE TABLE IF NOT EXISTS asset_favorites (
            asset_id TEXT PRIMARY KEY REFERENCES assets(id) ON DELETE CASCADE,
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS preview_artifacts (
            asset_id TEXT PRIMARY KEY REFERENCES assets(id) ON DELETE CASCADE,
            status TEXT NOT NULL,
            result_json TEXT NOT NULL DEFAULT '{}',
            error TEXT NOT NULL DEFAULT '',
            source_fingerprint TEXT NOT NULL DEFAULT '',
            decoder_version TEXT NOT NULL DEFAULT '',
            updated_at REAL NOT NULL,
            last_accessed REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cluster_runs (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            config_json TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cluster_members (
            run_id TEXT NOT NULL REFERENCES cluster_runs(id) ON DELETE CASCADE,
            asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
            cluster_id INTEGER NOT NULL,
            x REAL,
            y REAL,
            PRIMARY KEY(run_id, asset_id)
        );
        CREATE TABLE IF NOT EXISTS cluster_labels (
            run_id TEXT NOT NULL REFERENCES cluster_runs(id) ON DELETE CASCADE,
            cluster_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY(run_id, cluster_id)
        );
        CREATE TABLE IF NOT EXISTS augmentation_runs (
            id TEXT PRIMARY KEY,
            augmentation_type TEXT NOT NULL,
            params_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL,
            created_at REAL NOT NULL,
            completed_at REAL
        );
        CREATE TABLE IF NOT EXISTS augmentation_members (
            run_id TEXT NOT NULL REFERENCES augmentation_runs(id) ON DELETE CASCADE,
            original_asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
            augmented_asset_id TEXT REFERENCES assets(id) ON DELETE SET NULL,
            validation_json TEXT NOT NULL DEFAULT '{}',
            error TEXT NOT NULL DEFAULT '',
            PRIMARY KEY(run_id, original_asset_id)
        );
        CREATE TABLE IF NOT EXISTS augmentation_outputs (
            run_id TEXT NOT NULL REFERENCES augmentation_runs(id) ON DELETE CASCADE,
            original_asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
            duration_scale REAL NOT NULL,
            augmented_asset_id TEXT REFERENCES assets(id) ON DELETE SET NULL,
            validation_json TEXT NOT NULL DEFAULT '{}',
            error TEXT NOT NULL DEFAULT '',
            PRIMARY KEY(run_id, original_asset_id, duration_scale)
        );
        """
    )
    return db


def row(
    source: str,
    title: str,
    kind: str,
    fmt: str,
    locator_type: str,
    container: Path | str,
    inner_path: str = "",
    *,
    folder: str = "",
    aux: dict | None = None,
    animated: bool = False,
    status: str = "ready",
    size: int = 0,
    metadata: dict | None = None,
) -> dict:
    container = str(container)
    locator = f"{locator_type}|{container}|{inner_path}"
    return {
        "id": asset_id(source, locator),
        "source": source,
        "title": title,
        "folder": folder,
        "kind": kind,
        "format": fmt.lower(),
        "locator_type": locator_type,
        "container": container,
        "inner_path": inner_path,
        "aux_json": json.dumps(aux or {}, ensure_ascii=False),
        "animated": int(animated),
        "status": status,
        "size": int(size or 0),
        "metadata_json": json.dumps(metadata or {}, ensure_ascii=False),
    }


def replace_source(db: sqlite3.Connection, source: str, rows: list[dict]) -> None:
    keys = (
        "id source title folder kind format locator_type container inner_path "
        "aux_json animated status size metadata_json"
    ).split()
    with db:
        db.execute("CREATE TEMP TABLE IF NOT EXISTS scan_seen_assets(id TEXT PRIMARY KEY)")
        db.execute("DELETE FROM scan_seen_assets")
        db.executemany("INSERT INTO scan_seen_assets(id) VALUES(?)", ((item["id"],) for item in rows))
        db.executemany(
            f"""INSERT INTO assets ({','.join(keys)}) VALUES ({','.join('?' for _ in keys)})
                ON CONFLICT(id) DO UPDATE SET
                {','.join(f'{key}=excluded.{key}' for key in keys if key != 'id')}""",
            ([item[k] for k in keys] for item in rows),
        )
        db.execute(
            "DELETE FROM assets WHERE source=? AND id NOT IN (SELECT id FROM scan_seen_assets)",
            (source,),
        )


def set_state(db: sqlite3.Connection, key: str, value) -> None:
    with db:
        db.execute(
            "INSERT INTO state(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value, ensure_ascii=False)),
        )


def motion_metadata_from_npz(handle) -> dict:
    """Read only timing fields needed for library-level dataset statistics."""
    try:
        with np.load(handle, allow_pickle=False) as data:
            if "poses" not in data.files:
                return {}
            frames = int(data["poses"].shape[0])
            fps = None
            fps_field = ""
            for candidate in ("mocap_framerate", "mocap_frame_rate"):
                if candidate not in data.files:
                    continue
                values = np.asarray(data[candidate], dtype=np.float64).reshape(-1)
                if values.size and np.isfinite(values[0]) and values[0] > 0:
                    fps = float(values[0])
                    fps_field = candidate
                    break
            metadata = {"frame_count": frames}
            if fps is not None:
                metadata.update(
                    {
                        "fps": fps,
                        "fps_field": fps_field,
                        "duration_seconds": max(0.0, (frames - 1) / fps),
                    }
                )
            return metadata
    except (OSError, ValueError, zipfile.BadZipFile):
        return {}


def scan_humanrig(db: sqlite3.Connection) -> int:
    archive = DATA_ROOT / "HumanRig" / "humanrig.zip"
    rows: list[dict] = []
    if archive.exists():
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if info.filename.lower().endswith("/rigged.glb"):
                    parts = PurePosixPath(info.filename).parts
                    identity = parts[-2]
                    rows.append(
                        row(
                            "HumanRig",
                            identity,
                            "mesh_animation",
                            "glb",
                            "zip_member",
                            archive,
                            info.filename,
                            folder="rigged models",
                            # HumanRig's exported GLBs carry a ~0.066 s bind-pose
                            # clip, not a meaningful motion sequence.
                            animated=False,
                            size=info.file_size,
                        )
                    )
    replace_source(db, "HumanRig", rows)
    return len(rows)


def scan_mixamo(db: sqlite3.Connection) -> int:
    base = DATA_ROOT / "mixamo" / "animations"
    rows: list[dict] = []
    for archive in sorted(base.glob("*.zip")) if base.exists() else []:
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if info.filename.lower().endswith(".fbx"):
                    rows.append(
                        row(
                            "Mixamo",
                            PurePosixPath(info.filename).stem,
                            "mesh_animation",
                            "fbx",
                            "zip_member",
                            archive,
                            info.filename,
                            folder=archive.stem,
                            animated=True,
                            size=info.file_size,
                        )
                    )
    replace_source(db, "Mixamo", rows)
    return len(rows)


def scan_rignet(db: sqlite3.Connection) -> int:
    archive = DATA_ROOT / "3dbicar_rabit" / "rignet" / "processed.zip"
    rows: list[dict] = []
    if archive.exists():
        with zipfile.ZipFile(archive) as zf:
            infos = {i.filename: i for i in zf.infolist()}
            for name, info in infos.items():
                if not name.lower().endswith("/raw_data.npz"):
                    continue
                folder = str(PurePosixPath(name).parent)
                identity = PurePosixPath(folder).name
                mesh = f"{folder}/mesh.obj"
                rows.append(
                    row(
                        "RigNet / ModelsResource",
                        identity,
                        "rig_npz",
                        "npz+obj",
                        "zip_rig",
                        archive,
                        name,
                        folder="processed",
                        aux={"mesh": mesh if mesh in infos else ""},
                        size=info.file_size,
                    )
                )
    replace_source(db, "RigNet / ModelsResource", rows)
    return len(rows)


def seven_zip_entries(archive: Path) -> list[dict]:
    if not archive.exists() or SEVEN_ZIP is None or not SEVEN_ZIP.exists():
        return []
    proc = subprocess.run(
        [str(SEVEN_ZIP), "l", "-slt", str(archive)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    entries: list[dict] = []
    current: dict[str, str] = {}
    for line in proc.stdout.splitlines() + [""]:
        if not line.strip():
            if current.get("Path") and current.get("Path") != str(archive):
                entries.append(current)
            current = {}
        elif " = " in line:
            key, value = line.split(" = ", 1)
            current[key] = value
    return entries


def scan_rigxl(db: sqlite3.Connection) -> int:
    archive = DATA_ROOT / "rigxl" / "data" / "rigxl" / "processed.7z"
    rows: list[dict] = []
    for info in seven_zip_entries(archive):
        name = info.get("Path", "").replace("\\", "/")
        if not name.lower().endswith("/raw_data.npz"):
            continue
        identity = PurePosixPath(name).parent.name
        rows.append(
            row(
                "Rig-XL / UniRig",
                identity,
                "rig_npz",
                "npz",
                "7z_rig",
                archive,
                name,
                folder="processed",
                size=int(info.get("Size", 0) or 0),
            )
        )
    replace_source(db, "Rig-XL / UniRig", rows)
    return len(rows)


def scan_3dbicar(db: sqlite3.Connection) -> int:
    archive = DATA_ROOT / "3dbicar_rabit" / "3DBiCar_zip" / "3DBiCar.zip.001"
    grouped: dict[str, dict[str, str | int]] = {}
    for info in seven_zip_entries(archive):
        name = info.get("Path", "").replace("\\", "/")
        parts = PurePosixPath(name).parts
        if len(parts) < 4 or parts[0].lower() != "3dbicar":
            continue
        identity = parts[1]
        lower = name.lower()
        group = grouped.setdefault(identity, {})
        if lower.endswith("/pose/m.obj"):
            group["obj"] = name
            group["size"] = int(info.get("Size", 0) or 0)
        elif lower.endswith("/pose/m.mtl"):
            group["mtl"] = name
        elif lower.endswith("/params/pose.npy"):
            group["pose"] = name
    rows = [
        row(
            "3DBiCar",
            identity,
            "mesh_static",
            "obj",
            "split_zip_group",
            archive,
            str(group["obj"]),
            folder="posed rabbits",
            aux={k: v for k, v in group.items() if k not in {"obj", "size"}},
            size=int(group.get("size", 0)),
            metadata={"note": "Static posed rabbit mesh; pose parameters are included."},
        )
        for identity, group in grouped.items()
        if group.get("obj")
    ]
    replace_source(db, "3DBiCar", rows)
    return len(rows)


def scan_moyo(db: sqlite3.Connection) -> int:
    archive = DATA_ROOT / "MOYO_smplh_gendered.zip"
    rows: list[dict] = []
    if archive.exists():
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if info.filename.lower().endswith(".npz"):
                    path = PurePosixPath(info.filename)
                    with zf.open(info) as handle:
                        metadata = motion_metadata_from_npz(handle)
                    rows.append(
                        row(
                            "MOYO",
                            path.stem.replace("_stageii", ""),
                            "smplh_motion",
                            "npz",
                            "zip_member",
                            archive,
                            info.filename,
                            folder="/".join(path.parts[:-1]),
                            animated=True,
                            size=info.file_size,
                            metadata=metadata,
                        )
                    )
    replace_source(db, "MOYO", rows)
    return len(rows)


def scan_smplh_models(db: sqlite3.Connection) -> int:
    archive = DATA_ROOT / "smpl_smplh" / "smplh_300.zip"
    rows: list[dict] = []
    if archive.exists():
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if info.filename.lower().endswith(".npz"):
                    rows.append(
                        row(
                            "SMPL-H",
                            PurePosixPath(info.filename).stem,
                            "smplh_model",
                            "npz",
                            "zip_member",
                            archive,
                            info.filename,
                            folder="body models",
                            size=info.file_size,
                            metadata={"note": "Neutral-pose parametric body model; no clip attached."},
                        )
                    )
    replace_source(db, "SMPL-H", rows)
    return len(rows)


def scan_smal(db: sqlite3.Connection) -> int:
    base = DATA_ROOT / "smal"
    rows: list[dict] = []
    for archive in sorted(base.glob("*.tgz")) if base.exists() else []:
        try:
            with tarfile.open(archive, "r:gz") as tf:
                matches = [
                    m for m in tf.getmembers()
                    if m.isfile()
                    and not PurePosixPath(m.name).name.startswith("._")
                    and Path(m.name).suffix.lower() in {".pkl", ".npz", ".ply"}
                ]
            for info in matches:
                rows.append(
                    row(
                        "SMAL",
                        f"{archive.stem}: {PurePosixPath(info.name).name}",
                        "mesh_static" if PurePosixPath(info.name).suffix.lower() == ".ply" else "smal_model",
                        PurePosixPath(info.name).suffix,
                        "tar_member",
                        archive,
                        info.name,
                        folder=archive.name,
                        status="ready" if PurePosixPath(info.name).suffix.lower() == ".ply" else "needs_adapter",
                        size=info.size,
                        metadata={"note": "Model parameter file; SMAL code adapter is required."},
                    )
                )
        except tarfile.TarError:
            pass
    replace_source(db, "SMAL", rows)
    return len(rows)


def scan_articulation_metadata(db: sqlite3.Connection) -> int:
    metadata_path = DATA_ROOT / "articulation_xl" / "meta_Articulation_XL_2.0.csv"
    rows: list[dict] = []
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for item in csv.DictReader(handle):
                identity = item.get("uuid") or f"row-{len(rows)}"
                rows.append(
                    row(
                        "Articulation-XL 2.0",
                        identity,
                        "articulation_record",
                        item.get("fileType", "npz"),
                        "articulation_metadata",
                        metadata_path,
                        identity,
                        folder=item.get("category_label", "uncategorized"),
                        status="needs_shard_map",
                        metadata=item,
                    )
                )
    # Expose records from every structurally complete shard. The UUID metadata and
    # shard arrays are separate upstream files, so generic stable labels are used
    # until a shard has been opened and its embedded UUID is known.
    base = DATA_ROOT / "articulation_xl"
    for shard in sorted(base.glob("*.npz")) if base.exists() else []:
        try:
            with zipfile.ZipFile(shard) as zf, zf.open("arr_0.npy") as handle:
                version = __import__("numpy").lib.format.read_magic(handle)
                shape, _, _ = __import__("numpy").lib.format._read_array_header(handle, version)
            count = int(shape[0])
        except Exception:
            continue
        split = shard.stem.removeprefix("articulation_xlv2_")
        for index in range(count):
            rows.append(
                row(
                    "Articulation-XL 2.0",
                    f"{split} #{index:05d}",
                    "articulation_record",
                    "npz record",
                    "articulation_record",
                    shard,
                    str(index),
                    folder=f"decoded shards/{split}",
                    metadata={"record_index": index, "shard": shard.name, "first_open": "loads the source shard"},
                )
            )
    replace_source(db, "Articulation-XL 2.0", rows)
    return len(rows)


def scan_anymate_shards(db: sqlite3.Connection) -> int:
    base = DATA_ROOT / "anymate"
    if not base.exists():
        replace_source(db, "Anymate", [])
        return 0
    rows = [
        row(
            "Anymate",
            shard.stem,
            "anymate_shard",
            "pt",
            "torch_shard",
            shard,
            folder="dataset shards",
            status="index_required",
            size=shard.stat().st_size,
            metadata={"note": "Use Full index to expose individual assets."},
        )
        for shard in sorted(base.glob("*.pt"))
    ] if base.exists() else []
    keys = (
        "id source title folder kind format locator_type container inner_path "
        "aux_json animated status size metadata_json"
    ).split()
    # A fast rescan must not hide asset-level records that were already indexed.
    # Refresh only shard placeholders and keep completed anymate_record rows.
    with db:
        db.execute("DELETE FROM assets WHERE source='Anymate' AND kind='anymate_shard'")
        db.executemany(
            f"INSERT OR IGNORE INTO assets ({','.join(keys)}) VALUES ({','.join('?' for _ in keys)})",
            ([item[k] for k in keys] for item in rows),
        )
    return db.execute("SELECT COUNT(*) FROM assets WHERE source='Anymate'").fetchone()[0]


def scan_anymate_assets(db: sqlite3.Connection, pattern: str = "*.pt") -> int:
    import gc
    import torch

    base = DATA_ROOT / "anymate"
    rows: list[dict] = []
    for shard in sorted(base.glob(pattern)) if base.exists() else []:
        set_state(db, "full_scan", {"phase": f"Anymate / {shard.name}", "started": time.time()})
        records = torch.load(shard, map_location="cpu", weights_only=True, mmap=True)
        for index, item in enumerate(records):
            title = str(item.get("name") or f"{shard.stem} #{index:05d}")
            rows.append(
                row(
                    "Anymate",
                    title,
                    "anymate_record",
                    "pt record",
                    "anymate_record",
                    shard,
                    str(index),
                    folder=shard.stem,
                    metadata={"record_index": index, "shard": shard.name},
                )
            )
        del records
        gc.collect()
    replace_source(db, "Anymate", rows)
    return len(rows)


def scan_mvsydog(db: sqlite3.Connection, full: bool = False) -> int:
    archive = DATA_ROOT / "mvsydog" / "MVSyDog_full.tar.gz"
    rows: list[dict] = []
    if archive.exists():
        # Streaming mode avoids extracting the 121 GB archive. Quick indexing stops
        # after the first playable BVH; a full rescan walks the entire gzip stream.
        with tarfile.open(archive, "r|gz") as tf:
            for info in tf:
                if info.isfile() and info.name.lower().endswith(".bvh"):
                    path = PurePosixPath(info.name)
                    rows.append(
                        row(
                            "MVSyDog",
                            path.stem,
                            "mesh_animation",
                            "bvh",
                            "tar_member",
                            archive,
                            info.name,
                            folder="/".join(path.parts[:-1]),
                            animated=True,
                            size=info.size,
                        )
                    )
                    if not full:
                        break
    replace_source(db, "MVSyDog", rows)
    return len(rows)


def scan_direct_files(db: sqlite3.Connection) -> int:
    formats = {".glb", ".gltf", ".fbx", ".dae", ".blend", ".obj", ".bvh"}
    rows: list[dict] = []
    excluded = {"_preview_cache", "_tools", "_logs"}
    for path in DATA_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in formats:
            continue
        if any(part in excluded for part in path.relative_to(DATA_ROOT).parts):
            continue
        rel = path.relative_to(DATA_ROOT)
        source = f"Local / {rel.parts[0]}"
        rows.append(
            row(
                source,
                path.stem,
                "mesh_animation" if path.suffix.lower() in {".fbx", ".glb", ".gltf", ".dae", ".bvh"} else "mesh_static",
                path.suffix[1:],
                "direct",
                path,
                folder=str(rel.parent),
                animated=path.suffix.lower() in {".fbx", ".glb", ".gltf", ".dae", ".bvh"},
                size=path.stat().st_size,
            )
        )
    # Direct files use multiple source names, so replace the entire Local namespace.
    with db:
        db.execute("DELETE FROM assets WHERE source LIKE 'Local / %'")
        keys = (
            "id source title folder kind format locator_type container inner_path "
            "aux_json animated status size metadata_json"
        ).split()
        db.executemany(
            f"INSERT INTO assets ({','.join(keys)}) VALUES ({','.join('?' for _ in keys)})",
            ([item[k] for k in keys] for item in rows),
        )
    return len(rows)


def scan_amass(db: sqlite3.Connection) -> int:
    rows: list[dict] = []
    # Small *_demo archives were useful during development but are duplicates of
    # the real AMASS sources and should never appear in the product library.
    with db:
        db.execute("DELETE FROM assets WHERE source LIKE 'AMASS / %demo%'")
    for archive in sorted(DATA_ROOT.glob("*.tar.bz2")):
        if "demo" in archive.stem.lower():
            continue
        source = f"AMASS / {archive.name.removesuffix('.tar.bz2')}"
        set_state(db, "full_scan", {"phase": source, "started": time.time()})
        with tarfile.open(archive, "r:bz2") as tf:
            for info in tf:
                # AMASS archives also contain per-subject shape.npz support
                # files. They have no pose sequence and must not be advertised
                # as playable motion clips.
                if info.isfile() and info.name.lower().endswith("_poses.npz"):
                    path = PurePosixPath(info.name)
                    extracted = tf.extractfile(info)
                    metadata = {}
                    if extracted is not None:
                        with extracted:
                            metadata = motion_metadata_from_npz(io.BytesIO(extracted.read()))
                    rows.append(
                        row(
                            source,
                            path.stem,
                            "smplh_motion",
                            "npz",
                            "tar_member",
                            archive,
                            info.name,
                            folder="/".join(path.parts[:-1]),
                            animated=True,
                            size=info.size,
                            metadata=metadata,
                        )
                    )
        replace_source(db, source, [x for x in rows if x["source"] == source])
    return len(rows)


def build_index(full: bool = False) -> dict:
    db = connect()
    set_state(db, "indexing", {"active": True, "full": full, "started": time.time()})
    counts: dict[str, int] = {}
    scanners = [
        ("HumanRig", scan_humanrig),
        ("Mixamo", scan_mixamo),
        ("RigNet", scan_rignet),
        ("RigXL", scan_rigxl),
        ("3DBiCar", scan_3dbicar),
        ("MOYO", scan_moyo),
        ("SMPL-H", scan_smplh_models),
        ("SMAL", scan_smal),
        ("Articulation-XL", scan_articulation_metadata),
        ("Anymate", scan_anymate_shards),
        ("MVSyDog", lambda current: scan_mvsydog(current, full=False)),
        ("Local", scan_direct_files),
    ]
    try:
        for name, scanner in scanners:
            set_state(db, "indexing", {"active": True, "phase": name, "full": full})
            counts[name] = scanner(db)
        if full:
            counts["AMASS"] = scan_amass(db)
            counts["MVSyDog"] = scan_mvsydog(db, full=True)
            counts["Anymate"] = scan_anymate_assets(db)
        set_state(db, "last_index", {"finished": time.time(), "full": full, "counts": counts})
        return counts
    finally:
        set_state(db, "indexing", {"active": False, "finished": time.time(), "full": full})
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the read-only Body Data Studio index")
    parser.add_argument("--full", action="store_true", help="Also scan sequential AMASS tar.bz2 archives")
    args = parser.parse_args()
    print(json.dumps(build_index(args.full), indent=2, ensure_ascii=False))
