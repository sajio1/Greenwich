from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
import uuid

import numpy as np

from bodydata_augment import AUGMENT_ROOT, augment_assets
from bodydata_bone_augment import augment_bone_assets
from bodydata_config import CACHE_ROOT, DATA_ROOT
from bodydata_contact import CONTACT_EFFECTORS, CONTACT_ROOT, LEGACY_EFFECTORS, label_assets, load_contact_labels
from bodydata_decode import materialize
from bodydata_index import connect


PROCESS_ROOT = CACHE_ROOT / "process_runs"
EXPORTED_REGISTRY = CACHE_ROOT / "exported_runs"
MAX_TEMPORARY_RUNS = 30


def _expand_effector_selection(values: list[str] | tuple[str, ...] | None) -> list[str]:
    requested = list(CONTACT_EFFECTORS if values is None else values)
    expanded: list[str] = []
    for value in requested:
        if value == "left_foot":
            expanded.extend(("left_heel", "left_forefoot"))
        elif value == "right_foot":
            expanded.extend(("right_heel", "right_forefoot"))
        elif value in CONTACT_EFFECTORS:
            expanded.append(value)
    return list(dict.fromkeys(expanded))


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _run_path(run_id: str) -> Path:
    return PROCESS_ROOT / run_id / "run.json"


def load_process_run(run_id: str) -> dict | None:
    path = _run_path(run_id)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _binary_segments(values: np.ndarray) -> list[dict]:
    active = np.flatnonzero(np.asarray(values, dtype=bool))
    if not len(active):
        return []
    breaks = np.flatnonzero(np.diff(active) > 1)
    starts = np.r_[active[0], active[breaks + 1]]
    ends = np.r_[active[breaks], active[-1]]
    return [{"start_frame": int(start), "end_frame": int(end)} for start, end in zip(starts, ends)]


def load_process_contact_preview(run_id: str, asset_id: str) -> dict | None:
    """Load exactly the contact selection attached to one processing result."""
    result = load_process_run(run_id)
    labeling = (result or {}).get("labeling") or {}
    item = next((value for value in labeling.get("items", []) if value.get("asset_id") == asset_id), None)
    if not item:
        return None
    config = (result.get("config") or {}).get("labeling") or {}
    effectors = _expand_effector_selection(config.get("effectors"))
    channels = config.get("channels", ["binary", "confidence", "segments"])
    exported_path = Path(item.get("exported_label_path", ""))
    cached = load_contact_labels(asset_id)
    sidecar = exported_path if exported_path.is_file() else Path((cached or {}).get("sidecar_npz", ""))
    if not sidecar.is_file():
        return None
    with np.load(sidecar, allow_pickle=False) as archive:
        stored_effectors = [str(value) for value in archive["effectors"].tolist()] if "effectors" in archive else list(LEGACY_EFFECTORS)
        stored_map = {name: index for index, name in enumerate(stored_effectors)}
        array_keys = [key for key in ("contact", "contact_type", "contact_confidence", "foot_position_world_m", "foot_height_above_ground_m", "foot_speed_mps") if key in archive]
        frame_count = int(len(archive[array_keys[0]])) if array_keys else int(np.asarray(archive["frame_count"]).reshape(-1)[0]) if "frame_count" in archive else 0
        fps = float(np.asarray(archive["fps"]).reshape(-1)[0]) if "fps" in archive else float((cached or {}).get("fps", 0.0))
        contact_source = str(np.asarray(archive["contact_source"]).reshape(-1)[0]) if "contact_source" in archive else str((cached or {}).get("contact_source", "detected"))

        def expand(key: str, tail: tuple[int, ...] = (), dtype=np.float32) -> np.ndarray | None:
            if key not in archive:
                return None
            source = np.asarray(archive[key])
            output = np.zeros((frame_count, len(CONTACT_EFFECTORS), *tail), dtype=dtype)
            for column, name in enumerate(CONTACT_EFFECTORS):
                if name not in effectors:
                    continue
                stored_index = stored_map.get(name)
                if stored_index is None:
                    stored_index = stored_map.get("left_foot" if name.startswith("left_") else "right_foot")
                if stored_index is not None:
                    output[:, column] = source[:, stored_index]
            return output

        contact = expand("contact", dtype=np.uint8) if {"binary", "segments"} & set(channels) else None
        contact_type = expand("contact_type", dtype=np.uint8) if "contact_type" in archive else None
        confidence = expand("contact_confidence") if "confidence" in channels else None
        position = expand("foot_position_world_m", (3,)) if "position" in channels else None
        height = expand("foot_height_above_ground_m") if "height" in channels else None
        speed = expand("foot_speed_mps") if "speed" in channels else None

    stored_segments: dict = {}
    if "segments" in channels:
        if exported_path.is_file():
            segments_path = Path(item.get("exported_segments_path", ""))
            if segments_path.is_file():
                try:
                    stored_segments = json.loads(segments_path.read_text(encoding="utf-8")).get(asset_id, {})
                except (OSError, ValueError, TypeError):
                    stored_segments = {}
        else:
            stored_segments = dict((cached or {}).get("segments") or {})
    if contact is None and stored_segments:
        contact = np.zeros((frame_count, len(CONTACT_EFFECTORS)), dtype=np.uint8)
        for column, effector in enumerate(CONTACT_EFFECTORS):
            if effector not in effectors:
                continue
            segments = stored_segments.get(effector)
            if segments is None:
                segments = stored_segments.get("left" if effector.startswith("left_") else "right", [])
            for segment in segments:
                contact[int(segment["start_frame"]):int(segment["end_frame"]) + 1, column] = 1
    visual_segments = {
        name: _binary_segments(contact[:, index]) if contact is not None else []
        for index, name in enumerate(CONTACT_EFFECTORS)
    }
    visual_segments["left"] = _binary_segments(np.any(contact[:, :2], axis=1)) if contact is not None else []
    visual_segments["right"] = _binary_segments(np.any(contact[:, 2:], axis=1)) if contact is not None else []
    payload = {
        "asset_id": asset_id,
        "frame_count": frame_count,
        "fps": fps,
        "segments": visual_segments,
        "selected_effectors": effectors,
        "selected_channels": channels,
        "storage_state": "exported_sidecar" if exported_path.is_file() else "cache_sidecar",
        "contact_source": contact_source,
        "ground_height_m": float((cached or {}).get("ground_height_m", 0.0)),
        "legacy_compatibility": int((cached or {}).get("version", 0)) < 4,
    }
    if contact is not None:
        payload["contact"] = contact.tolist()
    if contact_type is not None:
        payload["contact_type"] = contact_type.tolist()
    if confidence is not None:
        payload["confidence"] = np.round(confidence.astype(np.float64), 4).tolist()
    if position is not None:
        payload["position_world_m"] = np.round(position.astype(np.float64), 6).tolist()
    if height is not None:
        payload["height_above_ground_m"] = np.round(height.astype(np.float64), 6).tolist()
    if speed is not None:
        payload["speed_mps"] = np.round(speed.astype(np.float64), 6).tolist()
    return payload


def _asset_summaries(asset_ids: list[str]) -> list[dict]:
    if not asset_ids:
        return []
    placeholders = ",".join("?" for _ in asset_ids)
    db = connect()
    rows = db.execute(
        f"SELECT id,source,title,folder,kind,format,animated,size FROM assets WHERE id IN ({placeholders})",
        asset_ids,
    ).fetchall()
    db.close()
    found = {row["id"]: dict(row) for row in rows}
    return [found[value] for value in asset_ids if value in found]


def _normalize_config(config: dict) -> dict:
    augmentation = dict(config.get("augmentation") or {})
    labeling = dict(config.get("labeling") or {})
    explicit_duration = augmentation.get("duration_enabled", augmentation.get("durationEnabled"))
    explicit_bone = augmentation.get("bone_enabled", augmentation.get("boneEnabled"))
    duration_enabled = bool(augmentation.get("enabled")) if explicit_duration is None and explicit_bone is None else bool(explicit_duration)
    bone_enabled = bool(explicit_bone)
    augmentation_enabled = duration_enabled or bone_enabled
    scales = list(dict.fromkeys(round(float(value), 4) for value in augmentation.get("duration_scales", augmentation.get("durationScales", [1.15]))))
    if duration_enabled and (not scales or len(scales) > 8 or any(value < 1.0 or value > 1.5 for value in scales)):
        raise ValueError("Duration multipliers must contain 1-8 values between 1.00 and 1.50")
    bone_variants = int(augmentation.get("bone_variants", augmentation.get("boneVariants", 1)))
    if bone_enabled and not 1 <= bone_variants <= 8:
        raise ValueError("Random bone length must contain 1-8 variants per motion")
    effectors = _expand_effector_selection(labeling.get("effectors"))
    channels = [
        value
        for value in labeling.get("channels", ["binary", "confidence", "segments"])
        if value in {"binary", "confidence", "segments", "position", "height", "speed"}
    ]
    if labeling.get("enabled") and (not effectors or not channels):
        raise ValueError("Foot-contact labeling requires at least one foot and one output channel")
    scope = str(labeling.get("scope", "both"))
    if scope not in {"original", "derived", "both"}:
        scope = "both"
    return {
        "augmentation": {
            "enabled": augmentation_enabled,
            "duration_enabled": duration_enabled,
            "bone_enabled": bone_enabled,
            "methods": (["duration_multiplier"] if duration_enabled else []) + (["random_bone_length"] if bone_enabled else []),
            "duration_scales": scales,
            "bone_variants": bone_variants,
        },
        "labeling": {
            "enabled": bool(labeling.get("enabled", False)),
            "methods": ["foot_contact"] if labeling.get("enabled") else [],
            "effectors": effectors,
            "channels": channels,
            "scope": scope,
        },
    }


def run_process(asset_ids: list[str], config: dict) -> dict:
    unique_ids = list(dict.fromkeys(str(value) for value in asset_ids if value))
    if not unique_ids:
        raise ValueError("Select at least one asset")
    normalized = _normalize_config(config)
    if not normalized["augmentation"]["enabled"] and not normalized["labeling"]["enabled"]:
        raise ValueError("Select at least one augmentation or labeling method")
    if normalized["labeling"]["enabled"] and normalized["labeling"]["scope"] == "derived" and not normalized["augmentation"]["enabled"]:
        raise ValueError("Derived-only labeling requires an enabled augmentation module")
    assets = _asset_summaries(unique_ids)
    if len(assets) != len(unique_ids):
        raise ValueError("One or more selected assets are no longer in the index")
    unsupported = [asset for asset in assets if not asset["animated"] or asset["kind"] != "smplh_motion" or asset["format"] != "npz"]
    if unsupported and (normalized["augmentation"]["enabled"] or normalized["labeling"]["enabled"]):
        raise ValueError(f"{len(unsupported)} selected assets do not support the configured SMPL-H processing modules")

    run_id = uuid.uuid4().hex[:20]
    created_at = time.time()
    result = {
        "run_id": run_id,
        "created_at": created_at,
        "updated_at": created_at,
        "status": "processing",
        "storage_state": "temporary",
        "pinned": False,
        "asset_ids": unique_ids,
        "assets": assets,
        "config": normalized,
        "augmentation": None,
        "labeling": None,
        "failures": [],
    }
    _atomic_json(_run_path(run_id), result)
    try:
        derived_ids: list[str] = []
        if normalized["augmentation"]["enabled"]:
            subruns = []
            if normalized["augmentation"]["duration_enabled"]:
                temporal = augment_assets(unique_ids, normalized["augmentation"]["duration_scales"], None, 0)
                temporal["run_ids"] = [temporal["run_id"]]
                temporal["output_directories"] = [temporal["output_directory"]]
                for pair in temporal.get("pairs", []):
                    pair["augmentation_type"] = "time"
                    pair["variant_label"] = f"Duration x{float(pair.get('actual_duration_scale', pair.get('duration_scale', 1))):.3f}"
                    pair["cache_path"] = str(Path(temporal["output_directory"]) / pair["augmented_filename"])
                subruns.append(temporal)
            if normalized["augmentation"]["bone_enabled"]:
                subruns.append(augment_bone_assets(unique_ids, normalized["augmentation"]["bone_variants"], 0))
            pairs = [pair for run in subruns for pair in run.get("pairs", [])]
            failures = [failure for run in subruns for failure in run.get("failures", [])]
            run_ids = [value for run in subruns for value in run.get("run_ids", [run.get("run_id")]) if value]
            output_directories = [value for run in subruns for value in run.get("output_directories", [run.get("output_directory")]) if value]
            augmented = {
                "run_id": run_ids[0] if run_ids else "",
                "run_ids": run_ids,
                "status": "ready" if pairs else "failed",
                "params": {"type": "multi" if len(subruns) > 1 else subruns[0]["params"].get("type", "unknown"), "methods": normalized["augmentation"]["methods"], "method": " + ".join(run["params"]["method"] for run in subruns)},
                "subruns": [{key: run.get(key) for key in ("run_id", "params", "status", "output_directory")} for run in subruns],
                "pairs": pairs,
                "failures": failures,
                "output_directory": output_directories[0] if output_directories else "",
                "output_directories": output_directories,
                "storage_state": "cache_draft",
                "saved": False,
                "passed": sum(bool(pair.get("validation", {}).get("passed")) for pair in pairs),
                "failed": len(failures) + sum(not bool(pair.get("validation", {}).get("passed")) for pair in pairs),
            }
            result["augmentation"] = augmented
            derived_ids = [pair["augmented_asset_id"] for pair in augmented.get("pairs", [])]
            result["failures"].extend(augmented.get("failures", []))

        if normalized["labeling"]["enabled"]:
            scope = normalized["labeling"]["scope"]
            targets = []
            if scope in {"original", "both"}:
                targets.extend(unique_ids)
            if scope in {"derived", "both"}:
                targets.extend(derived_ids)
            labels = label_assets(
                targets,
                False,
                {
                    "effectors": normalized["labeling"]["effectors"],
                    "channels": normalized["labeling"]["channels"],
                    "scope": scope,
                },
            )
            result["labeling"] = labels
            result["failures"].extend(labels.get("failures", []))

        result["status"] = "ready" if normalized["augmentation"]["enabled"] or normalized["labeling"]["enabled"] else "review_only"
    except Exception as exc:
        result["status"] = "failed"
        result["failures"].append({"error": f"{type(exc).__name__}: {exc}"})
    result["updated_at"] = time.time()
    result["summary"] = {
        "inputs": len(unique_ids),
        "derived": len((result.get("augmentation") or {}).get("pairs", [])),
        "labeled": int((result.get("labeling") or {}).get("completed", 0)),
        "failed": len(result["failures"]),
    }
    _atomic_json(_run_path(run_id), result)
    prune_temporary_runs()
    return result


def _iter_runs() -> list[dict]:
    runs = []
    if PROCESS_ROOT.is_dir():
        for path in PROCESS_ROOT.glob("*/run.json"):
            try:
                runs.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError):
                continue
    return sorted(runs, key=lambda item: float(item.get("updated_at", item.get("created_at", 0))), reverse=True)


def list_process_runs(scope: str = "temporary", query: str = "", limit: int = MAX_TEMPORARY_RUNS) -> list[dict]:
    needle = query.casefold().strip()
    runs = [run for run in _iter_runs() if (run.get("storage_state") == "exported") == (scope == "exported")]
    if needle:
        runs = [
            run for run in runs
            if needle in run.get("run_id", "").casefold()
            or any(needle in asset.get("title", "").casefold() or needle in asset.get("source", "").casefold() for asset in run.get("assets", []))
        ]
    compact = []
    for run in runs[: max(1, min(int(limit), MAX_TEMPORARY_RUNS))]:
        compact.append({key: run.get(key) for key in (
            "run_id", "created_at", "updated_at", "status", "storage_state", "pinned",
            "assets", "summary", "destination", "exported_at",
        )})
    return compact


def set_process_pinned(run_id: str, pinned: bool) -> dict:
    result = load_process_run(run_id)
    if not result:
        raise FileNotFoundError("Processing run not found")
    result["pinned"] = bool(pinned)
    result["updated_at"] = time.time()
    _atomic_json(_run_path(run_id), result)
    return result


def _safe_remove_tree(path: Path, root: Path) -> None:
    resolved, safe_root = path.resolve(), root.resolve()
    if resolved != safe_root and safe_root in resolved.parents and resolved.is_dir():
        shutil.rmtree(resolved)


def _purge_process_run_cache(run_id: str, result: dict) -> None:
    augmentation = result.get("augmentation") or {}
    augmentation_runs = list(dict.fromkeys(augmentation.get("run_ids") or [augmentation.get("run_id")]))
    for augmentation_run in (value for value in augmentation_runs if value):
        db = connect()
        output_ids = [row[0] for row in db.execute("SELECT augmented_asset_id FROM augmentation_outputs WHERE run_id=? AND augmented_asset_id IS NOT NULL", (augmentation_run,)).fetchall()]
        try:
            output_ids.extend(row[0] for row in db.execute("SELECT augmented_asset_id FROM augmentation_bone_outputs WHERE run_id=? AND augmented_asset_id IS NOT NULL", (augmentation_run,)).fetchall())
        except Exception:
            pass
        with db:
            if output_ids:
                placeholders = ",".join("?" for _ in output_ids)
                db.execute(f"DELETE FROM assets WHERE id IN ({placeholders})", output_ids)
            db.execute("DELETE FROM augmentation_outputs WHERE run_id=?", (augmentation_run,))
            try:
                db.execute("DELETE FROM augmentation_bone_outputs WHERE run_id=?", (augmentation_run,))
            except Exception:
                pass
            db.execute("DELETE FROM augmentation_runs WHERE id=?", (augmentation_run,))
        db.close()
        _safe_remove_tree(AUGMENT_ROOT / augmentation_run, AUGMENT_ROOT)
    labeling_run = (result.get("labeling") or {}).get("run_id")
    if labeling_run:
        manifest = (CONTACT_ROOT / "runs" / f"{labeling_run}.json").resolve()
        manifest_root = (CONTACT_ROOT / "runs").resolve()
        if manifest.parent == manifest_root and manifest.is_file():
            manifest.unlink()
    _safe_remove_tree(PROCESS_ROOT / run_id, PROCESS_ROOT)


def discard_process_run(run_id: str) -> None:
    result = load_process_run(run_id)
    if not result:
        return
    if result.get("storage_state") == "exported":
        raise ValueError("Exported runs require the explicit exported-delete action")
    _purge_process_run_cache(run_id, result)


def delete_exported_process_run(run_id: str) -> None:
    """Delete one verified exported bundle and its processing history/cache."""
    result = load_process_run(run_id)
    if not result:
        return
    if result.get("storage_state") != "exported":
        raise ValueError("Only exported runs can use the exported-delete action")

    raw_destination = str(result.get("destination") or "").strip()
    if raw_destination:
        target = Path(raw_destination).expanduser().resolve()
        forbidden = {
            DATA_ROOT.resolve(),
            CACHE_ROOT.resolve(),
            PROCESS_ROOT.resolve(),
            Path(target.anchor).resolve(),
        }
        if target in forbidden:
            raise ValueError("Refusing to delete an unsafe export destination")
        if target.exists():
            if not target.is_dir():
                raise ValueError("Export destination is not a directory")
            recipe_path = target / "processing_recipe.json"
            try:
                recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError) as exc:
                raise ValueError("Export destination is missing its verified processing recipe") from exc
            if recipe.get("process_run_id") != run_id:
                raise ValueError("Export destination does not belong to this processing run")
            shutil.rmtree(target)

    registry = EXPORTED_REGISTRY / f"{run_id}.json"
    if registry.is_file():
        registry.unlink()
    download = CACHE_ROOT / "downloads" / f"body-data-studio__{run_id}.zip"
    if download.is_file():
        download.unlink()
    _purge_process_run_cache(run_id, result)


def prune_temporary_runs() -> None:
    temporary = [run for run in _iter_runs() if run.get("storage_state") != "exported" and not run.get("pinned")]
    for run in temporary[MAX_TEMPORARY_RUNS:]:
        discard_process_run(run["run_id"])


def _selected_label_payload(source: Path, effectors: list[str], channels: list[str], destination: Path) -> None:
    requested_effectors = _expand_effector_selection(effectors)
    output = {}
    with np.load(source, allow_pickle=False) as archive:
        stored_effectors = [str(value) for value in archive["effectors"].tolist()] if "effectors" in archive else list(LEGACY_EFFECTORS)
        stored_map = {name: index for index, name in enumerate(stored_effectors)}
        indices, exported_effectors = [], []
        for value in requested_effectors:
            index = stored_map.get(value)
            if index is None:
                index = stored_map.get("left_foot" if value.startswith("left_") else "right_foot")
            if index is not None:
                indices.append(index)
                exported_effectors.append(value)
        if not indices:
            raise ValueError("No selected contact effectors exist in the sidecar")
        if "binary" in channels:
            output["contact"] = archive["contact"][:, indices]
            if "contact_type" in archive:
                output["contact_type"] = archive["contact_type"][:, indices]
        if "confidence" in channels:
            output["contact_confidence"] = archive["contact_confidence"][:, indices]
        if "position" in channels and "foot_position_world_m" in archive:
            output["foot_position_world_m"] = archive["foot_position_world_m"][:, indices]
        if "height" in channels:
            output["foot_height_above_ground_m"] = archive["foot_height_above_ground_m"][:, indices]
        if "speed" in channels:
            output["foot_speed_mps"] = archive["foot_speed_mps"][:, indices]
        output["fps"] = archive["fps"]
        output["frame_count"] = np.asarray(archive["contact"].shape[0], dtype=np.int64)
        output["effectors"] = np.asarray(exported_effectors)
        output["channels"] = np.asarray(channels)
    np.savez_compressed(destination, **output)


def save_process_run(run_id: str, destination: str | Path, include_original: bool = False) -> dict:
    result = load_process_run(run_id)
    if not result or result.get("status") not in {"ready", "review_only"}:
        raise ValueError("Only a ready temporary run can be saved")
    target = Path(destination).expanduser().resolve()
    if target == DATA_ROOT.resolve() or DATA_ROOT.resolve() in target.parents:
        raise ValueError("Choose an output folder outside the read-only input data root")
    if target.exists() and not target.is_dir():
        raise FileExistsError("The output destination must be a folder")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("The output folder must be new or empty")
    target.mkdir(parents=True, exist_ok=True)
    data_dir, labels_dir = target / "data", target / "labels" / "foot_contact"
    manifest = []

    if include_original:
        originals_dir = data_dir / "original"
        originals_dir.mkdir(parents=True, exist_ok=True)
        db = connect()
        for asset_id in result["asset_ids"]:
            row = db.execute("SELECT * FROM assets WHERE id=?", (asset_id,)).fetchone()
            if row is None:
                continue
            source = materialize(dict(row))
            output = originals_dir / f"{asset_id}__{source.name}"
            shutil.copy2(source, output)
            manifest.append({"asset_id": asset_id, "role": "original", "relative_path": str(output.relative_to(target))})
        db.close()

    augmentation = result.get("augmentation") or {}
    if augmentation.get("pairs"):
        augmented_dir = data_dir / "augmented"
        augmented_dir.mkdir(parents=True, exist_ok=True)
        for pair in augmentation["pairs"]:
            source = Path(pair.get("cache_path") or Path(augmentation["output_directory"]) / pair["augmented_filename"])
            # Asset ids keep same-named clips from different datasets collision-free.
            output = augmented_dir / f"{pair['augmented_asset_id']}__{pair['augmented_filename']}"
            shutil.copy2(source, output)
            pair["exported_path"] = str(output)
            db = connect()
            with db:
                db.execute(
                    "UPDATE assets SET locator_type='direct',container=?,inner_path='',size=? WHERE id=?",
                    (str(output), output.stat().st_size, pair["augmented_asset_id"]),
                )
            db.close()
            manifest.append({"asset_id": pair["augmented_asset_id"], "parent_asset_id": pair["original_asset_id"], "role": "augmented", "augmentation_type": pair.get("augmentation_type", "time"), "relative_path": str(output.relative_to(target))})

    labeling = result.get("labeling") or {}
    label_config = result["config"]["labeling"]
    if labeling.get("items"):
        labels_dir.mkdir(parents=True, exist_ok=True)
        if "segments" in label_config["channels"]:
            segment_manifest = {}
        else:
            segment_manifest = None
        for item in labeling["items"]:
            cached = load_contact_labels(item["asset_id"])
            if not cached:
                continue
            output = labels_dir / f"{item['asset_id']}__contact.npz"
            _selected_label_payload(Path(cached["sidecar_npz"]), label_config["effectors"], label_config["channels"], output)
            item["exported_label_path"] = str(output)
            if segment_manifest is not None:
                allowed = set(_expand_effector_selection(label_config["effectors"]))
                segment_manifest[item["asset_id"]] = {key: value for key, value in cached.get("segments", {}).items() if key in allowed}
            manifest.append({"asset_id": item["asset_id"], "role": "foot_contact", "relative_path": str(output.relative_to(target))})
        if segment_manifest is not None:
            _atomic_json(labels_dir / "segments.json", segment_manifest)
            for item in labeling["items"]:
                item["exported_segments_path"] = str(labels_dir / "segments.json")

    exported_at = time.time()
    recipe = {
        "process_run_id": run_id,
        "created_at": result["created_at"],
        "exported_at": exported_at,
        "config": result["config"],
        "source_asset_ids": result["asset_ids"],
        "manifest": manifest,
    }
    _atomic_json(target / "processing_recipe.json", recipe)
    _atomic_json(target / "manifest.json", {"items": manifest})
    result["storage_state"] = "exported"
    result["status"] = "exported"
    result["exported_at"] = exported_at
    result["destination"] = str(target)
    result["updated_at"] = exported_at
    _atomic_json(_run_path(run_id), result)
    _atomic_json(EXPORTED_REGISTRY / f"{run_id}.json", {"run_id": run_id, "destination": str(target), "exported_at": exported_at})
    return result
