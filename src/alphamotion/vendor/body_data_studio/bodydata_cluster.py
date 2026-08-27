from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import tarfile
import time
import uuid
from collections.abc import Callable

import numpy as np
from scipy.spatial.transform import Rotation

from bodydata_config import CACHE_ROOT
from bodydata_decode import _asset_dir, _safe_name, materialize
from bodydata_index import connect


CLUSTER_ROOT = CACHE_ROOT / "clusters"
FEATURE_CACHE_ROOT = CACHE_ROOT / "cluster_features_v2"
FEATURE_SAMPLE_COUNT = 32
FEATURE_TRANSITION_LIMIT = 1024

ProgressCallback = Callable[[dict], None]


def _report(progress: ProgressCallback | None, stage: str, completed: int = 0, total: int = 0, detail: str = "") -> None:
    if progress:
        progress({"stage": stage, "completed": int(completed), "total": int(total), "detail": detail})


def _materialize_folder(assets: list[dict], progress: ProgressCallback | None = None) -> dict[str, Path]:
    """Extract compressed tar members in one streaming pass per archive."""
    paths: dict[str, Path] = {}
    grouped: dict[Path, list[dict]] = {}
    for asset in assets:
        if asset["locator_type"] == "tar_member":
            target = _asset_dir(asset) / _safe_name(Path(asset["inner_path"]).name)
            if target.exists() and target.stat().st_size > 0:
                paths[asset["id"]] = target
            else:
                grouped.setdefault(Path(asset["container"]), []).append(asset)
        else:
            paths[asset["id"]] = materialize(asset)
        _report(progress, "materializing", len(paths), len(assets), "Preparing motion files")
    for archive, pending in grouped.items():
        wanted = {asset["inner_path"]: asset for asset in pending}
        mode = "r|bz2" if archive.name.lower().endswith(".tar.bz2") else "r|gz"
        with tarfile.open(archive, mode) as source:
            for info in source:
                asset = wanted.pop(info.name, None)
                if asset is None:
                    continue
                member = source.extractfile(info)
                if member is None:
                    continue
                target = _asset_dir(asset) / _safe_name(Path(asset["inner_path"]).name)
                temporary = target.with_suffix(target.suffix + ".tmp")
                with member, temporary.open("wb") as output:
                    shutil.copyfileobj(member, output, length=4 * 1024 * 1024)
                temporary.replace(target)
                paths[asset["id"]] = target
                _report(progress, "materializing", len(paths), len(assets), "Preparing motion files")
                if not wanted:
                    break
        if wanted:
            raise RuntimeError(f"Archive is missing {len(wanted)} requested motion files")
    return paths


def _resample(signal: np.ndarray, samples: int) -> np.ndarray:
    signal = np.asarray(signal, np.float32)
    if len(signal) == 1:
        return np.repeat(signal, samples, axis=0)
    source = np.linspace(0.0, 1.0, len(signal))
    target = np.linspace(0.0, 1.0, samples)
    return np.stack([np.interp(target, source, signal[:, column]) for column in range(signal.shape[1])], axis=1).astype(np.float32)


def _transition_indices(frame_count: int, limit: int = FEATURE_TRANSITION_LIMIT) -> tuple[np.ndarray, np.ndarray]:
    """Choose representative adjacent frame pairs without decoding every rotation."""
    count = min(max(frame_count - 1, 1), limit)
    ends = np.unique(np.linspace(1, frame_count - 1, count).round().astype(np.int64))
    return ends - 1, ends


def _motion_feature(path: Path, samples: int = FEATURE_SAMPLE_COUNT) -> tuple[list[np.ndarray], dict]:
    with np.load(path, allow_pickle=True) as data:
        if "poses" not in data.files:
            raise ValueError("NPZ has no poses array")
        poses = np.asarray(data["poses"], np.float32)
        translations = np.asarray(data["trans"], np.float32) if "trans" in data.files else np.zeros((len(poses), 3), np.float32)
        fps = 30.0
        for key in ("mocap_framerate", "mocap_frame_rate", "fps"):
            if key in data.files:
                fps = float(np.asarray(data[key]).item())
                break
    if poses.ndim != 2 or poses.shape[0] < 2 or poses.shape[1] < 66:
        raise ValueError(f"Expected body poses [frames,>=66], got {poses.shape}")
    # A clip is represented by joint synergies, not only flattened poses.
    # Exclude global orientation from local-joint coordination.
    starts, ends = _transition_indices(len(poses))
    posture_indices = np.unique(np.linspace(0, len(poses) - 1, samples).round().astype(np.int64))
    required = np.unique(np.concatenate((starts, ends, posture_indices)))
    body = poses[required, 3:66].reshape(len(required), 21, 3)
    matrices = Rotation.from_rotvec(body.reshape(-1, 3)).as_matrix().reshape(len(body), 21, 3, 3)
    lookup = np.full(len(poses), -1, np.int64)
    lookup[required] = np.arange(len(required))
    start_matrices, end_matrices = matrices[lookup[starts]], matrices[lookup[ends]]
    relative = np.einsum("tjki,tjkl->tjil", start_matrices, end_matrices)
    activity = Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude().reshape(len(starts), 21) * fps
    # Sparse mocap fitting spikes should be reported by QC, not dominate action identity.
    cap = np.quantile(activity, 0.99, axis=0, keepdims=True)
    robust_activity = np.minimum(activity, cap)
    temporal_activity = np.log1p(_resample(robust_activity, samples)).reshape(-1)

    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.corrcoef(robust_activity, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    synergy_correlation = correlation[np.triu_indices(21, 1)].astype(np.float32)
    spectrum = np.sort(np.maximum(np.linalg.eigvalsh(correlation), 0.0))[::-1]
    synergy_spectrum = (spectrum / max(float(spectrum.sum()), 1e-8)).astype(np.float32)
    energy = np.concatenate((np.log1p(robust_activity.mean(axis=0)), np.log1p(robust_activity.std(axis=0)))).astype(np.float32)

    posture_matrices = matrices[lookup[posture_indices]]
    rotation6d = posture_matrices[..., :2].reshape(len(posture_indices), -1)
    posture = np.concatenate((rotation6d.mean(axis=0), rotation6d.std(axis=0))).astype(np.float32)

    root_linear = (translations[ends] - translations[starts]) * fps
    horizontal_speed = np.linalg.norm(root_linear[:, :2], axis=1)
    vertical_speed = np.abs(root_linear[:, 2])
    root_start = Rotation.from_rotvec(poses[starts, :3]).as_matrix()
    root_end = Rotation.from_rotvec(poses[ends, :3]).as_matrix()
    root_relative = np.einsum("tki,tkl->til", root_start, root_end)
    root_angular_speed = Rotation.from_matrix(root_relative).magnitude() * fps
    root_motion = np.log1p(_resample(np.stack((horizontal_speed, vertical_speed, root_angular_speed), axis=1), samples)).reshape(-1)

    blocks = [temporal_activity, synergy_correlation, synergy_spectrum, energy, posture, root_motion]
    return blocks, {
        "frames": int(len(poses)), "fps": fps, "duration": float((len(poses) - 1) / max(fps, 1.0)),
        "synergy_activity_cap_quantile": 0.99,
    }


def _cached_motion_feature(asset: dict, path: Path) -> tuple[list[np.ndarray], dict, bool]:
    FEATURE_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_path = FEATURE_CACHE_ROOT / f"{_safe_name(asset['id'])}.npz"
    if cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                blocks = [np.asarray(data[f"block_{index}"], np.float32) for index in range(6)]
                metadata = {
                    "frames": int(data["frames"]), "fps": float(data["fps"]),
                    "duration": float(data["duration"]), "synergy_activity_cap_quantile": 0.99,
                }
            return blocks, metadata, True
        except (OSError, ValueError, KeyError):
            pass
    blocks, metadata = _motion_feature(path)
    temporary = cache_path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle, **{f"block_{index}": block for index, block in enumerate(blocks)},
            frames=np.int64(metadata["frames"]), fps=np.float64(metadata["fps"]),
            duration=np.float64(metadata["duration"]),
        )
    temporary.replace(cache_path)
    return blocks, metadata, False


def _pca(feature_blocks: list[list[np.ndarray]], variance_target: float = 0.9, max_components: int = 20) -> tuple[np.ndarray, np.ndarray]:
    # Equalize feature families so temporal samples do not win merely by having more columns.
    weights = (1.35, 2.0, 1.0, 1.0, 1.0, 1.0)
    normalized_blocks = []
    for block_index, weight in enumerate(weights):
        block = np.stack([features[block_index] for features in feature_blocks]).astype(np.float64)
        centered = block - block.mean(axis=0, keepdims=True)
        scale = centered.std(axis=0, keepdims=True)
        normalized = centered / np.where(scale > 1e-6, scale, 1.0)
        normalized_blocks.append(normalized * (weight / np.sqrt(max(1, block.shape[1]))))
    normalized = np.concatenate(normalized_blocks, axis=1)
    if len(feature_blocks) > 256 and min(normalized.shape) > max_components + 1:
        from scipy.sparse.linalg import svds

        u, singular, _ = svds(normalized, k=max_components, which="LM", random_state=7)
        order = np.argsort(singular)[::-1]
        u, singular = u[:, order], singular[order]
        total_variance = float(np.sum(normalized**2))
        full_ratio = singular**2 / max(total_variance, 1e-12)
    else:
        u, singular, _ = np.linalg.svd(normalized, full_matrices=False)
        variance = singular**2
        full_ratio = variance / max(float(variance.sum()), 1e-12)
    count = int(np.searchsorted(np.cumsum(full_ratio), variance_target) + 1)
    count = max(2, min(count, max_components, len(feature_blocks) - 1, u.shape[1]))
    embedding = u[:, :count] * singular[:count]
    ratio = full_ratio[:count]
    return embedding.astype(np.float32), ratio.astype(np.float32)


def _kmeans(points: np.ndarray, clusters: int, seed: int = 7, iterations: int = 100) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    clusters = max(1, min(int(clusters), len(points)))
    centers = [points[int(rng.integers(len(points)))]]
    for _ in range(1, clusters):
        distances = np.min(np.stack([np.sum((points - center) ** 2, axis=1) for center in centers]), axis=0)
        total = float(distances.sum())
        index = int(rng.choice(len(points), p=distances / total)) if total > 0 else int(rng.integers(len(points)))
        centers.append(points[index])
    centers = np.asarray(centers, np.float32)
    labels = np.zeros(len(points), np.int32)
    for _ in range(iterations):
        distance = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        updated = distance.argmin(axis=1).astype(np.int32)
        new_centers = centers.copy()
        for cluster in range(clusters):
            members = points[updated == cluster]
            if len(members):
                new_centers[cluster] = members.mean(axis=0)
        if np.array_equal(updated, labels) and np.allclose(new_centers, centers):
            labels = updated
            centers = new_centers
            break
        labels, centers = updated, new_centers
    inertia = float(np.sum((points - centers[labels]) ** 2))
    return labels, centers, inertia


def _silhouette(points: np.ndarray, labels: np.ndarray) -> float:
    if len(points) > 500:
        sample_indices = np.random.default_rng(7).choice(len(points), size=500, replace=False)
    else:
        sample_indices = np.arange(len(points))
    values = []
    for index in sample_indices:
        label = labels[index]
        distances = np.linalg.norm(points - points[index], axis=1)
        same = np.where(labels == label)[0]
        same = same[same != index]
        if len(same) == 0:
            values.append(0.0)
            continue
        within = float(distances[same].mean())
        other = [float(distances[labels == candidate].mean()) for candidate in np.unique(labels) if candidate != label]
        nearest = min(other)
        values.append((nearest - within) / max(within, nearest, 1e-12))
    return float(np.mean(values))


def _adjusted_rand(left: np.ndarray, right: np.ndarray) -> float:
    left_values, left_inverse = np.unique(left, return_inverse=True)
    right_values, right_inverse = np.unique(right, return_inverse=True)
    table = np.zeros((len(left_values), len(right_values)), np.int64)
    np.add.at(table, (left_inverse, right_inverse), 1)
    choose2 = lambda values: np.sum(values * (values - 1) / 2)
    sum_cells = float(choose2(table))
    sum_rows = float(choose2(table.sum(axis=1)))
    sum_cols = float(choose2(table.sum(axis=0)))
    total = len(left) * (len(left) - 1) / 2
    expected = sum_rows * sum_cols / max(total, 1.0)
    maximum = 0.5 * (sum_rows + sum_cols)
    return float((sum_cells - expected) / max(maximum - expected, 1e-12))


def _fit_k(points: np.ndarray, clusters: int) -> tuple[np.ndarray, np.ndarray, dict]:
    trials = [_kmeans(points, clusters, seed=seed) for seed in (7, 17, 31, 47, 73)]
    best_index = int(np.argmin([trial[2] for trial in trials]))
    labels, centers, inertia = trials[best_index]
    stability = float(np.mean([_adjusted_rand(labels, trial[0]) for index, trial in enumerate(trials) if index != best_index]))
    counts = np.bincount(labels, minlength=clusters)
    probabilities = counts / max(float(counts.sum()), 1.0)
    balance = float(-np.sum(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0])) / np.log(clusters)) if clusters > 1 else 1.0
    silhouette = _silhouette(points, labels)
    tiny_penalty = 0.25 if int(counts.min()) < 2 else 0.0
    score = 0.65 * silhouette + 0.25 * stability + 0.10 * balance - tiny_penalty
    order = np.argsort(centers[:, 0])
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    labels = remap[labels]
    centers = centers[order]
    return labels, centers, {
        "k": int(clusters), "score": float(score), "silhouette": silhouette, "stability": stability,
        "balance": balance, "minimum_cluster_size": int(counts.min()), "inertia": inertia,
    }


def _choose_clusters(points: np.ndarray, requested: int | None) -> tuple[np.ndarray, np.ndarray, int, list[dict]]:
    if requested is not None:
        labels, centers, metric = _fit_k(points, max(2, min(int(requested), len(points))))
        return labels, centers, int(metric["k"]), [metric]
    maximum = min(12, max(2, len(points) // 2))
    candidates = []
    fitted = {}
    for clusters in range(2, maximum + 1):
        labels, centers, metric = _fit_k(points, clusters)
        candidates.append(metric)
        fitted[clusters] = (labels, centers)
    best = max(candidates, key=lambda item: item["score"])
    labels, centers = fitted[int(best["k"])]
    return labels, centers, int(best["k"]), candidates


def cluster_scope(source: str, folder: str = "", clusters: int | None = None, progress: ProgressCallback | None = None) -> dict:
    _report(progress, "querying", detail="Finding supported motions")
    db = connect()
    if folder:
        rows = db.execute(
            "SELECT * FROM assets WHERE source=? AND folder=? AND status='ready' AND animated=1 ORDER BY title COLLATE NOCASE",
            (source, folder),
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT * FROM assets WHERE source=? AND status='ready' AND animated=1 ORDER BY folder COLLATE NOCASE,title COLLATE NOCASE",
            (source,),
        ).fetchall()
    db.close()
    assets = [dict(row) for row in rows]
    supported = [asset for asset in assets if asset["kind"] == "smplh_motion"]
    if len(supported) < 2:
        raise ValueError("This folder needs at least two supported SMPL-H motion clips")
    if clusters is None and len(supported) < 10:
        raise ValueError(f"Automatic clustering needs at least 10 motion clips; this scope contains only {len(supported)}")
    paths = _materialize_folder(supported, progress)
    features, metadata, included = [], [], []
    failures = []
    cache_hits = 0
    for index, asset in enumerate(supported, 1):
        try:
            feature, detail, cached = _cached_motion_feature(asset, paths[asset["id"]])
            cache_hits += int(cached)
            features.append(feature)
            metadata.append(detail)
            included.append(asset)
        except Exception as exc:
            failures.append({"id": asset["id"], "title": asset["title"], "error": str(exc)})
        _report(progress, "features", index, len(supported), f"{cache_hits} loaded from cache")
    if len(features) < 2:
        raise ValueError("Fewer than two clips produced valid motion features")
    _report(progress, "pca", 0, len(features), "Reducing the motion feature space")
    embedding, variance_ratio = _pca(features)
    _report(progress, "clustering", 0, len(features), "Fitting and validating clusters")
    labels, centers, cluster_count, model_selection = _choose_clusters(embedding, clusters)
    chosen_metric = next(metric for metric in model_selection if metric["k"] == cluster_count)
    if clusters is not None:
        structure_confidence = "manual"
    elif chosen_metric["silhouette"] >= 0.35 and chosen_metric["stability"] >= 0.70:
        structure_confidence = "high"
    elif chosen_metric["silhouette"] >= 0.20 and chosen_metric["stability"] >= 0.50:
        structure_confidence = "medium"
    else:
        structure_confidence = "low"
    run_id = uuid.uuid4().hex[:20]
    created_at = time.time()
    output_dir = CLUSTER_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    members = []
    for index, (asset, detail) in enumerate(zip(included, metadata)):
        members.append({
            "asset_id": asset["id"], "title": asset["title"], "cluster": int(labels[index]),
            "pca_x": float(embedding[index, 0]), "pca_y": float(embedding[index, 1] if embedding.shape[1] > 1 else 0.0),
            **detail,
        })
    config = {
        "method": "Synergy features + PCA + stable K-Means", "source": source, "folder": folder, "scope": "folder" if folder else "source", "clusters": cluster_count,
        "cluster_selection": "manual" if clusters is not None else "automatic multi-metric selection",
        "structure_confidence": structure_confidence,
        "selection_note": (
            "weak discrete structure; treat clusters as exploratory and preserve continuous PCA coordinates"
            if structure_confidence == "low" else "discrete cluster structure passed the configured confidence thresholds"
        ),
        "clips": len(included),
        "feature": "joint angular-velocity coactivation, correlation matrix, synergy spectrum, temporal energy, posture and root motion",
        "feature_blocks": ["temporal joint activity", "joint coactivation correlation", "synergy spectrum", "joint energy", "posture", "root motion"],
        "pca_components": int(embedding.shape[1]), "explained_variance_ratio": variance_ratio.astype(float).tolist(),
        "model_selection": model_selection,
        "failures": failures,
    }
    result = {"run_id": run_id, "created_at": created_at, "config": config, "members": members}
    _report(progress, "saving", len(included), len(included), "Saving analysis results")
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "members.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["asset_id", "title", "cluster", "pca_x", "pca_y", "frames", "fps", "duration", "synergy_activity_cap_quantile"])
        writer.writeheader()
        writer.writerows(members)
    db = connect()
    with db:
        db.execute("INSERT INTO cluster_runs(id,name,config_json,created_at) VALUES(?,?,?,?)", (run_id, f"{Path(folder).name if folder else source} PCA", json.dumps(config, ensure_ascii=False), created_at))
        db.executemany(
            "INSERT INTO cluster_members(run_id,asset_id,cluster_id,x,y) VALUES(?,?,?,?,?)",
            [(run_id, member["asset_id"], member["cluster"], member["pca_x"], member["pca_y"]) for member in members],
        )
    db.close()
    _report(progress, "complete", len(included), len(included), "Analysis ready")
    return result


def cluster_folder(source: str, folder: str, clusters: int | None = None) -> dict:
    """Backward-compatible wrapper for existing callers and saved workflows."""
    return cluster_scope(source, folder, clusters)


def load_cluster(run_id: str) -> dict | None:
    path = CLUSTER_ROOT / run_id / "result.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
