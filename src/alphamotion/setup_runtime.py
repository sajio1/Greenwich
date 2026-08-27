"""Reproducible first-run setup for public AlphaMotion checkouts.

AlphaMotion-owned weights are downloaded from Hugging Face. Robot visuals are
checked out from a pinned GMR revision so their original license files remain
next to the vendor assets. Licensed human body models are never downloaded or
redistributed here; users install files obtained from the official websites.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from .paths import data_dir

GMR_REPO = "https://github.com/YanjieZe/GMR.git"
GMR_REVISION = "bb1bbe40774794fceb2a7c579a3464a28e68c844"

ROBOT_XML = {
    "booster_k1": "assets/booster_k1/K1_serial.xml",
    "booster_t1": "assets/booster_t1/T1_serial.xml",
    "booster_t1_29dof": "assets/booster_t1_29dof/t1_mocap.xml",
    "engineai_pm01": "assets/engineai_pm01/pm_v2.xml",
    "fourier_gr3": "assets/fourier_gr3v2_1_1/mjcf/gr3v2_1_1.xml",
    "fourier_n1": "assets/fourier_n1/n1_mocap.xml",
    "hightorque_hi": "assets/hightorque_hi/hi_25dof.xml",
    "kuavo_s45": "assets/kuavo_s45/biped_s45_collision.xml",
    "pal_talos": "assets/pal_talos/talos.xml",
    "pnd_adam_lite": "assets/pnd_adam_lite/adam_lite.xml",
    "tienkung": "assets/tienkung/mjcf/tienkung.xml",
    "unitree_g1_23dof": "assets/unitree_g1/g1_mocap_29dof.xml",
    "unitree_g1_29dof": "assets/unitree_g1/g1_mocap_29dof.xml",
    "unitree_h1": "assets/unitree_h1/h1.xml",
    "unitree_h1_2": "assets/unitree_h1_2/h1_2_handless.xml",
}


def _run(args: list[str], cwd: Path | None = None) -> None:
    try:
        subprocess.run(args, cwd=cwd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"required program is not installed: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"command failed ({exc.returncode}): {' '.join(args)}") from exc


def _sparse_asset_roots() -> list[str]:
    roots = set()
    for value in ROBOT_XML.values():
        parts = Path(value).parts
        roots.add(str(Path(*parts[:2])))
    return sorted(roots)


def _checkout_gmr(vendor_root: Path, repo: str = GMR_REPO,
                  revision: str = GMR_REVISION) -> Path:
    """Create an immutable sparse checkout containing the renderable robots."""
    target = vendor_root / f"GMR-{revision[:12]}"
    if (target / ".alphamotion-revision").is_file():
        return target
    vendor_root.mkdir(parents=True, exist_ok=True)
    staging = vendor_root / f".GMR-{uuid.uuid4().hex}.tmp"
    try:
        _run(["git", "init", str(staging)])
        _run(["git", "remote", "add", "origin", repo], cwd=staging)
        _run(["git", "config", "extensions.partialClone", "origin"], cwd=staging)
        _run(["git", "config", "remote.origin.promisor", "true"], cwd=staging)
        _run(["git", "config", "remote.origin.partialclonefilter", "blob:none"],
             cwd=staging)
        _run(["git", "sparse-checkout", "init", "--cone"], cwd=staging)
        _run(["git", "sparse-checkout", "set", *_sparse_asset_roots()],
             cwd=staging)
        _run(["git", "fetch", "--depth", "1", "--filter=blob:none",
              "origin", revision], cwd=staging)
        _run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=staging)
        (staging / ".alphamotion-revision").write_text(
            revision + "\n", encoding="utf-8")
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def register_robot_assets(gmr_root: Path, output: Path | None = None) -> dict[str, str]:
    """Validate pinned robot files and write AlphaMotion's runtime mesh map."""
    mapping: dict[str, str] = {}
    missing: list[str] = []
    for name, relative in ROBOT_XML.items():
        path = (gmr_root / relative).resolve()
        if not path.is_file():
            missing.append(f"{name}: {relative}")
        else:
            mapping[name] = str(path)
    if missing:
        raise FileNotFoundError("robot assets are incomplete:\n  "
                                + "\n  ".join(missing))
    destination = output or data_dir() / "robot_meshes.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return mapping


def setup_robots(repo: str = GMR_REPO,
                 revision: str = GMR_REVISION) -> dict[str, str]:
    root = _checkout_gmr(data_dir() / "vendor", repo, revision)
    return register_robot_assets(root)


def _body_data_source() -> Path:
    return Path(__file__).parent / "vendor" / "body_data_studio"


def setup_data_studio(install_web: bool = True) -> Path:
    """Install the bundled Body Data Studio worker into writable user data."""
    source = _body_data_source()
    if not (source / "bodydata_server.py").is_file():
        raise FileNotFoundError(f"bundled Body Data Studio is missing: {source}")
    target = data_dir() / "vendor" / "body-data-studio"
    shutil.copytree(source, target, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    if install_web and (target / "package-lock.json").is_file():
        if shutil.which("npm") is None:
            raise RuntimeError(
                "Node.js/npm is required for the Data Studio web viewer. "
                "Install Node.js 20+ and run `alphamotion setup --no-weights "
                "--no-robots` again.")
        _run(["npm", "ci", "--omit=dev", "--no-audit", "--no-fund"], cwd=target)
    return target


def install_body_models(smplh_archive: Path | None = None,
                        smplx_model: Path | None = None) -> dict[str, str]:
    """Install user-downloaded licensed models without fetching them for users."""
    installed: dict[str, str] = {}
    if smplh_archive is not None:
        source = smplh_archive.expanduser().resolve()
        if not source.is_file() or source.suffix.lower() != ".zip":
            raise FileNotFoundError(f"SMPL-H archive not found: {source}")
        destination = data_dir() / "body_data" / "smpl_smplh" / "smplh_300.zip"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        installed["smplh"] = str(destination)
    if smplx_model is not None:
        source = smplx_model.expanduser().resolve()
        if not source.is_file() or source.suffix.lower() != ".npz":
            raise FileNotFoundError(f"SMPL-X neutral model not found: {source}")
        destination = data_dir() / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        installed["smplx"] = str(destination)
    if not installed:
        raise ValueError("provide --smplh-archive and/or --smplx-model")
    return installed


def import_smpl_library(input_root: Path, sources: list[str] | None = None) -> None:
    """Run the repository importer against user-downloaded AMASS archives."""
    project = Path(__file__).resolve().parents[2]
    script = project / "scripts" / "import_smpl_library.py"
    if not script.is_file():
        raise FileNotFoundError(
            "scripts/import_smpl_library.py is unavailable; run this command "
            "from a Git checkout installed with `pip install -e .`")
    args = [sys.executable, str(script), "--input", str(input_root.expanduser()),
            "--output", str(data_dir() / "imported_smpl")]
    if sources:
        args.extend(["--sources", *sources])
    _run(args, cwd=project)
