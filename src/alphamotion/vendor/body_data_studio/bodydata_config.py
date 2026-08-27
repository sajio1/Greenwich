from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys


APP_NAME = "body-data-studio"


def _platform_config_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "BodyDataStudio"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / APP_NAME


def _platform_cache_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "BodyDataStudio" / "cache"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / APP_NAME


CONFIG_DIR = _platform_config_dir()
CONFIG_PATH = CONFIG_DIR / "config.json"


def load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}


def save_config(config: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    temp = CONFIG_PATH.with_suffix(".tmp")
    temp.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(CONFIG_PATH)


def default_data_root() -> Path:
    env = os.environ.get("BODY_DATA_ROOT")
    if env:
        return Path(env).expanduser()
    configured = load_config().get("data_root")
    if configured:
        return Path(configured).expanduser()
    candidates = [Path(r"D:\body_data")] if sys.platform == "win32" else [Path("/data/body_data")]
    candidates.append(Path.home() / "body_data")
    return next((path for path in candidates if path.exists()), candidates[-1])


def resolve_cache_root(data_root: Path) -> Path:
    env = os.environ.get("BODY_DATA_CACHE")
    if env:
        return Path(env).expanduser()
    configured = load_config().get("cache_root")
    if configured:
        return Path(configured).expanduser()
    # Reuse an existing cache so upgrades do not force expensive re-decoding.
    legacy = data_root / "_preview_cache"
    if legacy.exists():
        return legacy
    return _platform_cache_dir()


def find_7zip() -> Path | None:
    explicit = os.environ.get("BODY_DATA_7ZIP")
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    for name in ("7zz", "7z", "7za"):
        found = shutil.which(name)
        if found:
            return Path(found)
    if sys.platform == "win32":
        candidate = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "7-Zip" / "7z.exe"
        if candidate.is_file():
            return candidate
    return None


DATA_ROOT = default_data_root()
CACHE_ROOT = resolve_cache_root(DATA_ROOT)
DB_PATH = CACHE_ROOT / "bodydata_studio.sqlite3"
SEVEN_ZIP = find_7zip()


def configure_paths(data_root: str | Path | None = None, cache_root: str | Path | None = None) -> None:
    """Persist user-selected paths for future launches."""
    config = load_config()
    if data_root is not None:
        config["data_root"] = str(Path(data_root).expanduser().resolve())
    if cache_root is not None:
        config["cache_root"] = str(Path(cache_root).expanduser().resolve())
    save_config(config)
