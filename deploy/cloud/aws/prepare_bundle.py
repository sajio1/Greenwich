#!/usr/bin/env python3
"""Create the small Docker build context without copying datasets or caches."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

IGNORED_NAMES = {
    ".git", ".pytest_cache", "__pycache__", "node_modules", ".venv",
    ".bundle", "env", "envs", "artifacts", "outputs", "MUJOCO_LOG.TXT",
}


def ignore(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names
            if name in IGNORED_NAMES or name.endswith((".pyc", ".log"))}


def main() -> None:
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--body-data-studio", type=Path,
        default=Path("/media/sajio/New Volume/BodyDataStudio"),
        help="BodyDataStudio checkout used by the current local Studio")
    parser.add_argument("--output", type=Path, default=here / ".bundle")
    args = parser.parse_args()

    body = args.body_data_studio.expanduser().resolve()
    if not (body / "bodydata_server.py").is_file():
        raise SystemExit(f"BodyDataStudio checkout not found: {body}")
    output = args.output.expanduser().resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    shutil.copytree(repo, output / "alphamotion", ignore=ignore)
    shutil.copytree(body, output / "body-data-studio", ignore=ignore)
    shutil.copy2(here / "Dockerfile", output / "Dockerfile")
    print(f"Prepared Docker context: {output}")


if __name__ == "__main__":
    main()
