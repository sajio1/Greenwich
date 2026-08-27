"""AlphaMotion CLI: serve / download / eval / bench / ingest-urdf / doctor."""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="AlphaMotion motion engine")


@app.command()
def doctor():
    """Self-check: platform, GPU, GL, ffmpeg, weights, DB, ports."""
    from . import __version__
    from .config import CONFIG
    from .paths import cache_dir, data_dir, db_path
    rows: list[tuple[str, str]] = [
        ("alphamotion", __version__),
        ("platform", f"{platform.system()} {platform.release()}"),
        ("python", sys.version.split()[0]),
    ]
    try:
        import torch
        rows.append(("torch", f"{torch.__version__} cuda={torch.cuda.is_available()}"))
    except Exception as e:
        rows.append(("torch", f"FAIL {e}"))
    try:
        import mujoco
        rows.append(("mujoco", mujoco.__version__))
    except Exception as e:
        rows.append(("mujoco", f"FAIL {e}"))
    try:
        import imageio_ffmpeg
        rows.append(("ffmpeg", imageio_ffmpeg.get_ffmpeg_exe()))
    except Exception as e:
        rows.append(("ffmpeg", f"FAIL {e}"))
    from .weights import status
    for k, ok in status().items():
        rows.append((f"weights/{k}", "ok" if ok else "not downloaded"))
    from .perception.genmo import status as perception_status
    perception = perception_status()
    rows.append(("perception/text", "ok (AlphaMotion)" if perception["text"]
                 else "not configured"))
    rows.append(("perception/video", "ok (AlphaMotion)" if perception["video"]
                 else "not configured"))
    rows.append(("cache", str(cache_dir())))
    rows.append(("data", str(data_dir())))
    rows.append(("db", str(db_path())))
    rows.append(("device", CONFIG.device))
    from .embodiment import registry
    rows.append(("robot meshes", f"{len(registry.mesh_map())} configured"))
    studio = Path(CONFIG.data_studio_repo) / "bodydata_server.py"
    rows.append(("Data Studio", "ok" if studio.is_file() else "not installed"))
    smplh = Path(CONFIG.data_studio_root) / "smpl_smplh" / "smplh_300.zip"
    rows.append(("SMPL-H", "installed" if smplh.is_file()
                 else "user download required"))
    smplx = data_dir() / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
    rows.append(("SMPL-X", "installed" if smplx.is_file()
                 else "user download required"))
    imported = data_dir() / "imported_smpl"
    count = len(list(imported.glob("*/library.npz"))) if imported.is_dir() else 0
    rows.append(("original motion shards", str(count)))
    width = max(len(k) for k, _ in rows)
    bad = 0
    for k, v in rows:
        flag = "!!" if "FAIL" in str(v) else "  "
        bad += "FAIL" in str(v)
        typer.echo(f" {flag} {k:<{width}}  {v}")
    raise typer.Exit(1 if bad else 0)


@app.command()
def download(third_party: bool = typer.Option(False, "--third-party",
                                              help="show perception setup")):
    """Fetch model weights from the HF hub (first run)."""
    from .weights import download_all
    out = download_all()
    for k, v in out.items():
        typer.echo(f"  {k}: {v}")
    if third_party:
        typer.echo("Text/video generation uses the separately configured "
                   "AlphaMotion generation worker; see deploy/README.md.")


@app.command()
def setup(
    weights: bool = typer.Option(True, "--weights/--no-weights"),
    robots: bool = typer.Option(True, "--robots/--no-robots"),
    data_studio: bool = typer.Option(True, "--data-studio/--no-data-studio"),
):
    """Prepare a public checkout for first use."""
    if weights:
        from .weights import download_all
        typer.echo("Downloading AlphaMotion release artifacts…")
        for name, path in download_all().items():
            typer.echo(f"  {name}: {path}")
    if robots:
        from .setup_runtime import setup_robots
        typer.echo("Installing pinned robot URDF/MJCF and mesh assets…")
        mapping = setup_robots()
        typer.echo(f"  {len(mapping)} renderable robot bodies configured")
    if data_studio:
        from .setup_runtime import setup_data_studio
        typer.echo("Installing Body Data Studio…")
        typer.echo(f"  {setup_data_studio()}")
    typer.echo("Setup complete. Licensed SMPL models and AMASS motions are "
               "optional user downloads; see docs/INSTALL.md.")


@app.command("install-body-models")
def install_body_models(
    smplh_archive: Path | None = typer.Option(
        None, "--smplh-archive", help="official smplh_300.zip"),
    smplx_model: Path | None = typer.Option(
        None, "--smplx-model", help="official SMPLX_NEUTRAL.npz"),
):
    """Install licensed SMPL files downloaded by the user."""
    from .setup_runtime import install_body_models as install
    for name, path in install(smplh_archive, smplx_model).items():
        typer.echo(f"  {name}: {path}")


@app.command("import-smpl-library")
def import_smpl_library(
    input_root: Path = typer.Option(
        ..., "--input", help="directory containing user-downloaded archives"),
    sources: list[str] = typer.Option([], "--source",
                                      help="KIT, CMU, BMLmovi, MOYO, DanceDB"),
):
    """Build exact-source motion shards from user-downloaded AMASS archives."""
    from .setup_runtime import import_smpl_library as run_import
    run_import(input_root, sources or None)


@app.command()
def serve(host: str = "", port: int = 0):
    """Start the AlphaMotion service (API + frontend + viewers)."""
    import uvicorn

    from .config import CONFIG
    from .service.app import create_app
    uvicorn.run(create_app(),
                host=host or CONFIG.host, port=port or CONFIG.port)


@app.command("ingest-urdf")
def ingest_urdf(path: str, name: str = ""):
    """Parse + check + register a user URDF (offline entry)."""
    from .embodiment.urdf_ingest import ingest
    report = ingest(path, name or None)
    typer.echo(json.dumps(report, indent=2, ensure_ascii=False))


@app.command()
def eval(out: str = "docs/BENCHMARK.md"):
    """Run the full release gate and write the benchmark table."""
    from .utils.eval_gate import run_gate
    ok = run_gate(out)
    raise typer.Exit(0 if ok else 1)


@app.command()
def bench():
    """Latency of the warm pool (encode/decode/bridge)."""
    from .utils.eval_gate import run_bench
    run_bench()


def main():
    app()


if __name__ == "__main__":
    main()
