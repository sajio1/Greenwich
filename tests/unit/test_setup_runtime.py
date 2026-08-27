from __future__ import annotations

import json
from pathlib import Path

from alphamotion.setup_runtime import (
    ROBOT_XML,
    _sparse_asset_roots,
    install_body_models,
    register_robot_assets,
    setup_data_studio,
)


def test_sparse_checkout_keeps_complete_robot_folders():
    roots = _sparse_asset_roots()
    assert "assets/unitree_g1" in roots
    assert "assets/fourier_gr3v2_1_1" in roots
    assert all(len(Path(value).parts) == 2 for value in roots)


def test_register_robot_assets_validates_and_writes_absolute_paths(tmp_path):
    gmr = tmp_path / "GMR"
    for relative in set(ROBOT_XML.values()):
        path = gmr / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<mujoco/>", encoding="utf-8")
    output = tmp_path / "data" / "robot_meshes.json"
    mapping = register_robot_assets(gmr, output)
    assert len(mapping) == len(ROBOT_XML)
    assert all(Path(value).is_absolute() for value in mapping.values())
    assert json.loads(output.read_text(encoding="utf-8")) == mapping


def test_install_body_models_uses_user_data_directory(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setenv("ALPHAMOTION_DATA", str(data))
    smplh = tmp_path / "smplh_300.zip"
    smplx = tmp_path / "SMPLX_NEUTRAL.npz"
    smplh.write_bytes(b"licensed-smplh")
    smplx.write_bytes(b"licensed-smplx")
    installed = install_body_models(smplh, smplx)
    assert Path(installed["smplh"]).read_bytes() == b"licensed-smplh"
    assert Path(installed["smplx"]).read_bytes() == b"licensed-smplx"
    assert Path(installed["smplh"]) == (
        data / "body_data" / "smpl_smplh" / "smplh_300.zip")
    assert Path(installed["smplx"]) == (
        data / "models" / "smplx" / "SMPLX_NEUTRAL.npz")


def test_setup_data_studio_copies_bundled_runtime(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setenv("ALPHAMOTION_DATA", str(data))
    installed = setup_data_studio(install_web=False)
    assert installed == data / "vendor" / "body-data-studio"
    assert (installed / "bodydata_server.py").is_file()
    assert (installed / "bodydata_web" / "index.html").is_file()
    assert (installed / "package-lock.json").is_file()
