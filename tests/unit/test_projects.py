import pytest

from alphamotion.projects import ProjectStore


def test_remove_motion_cleans_project_reference_and_bins(tmp_path):
    store = ProjectStore(tmp_path)
    project = store.create("editor")
    project = store.add_media(
        project["id"],
        motions=[
            {"asset_id": "motion-a", "library_id": 7, "name": "walk"},
            {"asset_id": "motion-b", "library_id": 8, "name": "kick"},
        ],
        bin_name="Imported",
    )

    project, removed = store.remove_media(
        project["id"], kind="motion", library_id=7
    )

    assert removed == 1
    assert [item["asset_id"] for item in project["assets"]["motions"]] == [
        "motion-b"
    ]
    assert project["assets"]["bins"][0]["asset_ids"] == ["motion-b"]


def test_remove_robot_and_reject_ambiguous_request(tmp_path):
    store = ProjectStore(tmp_path)
    project = store.create("editor")
    project = store.add_media(
        project["id"], bodies=[{"name": "unitree_h1"}, {"name": "booster_t1"}]
    )

    project, removed = store.remove_media(
        project["id"], kind="robot", name="unitree_h1"
    )

    assert removed == 1
    assert [item["name"] for item in project["assets"]["bodies"]] == [
        "booster_t1"
    ]
    with pytest.raises(ValueError, match="selector"):
        store.remove_media(project["id"], kind="motion")


def test_delete_removes_project_document_and_private_media(tmp_path):
    store = ProjectStore(tmp_path)
    project = store.create("disposable")
    media = store.media_dir(project["id"], "motions") / "take.npz"
    media.write_bytes(b"motion")

    deleted = store.delete(project["id"])

    assert deleted["id"] == project["id"]
    assert not (tmp_path / f'{project["id"]}.alphamotion-project.json').exists()
    assert not (tmp_path / project["id"]).exists()
    with pytest.raises(KeyError):
        store.get(project["id"])


def test_legacy_generation_branding_is_normalized_on_read(tmp_path):
    store = ProjectStore(tmp_path)
    project = store.create("editor")
    store.add_media(
        project["id"],
        motions=[{"asset_id": "old", "name": "GENMO · walk",
                  "origin": "genmo", "source": "GENMO"}],
        bin_name="GENMO SMPL",
    )

    restored = store.get(project["id"])
    motion = restored["assets"]["motions"][0]
    assert motion["name"] == "AlphaMotion · walk"
    assert motion["origin"] == "alphamotion"
    assert motion["source"] == "AlphaMotion"
    assert restored["assets"]["bins"][0]["name"] == "AlphaMotion SMPL"
