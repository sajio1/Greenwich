from types import SimpleNamespace

from alphamotion.service.app import (
    _library_preview_key,
    _resolve_project_library_index,
)


class FakeLibrary(SimpleNamespace):
    def __len__(self):
        return len(self.names)


def _library(name="kick", asset_id="asset-kick"):
    return FakeLibrary(
        asset_ids=[asset_id], origin_ids=[asset_id], names=[name],
        datasets=["imported_smpl"], sources=["BMLmovi"],
        source_models=["smplh"], source_genders=["female"],
        augmentations=[""], augmentation_values=[None],
        frames=lambda _index: 126,
    )


def test_hover_preview_cache_key_tracks_asset_identity():
    kick = _library()
    throw = _library(name="throw", asset_id="asset-throw")

    assert _library_preview_key(kick, 0) == _library_preview_key(kick, 0)
    assert _library_preview_key(kick, 0) != _library_preview_key(throw, 0)


def test_hover_preview_cache_key_supports_composite_metadata():
    composite = _library()
    del composite.source_models
    del composite.source_genders

    assert _library_preview_key(composite, 0)


def test_project_reference_does_not_rebind_to_unrelated_dense_row():
    library = _library(name="Imported/KIT__walk", asset_id="asset-new")
    library.sources = ["KIT"]

    assert _resolve_project_library_index({
        "library_id": 0, "asset_id": "library:0",
        "name": "SOMA__kick", "source": "SOMA",
    }, library) is None


def test_project_reference_recovers_by_stable_name_after_reorder():
    library = FakeLibrary(
        asset_ids=["asset-other", "asset-walk"],
        names=["Imported/CMU__other", "Imported/KIT__walk"],
        sources=["CMU", "KIT"],
    )

    assert _resolve_project_library_index({
        "library_id": 0, "asset_id": "library:0",
        "name": "KIT__walk", "source": "KIT",
    }, library) == 1
