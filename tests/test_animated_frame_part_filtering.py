import pytest

from src.datasets.animated_frame import (
    _filter_frame_parts,
    _selected_part_indices,
)


def _part(name):
    return {"surface_points": name, "surface_normals": name}


def test_filters_surfaces_and_all_aligned_manifest_fields():
    frame = {
        "surface_path": "frame.npy",
        "object_names": ["ball_0", "floor", "ball_1", "wall"],
        "visibility": [0.8, 1.0, 0.2, 1.0],
        "visibility_valid": [True, True, True, True],
        "observation_quality": [0.7, 1.0, 0.1, 1.0],
        "mask_touches_border": [False, False, False, True],
        "object_translation": [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
        "object_quaternion_xyzw": [[0, 0, 0, 1]] * 4,
        "visible_mask_paths": ["ball0-v", "floor-v", "ball1-v", "wall-v"],
        "amodal_mask_paths": ["ball0-a", "floor-a", "ball1-a", "wall-a"],
    }
    surface = {"parts": [_part(name) for name in frame["object_names"]]}

    filtered_frame, filtered_surface = _filter_frame_parts(
        frame, surface, ("ball_",), frozenset({"floor"})
    )

    assert filtered_frame["object_names"] == ["ball_0", "ball_1"]
    assert filtered_frame["visibility"] == [0.8, 0.2]
    assert filtered_frame["object_translation"] == [[0, 0, 0], [2, 0, 0]]
    assert filtered_frame["visible_mask_paths"] == ["ball0-v", "ball1-v"]
    assert filtered_frame["amodal_mask_paths"] == ["ball0-a", "ball1-a"]
    assert [part["surface_points"] for part in filtered_surface["parts"]] == [
        "ball_0", "ball_1"
    ]


def test_filter_requires_names_and_rejects_empty_selection():
    with pytest.raises(ValueError, match="requires object_names"):
        _selected_part_indices({}, ("ball_",), frozenset(), expected_parts=2)
    with pytest.raises(ValueError, match="removed every object"):
        _selected_part_indices(
            {"object_names": ["floor", "wall"]}, ("ball_",), frozenset(), expected_parts=2
        )


def test_filter_rejects_misaligned_metadata():
    frame = {"object_names": ["ball_0", "floor"], "visibility": [1.0]}
    surface = {"parts": [_part("ball_0"), _part("floor")]}
    with pytest.raises(ValueError, match="visibility must contain one value"):
        _filter_frame_parts(frame, surface, ("ball_",), frozenset())
