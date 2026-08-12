from __future__ import annotations

import numpy as np
import pytest

from trajplayer.interaction.models import AnalysisResult, SelectionSnapshot, TimeAxis
from trajplayer.interaction.picking import (
    decode_pick_id_rgba8,
    encode_pick_id_rgba8,
    physical_pick_pixel,
)
from trajplayer.interaction.selection_manager import SelectionManager, SelectionOp


def test_time_axis_never_invents_physical_time() -> None:
    axis = TimeAxis(frame_count=4)
    np.testing.assert_array_equal(axis.values_for_frames(np.array([0, 2, 3])), [0.0, 2.0, 3.0])
    assert axis.unit == "frame"

    physical = TimeAxis(frame_count=4, unit="ps", dt=0.5)
    np.testing.assert_allclose(physical.values_for_frames(np.array([0, 2, 3])), [0.0, 1.0, 1.5])


def test_selection_snapshot_and_analysis_result_are_read_only() -> None:
    snapshot = SelectionSnapshot(
        atom_indices=np.array([1, 4], dtype=np.uint32),
        primary_atom=4,
        revision=2,
        trajectory_generation=3,
    )
    with pytest.raises(ValueError):
        snapshot.atom_indices[0] = 3

    result = AnalysisResult(
        kind="rmsd",
        x=np.arange(3),
        y=np.arange(3, dtype=np.float32),
        x_unit="frame",
        y_unit="A",
        source_frames=(0, 3, 1),
        selection_revision=2,
        trajectory_generation=3,
    )
    with pytest.raises(ValueError):
        result.y[0] = 4.0


def test_selection_manager_uses_canonical_sorted_indices() -> None:
    manager = SelectionManager()
    manager.begin_trajectory(8, 11)
    manager.replace([5, 1, 5])
    np.testing.assert_array_equal(manager.snapshot().atom_indices, [1, 5])
    np.testing.assert_array_equal(manager.selection_order(), [5, 1])
    assert manager.snapshot().primary_atom == 5

    manager.select_atom(3, SelectionOp.ADD)
    np.testing.assert_array_equal(manager.snapshot().atom_indices, [1, 3, 5])
    np.testing.assert_array_equal(manager.selection_order(), [5, 1, 3])
    manager.select_atom(1, SelectionOp.TOGGLE)
    np.testing.assert_array_equal(manager.snapshot().atom_indices, [3, 5])
    manager.remove([5])
    np.testing.assert_array_equal(manager.snapshot().atom_indices, [3])
    manager.clear()
    assert manager.snapshot().atom_indices.size == 0


def test_selection_manager_resets_on_trajectory_generation() -> None:
    manager = SelectionManager()
    manager.begin_trajectory(3, 1)
    manager.replace([2])
    manager.begin_trajectory(2, 2)
    snapshot = manager.snapshot()
    assert snapshot.trajectory_generation == 2
    assert snapshot.atom_indices.size == 0
    with pytest.raises(IndexError):
        manager.replace([2])


def test_rgba8_pick_ids_round_trip_canonical_atom_indices() -> None:
    for atom_index in (0, 1, 999_999, 0xFFFFFF - 1):
        assert decode_pick_id_rgba8(encode_pick_id_rgba8(atom_index)) == atom_index
    assert decode_pick_id_rgba8((0, 0, 0, 255)) is None


@pytest.mark.parametrize(
    ("dpr", "expected"),
    [(1.0, (25, 59, 100, 80)), (1.25, (31, 74, 125, 100)), (2.0, (50, 119, 200, 160))],
)
def test_pick_pixel_is_hidpi_aware(dpr: float, expected: tuple[int, int, int, int]) -> None:
    assert physical_pick_pixel(25.0, 20.0, 100, 80, dpr) == expected
