from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest
from ase.io import read

from trajplayer.exporter import write_analysis_csv, write_xyz_frame
from trajplayer.interaction.models import AnalysisResult


def test_extxyz_frame_export_round_trips_float32_coordinates(tmp_path) -> None:
    path = tmp_path / "frame.extxyz"
    positions = np.array([[0.1, 0.2, 0.3], [9.9, 8.8, 7.7]], dtype=np.float32)
    cell = np.diag([10.0, 11.0, 12.0]).astype(np.float32)
    write_xyz_frame(path, np.array([1, 8], dtype=np.uint16), positions, cell)
    atoms = read(path)
    np.testing.assert_allclose(atoms.positions, positions, rtol=0, atol=1e-7)
    np.testing.assert_allclose(atoms.cell.array, cell)


def test_analysis_csv_contains_full_result(tmp_path) -> None:
    result = AnalysisResult(
        kind="com",
        x=np.array([0.0, 2.0]),
        y=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        x_unit="frame",
        y_unit="A",
        source_frames=(0, 4, 2),
        selection_revision=1,
        trajectory_generation=1,
        metadata=MappingProxyType({"series": ["COM X", "COM Y", "COM Z"]}),
    )
    path = tmp_path / "com.csv"
    write_analysis_csv(path, result)
    values = np.loadtxt(path, delimiter=",", skiprows=1)
    np.testing.assert_allclose(values, np.column_stack((result.x, result.y)))


def test_analysis_csv_cancellation_stops_between_bounded_chunks(tmp_path) -> None:
    result = AnalysisResult(
        kind="rmsd",
        x=np.arange(50_001, dtype=np.float64),
        y=np.arange(50_001, dtype=np.float64),
        x_unit="frame",
        y_unit="A",
        source_frames=(0, 50_001, 1),
        selection_revision=1,
        trajectory_generation=1,
    )
    checks = 0

    def cancelled() -> bool:
        nonlocal checks
        checks += 1
        return checks > 1

    with pytest.raises(RuntimeError, match="cancelled"):
        write_analysis_csv(tmp_path / "cancelled.csv", result, cancelled=cancelled)
