from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from trajplayer.analysis.runner import AnalysisCancelled, run_analysis
from trajplayer.interaction.measurements import Measurement, MeasurementKind
from trajplayer.interaction.models import AnalysisRequest, SelectionSnapshot


class ArrayStore:
    def __init__(self, positions: np.ndarray, cells: np.ndarray | None) -> None:
        self.frames = np.ascontiguousarray(positions, dtype=np.float32)
        self.cells = None if cells is None else np.ascontiguousarray(cells, dtype=np.float32)
        self.root = Path("array")
        self.atom_numbers = np.full(self.frames.shape[1], 6, dtype=np.uint16)
        self.metadata = {}

    @property
    def frame_count(self):
        return self.frames.shape[0]

    frame_count_is_final = True
    supports_random_access = True
    is_complete = True

    @property
    def atom_count(self):
        return self.frames.shape[1]

    @property
    def has_cells(self):
        return self.cells is not None

    @property
    def available_frame_count(self):
        return self.frame_count

    @property
    def navigable_frame_count(self):
        return self.frame_count

    def read_frame_into(self, frame_index, positions, cell):
        np.copyto(positions, self.frames[frame_index])
        if cell is not None:
            np.copyto(cell, self.cells[frame_index])


def request(kind: str, frame_count: int, atom_count: int, **parameters) -> AnalysisRequest:
    return AnalysisRequest(
        kind=kind,
        source_frames=(0, frame_count, 1),
        selection=SelectionSnapshot(
            atom_indices=np.arange(atom_count, dtype=np.uint32),
            primary_atom=atom_count - 1,
            revision=2,
            trajectory_generation=7,
        ),
        parameters=parameters,
    )


def run(store: ArrayStore, analysis_request: AnalysisRequest):
    return run_analysis(
        store,
        analysis_request,
        store.atom_numbers,
        threading.Event(),
        threading.Event(),
        lambda _done, _total: None,
    )


def test_runner_density_rmsd_com_and_measurement() -> None:
    base = np.array([[0, 0, 0], [2, 0, 0]], dtype=np.float32)
    frames = np.stack([base, base + [1, 2, 3], base + [2, 4, 6]])
    cells = np.repeat(np.eye(3, dtype=np.float32)[None] * 10, 3, axis=0)
    store = ArrayStore(frames, cells)

    density = run(store, request("number_density", 3, 2, mass_density=False))
    np.testing.assert_allclose(density.y, 0.002)
    assert density.metadata["selection_scope"] == "all"
    all_atoms = run(
        store,
        AnalysisRequest(
            kind="number_density",
            source_frames=(0, 3, 1),
            selection=SelectionSnapshot(
                atom_indices=np.empty(0, dtype=np.uint32),
                primary_atom=None,
                revision=3,
                trajectory_generation=7,
            ),
            parameters={"mass_density": False},
        ),
    )
    assert all_atoms.metadata["selection_scope"] == "all"

    subset_density = run(
        store,
        AnalysisRequest(
            kind="number_density",
            source_frames=(0, 3, 1),
            selection=SelectionSnapshot(
                atom_indices=np.array([0], dtype=np.uint32),
                primary_atom=0,
                revision=4,
                trajectory_generation=7,
            ),
            parameters={"mass_density": False},
        ),
    )
    np.testing.assert_allclose(subset_density.y, 0.002)
    assert subset_density.metadata["selection_scope"] == "all"

    fitted = run(store, request("rmsd", 3, 2, fit=True))
    np.testing.assert_allclose(fitted.y, 0.0, atol=1e-12)

    com = run(store, request("com", 3, 2, make_whole=False))
    np.testing.assert_allclose(com.y[:, 0], [1, 2, 3])

    measurement = Measurement(MeasurementKind.DISTANCE, (0, 1), pbc_mode="raw")
    distance = run(store, request("measurement", 3, 2, measurement=measurement))
    np.testing.assert_allclose(distance.y, 2.0)


def test_runner_origin_msd_and_cancellation() -> None:
    frames = np.zeros((4, 2, 3), dtype=np.float32)
    frames[:, :, 0] = np.arange(4, dtype=np.float32)[:, None]
    store = ArrayStore(frames, None)
    result = run(store, request("msd", 4, 2, unwrap_pbc=False))
    np.testing.assert_allclose(result.y, [0, 1, 4, 9])
    physical = run(
        store,
        request("msd", 4, 2, unwrap_pbc=False, timestep=0.5, time_unit="ps"),
    )
    np.testing.assert_allclose(physical.x, [0.0, 0.5, 1.0, 1.5])
    np.testing.assert_array_equal(physical.metadata["frame_indices"], [0, 1, 2, 3])

    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(AnalysisCancelled):
        run_analysis(
            store,
            request("msd", 4, 2, unwrap_pbc=False),
            store.atom_numbers,
            cancelled,
            threading.Event(),
            lambda _done, _total: None,
        )


def test_windowed_msd_honors_max_lag_and_optional_com_drift_removal() -> None:
    base = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    frames = np.stack([base + [float(step), 0.0, 0.0] for step in range(6)])
    store = ArrayStore(frames, None)

    drifting = run(
        store,
        request(
            "msd_windowed",
            6,
            2,
            unwrap_pbc=False,
            max_lag=3,
            remove_com_drift=False,
        ),
    )
    np.testing.assert_allclose(drifting.y, [0.0, 1.0, 4.0, 9.0])

    corrected = run(
        store,
        request(
            "msd_windowed",
            6,
            2,
            unwrap_pbc=False,
            max_lag=3,
            remove_com_drift=True,
        ),
    )
    np.testing.assert_allclose(corrected.y, 0.0, atol=1.0e-12)


def test_windowed_msd_can_cancel_after_the_coordinate_scan() -> None:
    frames = np.zeros((8, 2, 3), dtype=np.float32)
    frames[:, :, 0] = np.arange(8, dtype=np.float32)[:, None]
    store = ArrayStore(frames, None)
    cancelled = threading.Event()

    def cancel_after_scan(done: int, total: int) -> None:
        if done == store.frame_count and total > store.frame_count:
            cancelled.set()

    with pytest.raises(AnalysisCancelled):
        run_analysis(
            store,
            request("msd_windowed", 8, 2, unwrap_pbc=False),
            store.atom_numbers,
            cancelled,
            threading.Event(),
            cancel_after_scan,
        )
