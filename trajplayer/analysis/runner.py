from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable

import numpy as np

from trajplayer.frame_store import FrameStore
from trajplayer.interaction.measurements import Measurement, evaluate_measurement
from trajplayer.interaction.models import AnalysisRequest, AnalysisResult
from trajplayer.interaction.models import TimeAxis

from .alignment import align_positions, rmsd
from .density import AMU_PER_ANGSTROM3_TO_G_CM3, density_profile
from .geometry import center_of_mass, radius_of_gyration
from .msd import bounded_fft_atom_chunk, windowed_msd_direct, windowed_msd_fft
from .pbc import (
    cartesian_to_fractional,
    cell_volume,
    make_whole_relative_to_anchor,
    unwrap_fractional_step,
)


class AnalysisCancelled(RuntimeError):
    pass


class AnalysisFrameProvider:
    """One reusable full-frame slab for a serialized analysis scan."""

    def __init__(
        self,
        store: FrameStore,
        atom_indices: np.ndarray,
        cancel_event: threading.Event,
        playback_event: threading.Event,
    ) -> None:
        self.store = store
        self.atom_indices = np.ascontiguousarray(atom_indices, dtype=np.int64)
        self.cancel_event = cancel_event
        self.playback_event = playback_event
        self.positions = np.empty((store.atom_count, 3), dtype=np.float32)
        self.cell = np.empty((3, 3), dtype=np.float32) if store.has_cells else None

    def _read_frame(self, frame_index: int) -> None:
        while self.playback_event.is_set():
            if self.cancel_event.wait(0.02):
                raise AnalysisCancelled()
        if self.cancel_event.is_set():
            raise AnalysisCancelled()
        self.store.read_frame_into(int(frame_index), self.positions, self.cell)

    def read(self, frame_index: int) -> tuple[np.ndarray, np.ndarray | None]:
        self._read_frame(frame_index)
        selected = np.ascontiguousarray(self.positions[self.atom_indices], dtype=np.float64)
        cell = None if self.cell is None else np.array(self.cell, dtype=np.float64, copy=True)
        return selected, cell

    def read_cell(self, frame_index: int) -> np.ndarray | None:
        self._read_frame(frame_index)
        return None if self.cell is None else np.array(self.cell, dtype=np.float64, copy=True)


class _Progress:
    def __init__(self, total: int, callback: Callable[[int, int], None]) -> None:
        self.total = max(1, int(total))
        self.callback = callback
        self.last_emit = 0.0

    def update(self, done: int, *, force: bool = False) -> None:
        now = time.monotonic()
        if force or done >= self.total or now - self.last_emit >= 0.05:
            self.callback(int(done), self.total)
            self.last_emit = now


def run_analysis(
    store: FrameStore,
    request: AnalysisRequest,
    atom_numbers: np.ndarray,
    cancel_event: threading.Event,
    playback_event: threading.Event,
    progress_callback: Callable[[int, int], None],
) -> AnalysisResult:
    start, stop, stride = request.source_frames
    stop = min(int(stop), store.navigable_frame_count)
    frames = np.arange(start, stop, stride, dtype=np.int64)
    if frames.size == 0:
        raise ValueError("The selected analysis range contains no available frames")
    selected = request.selection.atom_indices.astype(np.int64, copy=False)
    if selected.size == 0:
        selected = np.arange(store.atom_count, dtype=np.int64)
    if int(selected.max()) >= store.atom_count:
        raise IndexError("Analysis selection is outside the current trajectory")
    provider = AnalysisFrameProvider(store, selected, cancel_event, playback_event)
    numbers = np.asarray(atom_numbers, dtype=np.int64)[selected]
    progress = _Progress(int(frames.size), progress_callback)
    kind = request.kind

    if kind in {"density", "number_density", "mass_density"}:
        mass_mode = kind != "number_density" and bool(request.parameters.get("mass_density", True))
        masses = _atomic_masses(numbers) if mass_mode else None
        total_weight = float(masses.sum()) if mass_mode else float(selected.size)
        values = np.empty(frames.size, dtype=np.float64)
        for row, frame_index in enumerate(frames):
            cell = provider.read_cell(int(frame_index))
            if cell is None:
                raise ValueError("Density analysis requires cell information")
            values[row] = total_weight / cell_volume(cell)
            if mass_mode:
                values[row] *= AMU_PER_ANGSTROM3_TO_G_CM3
            progress.update(row + 1)
        return _result(request, frames, values, y_unit="g/cm3" if mass_mode else "1/A3", metadata={"x_kind": "frame", "series": ["Density"]})

    if kind == "density_profile":
        bins = max(4, min(4096, int(request.parameters.get("bins", 100))))
        axis_name = str(request.parameters.get("axis", "z")).lower()
        axis = {"x": 0, "y": 1, "z": 2}.get(axis_name)
        if axis is None:
            raise ValueError("Density profile axis must be X, Y, or Z")
        mass_mode = bool(request.parameters.get("mass_density", True))
        masses = _atomic_masses(numbers) if mass_mode else None
        sampled_frames = frames[
            np.linspace(0, frames.size - 1, 2000, dtype=np.int64)
        ] if frames.size > 2000 else frames
        values = np.empty((sampled_frames.size, bins), dtype=np.float64)
        for row, frame_index in enumerate(sampled_frames):
            positions, cell = provider.read(int(frame_index))
            if cell is None:
                raise ValueError("Density profile analysis requires cell information")
            values[row] = density_profile(
                positions,
                cell,
                axis=axis,
                bins=bins,
                weights=masses,
                mass_density=mass_mode,
            )
            progress.update(min(row + 1, progress.total))
        progress.update(progress.total, force=True)
        return _result(
            request,
            sampled_frames,
            values,
            y_unit="g/cm3" if mass_mode else "1/A3",
            metadata={
                "x_kind": "frame",
                "heatmap": True,
                "profile_axis": axis_name,
                "profile_fraction": np.linspace(0.0, 1.0, bins, endpoint=False).tolist(),
            },
        )

    if kind in {"msd", "msd_origin"}:
        dimensions = str(request.parameters.get("dimensions", "xyz"))
        values = _origin_msd(
            provider,
            frames,
            dimensions=dimensions,
            unwrap=bool(request.parameters.get("unwrap_pbc", store.has_cells)),
            remove_com_drift=bool(request.parameters.get("remove_com_drift", False)),
            progress=progress,
        )
        return _result(request, frames, values, y_unit="A2", metadata={"x_kind": "frame", "series": [f"MSD {dimensions.upper()}"]})

    if kind == "msd_windowed":
        values = _windowed_msd(
            provider,
            frames,
            dimensions=str(request.parameters.get("dimensions", "xyz")),
            unwrap=bool(request.parameters.get("unwrap_pbc", store.has_cells)),
            max_lag=request.parameters.get("max_lag"),
            remove_com_drift=bool(
                request.parameters.get("remove_com_drift", False)
            ),
            progress=progress,
        )
        lags = np.arange(values.size, dtype=np.float64) * stride
        return _result(request, lags, values, y_unit="A2", metadata={"x_kind": "lag", "series": ["Time-averaged MSD"]})

    if kind == "rmsd":
        reference_frame = int(request.parameters.get("reference_frame", int(frames[0])))
        reference, reference_cell = provider.read(reference_frame)
        use_pbc = bool(request.parameters.get("make_whole", False))
        if use_pbc and reference_cell is not None:
            reference = make_whole_relative_to_anchor(reference, reference_cell)
        fit = bool(request.parameters.get("fit", True))
        masses = (
            _atomic_masses(numbers)
            if bool(request.parameters.get("mass_weighted", False))
            else None
        )
        values = np.empty(frames.size, dtype=np.float64)
        for row, frame_index in enumerate(frames):
            positions, cell = provider.read(int(frame_index))
            if use_pbc and cell is not None:
                positions = make_whole_relative_to_anchor(positions, cell)
            values[row] = rmsd(positions, reference, weights=masses, translate=fit, rotate=fit)
            progress.update(row + 1)
        return _result(request, frames, values, y_unit="A", metadata={"x_kind": "frame", "series": ["RMSD"]})

    if kind == "rmsf":
        reference_frame = int(request.parameters.get("reference_frame", int(frames[0])))
        reference, reference_cell = provider.read(reference_frame)
        use_pbc = bool(request.parameters.get("make_whole", False))
        if use_pbc and reference_cell is not None:
            reference = make_whole_relative_to_anchor(reference, reference_cell)
        fit = bool(request.parameters.get("fit", True))
        masses = (
            _atomic_masses(numbers)
            if bool(request.parameters.get("mass_weighted", False))
            else None
        )
        mean = np.zeros_like(reference, dtype=np.float64)
        m2 = np.zeros_like(reference, dtype=np.float64)
        for row, frame_index in enumerate(frames):
            positions, cell = provider.read(int(frame_index))
            if use_pbc and cell is not None:
                positions = make_whole_relative_to_anchor(positions, cell)
            if fit:
                positions = align_positions(positions, reference, weights=masses)
            delta = positions - mean
            mean += delta / (row + 1)
            m2 += delta * (positions - mean)
            progress.update(row + 1)
        values = np.sqrt(np.sum(m2 / frames.size, axis=1, dtype=np.float64))
        return _result(request, selected.astype(np.float64) + 1.0, values, y_unit="A", metadata={"x_kind": "atom", "series": ["RMSF"]})

    if kind in {"com", "center_of_mass", "rg", "radius_of_gyration"}:
        com_mode = kind in {"com", "center_of_mass"}
        masses = _atomic_masses(numbers)
        values = np.empty((frames.size, 3), dtype=np.float64) if com_mode else np.empty(frames.size, dtype=np.float64)
        use_pbc = bool(request.parameters.get("make_whole", store.has_cells))
        for row, frame_index in enumerate(frames):
            positions, cell = provider.read(int(frame_index))
            if use_pbc and cell is not None:
                positions = make_whole_relative_to_anchor(positions, cell)
            values[row] = center_of_mass(positions, masses) if com_mode else radius_of_gyration(positions, masses)
            progress.update(row + 1)
        return _result(
            request,
            frames,
            values,
            y_unit="A",
            metadata={"x_kind": "frame", "series": ["COM X", "COM Y", "COM Z"] if com_mode else ["Rg"]},
        )

    if kind == "measurement":
        measurement = request.parameters.get("measurement")
        if not isinstance(measurement, Measurement):
            raise ValueError("Measurement analysis requires a pinned measurement")
        local_map = {int(atom): index for index, atom in enumerate(selected)}
        try:
            local_indices = tuple(local_map[index] for index in measurement.atom_indices)
        except KeyError as exc:
            raise ValueError("Measurement atoms are not present in the analysis selection") from exc
        local_measurement = Measurement(
            kind=measurement.kind,
            atom_indices=local_indices,
            pbc_mode=measurement.pbc_mode,
            measurement_id=measurement.measurement_id,
        )
        values = np.empty(frames.size, dtype=np.float64)
        for row, frame_index in enumerate(frames):
            positions, cell = provider.read(int(frame_index))
            values[row] = evaluate_measurement(local_measurement, positions, cell).value
            progress.update(row + 1)
        return _result(request, frames, values, y_unit="A" if measurement.kind.value == "distance" else "deg", metadata={"x_kind": "frame", "series": [measurement.kind.value.title()]})

    raise ValueError(f"Unsupported analysis: {kind}")


def _atomic_masses(numbers: np.ndarray) -> np.ndarray:
    from ase.data import atomic_masses

    values = np.asarray(numbers, dtype=np.int64)
    if values.size == 0 or np.any(values <= 0) or np.any(values >= len(atomic_masses)):
        raise ValueError("One or more selected elements have no reliable atomic mass")
    masses = np.asarray(atomic_masses, dtype=np.float64)[values]
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("One or more selected elements have no reliable atomic mass")
    return np.ascontiguousarray(masses, dtype=np.float64)


def _origin_msd(provider, frames, *, dimensions, unwrap, remove_com_drift, progress):
    axes = {"x": (0,), "y": (1,), "z": (2,), "xy": (0, 1), "xyz": (0, 1, 2)}.get(dimensions.lower())
    if axes is None:
        raise ValueError("MSD dimensions must be X, Y, Z, XY, or XYZ")
    values = np.empty(frames.size, dtype=np.float64)
    origin = previous_fractional = unwrapped = None
    for row, frame_index in enumerate(frames):
        positions, cell = provider.read(int(frame_index))
        if origin is None:
            origin = positions.copy()
            unwrapped = positions.copy()
            if unwrap:
                if cell is None:
                    raise ValueError("MSD PBC unwrapping requires cell information")
                previous_fractional = cartesian_to_fractional(positions, cell)
        elif unwrap:
            if cell is None:
                raise ValueError("MSD PBC unwrapping requires a cell for every frame")
            current_fractional = cartesian_to_fractional(positions, cell)
            unwrapped = unwrap_fractional_step(previous_fractional, current_fractional, unwrapped, cell)
            previous_fractional = current_fractional
        else:
            unwrapped = positions
        displacement = unwrapped - origin
        if remove_com_drift:
            displacement -= np.mean(displacement, axis=0)
        values[row] = np.mean(np.sum(displacement[:, axes] ** 2, axis=1))
        progress.update(row + 1)
    return values


def _windowed_msd(
    provider,
    frames,
    *,
    dimensions,
    unwrap,
    max_lag,
    remove_com_drift,
    progress,
):
    path = ""
    mapped = None
    try:
        lag = None if max_lag is None else max(0, int(max_lag))
        lag_count = frames.size if lag is None else min(frames.size, lag + 1)
        work = frames.size * provider.atom_indices.size
        use_direct = work <= 5_000_000 and (lag is None or lag <= 2048)
        if use_direct:
            compute_steps = max(1, int(lag_count) - 1)
        else:
            chunk_size = bounded_fft_atom_chunk(
                frames.size,
                dimensions,
                4096,
            )
            compute_steps = int(max(
                1,
                (provider.atom_indices.size + chunk_size - 1) // chunk_size,
            ))
        scan_steps = int(frames.size)
        progress.total = int(scan_steps + compute_steps)

        required_bytes = int(frames.size) * int(provider.atom_indices.size) * 3 * 4
        temp_root = tempfile.gettempdir()
        free_bytes = shutil.disk_usage(temp_root).free
        disk_budget = max(0, min(8 * 1024**3, free_bytes // 2))
        if required_bytes > disk_budget:
            required_gib = required_bytes / 1024**3
            budget_gib = disk_budget / 1024**3
            raise ValueError(
                "Time-averaged MSD requires "
                f"{required_gib:.2f} GiB of temporary storage; the safe budget is "
                f"{budget_gib:.2f} GiB. Reduce the selection, range, or stride."
            )
        handle = tempfile.NamedTemporaryFile(prefix="trajplayer-msd-", suffix=".bin", delete=False)
        path = handle.name
        handle.close()
        shape = (frames.size, provider.atom_indices.size, 3)
        mapped = np.memmap(path, mode="w+", dtype=np.float32, shape=shape)
        previous_fractional = unwrapped = None
        for row, frame_index in enumerate(frames):
            positions, cell = provider.read(int(frame_index))
            if row == 0:
                unwrapped = positions.copy()
                if unwrap:
                    if cell is None:
                        raise ValueError("MSD PBC unwrapping requires cell information")
                    previous_fractional = cartesian_to_fractional(positions, cell)
            elif unwrap:
                if cell is None:
                    raise ValueError("MSD PBC unwrapping requires a cell for every frame")
                current_fractional = cartesian_to_fractional(positions, cell)
                unwrapped = unwrap_fractional_step(previous_fractional, current_fractional, unwrapped, cell)
                previous_fractional = current_fractional
            else:
                unwrapped = positions
            if remove_com_drift:
                mapped[row] = unwrapped - np.mean(unwrapped, axis=0)
            else:
                mapped[row] = unwrapped
            progress.update(row + 1)
        progress.update(scan_steps, force=True)

        def check_cancel() -> None:
            if provider.cancel_event.is_set():
                raise AnalysisCancelled()

        def compute_progress(done: int) -> None:
            progress.update(scan_steps + min(compute_steps, int(done)))

        check_cancel()
        if use_direct:
            result = windowed_msd_direct(
                mapped,
                dimensions=dimensions,
                max_lag=lag,
                check_cancel=check_cancel,
                progress_callback=compute_progress,
            )
        else:
            result = windowed_msd_fft(
                mapped,
                dimensions=dimensions,
                max_lag=lag,
                atom_chunk=4096,
                check_cancel=check_cancel,
                progress_callback=compute_progress,
            )
        progress.update(progress.total, force=True)
        return result
    finally:
        if mapped is not None:
            del mapped
        if path:
            try:
                os.remove(path)
            except OSError:
                pass


def _result(request, x, y, *, y_unit: str, metadata: dict[str, object]) -> AnalysisResult:
    x_values = np.asarray(x, dtype=np.float64)
    result_metadata = dict(metadata)
    result_metadata["selection_scope"] = (
        "all" if request.selection.atom_indices.size == 0 else "selection"
    )
    x_kind = str(result_metadata.get("x_kind", "frame"))
    x_unit = "atom" if x_kind == "atom" else "frame"
    if x_kind == "frame":
        frame_indices = np.ascontiguousarray(np.rint(x_values), dtype=np.int64)
        frame_indices.setflags(write=False)
        result_metadata["frame_indices"] = frame_indices
    timestep = float(request.parameters.get("timestep", 0.0) or 0.0)
    time_unit = str(request.parameters.get("time_unit", "ps"))
    if x_kind in {"frame", "lag"} and timestep > 0.0:
        axis_count = max(1, int(np.ceil(float(np.max(x_values)))) + 1)
        axis = TimeAxis(frame_count=axis_count, unit=time_unit, dt=timestep)
        x_values = axis.values_for_frames(np.rint(x_values).astype(np.int64))
        x_unit = time_unit
    return AnalysisResult(
        kind=request.kind,
        x=x_values,
        y=np.asarray(y),
        x_unit=x_unit,
        y_unit=y_unit,
        source_frames=request.source_frames,
        selection_revision=request.selection.revision,
        trajectory_generation=request.selection.trajectory_generation,
        parameters=request.parameters,
        metadata=result_metadata,
    )
