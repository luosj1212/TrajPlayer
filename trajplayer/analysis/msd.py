from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from .pbc import cartesian_to_fractional, unwrap_fractional_step


DIMENSION_AXES = {
    "x": (0,),
    "y": (1,),
    "z": (2,),
    "xy": (0, 1),
    "xyz": (0, 1, 2),
}
FFT_WORKING_SET_BYTES = 64 * 1024 * 1024


def _axes(dimensions: str) -> tuple[int, ...]:
    try:
        return DIMENSION_AXES[str(dimensions).lower()]
    except KeyError as exc:
        raise ValueError("dimensions must be X, Y, Z, XY, or XYZ") from exc


def _coordinate_view(positions: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    if axes == (0, 1, 2):
        return positions
    if axes == (0, 1):
        return positions[..., :2]
    axis = axes[0]
    return positions[..., axis : axis + 1]


def bounded_fft_atom_chunk(
    frame_count: int,
    dimensions: str,
    requested: int,
    *,
    memory_bytes: int = FFT_WORKING_SET_BYTES,
) -> int:
    frames = max(1, int(frame_count))
    axes = len(_axes(dimensions))
    nfft = 1 << max(1, (2 * frames - 1).bit_length())
    frequency_count = nfft // 2 + 1
    # float64 coordinates, complex spectrum plus product, and irfft output.
    bytes_per_atom = axes * (
        frames * 8 + frequency_count * 32 + nfft * 8
    )
    budget_chunk = max(1, int(memory_bytes) // max(1, bytes_per_atom))
    return max(1, min(int(requested), budget_chunk))


def msd_from_origin(
    frames: Iterable[tuple[np.ndarray, np.ndarray | None]],
    *,
    dimensions: str = "xyz",
    unwrap_pbc: bool = True,
    remove_com_drift: bool = False,
) -> np.ndarray:
    selected_axes = _axes(dimensions)
    origin: np.ndarray | None = None
    previous_fractional: np.ndarray | None = None
    unwrapped: np.ndarray | None = None
    values: list[float] = []
    for positions, cell in frames:
        current = np.asarray(positions, dtype=np.float64)
        if current.ndim != 2 or current.shape[1] != 3:
            raise ValueError("MSD frames must have shape (N, 3)")
        if origin is None:
            origin = np.ascontiguousarray(current, dtype=np.float64)
            unwrapped = origin.copy()
            if unwrap_pbc:
                if cell is None:
                    raise ValueError("MSD PBC unwrapping requires cell information")
                previous_fractional = cartesian_to_fractional(current, cell)
        elif unwrap_pbc:
            if cell is None or previous_fractional is None or unwrapped is None:
                raise ValueError("MSD PBC unwrapping requires a cell for every frame")
            current_fractional = cartesian_to_fractional(current, cell)
            unwrapped = unwrap_fractional_step(
                previous_fractional,
                current_fractional,
                unwrapped,
                cell,
            )
            previous_fractional = current_fractional
        else:
            unwrapped = current
        displacement = unwrapped - origin
        if remove_com_drift and displacement.shape[0]:
            displacement = displacement - np.mean(displacement, axis=0)
        chosen = displacement[:, selected_axes]
        values.append(float(np.mean(np.sum(chosen * chosen, axis=1, dtype=np.float64))))
    if origin is None:
        raise ValueError("MSD requires at least one frame")
    return np.ascontiguousarray(values, dtype=np.float64)


def windowed_msd_direct(
    unwrapped_positions: np.ndarray,
    *,
    dimensions: str = "xyz",
    max_lag: int | None = None,
    check_cancel: Callable[[], None] | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> np.ndarray:
    positions = np.asarray(unwrapped_positions)
    if positions.ndim != 3 or positions.shape[2] != 3 or positions.shape[0] == 0:
        raise ValueError("unwrapped_positions must have shape (F, N, 3)")
    axes = _axes(dimensions)
    frame_count = positions.shape[0]
    lag_count = frame_count if max_lag is None else min(frame_count, int(max_lag) + 1)
    if lag_count <= 0:
        raise ValueError("max_lag must be non-negative")
    selected = _coordinate_view(positions, axes)
    result = np.empty(lag_count, dtype=np.float64)
    result[0] = 0.0
    for lag in range(1, lag_count):
        if check_cancel is not None:
            check_cancel()
        delta = selected[lag:] - selected[:-lag]
        result[lag] = float(np.mean(np.sum(delta * delta, axis=2, dtype=np.float64)))
        if progress_callback is not None:
            progress_callback(lag)
    return result


def windowed_msd_fft(
    unwrapped_positions: np.ndarray,
    *,
    dimensions: str = "xyz",
    max_lag: int | None = None,
    atom_chunk: int = 4096,
    check_cancel: Callable[[], None] | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> np.ndarray:
    positions = np.asarray(unwrapped_positions)
    if positions.ndim != 3 or positions.shape[2] != 3 or positions.shape[0] == 0:
        raise ValueError("unwrapped_positions must have shape (F, N, 3)")
    axes = _axes(dimensions)
    frame_count, atom_count, _ = positions.shape
    if atom_count == 0:
        raise ValueError("MSD requires at least one atom")
    lag_count = frame_count if max_lag is None else min(frame_count, int(max_lag) + 1)
    if lag_count <= 0:
        raise ValueError("max_lag must be non-negative")
    nfft = 1 << max(1, (2 * frame_count - 1).bit_length())
    chunk_size = bounded_fft_atom_chunk(
        frame_count,
        dimensions,
        atom_chunk,
    )
    autocorrelation = np.zeros(lag_count, dtype=np.float64)
    squared_per_frame = np.zeros(frame_count, dtype=np.float64)
    for chunk_index, start in enumerate(range(0, atom_count, chunk_size), start=1):
        if check_cancel is not None:
            check_cancel()
        source_chunk = _coordinate_view(
            positions[:, start : start + chunk_size, :],
            axes,
        )
        chunk = np.asarray(source_chunk, dtype=np.float64)
        squared_per_frame += np.sum(chunk * chunk, axis=(1, 2), dtype=np.float64)
        transformed = np.fft.rfft(chunk, n=nfft, axis=0)
        transformed *= np.conjugate(transformed)
        correlation = np.fft.irfft(
            transformed,
            n=nfft,
            axis=0,
        )[:lag_count]
        autocorrelation += np.sum(correlation, axis=(1, 2), dtype=np.float64)
        if progress_callback is not None:
            progress_callback(chunk_index)
    prefix = np.concatenate(([0.0], np.cumsum(squared_per_frame, dtype=np.float64)))
    result = np.empty(lag_count, dtype=np.float64)
    for lag in range(lag_count):
        if check_cancel is not None and lag % 1024 == 0:
            check_cancel()
        paired_squared = (prefix[frame_count] - prefix[lag]) + prefix[frame_count - lag]
        denominator = atom_count * (frame_count - lag)
        result[lag] = (paired_squared - 2.0 * autocorrelation[lag]) / denominator
    result[0] = 0.0
    np.maximum(result, 0.0, out=result)
    return result
