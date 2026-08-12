from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping
from uuid import UUID, uuid4

import numpy as np


def _readonly_array(
    values: np.ndarray,
    *,
    dtype: np.dtype | type,
    ndim: int | None = None,
) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=dtype)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"Expected a {ndim}D array, got shape {array.shape}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class TimeAxis:
    frame_count: int
    values: np.ndarray | None = None
    unit: str = "frame"
    dt: float | None = None

    def __post_init__(self) -> None:
        count = int(self.frame_count)
        if count < 0:
            raise ValueError("frame_count must be non-negative")
        if self.unit not in {"frame", "fs", "ps", "ns"}:
            raise ValueError(f"Unsupported time unit: {self.unit}")
        if self.dt is not None and (not np.isfinite(self.dt) or float(self.dt) <= 0.0):
            raise ValueError("dt must be a finite positive value")
        object.__setattr__(self, "frame_count", count)
        if self.values is not None:
            values = _readonly_array(self.values, dtype=np.float64, ndim=1)
            if values.shape != (count,):
                raise ValueError("Time-axis values must contain one value per frame")
            if values.size > 1 and np.any(np.diff(values) <= 0.0):
                raise ValueError("Time-axis values must be strictly increasing")
            object.__setattr__(self, "values", values)

    def values_for_frames(self, frames: np.ndarray) -> np.ndarray:
        indices = np.asarray(frames, dtype=np.int64)
        if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= self.frame_count):
            raise IndexError("Frame index is outside the time axis")
        if self.values is not None:
            return np.ascontiguousarray(self.values[indices], dtype=np.float64)
        if self.dt is not None:
            return np.ascontiguousarray(indices * float(self.dt), dtype=np.float64)
        return np.ascontiguousarray(indices, dtype=np.float64)


@dataclass(frozen=True)
class SelectionSnapshot:
    atom_indices: np.ndarray
    primary_atom: int | None
    revision: int
    trajectory_generation: int

    def __post_init__(self) -> None:
        indices = _readonly_array(self.atom_indices, dtype=np.uint32, ndim=1)
        if indices.size > 1 and np.any(indices[1:] <= indices[:-1]):
            raise ValueError("atom_indices must be sorted and unique")
        primary = None if self.primary_atom is None else int(self.primary_atom)
        if primary is not None:
            location = int(np.searchsorted(indices, np.uint32(primary)))
            if location >= indices.size or int(indices[location]) != primary:
                raise ValueError("primary_atom must belong to atom_indices")
        if int(self.revision) < 0 or int(self.trajectory_generation) < 0:
            raise ValueError("Selection revisions and generations must be non-negative")
        object.__setattr__(self, "atom_indices", indices)
        object.__setattr__(self, "primary_atom", primary)
        object.__setattr__(self, "revision", int(self.revision))
        object.__setattr__(self, "trajectory_generation", int(self.trajectory_generation))


@dataclass(frozen=True)
class AnalysisRequest:
    kind: str
    source_frames: tuple[int, int, int]
    selection: SelectionSnapshot
    parameters: Mapping[str, object] = field(default_factory=dict)
    request_id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        start, stop, stride = (int(value) for value in self.source_frames)
        if start < 0 or stop <= start or stride <= 0:
            raise ValueError("source_frames must be a non-empty start/stop/stride range")
        if not str(self.kind).strip():
            raise ValueError("Analysis kind cannot be empty")
        object.__setattr__(self, "kind", str(self.kind).strip().lower())
        object.__setattr__(self, "source_frames", (start, stop, stride))
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


@dataclass(frozen=True)
class AnalysisResult:
    kind: str
    x: np.ndarray
    y: np.ndarray
    x_unit: str
    y_unit: str
    source_frames: tuple[int, int, int]
    selection_revision: int
    trajectory_generation: int
    parameters: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)
    result_id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        x = _readonly_array(self.x, dtype=np.float64, ndim=1)
        y = np.ascontiguousarray(self.y)
        if y.ndim not in {1, 2}:
            raise ValueError("Analysis y data must be one- or two-dimensional")
        if y.shape[0] != x.shape[0]:
            raise ValueError("Analysis x and y must share the first dimension")
        if not np.issubdtype(y.dtype, np.floating):
            y = y.astype(np.float64)
        y.setflags(write=False)
        object.__setattr__(self, "kind", str(self.kind).strip().lower())
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "selection_revision", int(self.selection_revision))
        object.__setattr__(self, "trajectory_generation", int(self.trajectory_generation))
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
