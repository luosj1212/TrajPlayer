from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4

import numpy as np
from PySide6.QtCore import QObject, Signal

from trajplayer.analysis.geometry import angle, dihedral, distance


class MeasurementKind(str, Enum):
    DISTANCE = "distance"
    ANGLE = "angle"
    DIHEDRAL = "dihedral"

    @property
    def atom_count(self) -> int:
        return {
            MeasurementKind.DISTANCE: 2,
            MeasurementKind.ANGLE: 3,
            MeasurementKind.DIHEDRAL: 4,
        }[self]


@dataclass(frozen=True)
class Measurement:
    kind: MeasurementKind
    atom_indices: tuple[int, ...]
    pbc_mode: str = "minimum_image"
    pinned: bool = True
    measurement_id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        kind = MeasurementKind(self.kind)
        indices = tuple(int(index) for index in self.atom_indices)
        if len(indices) != kind.atom_count or len(set(indices)) != len(indices):
            raise ValueError(f"{kind.value} requires {kind.atom_count} distinct atoms")
        if min(indices) < 0:
            raise ValueError("Atom indices must be non-negative")
        if self.pbc_mode not in {"minimum_image", "raw"}:
            raise ValueError("pbc_mode must be minimum_image or raw")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "atom_indices", indices)


@dataclass(frozen=True)
class MeasurementValue:
    measurement: Measurement
    value: float
    unit: str


def measurement_kind_for_count(atom_count: int) -> MeasurementKind:
    try:
        return {
            2: MeasurementKind.DISTANCE,
            3: MeasurementKind.ANGLE,
            4: MeasurementKind.DIHEDRAL,
        }[int(atom_count)]
    except KeyError as exc:
        raise ValueError("Select exactly 2, 3, or 4 atoms to create a measurement") from exc


def evaluate_measurement(
    measurement: Measurement,
    positions: np.ndarray,
    cell: np.ndarray | None,
) -> MeasurementValue:
    frame = np.asarray(positions, dtype=np.float64)
    if frame.ndim != 2 or frame.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    indices = np.asarray(measurement.atom_indices, dtype=np.int64)
    if int(indices.max()) >= frame.shape[0]:
        raise IndexError("Measurement atom is outside the frame")
    effective_cell = cell if measurement.pbc_mode == "minimum_image" else None
    selected = frame[indices]
    if measurement.kind is MeasurementKind.DISTANCE:
        value = distance(selected[0], selected[1], effective_cell)
        unit = "A"
    elif measurement.kind is MeasurementKind.ANGLE:
        value = angle(selected[0], selected[1], selected[2], effective_cell)
        unit = "deg"
    else:
        value = dihedral(selected[0], selected[1], selected[2], selected[3], effective_cell)
        unit = "deg"
    return MeasurementValue(measurement=measurement, value=value, unit=unit)


class MeasurementManager(QObject):
    measurementsChanged = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._measurements: list[Measurement] = []
        self._trajectory_generation = 0

    @property
    def measurements(self) -> tuple[Measurement, ...]:
        return tuple(self._measurements)

    def begin_trajectory(self, generation: int) -> None:
        self._trajectory_generation = int(generation)
        self.clear()

    def create(
        self,
        atom_indices: np.ndarray | tuple[int, ...] | list[int],
        *,
        pbc_mode: str = "minimum_image",
    ) -> Measurement:
        indices = tuple(int(index) for index in np.asarray(atom_indices).tolist())
        measurement = Measurement(
            kind=measurement_kind_for_count(len(indices)),
            atom_indices=indices,
            pbc_mode=pbc_mode,
            measurement_id=uuid4(),
        )
        self._measurements.append(measurement)
        self.measurementsChanged.emit(self.measurements)
        return measurement

    def remove(self, measurement_id: UUID) -> None:
        remaining = [item for item in self._measurements if item.measurement_id != measurement_id]
        if len(remaining) == len(self._measurements):
            return
        self._measurements = remaining
        self.measurementsChanged.emit(self.measurements)

    def clear(self) -> None:
        if not self._measurements:
            return
        self._measurements.clear()
        self.measurementsChanged.emit(self.measurements)

    def evaluate_all(
        self,
        positions: np.ndarray,
        cell: np.ndarray | None,
    ) -> tuple[MeasurementValue, ...]:
        return tuple(
            evaluate_measurement(measurement, positions, cell)
            for measurement in self._measurements
        )
