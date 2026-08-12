from __future__ import annotations

from enum import Enum

import numpy as np
from PySide6.QtCore import QObject, Signal

from .models import SelectionSnapshot


class SelectionOp(str, Enum):
    REPLACE = "replace"
    ADD = "add"
    TOGGLE = "toggle"
    REMOVE = "remove"


class SelectionManager(QObject):
    selectionChanged = Signal(object)
    primaryChanged = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._atom_count = 0
        self._trajectory_generation = 0
        self._revision = 0
        self._indices = np.empty((0,), dtype=np.uint32)
        self._order: tuple[int, ...] = ()
        self._primary: int | None = None
        self._component_ids = np.empty((0,), dtype=np.int32)

    @property
    def atom_count(self) -> int:
        return self._atom_count

    @property
    def revision(self) -> int:
        return self._revision

    def begin_trajectory(self, atom_count: int, trajectory_generation: int) -> None:
        count = int(atom_count)
        generation = int(trajectory_generation)
        if count < 0 or generation < 0:
            raise ValueError("atom_count and trajectory_generation must be non-negative")
        self._atom_count = count
        self._trajectory_generation = generation
        self._component_ids = np.empty((0,), dtype=np.int32)
        self._apply(np.empty((0,), dtype=np.uint32), None, force=True)

    def set_component_ids(self, component_ids: np.ndarray | None) -> None:
        if component_ids is None:
            self._component_ids = np.empty((0,), dtype=np.int32)
            return
        values = np.ascontiguousarray(component_ids, dtype=np.int32)
        if values.shape != (self._atom_count,):
            raise ValueError("component_ids must contain one entry per atom")
        self._component_ids = values

    def snapshot(self) -> SelectionSnapshot:
        return SelectionSnapshot(
            atom_indices=self._indices,
            primary_atom=self._primary,
            revision=self._revision,
            trajectory_generation=self._trajectory_generation,
        )

    def selection_order(self) -> np.ndarray:
        values = np.asarray(self._order, dtype=np.uint32)
        values.setflags(write=False)
        return values

    def contains(self, atom_index: int) -> bool:
        index = int(atom_index)
        if index < 0 or index >= self._atom_count or self._indices.size == 0:
            return False
        location = int(np.searchsorted(self._indices, np.uint32(index)))
        return location < self._indices.size and int(self._indices[location]) == index

    def replace(self, atom_indices: np.ndarray | list[int] | tuple[int, ...]) -> None:
        order = self._input_order(atom_indices)
        values = np.asarray(sorted(order), dtype=np.uint32)
        primary = int(values[-1]) if values.size else None
        self._apply(values, primary, order=order)

    def add(self, atom_indices: np.ndarray | list[int] | tuple[int, ...]) -> None:
        order = self._input_order(atom_indices)
        values = np.asarray(sorted(order), dtype=np.uint32)
        combined = np.union1d(self._indices, values).astype(np.uint32, copy=False)
        combined_order = self._order + tuple(index for index in order if index not in self._order)
        primary = order[-1] if order else self._primary
        self._apply(combined, primary, order=combined_order)

    def toggle(self, atom_indices: np.ndarray | list[int] | tuple[int, ...]) -> None:
        order = self._input_order(atom_indices)
        values = np.asarray(sorted(order), dtype=np.uint32)
        combined = np.setxor1d(self._indices, values, assume_unique=True).astype(
            np.uint32,
            copy=False,
        )
        combined_order = list(self._order)
        for index in order:
            if index in combined_order:
                combined_order.remove(index)
            else:
                combined_order.append(index)
        primary = order[-1] if order and order[-1] in combined_order else None
        if primary is None and combined.size:
            primary = combined_order[-1]
        self._apply(combined, primary, order=tuple(combined_order))

    def remove(self, atom_indices: np.ndarray | list[int] | tuple[int, ...]) -> None:
        order = self._input_order(atom_indices)
        values = np.asarray(sorted(order), dtype=np.uint32)
        remaining = np.setdiff1d(self._indices, values, assume_unique=True).astype(
            np.uint32,
            copy=False,
        )
        primary = self._primary if self._primary is not None and self._primary in remaining else None
        if primary is None and remaining.size:
            remaining_order = tuple(index for index in self._order if index not in order)
            primary = remaining_order[-1]
        else:
            remaining_order = tuple(index for index in self._order if index not in order)
        self._apply(remaining, primary, order=remaining_order)

    def clear(self) -> None:
        self._apply(np.empty((0,), dtype=np.uint32), None, order=())

    def select_atom(self, atom_index: int, op: SelectionOp = SelectionOp.REPLACE) -> None:
        values = np.asarray([int(atom_index)], dtype=np.int64)
        if op is SelectionOp.REPLACE:
            self.replace(values)
        elif op is SelectionOp.ADD:
            self.add(values)
        elif op is SelectionOp.TOGGLE:
            self.toggle(values)
        elif op is SelectionOp.REMOVE:
            self.remove(values)
        else:
            raise ValueError(f"Unsupported selection operation: {op}")

    def select_component(self, component_index: int, op: SelectionOp) -> None:
        if self._component_ids.shape != (self._atom_count,):
            raise ValueError("No component topology is available")
        component = int(component_index)
        atom_indices = np.flatnonzero(self._component_ids == component)
        if atom_indices.size == 0:
            raise IndexError(component)
        if op is SelectionOp.REPLACE:
            self.replace(atom_indices)
        elif op is SelectionOp.ADD:
            self.add(atom_indices)
        elif op is SelectionOp.TOGGLE:
            self.toggle(atom_indices)
        elif op is SelectionOp.REMOVE:
            self.remove(atom_indices)

    def _input_order(self, atom_indices) -> tuple[int, ...]:
        values = np.asarray(atom_indices, dtype=np.int64)
        if values.ndim != 1:
            raise ValueError("atom_indices must be one-dimensional")
        if values.size and (int(values.min()) < 0 or int(values.max()) >= self._atom_count):
            raise IndexError("Selection contains an atom outside the current trajectory")
        return tuple(dict.fromkeys(int(value) for value in values))

    def _apply(
        self,
        indices: np.ndarray,
        primary: int | None,
        *,
        force: bool = False,
        order: tuple[int, ...] | None = None,
    ) -> None:
        normalized = np.ascontiguousarray(indices, dtype=np.uint32)
        old_primary = self._primary
        normalized_order = (
            tuple(int(index) for index in order)
            if order is not None
            else tuple(index for index in self._order if index in normalized)
        )
        changed = (
            force
            or primary != self._primary
            or normalized_order != self._order
            or not np.array_equal(normalized, self._indices)
        )
        if not changed:
            return
        self._indices = normalized
        self._order = normalized_order
        self._primary = None if primary is None else int(primary)
        self._revision += 1
        snapshot = self.snapshot()
        self.selectionChanged.emit(snapshot)
        if force or old_primary != self._primary:
            self.primaryChanged.emit(self._primary)
