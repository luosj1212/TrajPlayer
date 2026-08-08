from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class BondSource(str, Enum):
    DISABLED = "disabled"
    INFERENCE_PENDING = "inference_pending"
    INFERENCE_FAILED = "inference_failed"
    INFERRED_STATIC = "inferred_static"
    FILE = "file"
    GENERATED = "generated"


@dataclass(frozen=True)
class BondTopology:
    bonds: np.ndarray
    component_ids: np.ndarray
    component_sizes: np.ndarray
    source: BondSource
    source_frame: int | None = None

    def __post_init__(self) -> None:
        bonds = np.asarray(self.bonds, dtype=np.int32)
        component_ids = np.asarray(self.component_ids, dtype=np.int32)
        component_sizes = np.asarray(self.component_sizes, dtype=np.int32)
        if bonds.size == 0:
            bonds = np.empty((0, 2), dtype=np.int32)
        if bonds.ndim != 2 or bonds.shape[1] != 2:
            raise ValueError("bonds must have shape (M, 2)")
        if component_ids.ndim != 1 or component_sizes.ndim != 1:
            raise ValueError("component arrays must be one-dimensional")
        object.__setattr__(self, "bonds", np.ascontiguousarray(bonds, dtype=np.int32))
        object.__setattr__(
            self,
            "component_ids",
            np.ascontiguousarray(component_ids, dtype=np.int32),
        )
        object.__setattr__(
            self,
            "component_sizes",
            np.ascontiguousarray(component_sizes, dtype=np.int32),
        )

    @property
    def description(self) -> str:
        if self.source is BondSource.DISABLED:
            return "disabled"
        if self.source is BondSource.INFERENCE_PENDING:
            return "inferring from frame 1"
        if self.source is BondSource.INFERENCE_FAILED:
            return "inference failed"
        if self.source is BondSource.INFERRED_STATIC:
            frame_number = 1 if self.source_frame is None else self.source_frame + 1
            return f"inferred from frame {frame_number}"
        if self.source is BondSource.FILE:
            return "from file topology"
        return "generated benchmark topology"

    @property
    def chain_selection_available(self) -> bool:
        return self.bonds.shape[0] > 0 and self.component_sizes.size > 0


def empty_topology(source: BondSource = BondSource.DISABLED) -> BondTopology:
    return BondTopology(
        bonds=np.empty((0, 2), dtype=np.int32),
        component_ids=np.empty((0,), dtype=np.int32),
        component_sizes=np.empty((0,), dtype=np.int32),
        source=source,
    )
