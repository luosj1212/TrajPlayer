from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class FrameStore(Protocol):
    root: Path
    atom_numbers: np.ndarray
    metadata: dict[str, object]

    @property
    def frame_count(self) -> int: ...

    @property
    def frame_count_is_final(self) -> bool: ...

    @property
    def atom_count(self) -> int: ...

    @property
    def has_cells(self) -> bool: ...

    @property
    def supports_random_access(self) -> bool: ...

    @property
    def is_complete(self) -> bool: ...

    @property
    def available_frame_count(self) -> int: ...

    @property
    def navigable_frame_count(self) -> int: ...

    def is_frame_available(self, frame_index: int) -> bool: ...

    def read_frame_arrays(
        self,
        frame_index: int,
    ) -> tuple[np.ndarray, np.ndarray | None]: ...

    def read_frame_into(
        self,
        frame_index: int,
        positions: np.ndarray,
        cell: np.ndarray | None,
    ) -> None: ...

    def close(self) -> None: ...

