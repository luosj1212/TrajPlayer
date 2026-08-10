from __future__ import annotations

from pathlib import Path

import numpy as np


class ChemfilesGromacsReader:
    """Small native XTC/TRR reader backed by the chemfiles C++ library."""

    def __init__(self, path: Path, *, expected_atom_count: int) -> None:
        try:
            from chemfiles import Trajectory
        except ImportError as exc:
            raise RuntimeError(
                "Gromacs XTC/TRR support requires the chemfiles package"
            ) from exc

        self.path = path.resolve()
        self._trajectory = Trajectory(str(self.path), mode="r")
        self._prefetched_first_frame: tuple[np.ndarray, np.ndarray | None] | None = None
        self.frame_count = int(self._trajectory.nsteps)
        if self.frame_count <= 0:
            self.close()
            raise ValueError("No frames found in Gromacs trajectory")
        first_positions, first_cell = self._decode_frame(0)
        self.atom_count = int(first_positions.shape[0])
        if self.atom_count != int(expected_atom_count):
            self.close()
            raise ValueError(
                f"GRO topology has {expected_atom_count} atoms but {self.path.name} "
                f"has {self.atom_count} atoms"
            )
        self.has_cell = first_cell is not None
        self._prefetched_first_frame = (first_positions, first_cell)

    def read_frame(self, frame_index: int) -> tuple[np.ndarray, np.ndarray | None]:
        index = int(frame_index)
        if index < 0 or index >= self.frame_count:
            raise IndexError(index)
        if index == 0 and self._prefetched_first_frame is not None:
            prefetched = self._prefetched_first_frame
            self._prefetched_first_frame = None
            return prefetched
        return self._decode_frame(index)

    def _decode_frame(self, index: int) -> tuple[np.ndarray, np.ndarray | None]:
        frame = self._trajectory.read_step(index)
        positions = np.ascontiguousarray(frame.positions, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(
                f"Frame {index} has position shape {positions.shape}; expected (N, 3)"
            )
        cell = np.asarray(frame.cell.matrix, dtype=np.float32)
        # Chemfiles exposes basis vectors as columns; TrajPlayer uses row vectors.
        cell = np.ascontiguousarray(cell.T, dtype=np.float32)
        if cell.shape != (3, 3) or not np.any(cell):
            return positions, None
        return positions, cell

    def close(self) -> None:
        self._prefetched_first_frame = None
        trajectory = getattr(self, "_trajectory", None)
        if trajectory is not None:
            trajectory.close()
            self._trajectory = None
