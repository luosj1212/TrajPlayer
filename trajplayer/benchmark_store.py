from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from .binary_store import BinaryTrajectoryStore


def create_synthetic_store(
    root: Path,
    *,
    frame_count: int,
    atom_count: int,
    chunk_frames: int = 8,
) -> BinaryTrajectoryStore:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if atom_count <= 0:
        raise ValueError("atom_count must be positive")
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")

    root = root.resolve()
    if root.exists():
        shutil.rmtree(root)

    atom_numbers = np.resize(np.array([6, 1, 8, 7, 16], dtype=np.uint16), atom_count)
    store = BinaryTrajectoryStore.create(
        root,
        frame_count=frame_count,
        atom_numbers=atom_numbers,
        symbols=["C", "H", "O", "N", "S"],
        source_path=None,
        source_mtime_ns=0,
        source_size=frame_count * atom_count * 3 * 4,
    )
    store.metadata["synthetic"] = True
    store.metadata["benchmark_atom_count"] = atom_count
    store.metadata["benchmark_frame_count"] = frame_count
    (root / "metadata.json").write_text(
        __import__("json").dumps(store.metadata, indent=2),
        encoding="utf-8",
    )

    atom_index = np.arange(atom_count, dtype=np.float32)
    side = max(1, int(np.ceil(atom_count ** (1.0 / 3.0))))
    base = np.empty((atom_count, 3), dtype=np.float32)
    base[:, 0] = (atom_index % side) * 0.45
    base[:, 1] = ((atom_index // side) % side) * 0.45
    base[:, 2] = (atom_index // (side * side)) * 0.45

    for start in range(0, frame_count, chunk_frames):
        stop = min(frame_count, start + chunk_frames)
        frames = np.arange(start, stop, dtype=np.float32)
        chunk = np.empty((stop - start, atom_count, 3), dtype=np.float32)
        phase = frames[:, None] * 0.13 + atom_index[None, :] * 0.002
        wobble = np.sin(phase, dtype=np.float32) * 0.035
        chunk[:, :, 0] = base[None, :, 0] + wobble
        chunk[:, :, 1] = base[None, :, 1] + np.cos(phase, dtype=np.float32) * 0.035
        chunk[:, :, 2] = base[None, :, 2] + np.sin(phase * 0.37, dtype=np.float32) * 0.035
        store.positions[start:stop, :, :] = chunk

    store.flush()
    return store
