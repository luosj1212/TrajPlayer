from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from .vector_export import write_molecule_svg


def write_xyz_frame(
    path: Path,
    atom_numbers: np.ndarray,
    positions: np.ndarray,
    cell: np.ndarray | None,
    cancelled=None,
) -> None:
    from ase.data import chemical_symbols

    target = Path(path)
    numbers = np.asarray(atom_numbers, dtype=np.int64)
    frame = np.asarray(positions, dtype=np.float32)
    if frame.shape != (numbers.size, 3):
        raise ValueError("positions and atom_numbers have incompatible shapes")
    if np.any(numbers < 0) or np.any(numbers >= len(chemical_symbols)):
        raise ValueError("An atom has no valid chemical symbol")
    extxyz = target.suffix.lower() == ".extxyz"
    if extxyz:
        comment = "Properties=species:S:1:pos:R:3"
        if cell is not None:
            lattice = " ".join(f"{float(value):.9g}" for value in np.ravel(cell))
            comment = f'Lattice="{lattice}" {comment}'
    else:
        comment = "Exported by TrajPlayer"
    symbols = np.asarray(chemical_symbols, dtype="U3")[numbers]
    with target.open("wb", buffering=1024 * 1024) as handle:
        handle.write(f"{frame.shape[0]}\n{comment}\n".encode("ascii"))
        for start in range(0, frame.shape[0], 65_536):
            if cancelled is not None and cancelled():
                raise RuntimeError("Export cancelled")
            stop = min(frame.shape[0], start + 65_536)
            chunk = frame[start:stop]
            lines = np.char.add(symbols[start:stop], " ")
            for axis in range(3):
                values = np.char.mod("%.9g", chunk[:, axis])
                lines = np.char.add(np.char.add(lines, values), " " if axis < 2 else "")
            handle.write(("\n".join(lines.tolist()) + "\n").encode("ascii"))


def write_analysis_csv(path: Path, result, cancelled=None) -> None:
    y = np.asarray(result.y)
    values = y[:, None] if y.ndim == 1 else y
    columns = list(result.metadata.get("series", []))
    if len(columns) != values.shape[1]:
        columns = [f"value_{index + 1}" for index in range(values.shape[1])]
    header = ",".join([f"x_{result.x_unit}", *columns])
    column_count = values.shape[1] + 1
    rows_per_chunk = max(1, 100_000 // column_count)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        handle.write(header + "\n")
        for start in range(0, result.x.shape[0], rows_per_chunk):
            if cancelled is not None and cancelled():
                raise RuntimeError("Export cancelled")
            stop = min(result.x.shape[0], start + rows_per_chunk)
            matrix = np.column_stack((result.x[start:stop], values[start:stop]))
            np.savetxt(
                handle,
                matrix,
                delimiter=",",
                fmt="%.10g",
            )


class ExportThread(QThread):
    succeeded = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        *,
        kind: str,
        path: Path,
        payload,
        release_callback=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.kind = str(kind)
        self.path = Path(path)
        self.payload = payload
        self.release_callback = release_callback

    def run(self) -> None:  # type: ignore[override]
        temporary_path: Path | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=self.path.parent,
                prefix=f".{self.path.stem}-",
                suffix=self.path.suffix or ".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            if self.kind == "frame":
                atom_numbers, positions, cell = self.payload
                write_xyz_frame(
                    temporary_path,
                    atom_numbers,
                    positions,
                    cell,
                    cancelled=self.isInterruptionRequested,
                )
            elif self.kind == "analysis":
                write_analysis_csv(
                    temporary_path,
                    self.payload,
                    cancelled=self.isInterruptionRequested,
                )
            elif self.kind == "image":
                image = self.payload
                if not isinstance(image, QImage) or not image.save(str(temporary_path), "PNG"):
                    raise OSError("Qt could not encode the PNG image")
            elif self.kind == "molecule_svg":
                write_molecule_svg(
                    temporary_path,
                    self.payload,
                    cancelled=self.isInterruptionRequested,
                )
            else:
                raise ValueError(f"Unsupported export kind: {self.kind}")
            if self.isInterruptionRequested():
                raise RuntimeError("Export cancelled")
            os.replace(temporary_path, self.path)
            temporary_path = None
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.succeeded.emit(str(self.path))
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            if self.release_callback is not None:
                self.release_callback()
                self.release_callback = None
