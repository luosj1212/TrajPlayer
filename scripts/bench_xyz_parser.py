from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trajplayer import __display_version__
from trajplayer import trajcore
from trajplayer.xyz_reader import read_xyz_frame


def benchmark_xyz_parser(*, atoms: int, repeat: int) -> dict[str, object]:
    atom_count = max(1, int(atoms))
    repetitions = max(1, int(repeat))
    rows = b"C 1.2 2.3 3.4\n" * atom_count
    payload = (
        f"{atom_count}\nProperties=species:S:1:pos:R:3\n".encode("ascii")
        + rows
    )
    positions = np.empty((atom_count, 3), dtype=np.float32)
    expected_numbers = np.full(atom_count, 6, dtype=np.uint16)

    native_ms = _samples_ms(
        lambda: trajcore.xyz_read_frame_into(
            rows,
            data_offset=0,
            data_end=len(rows),
            positions=positions,
            identity_column=0,
            identity_is_atomic_number=False,
            position_columns=(1, 2, 3),
            expected_columns=4,
            expected_atom_numbers=expected_numbers,
        ),
        repetitions,
    )
    if not trajcore.NATIVE_XYZ_READ_AVAILABLE:
        raise RuntimeError("native trajcore XYZ parsing is unavailable")
    python_ms = _samples_ms(
        lambda: read_xyz_frame(BytesIO(payload), atom_count, 0),
        repetitions,
    )
    expected_sum = np.float32(6.9) * np.float32(atom_count)
    if not np.isclose(np.sum(positions, dtype=np.float64), expected_sum, rtol=1.0e-5):
        raise RuntimeError("native XYZ parser produced unexpected coordinates")

    native_median = statistics.median(native_ms)
    python_median = statistics.median(python_ms)
    return {
        "trajplayer_version": __display_version__,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "atoms": atom_count,
        "repeat": repetitions,
        "source_bytes": len(payload),
        "native_ms": _summary(native_ms),
        "python_ms": _summary(python_ms),
        "native_matoms_s": atom_count / native_median / 1000.0,
        "speedup": python_median / native_median,
    }


def _samples_ms(operation, repeat: int) -> list[float]:
    samples: list[float] = []
    for _ in range(repeat):
        started_s = time.perf_counter()
        operation()
        samples.append((time.perf_counter() - started_s) * 1000.0)
    return samples


def _summary(samples: list[float]) -> dict[str, float]:
    return {
        "min": min(samples),
        "median": statistics.median(samples),
        "max": max(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark native and Python XYZ parsing")
    parser.add_argument("--atoms", type=int, default=1_000_000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = benchmark_xyz_parser(atoms=args.atoms, repeat=args.repeat)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
