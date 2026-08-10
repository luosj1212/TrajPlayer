from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trajplayer import __display_version__, trajcore


def _measure(callback, repeats: int) -> tuple[float, np.ndarray]:
    samples: list[float] = []
    result = np.empty((0, 2), dtype=np.int32)
    for _ in range(repeats):
        started = time.perf_counter()
        result = callback()
        samples.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(samples), result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark native valence bond selection.")
    parser.add_argument("--atoms", type=int, default=200_000)
    parser.add_argument("--candidates", type=int, default=1_000_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    atom_count = max(2, int(args.atoms))
    candidate_count = max(1, int(args.candidates))
    repeats = max(1, int(args.repeats))

    rng = np.random.default_rng(20260810)
    left = rng.integers(0, atom_count - 1, candidate_count, dtype=np.int32)
    right = rng.integers(1, atom_count, candidate_count, dtype=np.int32)
    low = np.minimum(left, right)
    high = np.maximum(left, right)
    distance2 = rng.random(candidate_count, dtype=np.float32)
    caps = np.full(atom_count, 4, dtype=np.uint8)

    if not trajcore.NATIVE_VALENCE_SELECTION_AVAILABLE:
        raise SystemExit("native select_valence_bonds is not available")
    native_ms, native = _measure(
        lambda: trajcore._native.select_valence_bonds(distance2, low, high, caps),
        repeats,
    )
    python_ms, reference = _measure(
        lambda: trajcore._python_select_valence_bonds(distance2, low, high, caps),
        repeats,
    )
    report = {
        "trajplayer_version": __display_version__,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "atoms": atom_count,
        "candidates": candidate_count,
        "repeats": repeats,
        "native_median_ms": round(native_ms, 2),
        "python_median_ms": round(python_ms, 2),
        "speedup": round(python_ms / native_ms, 2),
        "bonds": len(native),
        "equal": bool(np.array_equal(native, reference)),
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
