from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trajplayer import trajcore


def _median_ms(operation, *, repeat: int) -> float:
    operation()
    samples: list[float] = []
    for _ in range(max(1, int(repeat))):
        started_s = time.perf_counter()
        operation()
        samples.append((time.perf_counter() - started_s) * 1000.0)
    return float(statistics.median(samples))


def benchmark_depth_order(*, atoms: int, repeat: int, seed: int) -> dict[str, object]:
    depth = np.random.default_rng(seed).normal(size=max(1, int(atoms))).astype(np.float32)
    reference = trajcore._python_coarse_depth_order(depth)
    actual = trajcore.coarse_depth_order(depth)
    exact = bool(np.array_equal(actual, reference))
    public_ms = _median_ms(lambda: trajcore.coarse_depth_order(depth), repeat=repeat)
    fallback_ms = _median_ms(lambda: trajcore._python_coarse_depth_order(depth), repeat=repeat)
    return {
        "atoms": int(depth.size),
        "repeat": max(1, int(repeat)),
        "native": trajcore.NATIVE_DEPTH_ORDER_AVAILABLE,
        "exact_reference_order": exact,
        "public_median_ms": public_ms,
        "stable_argsort_median_ms": fallback_ms,
        "speedup": 0.0 if public_ms <= 0.0 else fallback_ms / public_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the 256-bin depth-order hot path")
    parser.add_argument("--atoms", type=int, default=1_000_000)
    parser.add_argument("--repeat", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--require-native", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = benchmark_depth_order(atoms=args.atoms, repeat=args.repeat, seed=args.seed)
    if args.require_native and not report["native"]:
        raise SystemExit("native trajcore depth ordering is unavailable")
    if not report["exact_reference_order"]:
        raise SystemExit("depth ordering differs from the visual reference")
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
