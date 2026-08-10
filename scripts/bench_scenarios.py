from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trajplayer import __display_version__
from trajplayer.process_memory import process_memory_snapshot
from trajplayer.random_access_cache import open_direct_random_access_store
from trajplayer.streaming import FrameStreamer
from trajplayer.trajectory_source import TrajectorySource, resolve_trajectory_source


SCHEMA_VERSION = 1


def latency_summary(samples_ms: list[float]) -> dict[str, float | int]:
    if not samples_ms:
        return {"count": 0, "min": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    values = np.asarray(samples_ms, dtype=np.float64)
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50.0)),
        "p95": float(np.percentile(values, 95.0)),
        "p99": float(np.percentile(values, 99.0)),
        "max": float(np.max(values)),
    }


def benchmark_open(
    source: TrajectorySource,
    *,
    repeat: int,
    wait_index: bool,
    index_timeout_s: float,
) -> dict[str, object]:
    metadata_samples: list[float] = []
    first_frame_samples: list[float] = []
    index_samples: list[float] = []
    stages: list[str] = []
    details: dict[str, object] = {}

    for _ in range(max(1, int(repeat))):
        started_s = time.perf_counter()
        store = open_direct_random_access_store(source, status_callback=stages.append)
        metadata_samples.append((time.perf_counter() - started_s) * 1000.0)
        try:
            known_at_open = store.frame_count
            first_started_s = time.perf_counter()
            positions, cell = store.read_frame_arrays(0)
            first_frame_samples.append((time.perf_counter() - first_started_s) * 1000.0)
            index_ms = 0.0
            if wait_index and not store.frame_count_is_final:
                index_started_s = time.perf_counter()
                _wait_for_index(store, timeout_s=index_timeout_s)
                index_ms = (time.perf_counter() - index_started_s) * 1000.0
            if wait_index:
                index_samples.append(index_ms)
            details = {
                "atom_count": store.atom_count,
                "frame_count_at_open": known_at_open,
                "frame_count_after_wait": store.frame_count,
                "frame_count_is_final": store.frame_count_is_final,
                "has_cells": store.has_cells,
                "first_frame_bytes": int(positions.nbytes + (0 if cell is None else cell.nbytes)),
                "reader_root": store.root.name,
            }
        finally:
            store.close()

    report = _report_header("open", source)
    report["parameters"] = {
        "repeat": max(1, int(repeat)),
        "wait_index": bool(wait_index),
        "index_timeout_s": float(index_timeout_s),
        "os_cache_state": "unmanaged",
    }
    report["metrics"] = {
        "metadata_ms": latency_summary(metadata_samples),
        "first_frame_ms": latency_summary(first_frame_samples),
        "index_complete_ms": latency_summary(index_samples),
        "rss_mib": process_memory_snapshot().rss_mib,
    }
    report["trajectory"] = details
    report["stages"] = list(dict.fromkeys(stages))
    return report


def benchmark_seek(
    source: TrajectorySource,
    *,
    samples: int,
    pattern: str,
    seed: int,
    wait_index: bool,
    index_timeout_s: float,
) -> dict[str, object]:
    open_started_s = time.perf_counter()
    store = open_direct_random_access_store(source)
    open_ms = (time.perf_counter() - open_started_s) * 1000.0
    try:
        if wait_index and not store.frame_count_is_final:
            _wait_for_index(store, timeout_s=index_timeout_s)
        frame_count = store.frame_count
        if frame_count <= 0:
            raise ValueError("trajectory has no indexed frames")
        positions = np.empty((store.atom_count, 3), dtype=np.float32)
        cell = np.empty((3, 3), dtype=np.float32) if store.has_cells else None
        frame_indices = _seek_indices(
            frame_count,
            sample_count=max(1, int(samples)),
            pattern=pattern,
            seed=int(seed),
        )
        latencies_ms: list[float] = []
        total_started_s = time.perf_counter()
        for frame_index in frame_indices:
            read_started_s = time.perf_counter()
            store.read_frame_into(int(frame_index), positions, cell)
            latencies_ms.append((time.perf_counter() - read_started_s) * 1000.0)
        elapsed_s = max(time.perf_counter() - total_started_s, 1.0e-12)
        frame_bytes = int(positions.nbytes + (0 if cell is None else cell.nbytes))
        rss = process_memory_snapshot()
        report = _report_header("seek", source)
        report["parameters"] = {
            "samples": len(frame_indices),
            "pattern": pattern,
            "seed": int(seed),
            "wait_index": bool(wait_index),
            "index_timeout_s": float(index_timeout_s),
        }
        report["metrics"] = {
            "open_ms": open_ms,
            "frame_read_ms": latency_summary(latencies_ms),
            "decode_mib_s": (frame_bytes * len(frame_indices)) / elapsed_s / (1024.0 * 1024.0),
            "elapsed_s": elapsed_s,
            "rss_mib": rss.rss_mib,
            "peak_rss_mib": rss.peak_rss_mib,
        }
        report["trajectory"] = {
            "atom_count": store.atom_count,
            "frame_count": frame_count,
            "frame_count_is_final": store.frame_count_is_final,
            "has_cells": store.has_cells,
            "frame_bytes": frame_bytes,
        }
        return report
    finally:
        store.close()


def benchmark_soak(
    source: TrajectorySource,
    *,
    duration_s: float,
    fps: float,
    prefetch_radius: int,
    wait_index: bool,
    index_timeout_s: float,
) -> dict[str, object]:
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    store = open_direct_random_access_store(source)
    streamer: FrameStreamer | None = None
    try:
        if wait_index and not store.frame_count_is_final:
            _wait_for_index(store, timeout_s=index_timeout_s)
        if store.frame_count <= 0:
            raise ValueError("trajectory has no indexed frames")
        streamer = FrameStreamer(store, prefetch_radius=max(0, int(prefetch_radius)))
        streamer.set_playback_fps(float(fps))
        streamer.start()
        started_s = time.perf_counter()
        deadline_s = started_s
        finish_s = started_s + float(duration_s)
        next_memory_sample_s = started_s
        memory_samples: list[tuple[float, int]] = []
        frame_index = 0
        presented_frames = 0
        wait_timeouts = 0
        deadline_misses = 0

        while time.perf_counter() < finish_s:
            streamer.seek(frame_index, direction=1)
            lease = streamer.wait_for_lease(
                frame_index,
                timeout_s=max(1.0, 2.0 / float(fps)),
            )
            if lease is None:
                wait_timeouts += 1
                continue
            lease.release()
            presented_frames += 1
            frame_index = (frame_index + 1) % store.frame_count
            deadline_s += 1.0 / float(fps)
            now_s = time.perf_counter()
            if now_s > deadline_s:
                deadline_misses += 1
            else:
                time.sleep(deadline_s - now_s)
            now_s = time.perf_counter()
            if now_s >= next_memory_sample_s:
                memory_samples.append((now_s - started_s, process_memory_snapshot().rss_bytes))
                next_memory_sample_s = now_s + 1.0

        memory_samples.append(
            (time.perf_counter() - started_s, process_memory_snapshot().rss_bytes)
        )
        stats = streamer.stats_snapshot()
        elapsed_s = max(time.perf_counter() - started_s, 1.0e-12)
        report = _report_header("soak", source)
        report["parameters"] = {
            "duration_s": float(duration_s),
            "fps": float(fps),
            "prefetch_radius": max(0, int(prefetch_radius)),
            "wait_index": bool(wait_index),
            "index_timeout_s": float(index_timeout_s),
        }
        report["metrics"] = {
            "elapsed_s": elapsed_s,
            "presented_frames": presented_frames,
            "cadence_fps": presented_frames / elapsed_s,
            "deadline_misses": deadline_misses,
            "wait_timeouts": wait_timeouts,
            "rss_start_mib": memory_samples[0][1] / (1024.0 * 1024.0),
            "rss_end_mib": memory_samples[-1][1] / (1024.0 * 1024.0),
            "rss_slope_mib_min": _memory_slope_mib_min(memory_samples),
            "cache_hit_rate": stats.cache_hit_rate,
            "decode_mib_s": stats.decode_megabytes_per_second,
            "read_ms_p95": stats.read_latency_ms_p95,
            "allocated_cache_mib": stats.allocated_cache_bytes / (1024.0 * 1024.0),
            "memory_target_mib": stats.memory_target_bytes / (1024.0 * 1024.0),
            "memory_target_reason": stats.memory_target_reason,
            "lease_balance": stats.lease_acquisitions - stats.lease_releases,
            "stale_lease_releases": stats.stale_lease_releases,
        }
        report["trajectory"] = {
            "atom_count": store.atom_count,
            "frame_count": store.frame_count,
            "frame_count_is_final": store.frame_count_is_final,
            "has_cells": store.has_cells,
        }
        return report
    finally:
        if streamer is not None:
            streamer.set_playback_fps(0.0)
            streamer.stop()
        store.close()


def _wait_for_index(store, *, timeout_s: float) -> None:
    deadline_s = time.monotonic() + max(0.0, float(timeout_s))
    known_count = store.frame_count
    while not store.frame_count_is_final:
        remaining_s = deadline_s - time.monotonic()
        if remaining_s <= 0.0:
            raise TimeoutError(
                f"trajectory index did not finish within {float(timeout_s):.1f} seconds"
            )
        known_count, _complete = store.wait_for_index_update(
            known_count,
            timeout_s=min(0.2, remaining_s),
        )


def _seek_indices(
    frame_count: int,
    *,
    sample_count: int,
    pattern: str,
    seed: int,
) -> np.ndarray:
    if pattern == "sequential":
        return np.arange(sample_count, dtype=np.int64) % frame_count
    if pattern == "storm":
        sequence = np.arange(sample_count, dtype=np.int64)
        near = (sequence // 2) % frame_count
        far = (frame_count - 1 - near) % frame_count
        return np.where(sequence % 2 == 0, near, far)
    if pattern == "random":
        return np.random.default_rng(seed).integers(0, frame_count, size=sample_count, dtype=np.int64)
    raise ValueError(f"unknown seek pattern: {pattern}")


def _memory_slope_mib_min(samples: list[tuple[float, int]]) -> float:
    if len(samples) < 2:
        return 0.0
    x = np.asarray([sample[0] for sample in samples], dtype=np.float64)
    y = np.asarray([sample[1] for sample in samples], dtype=np.float64) / (1024.0 * 1024.0)
    centered = x - np.mean(x)
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(centered, y - np.mean(y)) / denominator * 60.0)


def _report_header(scenario: str, source: TrajectorySource) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario": scenario,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "trajplayer_version": __display_version__,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cpu_count": os.cpu_count(),
        },
        "source": [
            {
                "name": path.name,
                "suffix": path.suffix.lower(),
                "bytes": path.stat().st_size,
            }
            for path in source.paths
        ],
    }


def _source_from_args(args: argparse.Namespace) -> TrajectorySource:
    paths = [args.trajectory]
    if args.topology is not None:
        paths.append(args.topology)
    return resolve_trajectory_source(paths)


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--topology", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-timeout", type=float, default=600.0)
    parser.add_argument(
        "--wait-index",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait for progressive XYZ/extXYZ indexing before the measured workload",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark TrajPlayer with real trajectory files")
    subparsers = parser.add_subparsers(dest="scenario", required=True)

    open_parser = subparsers.add_parser("open", help="Measure metadata and first-frame latency")
    _add_source_arguments(open_parser)
    open_parser.set_defaults(wait_index=False)
    open_parser.add_argument("--repeat", type=int, default=5)

    seek_parser = subparsers.add_parser("seek", help="Measure direct real-file frame reads")
    _add_source_arguments(seek_parser)
    seek_parser.add_argument("--samples", type=int, default=500)
    seek_parser.add_argument("--pattern", choices=("random", "sequential", "storm"), default="random")
    seek_parser.add_argument("--seed", type=int, default=20260810)

    soak_parser = subparsers.add_parser("soak", help="Run a bounded no-skip streaming soak")
    _add_source_arguments(soak_parser)
    soak_parser.add_argument("--minutes", type=float, default=30.0)
    soak_parser.add_argument("--fps", type=float, default=60.0)
    soak_parser.add_argument("--prefetch-radius", type=int, default=200)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source = _source_from_args(args)
    if args.scenario == "open":
        report = benchmark_open(
            source,
            repeat=args.repeat,
            wait_index=args.wait_index,
            index_timeout_s=args.index_timeout,
        )
    elif args.scenario == "seek":
        report = benchmark_seek(
            source,
            samples=args.samples,
            pattern=args.pattern,
            seed=args.seed,
            wait_index=args.wait_index,
            index_timeout_s=args.index_timeout,
        )
    else:
        report = benchmark_soak(
            source,
            duration_s=float(args.minutes) * 60.0,
            fps=args.fps,
            prefetch_radius=args.prefetch_radius,
            wait_index=args.wait_index,
            index_timeout_s=args.index_timeout,
        )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.scenario} benchmark: {output}")


if __name__ == "__main__":
    main()
