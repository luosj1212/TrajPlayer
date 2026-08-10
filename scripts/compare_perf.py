from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


Direction = Literal["lower", "higher"]

METRICS: dict[str, Direction] = {
    "startup.process_to_qapplication_ms": "lower",
    "startup.process_to_window_visible_ms": "lower",
    "startup.process_to_first_gl_frame_ms": "lower",
    "open.metadata_ms": "lower",
    "open.first_frame_ms": "lower",
    "open.index_complete_ms": "lower",
    "render.cadence_fps": "higher",
    "render.paint_ms_p50": "lower",
    "render.paint_ms_p95": "lower",
    "render.paint_ms_p99": "lower",
    "render.upload_ms_p50": "lower",
    "render.upload_ms_p95": "lower",
    "render.upload_ms_p99": "lower",
    "render.depth_sort_ms_p50": "lower",
    "render.depth_sort_ms_p95": "lower",
    "render.depth_sort_ms_p99": "lower",
    "pipeline.present_latency_ms_p50": "lower",
    "pipeline.present_latency_ms_p95": "lower",
    "pipeline.present_latency_ms_p99": "lower",
    "io.frame_read_ms_p50": "lower",
    "io.frame_read_ms_p95": "lower",
    "io.frame_read_ms_p99": "lower",
    "io.decode_mb_s": "higher",
    "io.cache_hit_rate": "higher",
    "memory.rss_playback_mib": "lower",
    "memory.rss_peak_mib": "lower",
}

ZERO_INVARIANTS = (
    "pipeline.dropped_frames_total",
    "pipeline.duplicate_frames",
    "io.stale_lease_releases",
    "copies.renderer_full_frame_copy_bytes",
    "copies.renderer_full_frame_copy_bytes_per_frame",
)

SCENARIO_METRICS: dict[str, dict[str, Direction]] = {
    "open": {
        "metrics.metadata_ms.p50": "lower",
        "metrics.metadata_ms.p95": "lower",
        "metrics.first_frame_ms.p50": "lower",
        "metrics.first_frame_ms.p95": "lower",
        "metrics.index_complete_ms.p95": "lower",
        "metrics.rss_mib": "lower",
    },
    "seek": {
        "metrics.open_ms": "lower",
        "metrics.frame_read_ms.p50": "lower",
        "metrics.frame_read_ms.p95": "lower",
        "metrics.frame_read_ms.p99": "lower",
        "metrics.decode_mib_s": "higher",
        "metrics.rss_mib": "lower",
        "metrics.peak_rss_mib": "lower",
    },
    "soak": {
        "metrics.cadence_fps": "higher",
        "metrics.deadline_misses": "lower",
        "metrics.wait_timeouts": "lower",
        "metrics.cache_hit_rate": "higher",
        "metrics.decode_mib_s": "higher",
        "metrics.read_ms_p95": "lower",
        "metrics.rss_end_mib": "lower",
    },
}

SCENARIO_ZERO_INVARIANTS = {
    "soak": (
        "metrics.lease_balance",
        "metrics.stale_lease_releases",
    )
}


@dataclass(frozen=True, slots=True)
class PerformanceRegression:
    metric: str
    baseline: float
    current: float
    regression_percent: float
    reason: str


def compare_performance(
    baseline: dict[str, object],
    current: dict[str, object],
    *,
    fail_regression_percent: float = 10.0,
) -> list[PerformanceRegression]:
    if fail_regression_percent < 0.0:
        raise ValueError("fail_regression_percent must be non-negative")
    baseline_scenario = _value_at_path(baseline, "scenario")
    current_scenario = _value_at_path(current, "scenario")
    if baseline_scenario != current_scenario:
        raise ValueError(
            f"benchmark scenarios differ: {baseline_scenario!r} != {current_scenario!r}"
        )
    scenario = baseline_scenario if isinstance(baseline_scenario, str) else None
    metrics = SCENARIO_METRICS.get(scenario, METRICS)
    zero_invariants = SCENARIO_ZERO_INVARIANTS.get(scenario, ZERO_INVARIANTS)
    regressions: list[PerformanceRegression] = []
    for path, direction in metrics.items():
        baseline_value = _numeric_value(baseline, path)
        current_value = _numeric_value(current, path)
        if baseline_value is None or current_value is None or baseline_value <= 0.0:
            continue
        if direction == "lower":
            regression = ((current_value / baseline_value) - 1.0) * 100.0
        else:
            regression = (1.0 - (current_value / baseline_value)) * 100.0
        if regression > fail_regression_percent:
            regressions.append(
                PerformanceRegression(
                    metric=path,
                    baseline=baseline_value,
                    current=current_value,
                    regression_percent=regression,
                    reason=f"relative {direction}-is-better budget exceeded",
                )
            )

    for path in zero_invariants:
        current_value = _numeric_value(current, path)
        if current_value is not None and current_value != 0.0:
            regressions.append(
                PerformanceRegression(
                    metric=path,
                    baseline=0.0,
                    current=current_value,
                    regression_percent=float("inf"),
                    reason="zero invariant violated",
                )
            )
    if scenario is None and (
        _value_at_path(baseline, "render.single_draw_call_per_frame") is True
        and _value_at_path(current, "render.single_draw_call_per_frame") is False
    ):
        regressions.append(
            PerformanceRegression(
                metric="render.single_draw_call_per_frame",
                baseline=1.0,
                current=0.0,
                regression_percent=100.0,
                reason="single draw-call invariant violated",
            )
        )
    return regressions


def comparison_context_warnings(
    baseline: dict[str, object],
    current: dict[str, object],
) -> list[str]:
    warnings: list[str] = []
    for path in ("environment.machine", "environment.processor", "environment.platform"):
        baseline_value = _value_at_path(baseline, path)
        current_value = _value_at_path(current, path)
        if baseline_value is not None and current_value is not None and baseline_value != current_value:
            warnings.append(f"{path} differs: {baseline_value!r} != {current_value!r}")
    return warnings


def _value_at_path(document: dict[str, object], path: str) -> object | None:
    value: object = document
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _numeric_value(document: dict[str, object], path: str) -> float | None:
    value = _value_at_path(document, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two TrajPlayer benchmark reports")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--fail-regression-percent", type=float, default=10.0)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    current = json.loads(args.current.read_text(encoding="utf-8"))
    for warning in comparison_context_warnings(baseline, current):
        print(f"WARNING {warning}")
    regressions = compare_performance(
        baseline,
        current,
        fail_regression_percent=args.fail_regression_percent,
    )
    if regressions:
        for regression in regressions:
            percentage = (
                "invariant"
                if regression.regression_percent == float("inf")
                else f"{regression.regression_percent:.2f}%"
            )
            print(
                f"REGRESSION {regression.metric}: {regression.baseline:g} -> "
                f"{regression.current:g} ({percentage}; {regression.reason})"
            )
        raise SystemExit(1)
    print("Performance comparison passed")


if __name__ == "__main__":
    main()
