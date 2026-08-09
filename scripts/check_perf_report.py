from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_perf import _numeric_value, _value_at_path


REQUIRED_NUMERIC_METRICS = (
    "startup.process_to_qapplication_ms",
    "startup.process_to_window_visible_ms",
    "startup.process_to_first_gl_frame_ms",
    "open.metadata_ms",
    "open.first_frame_ms",
    "render.cadence_fps",
    "render.paint_ms_p50",
    "render.paint_ms_p95",
    "render.paint_ms_p99",
    "render.upload_ms_p50",
    "render.upload_ms_p95",
    "render.upload_ms_p99",
    "render.depth_sort_ms_p50",
    "render.depth_sort_ms_p95",
    "render.depth_sort_ms_p99",
    "pipeline.present_latency_ms_p50",
    "pipeline.present_latency_ms_p95",
    "pipeline.present_latency_ms_p99",
    "pipeline.dropped_frames_total",
    "pipeline.duplicate_frames",
    "io.frame_read_ms_p50",
    "io.frame_read_ms_p95",
    "io.frame_read_ms_p99",
    "io.decode_mb_s",
    "io.cache_hit_rate",
    "memory.rss_idle_mib",
    "memory.rss_playback_mib",
    "memory.rss_peak_mib",
    "memory.frame_cache_mib",
    "copies.renderer_full_frame_copy_bytes",
    "copies.renderer_full_frame_copy_bytes_per_frame",
)


def performance_report_errors(report: dict[str, object]) -> list[str]:
    errors = [
        f"missing numeric metric: {path}"
        for path in REQUIRED_NUMERIC_METRICS
        if _numeric_value(report, path) is None
    ]
    if _numeric_value(report, "render.frames") in {None, 0.0}:
        errors.append("render.frames must be positive")
    if _numeric_value(report, "pipeline.dropped_frames_total") not in {None, 0.0}:
        errors.append("playback dropped trajectory frames")
    if _numeric_value(report, "pipeline.duplicate_frames") not in {None, 0.0}:
        errors.append("playback presented duplicate trajectory frames")
    if _numeric_value(report, "copies.renderer_full_frame_copy_bytes") not in {None, 0.0}:
        errors.append("renderer performed a full-frame CPU copy")
    if _numeric_value(report, "io.stale_lease_releases") not in {None, 0.0}:
        errors.append("stale frame lease release detected")
    if _value_at_path(report, "timed_out") is True:
        errors.append("benchmark timed out")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a TrajPlayer benchmark JSON report")
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    errors = performance_report_errors(report)
    if errors:
        raise SystemExit("Performance report check failed:\n- " + "\n- ".join(errors))
    print(f"Performance report passed: {args.report}")


if __name__ == "__main__":
    main()
