from __future__ import annotations

from dataclasses import dataclass, field

from .telemetry import RollingLatency


@dataclass
class RenderStats:
    """Constant-memory render telemetry with lifetime counters."""

    paint_ms: RollingLatency = field(default_factory=RollingLatency)
    upload_ms: RollingLatency = field(default_factory=RollingLatency)
    depth_sort_ms: RollingLatency = field(default_factory=RollingLatency)
    timestamps_s: RollingLatency = field(default_factory=RollingLatency)
    _frames: int = 0
    _paint_ms_total: float = 0.0
    _upload_ms_total: float = 0.0
    _paint_ms_max: float = 0.0
    _upload_ms_max: float = 0.0
    _renderer_copy_bytes: int = 0
    _draw_calls_min: int | None = None
    _draw_calls_max: int = 0

    def record_frame(
        self,
        *,
        paint_ms: float,
        upload_ms: float,
        draw_calls: int,
        timestamp_s: float | None = None,
        depth_sort_ms: float = 0.0,
        renderer_copy_bytes: int = 0,
    ) -> None:
        paint = max(0.0, float(paint_ms))
        upload = max(0.0, float(upload_ms))
        calls = max(0, int(draw_calls))
        self.paint_ms.record(paint)
        self.upload_ms.record(upload)
        self.depth_sort_ms.record(depth_sort_ms)
        if timestamp_s is not None:
            self.timestamps_s.record(timestamp_s)
        self._frames += 1
        self._paint_ms_total += paint
        self._upload_ms_total += upload
        self._paint_ms_max = max(self._paint_ms_max, paint)
        self._upload_ms_max = max(self._upload_ms_max, upload)
        self._renderer_copy_bytes += max(0, int(renderer_copy_bytes))
        self._draw_calls_min = (
            calls if self._draw_calls_min is None else min(self._draw_calls_min, calls)
        )
        self._draw_calls_max = max(self._draw_calls_max, calls)

    def summary(self) -> dict[str, float | int | bool]:
        frame_span_s = self.frame_span_s
        paint_avg = self._paint_ms_total / self._frames if self._frames else 0.0
        upload_avg = self._upload_ms_total / self._frames if self._frames else 0.0
        budget_ms_avg = paint_avg + upload_avg
        return {
            "frames": self._frames,
            "paint_ms_avg": paint_avg,
            "paint_ms_max": self._paint_ms_max,
            "paint_ms_p50": self.paint_ms.percentile(50.0),
            "paint_ms_p95": self.paint_ms.percentile(95.0),
            "paint_ms_p99": self.paint_ms.percentile(99.0),
            "upload_ms_avg": upload_avg,
            "upload_ms_max": self._upload_ms_max,
            "upload_ms_p50": self.upload_ms.percentile(50.0),
            "upload_ms_p95": self.upload_ms.percentile(95.0),
            "upload_ms_p99": self.upload_ms.percentile(99.0),
            "depth_sort_ms_p50": self.depth_sort_ms.percentile(50.0),
            "depth_sort_ms_p95": self.depth_sort_ms.percentile(95.0),
            "depth_sort_ms_p99": self.depth_sort_ms.percentile(99.0),
            "renderer_full_frame_copy_bytes": self._renderer_copy_bytes,
            "renderer_full_frame_copy_bytes_per_frame": (
                self._renderer_copy_bytes / self._frames if self._frames else 0.0
            ),
            "render_budget_ms_avg": budget_ms_avg,
            "render_budget_fps_avg": (1000.0 / budget_ms_avg) if budget_ms_avg > 0 else 0.0,
            "frame_span_s": frame_span_s,
            "cadence_fps": (
                (self.timestamps_s.sample_count - 1) / frame_span_s
                if self.timestamps_s.sample_count > 1 and frame_span_s > 0
                else 0.0
            ),
            "draw_calls_max": self._draw_calls_max,
            "single_draw_call_per_frame": self._draw_calls_min == 1
            and self._draw_calls_max == 1,
        }

    @property
    def frame_span_s(self) -> float:
        return self.timestamps_s.span()
