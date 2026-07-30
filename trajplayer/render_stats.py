from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class RenderStats:
    paint_ms: list[float] = field(default_factory=list)
    upload_ms: list[float] = field(default_factory=list)
    draw_calls: list[int] = field(default_factory=list)
    timestamps_s: list[float] = field(default_factory=list)

    def record_frame(
        self,
        *,
        paint_ms: float,
        upload_ms: float,
        draw_calls: int,
        timestamp_s: float | None = None,
    ) -> None:
        self.paint_ms.append(float(paint_ms))
        self.upload_ms.append(float(upload_ms))
        self.draw_calls.append(int(draw_calls))
        if timestamp_s is not None:
            self.timestamps_s.append(float(timestamp_s))

    def summary(self) -> dict[str, float | int | bool]:
        frame_span_s = self.frame_span_s
        paint_avg = _average(self.paint_ms)
        upload_avg = _average(self.upload_ms)
        budget_ms_avg = paint_avg + upload_avg
        return {
            "frames": len(self.paint_ms),
            "paint_ms_avg": paint_avg,
            "paint_ms_max": max(self.paint_ms) if self.paint_ms else 0.0,
            "paint_ms_p95": _percentile(self.paint_ms, 95.0),
            "upload_ms_avg": upload_avg,
            "upload_ms_max": max(self.upload_ms) if self.upload_ms else 0.0,
            "render_budget_ms_avg": budget_ms_avg,
            "render_budget_fps_avg": (1000.0 / budget_ms_avg) if budget_ms_avg > 0 else 0.0,
            "frame_span_s": frame_span_s,
            "cadence_fps": ((len(self.paint_ms) - 1) / frame_span_s)
            if len(self.paint_ms) > 1 and frame_span_s > 0
            else 0.0,
            "draw_calls_max": max(self.draw_calls) if self.draw_calls else 0,
            "single_draw_call_per_frame": bool(self.draw_calls)
            and max(self.draw_calls) == 1
            and min(self.draw_calls) == 1,
        }

    @property
    def frame_span_s(self) -> float:
        if len(self.timestamps_s) < 2:
            return 0.0
        return max(0.0, self.timestamps_s[-1] - self.timestamps_s[0])


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil((percentile / 100.0) * len(ordered)) - 1))
    return float(ordered[index])
