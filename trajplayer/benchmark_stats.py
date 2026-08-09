from __future__ import annotations

from dataclasses import dataclass, field

from .telemetry import RollingLatency


@dataclass
class BenchmarkDiagnostics:
    tick_times_s: RollingLatency = field(default_factory=RollingLatency)
    decision_times_s: RollingLatency = field(default_factory=RollingLatency)
    upload_times_s: RollingLatency = field(default_factory=RollingLatency)
    present_latency_ms: RollingLatency = field(default_factory=RollingLatency)
    dropped_frames_total: int = 0
    no_frame_count: int = 0
    duplicate_frame_count: int = 0
    no_decision_count: int = 0

    def record_tick(self, *, timestamp_s: float) -> None:
        self.tick_times_s.record(timestamp_s)

    def record_decision(self, *, timestamp_s: float, dropped_frames: int) -> None:
        self.decision_times_s.record(timestamp_s)
        self.dropped_frames_total += int(dropped_frames)

    def record_upload(self, *, timestamp_s: float) -> None:
        self.upload_times_s.record(timestamp_s)

    def record_present_latency(self, latency_ms: float) -> None:
        self.present_latency_ms.record(latency_ms)

    def record_no_frame(self) -> None:
        self.no_frame_count += 1

    def record_duplicate_frame(self) -> None:
        self.duplicate_frame_count += 1

    def record_no_decision(self) -> None:
        self.no_decision_count += 1

    def summary(self) -> dict[str, float | int]:
        return {
            "timer_ticks": self.tick_times_s.total_count,
            "timer_span_s": self.tick_times_s.span(),
            "timer_hz": _hz(self.tick_times_s),
            "playback_decisions": self.decision_times_s.total_count,
            "decision_span_s": self.decision_times_s.span(),
            "decision_hz": _hz(self.decision_times_s),
            "uploaded_frames": self.upload_times_s.total_count,
            "upload_span_s": self.upload_times_s.span(),
            "upload_hz": _hz(self.upload_times_s),
            "dropped_frames_total": self.dropped_frames_total,
            "dropped_frames": self.dropped_frames_total,
            "no_frame_count": self.no_frame_count,
            "duplicate_frame_count": self.duplicate_frame_count,
            "duplicate_frames": self.duplicate_frame_count,
            "no_decision_count": self.no_decision_count,
            **self.present_latency_ms.summary(prefix="present_latency_ms"),
        }


def _hz(values: RollingLatency) -> float:
    span = values.span()
    if values.sample_count < 2 or span <= 0.0:
        return 0.0
    return (values.sample_count - 1) / span
