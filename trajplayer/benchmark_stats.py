from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BenchmarkDiagnostics:
    tick_times_s: list[float] = field(default_factory=list)
    decision_times_s: list[float] = field(default_factory=list)
    upload_times_s: list[float] = field(default_factory=list)
    dropped_frames_total: int = 0
    no_frame_count: int = 0
    duplicate_frame_count: int = 0
    no_decision_count: int = 0

    def record_tick(self, *, timestamp_s: float) -> None:
        self.tick_times_s.append(float(timestamp_s))

    def record_decision(self, *, timestamp_s: float, dropped_frames: int) -> None:
        self.decision_times_s.append(float(timestamp_s))
        self.dropped_frames_total += int(dropped_frames)

    def record_upload(self, *, timestamp_s: float) -> None:
        self.upload_times_s.append(float(timestamp_s))

    def record_no_frame(self) -> None:
        self.no_frame_count += 1

    def record_duplicate_frame(self) -> None:
        self.duplicate_frame_count += 1

    def record_no_decision(self) -> None:
        self.no_decision_count += 1

    def summary(self) -> dict[str, float | int]:
        return {
            "timer_ticks": len(self.tick_times_s),
            "timer_span_s": _span(self.tick_times_s),
            "timer_hz": _hz(self.tick_times_s),
            "playback_decisions": len(self.decision_times_s),
            "decision_span_s": _span(self.decision_times_s),
            "decision_hz": _hz(self.decision_times_s),
            "uploaded_frames": len(self.upload_times_s),
            "upload_span_s": _span(self.upload_times_s),
            "upload_hz": _hz(self.upload_times_s),
            "dropped_frames_total": self.dropped_frames_total,
            "no_frame_count": self.no_frame_count,
            "duplicate_frame_count": self.duplicate_frame_count,
            "no_decision_count": self.no_decision_count,
        }


def _span(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return max(0.0, values[-1] - values[0])


def _hz(values: list[float]) -> float:
    span = _span(values)
    if len(values) < 2 or span <= 0.0:
        return 0.0
    return (len(values) - 1) / span
