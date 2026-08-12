from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlaybackDecision:
    frame_index: int
    dropped_frames: int
    stop_playback: bool = False


class PlaybackEngine:
    """Fixed-rate sequential scheduler that never skips trajectory frames."""

    def __init__(
        self,
        *,
        total_frames: int,
        fps: float = 60.0,
        loop: bool = True,
        range_start: int = 0,
        range_end: int | None = None,
    ) -> None:
        if total_frames < 0:
            raise ValueError("total_frames must be non-negative")
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.total_frames = int(total_frames)
        self.fps = float(fps)
        self.loop = bool(loop)
        last = max(0, self.total_frames - 1)
        self.range_start = max(0, min(int(range_start), last))
        self.range_end = last if range_end is None else max(0, min(int(range_end), last))
        if self.range_start > self.range_end:
            self.range_start, self.range_end = self.range_end, self.range_start
        self._running = False
        self._last_frame_index = 0
        self._next_frame_time_s = 0.0

    @property
    def running(self) -> bool:
        return self._running

    def start(self, *, frame_index: int, now_s: float) -> None:
        if self.total_frames <= 0:
            self._running = False
            return
        frame_index = max(self.range_start, min(int(frame_index), self.range_end))
        self._running = True
        self._last_frame_index = frame_index
        self._next_frame_time_s = float(now_s) + (1.0 / self.fps)

    def stop(self) -> None:
        self._running = False

    def next_frame_delay_s(self, now_s: float) -> float | None:
        if not self._running or self.total_frames <= 0:
            return None
        return max(0.0, self._next_frame_time_s - float(now_s))

    def schedule(self, now_s: float) -> PlaybackDecision | None:
        if not self._running or self.total_frames <= 0:
            return None

        now_s = float(now_s)
        if now_s + 1.0e-9 < self._next_frame_time_s:
            return None

        stop_playback = False
        frame_index = self._last_frame_index + 1
        if frame_index <= self.range_end:
            if not self.loop and frame_index == self.range_end:
                stop_playback = True
        elif self.loop:
            frame_index = self.range_start
        else:
            frame_index = self.range_end
            stop_playback = True

        self._last_frame_index = int(frame_index)
        next_deadline_s = self._next_frame_time_s + (1.0 / self.fps)
        if next_deadline_s <= now_s:
            next_deadline_s = now_s + (1.0 / self.fps)
        self._next_frame_time_s = next_deadline_s
        if stop_playback:
            self._running = False
        return PlaybackDecision(
            frame_index=int(frame_index),
            dropped_frames=0,
            stop_playback=stop_playback,
        )
