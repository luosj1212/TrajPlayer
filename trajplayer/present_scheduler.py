from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


class PlaybackClock(Protocol):
    @property
    def running(self) -> bool: ...

    def next_frame_delay_s(self, now_s: float) -> float | None: ...


@dataclass(frozen=True, slots=True)
class RenderTicket:
    generation: int
    sequence: int
    frame_index: int
    lease_epoch: int
    submitted_at_s: float


PresentToken = RenderTicket


@dataclass(frozen=True, slots=True)
class PresentAcknowledgement:
    token: RenderTicket
    latency_ms: float
    accepted: bool


class PresentScheduler:
    """Own the one-frame present state shared by playback and the GL widget."""

    def __init__(self) -> None:
        self._generation = 0
        self._sequence = 0
        self._target_frame = 0
        self._displayed_frame = -1
        self._pending: RenderTicket | None = None
        self._painted: RenderTicket | None = None

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def target_frame(self) -> int:
        return self._target_frame

    @property
    def displayed_frame(self) -> int:
        return self._displayed_frame

    @property
    def pending_token(self) -> RenderTicket | None:
        return self._pending

    @property
    def painted_ticket(self) -> RenderTicket | None:
        return self._painted

    @property
    def pending_frame(self) -> int | None:
        return None if self._pending is None else self._pending.frame_index

    @property
    def has_pending_frame(self) -> bool:
        return self._pending is not None

    def begin_generation(self, *, target_frame: int = 0) -> int:
        self._generation += 1
        self._target_frame = int(target_frame)
        self._displayed_frame = -1
        self._pending = None
        self._painted = None
        return self._generation

    def set_target_frame(self, frame_index: int) -> None:
        self._target_frame = int(frame_index)

    def invalidate_display(self) -> None:
        self._displayed_frame = -1

    def set_displayed_frame(self, frame_index: int) -> None:
        """Set display state during initialization and compatibility migrations."""

        self._displayed_frame = int(frame_index)

    def submit(
        self,
        frame_index: int,
        *,
        lease_epoch: int,
        now_s: float,
    ) -> RenderTicket | None:
        if self._pending is not None:
            return None
        self._sequence += 1
        token = RenderTicket(
            generation=self._generation,
            sequence=self._sequence,
            frame_index=int(frame_index),
            lease_epoch=int(lease_epoch),
            submitted_at_s=float(now_s),
        )
        self._pending = token
        return token

    def mark_painted(self, ticket: RenderTicket) -> bool:
        if ticket != self._pending or ticket.generation != self._generation:
            return False
        self._painted = ticket
        return True

    def acknowledge_swap(self, *, now_s: float) -> PresentAcknowledgement | None:
        token = self._painted
        if token is None:
            return None
        self._pending = None
        self._painted = None
        accepted = token.generation == self._generation
        if accepted:
            self._displayed_frame = token.frame_index
        return PresentAcknowledgement(
            token=token,
            latency_ms=max(0.0, (float(now_s) - token.submitted_at_s) * 1000.0),
            accepted=accepted,
        )

    def clear_pending(self) -> RenderTicket | None:
        token = self._pending
        self._pending = None
        self._painted = None
        return token

    def next_timer_delay_ms(
        self,
        *,
        playback: PlaybackClock | None,
        frame_available: bool,
        now_s: float,
    ) -> int | None:
        if self._pending is not None:
            return None
        if self._displayed_frame != self._target_frame:
            return 0 if frame_available else None
        if playback is None or not playback.running:
            return None
        delay_s = playback.next_frame_delay_s(float(now_s))
        if delay_s is None:
            return None
        return max(1, int(math.ceil(delay_s * 1000.0)))
