from __future__ import annotations

import math

import numpy as np


class RollingLatency:
    """Fixed-memory latency sample ring used by long-running sessions."""

    def __init__(self, capacity: int = 4096) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._values = np.empty(int(capacity), dtype=np.float64)
        self._count = 0
        self._cursor = 0

    @property
    def sample_count(self) -> int:
        return min(self._count, self._values.size)

    @property
    def total_count(self) -> int:
        return self._count

    def record(self, value: float) -> None:
        self._values[self._cursor] = max(0.0, float(value))
        self._cursor = (self._cursor + 1) % self._values.size
        self._count += 1

    def percentile(self, percentile: float) -> float:
        count = self.sample_count
        if count == 0:
            return 0.0
        values = self._values[:count].copy()
        values.sort()
        index = max(
            0,
            min(count - 1, math.ceil((float(percentile) / 100.0) * count) - 1),
        )
        return float(values[index])

    def average(self) -> float:
        count = self.sample_count
        if count == 0:
            return 0.0
        return float(np.mean(self._values[:count]))

    def maximum(self) -> float:
        count = self.sample_count
        if count == 0:
            return 0.0
        return float(np.max(self._values[:count]))

    def span(self) -> float:
        count = self.sample_count
        if count < 2:
            return 0.0
        if self._count <= self._values.size:
            oldest = self._values[0]
            newest = self._values[count - 1]
        else:
            oldest = self._values[self._cursor]
            newest = self._values[(self._cursor - 1) % self._values.size]
        return max(0.0, float(newest - oldest))

    def summary(self, *, prefix: str = "latency_ms") -> dict[str, float | int]:
        return {
            f"{prefix}_samples": self.sample_count,
            f"{prefix}_p50": self.percentile(50.0),
            f"{prefix}_p95": self.percentile(95.0),
            f"{prefix}_p99": self.percentile(99.0),
        }
