"""Clock abstractions.

Rule (master plan §5.1): in-process scheduling and age calculations use a
*monotonic* clock only.  Device timestamps are data, never scheduling inputs.
The fake clock makes the control loop and all timeout logic deterministic in
tests.
"""

from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    def now_ns(self) -> int:
        """Monotonic nanoseconds."""

    def sleep(self, duration_ns: int) -> None:
        """Sleep for at least ``duration_ns``."""


class MonotonicClock:
    """Production clock backed by ``time.monotonic_ns``."""

    __slots__ = ()

    def now_ns(self) -> int:
        return time.monotonic_ns()

    def sleep(self, duration_ns: int) -> None:
        if duration_ns > 0:
            time.sleep(duration_ns / 1_000_000_000)


class FakeClock:
    """Deterministic clock; time only moves when the test says so."""

    __slots__ = ("_now_ns",)

    def __init__(self, start_ns: int = 1_000_000_000) -> None:
        if start_ns <= 0:
            raise ValueError("start_ns must be positive")
        self._now_ns = int(start_ns)

    def now_ns(self) -> int:
        return self._now_ns

    def advance(self, duration_ns: int) -> int:
        if duration_ns < 0:
            raise ValueError("cannot move a monotonic clock backwards")
        self._now_ns += int(duration_ns)
        return self._now_ns

    def sleep(self, duration_ns: int) -> None:
        self.advance(duration_ns)

    def set(self, now_ns: int) -> None:
        """Test helper: jump forward to an absolute time."""
        if now_ns < self._now_ns:
            raise ValueError("cannot move a monotonic clock backwards")
        self._now_ns = int(now_ns)


def seconds_to_ns(seconds: float) -> int:
    return int(round(seconds * 1_000_000_000))


def ns_to_seconds(ns: int) -> float:
    return ns / 1_000_000_000
