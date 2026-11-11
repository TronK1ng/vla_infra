"""Thread-safe, bounded ring buffer for timestamped robot state.

The buffer enforces strictly increasing *host* timestamps and never invents a
sample.  The synchronizer asks it for the two samples bracketing a target time;
a target newer than the newest sample yields ``None`` (never the newest sample
dressed up as a future measurement).
"""

from __future__ import annotations

import bisect
import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

from ..contracts.observation import RobotStateSample
from ..errors import ErrorCode, VlaInfraError


class NonMonotonicTimestampError(VlaInfraError):
    code = ErrorCode.BUFFER_NOT_MONOTONIC


@dataclass(frozen=True)
class Bracket:
    """Two samples bracketing a target host time."""

    before: RobotStateSample
    after: RobotStateSample
    exact: bool
    gap_ns: int


@dataclass(frozen=True)
class RingBufferStats:
    capacity: int
    size: int
    appended: int
    rejected: int
    evicted: int
    oldest_host_ns: Optional[int]
    newest_host_ns: Optional[int]

    def as_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "size": self.size,
            "appended": self.appended,
            "rejected": self.rejected,
            "evicted": self.evicted,
            "oldest_host_ns": self.oldest_host_ns,
            "newest_host_ns": self.newest_host_ns,
        }


class TimestampRingBuffer:
    def __init__(self, capacity: int = 2048, *, name: str = "robot_state") -> None:
        if capacity < 2:
            raise ValueError("ring buffer capacity must be >= 2")
        self._name = name
        self._capacity = int(capacity)
        self._lock = threading.RLock()
        self._samples: Deque[RobotStateSample] = deque(maxlen=self._capacity)
        self._appended = 0
        self._rejected = 0
        self._evicted = 0

    def append(self, sample: RobotStateSample) -> None:
        if not isinstance(sample, RobotStateSample):
            raise TypeError("ring buffer accepts RobotStateSample objects")
        with self._lock:
            if self._samples and sample.host_time_ns <= self._samples[-1].host_time_ns:
                self._rejected += 1
                raise NonMonotonicTimestampError(
                    f"{self._name}: host timestamp did not increase "
                    f"({sample.host_time_ns} <= {self._samples[-1].host_time_ns})",
                    code=ErrorCode.BUFFER_NOT_MONOTONIC,
                    details={
                        "buffer": self._name,
                        "new": sample.host_time_ns,
                        "last": self._samples[-1].host_time_ns,
                    },
                )
            if len(self._samples) == self._capacity:
                self._evicted += 1
            self._samples.append(sample)
            self._appended += 1

    def latest(self) -> Optional[RobotStateSample]:
        with self._lock:
            return self._samples[-1] if self._samples else None

    def oldest(self) -> Optional[RobotStateSample]:
        with self._lock:
            return self._samples[0] if self._samples else None

    def bracket(self, target_ns: int) -> Optional[Bracket]:
        """Return the samples bracketing ``target_ns``.

        ``before.host_time_ns <= target_ns <= after.host_time_ns``.  Returns
        ``None`` when the target is outside the buffered range (including the
        "target in the future" case).
        """
        if not isinstance(target_ns, int) or target_ns <= 0:
            raise ValueError("target_ns must be a positive integer")
        with self._lock:
            if not self._samples:
                return None
            samples: Tuple[RobotStateSample, ...] = tuple(self._samples)
        keys = [s.host_time_ns for s in samples]
        index = bisect.bisect_left(keys, target_ns)
        if index >= len(samples):
            return None
        if keys[index] == target_ns:
            return Bracket(samples[index], samples[index], True, 0)
        if index == 0:
            return None
        before = samples[index - 1]
        after = samples[index]
        return Bracket(before, after, False, after.host_time_ns - before.host_time_ns)

    def snapshot(self) -> Tuple[RobotStateSample, ...]:
        with self._lock:
            return tuple(self._samples)

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()

    @property
    def stats(self) -> RingBufferStats:
        with self._lock:
            return RingBufferStats(
                capacity=self._capacity,
                size=len(self._samples),
                appended=self._appended,
                rejected=self._rejected,
                evicted=self._evicted,
                oldest_host_ns=self._samples[0].host_time_ns if self._samples else None,
                newest_host_ns=self._samples[-1].host_time_ns if self._samples else None,
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._samples)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        stats = self.stats
        return f"TimestampRingBuffer(name={self._name!r}, size={stats.size}, capacity={stats.capacity})"


__all__ = [
    "Bracket",
    "NonMonotonicTimestampError",
    "RingBufferStats",
    "TimestampRingBuffer",
]
