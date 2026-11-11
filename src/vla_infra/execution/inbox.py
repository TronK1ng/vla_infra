"""Bounded, latest-only action inbox.

A chunk is bound to the control epoch in which its *observation* was produced.
The policy worker captures the epoch before inference; a chunk inferred before a
HOLD can therefore never be executed after a later ARM.

M1 deliberately implements whole-chunk replacement (capacity 1) and fixed
consumption; no temporal ensemble.  Sequence regressions are rejected even
after the queue was cleared, so a delayed response cannot be replayed.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from typing import Optional

from ..contracts.action import ActionChunk
from ..errors import ConfigError
from .authority import ControlAuthority


@dataclass(frozen=True)
class InboxEntry:
    chunk: ActionChunk
    epoch: int
    received_time_ns: int


@dataclass(frozen=True)
class InboxStats:
    submitted: int = 0
    replaced: int = 0
    taken: int = 0
    rejected_epoch: int = 0
    rejected_sequence: int = 0
    rejected_stale: int = 0
    cleared: int = 0

    def as_dict(self) -> dict:
        return {
            "submitted": self.submitted,
            "replaced": self.replaced,
            "taken": self.taken,
            "rejected_epoch": self.rejected_epoch,
            "rejected_sequence": self.rejected_sequence,
            "rejected_stale": self.rejected_stale,
            "cleared": self.cleared,
        }


class ActionInbox:
    def __init__(self, authority: ControlAuthority, *, capacity: int = 1) -> None:
        if capacity != 1:
            raise ConfigError(
                "M1 ActionInbox is latest-only (capacity must be 1); "
                "a different replacement policy must be designed and tested first",
                details={"capacity": capacity},
            )
        self._authority = authority
        self._lock = threading.Lock()
        self._pending: Optional[InboxEntry] = None
        self._last_submitted_seq: Optional[int] = None
        self._stats = InboxStats()

    def submit(self, chunk: ActionChunk, *, epoch: int, received_time_ns: int) -> bool:
        """Submit a chunk; returns True if it became the pending chunk."""
        if not isinstance(chunk, ActionChunk):
            raise TypeError("inbox accepts ActionChunk objects")
        if not isinstance(received_time_ns, int) or received_time_ns <= 0:
            raise ValueError("received_time_ns must be a positive integer")
        with self._authority.guard():
            if not self._authority.is_current(epoch):
                with self._lock:
                    self._stats = _bump(self._stats, rejected_epoch=1)
                return False
            with self._lock:
                seq = chunk.request_sequence_id
                if self._last_submitted_seq is not None and seq <= self._last_submitted_seq:
                    self._stats = _bump(self._stats, rejected_sequence=1)
                    return False
                replaced = self._pending is not None
                self._pending = InboxEntry(
                    chunk=chunk,
                    epoch=int(epoch),
                    received_time_ns=int(received_time_ns),
                )
                self._last_submitted_seq = seq
                self._stats = _bump(
                    self._stats,
                    submitted=1,
                    replaced=1 if replaced else 0,
                )
                return True

    def take(self, *, now_ns: int, epoch: int, max_age_ns: int) -> Optional[InboxEntry]:
        """Pop the pending chunk if it belongs to ``epoch`` and is not stale."""
        if max_age_ns <= 0:
            raise ValueError("max_age_ns must be positive")
        with self._authority.guard():
            if not self._authority.is_current(epoch):
                with self._lock:
                    self._stats = _bump(self._stats, rejected_epoch=1)
                return None
            with self._lock:
                entry = self._pending
                self._pending = None
                if entry is None:
                    return None
                if now_ns - entry.received_time_ns > max_age_ns:
                    self._stats = _bump(self._stats, rejected_stale=1)
                    return None
                self._stats = _bump(self._stats, taken=1)
                return entry

    def clear(self, reason: str = "") -> None:
        with self._authority.guard():
            with self._lock:
                had_pending = self._pending is not None
                self._pending = None
                if had_pending:
                    self._stats = _bump(self._stats, cleared=1)

    @property
    def pending(self) -> Optional[InboxEntry]:
        with self._lock:
            return self._pending

    @property
    def pending_sequence(self) -> Optional[int]:
        entry = self.pending
        return None if entry is None else entry.chunk.request_sequence_id

    @property
    def last_submitted_sequence(self) -> Optional[int]:
        with self._lock:
            return self._last_submitted_seq

    @property
    def stats(self) -> InboxStats:
        with self._lock:
            return self._stats

    def depth(self) -> int:
        with self._lock:
            return 1 if self._pending is not None else 0


def _bump(stats: InboxStats, **increments: int) -> InboxStats:
    return replace(
        stats,
        **{name: getattr(stats, name) + int(delta) for name, delta in increments.items()},
    )


__all__ = ["ActionInbox", "InboxEntry", "InboxStats"]
