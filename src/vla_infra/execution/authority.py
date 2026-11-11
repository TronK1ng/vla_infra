"""Single control authority / write lease.

Only the supervisor mutates the authority.  Executors perform every device write
inside :meth:`ControlAuthority.write_permit`, which holds the authority lock.
Revoking (HOLD/FAULT) therefore cannot return while a write is still in flight,
and no write can *begin* after revocation returns.  This is the mechanism that
makes HOLD block motion rather than merely changing a status flag.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from ..errors import ErrorCode, NotArmedError


@dataclass(frozen=True)
class AuthoritySnapshot:
    epoch: int
    valid: bool
    running: bool
    reason: str
    owner: str


class ControlAuthority:
    def __init__(self, owner: str = "supervisor") -> None:
        self._owner = owner
        self._lock = threading.RLock()
        self._epoch = 0
        self._valid = False
        self._running = False
        self._reason = "init"

    # -- state --------------------------------------------------------------
    @property
    def epoch(self) -> int:
        with self._lock:
            return self._epoch

    @property
    def valid(self) -> bool:
        with self._lock:
            return self._valid

    @property
    def running(self) -> bool:
        with self._lock:
            return self._valid and self._running

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    @property
    def owner(self) -> str:
        return self._owner

    def snapshot(self) -> AuthoritySnapshot:
        with self._lock:
            return AuthoritySnapshot(
                epoch=self._epoch,
                valid=self._valid,
                running=self._running,
                reason=self._reason,
                owner=self._owner,
            )

    # -- transitions --------------------------------------------------------
    def grant(self, reason: str = "armed") -> int:
        """Create a new epoch and mark the system write-capable (not yet running)."""
        with self._lock:
            self._epoch += 1
            self._valid = True
            self._running = False
            self._reason = reason
            return self._epoch

    def activate(self, reason: str = "running") -> int:
        with self._lock:
            if not self._valid:
                raise NotArmedError(
                    "cannot activate execution without a valid control epoch",
                    code=ErrorCode.NOT_ARMED,
                    details={"reason": self._reason},
                )
            self._running = True
            self._reason = reason
            return self._epoch

    def deactivate(self, reason: str = "not_running") -> None:
        with self._lock:
            self._running = False
            self._reason = reason

    def revoke(self, reason: str) -> int:
        """Invalidate the epoch; waits for any in-flight write to finish."""
        with self._lock:
            self._epoch += 1
            self._valid = False
            self._running = False
            self._reason = str(reason)
            return self._epoch

    def is_current(self, epoch: int) -> bool:
        with self._lock:
            return self._valid and int(epoch) == self._epoch

    # -- locking ------------------------------------------------------------
    @contextmanager
    def guard(self) -> Iterator["ControlAuthority"]:
        """Hold the authority lock (lock order: authority -> inbox)."""
        with self._lock:
            yield self

    @contextmanager
    def write_permit(self, epoch: int, *, require_running: bool = True) -> Iterator[int]:
        """Context manager held across a device write.

        Raises :class:`NotArmedError` if the epoch is stale or execution is not
        active.  Revocation waits for the context to exit.
        """
        with self._lock:
            if not self._valid or int(epoch) != self._epoch:
                raise NotArmedError(
                    "control epoch is stale; command refused",
                    code=ErrorCode.EPOCH_MISMATCH,
                    details={
                        "requested_epoch": int(epoch),
                        "current_epoch": self._epoch,
                        "valid": self._valid,
                        "reason": self._reason,
                    },
                )
            if require_running and not self._running:
                raise NotArmedError(
                    "execution is not active; command refused",
                    code=ErrorCode.NOT_ARMED,
                    details={"epoch": self._epoch, "reason": self._reason},
                )
            yield self._epoch


__all__ = ["AuthoritySnapshot", "ControlAuthority"]
