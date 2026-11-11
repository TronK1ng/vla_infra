"""Shared session safety gate for M5 producers/consumers."""

from __future__ import annotations

import threading

from ..errors import DeviceError, ErrorCode


class SessionSafetyGate:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._quarantined = False
        self._reason = ""

    @property
    def quarantined(self) -> bool:
        with self._lock:
            return self._quarantined

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    def quarantine(self, reason: str) -> None:
        with self._lock:
            self._quarantined = True
            self._reason = str(reason or "quarantined")

    def require_usable(self) -> None:
        with self._lock:
            if self._quarantined:
                raise DeviceError(
                    f"session is quarantined: {self._reason}",
                    code=ErrorCode.STOP_FAILED,
                )


__all__ = ["SessionSafetyGate"]
