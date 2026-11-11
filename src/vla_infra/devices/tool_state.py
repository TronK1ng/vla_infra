"""Shared tool readiness gate (M5 Round 01R4).

Tool.Switch succeeding only means the RDK accepted the tool change; it does not
prove the TCP/payload calibration is correct.  Motion requires an explicit
operator/preflight verification of tool name + TCP/payload hashes.
"""

from __future__ import annotations

import threading
from typing import Optional

from ..errors import DeviceError, ErrorCode


class ToolReadiness:
    def __init__(
        self,
        *,
        expected_name: Optional[str] = None,
        expected_tcp_hash: Optional[str] = None,
        expected_payload_hash: Optional[str] = None,
    ) -> None:
        self._expected_name = expected_name
        self._expected_tcp_hash = expected_tcp_hash
        self._expected_payload_hash = expected_payload_hash
        self._lock = threading.Lock()
        self._switched = False
        self._verified = False
        self._name: Optional[str] = None
        self._tcp_hash: Optional[str] = None
        self._payload_hash: Optional[str] = None

    @property
    def expected_name(self):
        return self._expected_name

    @property
    def expected_tcp_hash(self):
        return self._expected_tcp_hash

    @property
    def expected_payload_hash(self):
        return self._expected_payload_hash

    @property
    def switched(self) -> bool:
        with self._lock:
            return self._switched

    @property
    def verified(self) -> bool:
        with self._lock:
            return self._verified

    def mark_switched(self, name: str) -> None:
        with self._lock:
            self._switched = True
            self._verified = False
            self._name = name

    def verify(self, *, name: str, tcp_hash: str, payload_hash: str) -> None:
        if not name or not tcp_hash or not payload_hash:
            raise DeviceError(
                "tool verification requires name, TCP hash and payload hash",
                code=ErrorCode.CONFIG_INVALID,
            )
        with self._lock:
            if not self._switched:
                raise DeviceError(
                    "cannot verify a tool that has not been switched",
                    code=ErrorCode.NOT_READY,
                )
            if self._name is not None and name != self._name:
                raise DeviceError(
                    "tool verification name does not match switched tool",
                    code=ErrorCode.CONFIG_INVALID,
                    details={"switched": self._name, "verified": name},
                )
            if self._expected_name is not None and name != self._expected_name:
                raise DeviceError("tool name does not match approved profile", code=ErrorCode.CONFIG_INVALID)
            if self._expected_tcp_hash is not None and tcp_hash != self._expected_tcp_hash:
                raise DeviceError("TCP hash does not match approved profile", code=ErrorCode.CONFIG_INVALID)
            if self._expected_payload_hash is not None and payload_hash != self._expected_payload_hash:
                raise DeviceError("payload hash does not match approved profile", code=ErrorCode.CONFIG_INVALID)
            self._tcp_hash = tcp_hash
            self._payload_hash = payload_hash
            self._verified = True

    def reset(self) -> None:
        with self._lock:
            self._switched = False
            self._verified = False
            self._name = None
            self._tcp_hash = None
            self._payload_hash = None

    def require_motion_ready(self) -> None:
        with self._lock:
            if not (self._switched and self._verified):
                raise DeviceError(
                    "tool/TCP/payload is not switched and verified; motion is blocked",
                    code=ErrorCode.NOT_READY,
                )


__all__ = ["ToolReadiness"]
