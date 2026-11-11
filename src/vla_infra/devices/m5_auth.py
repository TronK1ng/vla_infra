"""M5 capability/authorization skeleton.

A preflight report with PASS_ENABLE_ALLOWED is not sufficient by itself.  The
issuer binds station/config/report hashes, operator ACK, freshness and a
process-local HMAC token.  The writer requires a verifier; a fabricated
dataclass without the issuer secret fails.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from ..errors import RealHardwareDisabledError


@dataclass(frozen=True)
class MotionAuthorization:
    station_id: str
    config_hash: str
    report_hash: str
    issued_at_ns: int
    expires_at_ns: int
    operator_ack: bool
    token: str


class M5AuthorizationIssuer:
    def __init__(
        self,
        *,
        station_id: str,
        config_hash: str,
        secret: Optional[bytes] = None,
    ) -> None:
        if not station_id or not config_hash:
            raise ValueError("issuer requires expected station_id/config_hash")
        self._station_id = station_id
        self._config_hash = config_hash
        self._secret = secret or secrets.token_bytes(32)

    def _signature(
        self,
        station_id: str,
        config_hash: str,
        report_hash: str,
        issued_at_ns: int,
        expires_at_ns: int,
    ) -> str:
        payload = f"{station_id}|{config_hash}|{report_hash}|{issued_at_ns}|{expires_at_ns}".encode()
        return hmac.new(self._secret, payload, hashlib.sha256).hexdigest()

    def issue(self, report: Any, *, operator_ack: bool, now_ns: Optional[int] = None) -> MotionAuthorization:
        from .preflight import M5PreflightReport

        if not isinstance(report, M5PreflightReport):
            raise RealHardwareDisabledError("Enable authorization requires an M5PreflightReport")
        if operator_ack is not True:
            raise RealHardwareDisabledError("Enable authorization requires explicit operator ACK")
        now = int(time.monotonic_ns() if now_ns is None else now_ns)
        try:
            report.validate_for_enable(
                expected_station_id=self._station_id,
                expected_config_hash=self._config_hash,
                now_ns=now,
            )
        except ValueError as exc:
            raise RealHardwareDisabledError(
                "preflight report failed enable validation",
                details={"reason": str(exc)},
            ) from exc
        report_hash = report.report_hash()
        token = self._signature(report.station_id, report.config_hash, report_hash, now, report.expires_at_ns)
        return MotionAuthorization(
            station_id=report.station_id,
            config_hash=report.config_hash,
            report_hash=report_hash,
            issued_at_ns=now,
            expires_at_ns=report.expires_at_ns,
            operator_ack=True,
            token=token,
        )

    def verify(self, authorization: Any, *, now_ns: Optional[int] = None) -> bool:
        if not isinstance(authorization, MotionAuthorization):
            return False
        if authorization.station_id != self._station_id or authorization.config_hash != self._config_hash:
            return False
        if authorization.operator_ack is not True:
            return False
        now = int(time.monotonic_ns() if now_ns is None else now_ns)
        if now < authorization.issued_at_ns or now > authorization.expires_at_ns:
            return False
        expected = self._signature(
            authorization.station_id,
            authorization.config_hash,
            authorization.report_hash,
            authorization.issued_at_ns,
            authorization.expires_at_ns,
        )
        return hmac.compare_digest(expected, authorization.token)


class MotionCapability:
    """Shared arm+gripper revocation/epoch state.

    Possession of the object is not enough: the authorization verifier must
    still accept the stored MotionAuthorization.  Revoking once revokes both
    device sinks that share this object.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._authorization: Optional[MotionAuthorization] = None
        self._epoch: Optional[int] = None
        self._active = False

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    @property
    def epoch(self) -> Optional[int]:
        with self._lock:
            return self._epoch

    def grant(self, authorization: MotionAuthorization, epoch: int) -> None:
        if not isinstance(authorization, MotionAuthorization):
            raise ValueError("authorization must be MotionAuthorization")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be non-negative")
        with self._lock:
            self._authorization = authorization
            self._epoch = int(epoch)
            self._active = True

    def check(self, authorization: Any, epoch: int) -> bool:
        with self._lock:
            return (
                self._active
                and self._authorization is not None
                and authorization is self._authorization
                and self._epoch == epoch
            )

    def revoke(self) -> None:
        with self._lock:
            self._authorization = None
            self._epoch = None
            self._active = False


__all__ = ["M5AuthorizationIssuer", "MotionAuthorization", "MotionCapability"]
