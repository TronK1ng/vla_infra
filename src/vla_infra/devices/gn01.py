"""GN01 gripper adapter skeleton (M5 Round 01; injected SDK only).

Normalized channel ``0..1`` maps monotonically to physical width
``width_min..width_max`` metres.  Commands are de-duplicated, range-checked and
always carry explicit speed/force parameters.  No automatic enable, recover or
retry is performed here.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional

import numpy as np

from ..clock import Clock, MonotonicClock
from ..errors import DeviceError, ErrorCode, RealHardwareDisabledError
from .flexiv_rdk import assert_real_hardware_allowed
from .interfaces import SingleWriterGuard
from .m5_auth import MotionCapability
from .tool_state import ToolReadiness


@dataclass(frozen=True)
class Gn01Config:
    width_min_m: float
    width_max_m: float
    speed_m_s: float
    force_n: float
    dedup_width_m: float = 0.0001

    def __post_init__(self) -> None:
        if not (0.0 <= self.width_min_m < self.width_max_m):
            raise DeviceError("GN01 width range is invalid", code=ErrorCode.CONFIG_INVALID)
        for name, value in (("speed", self.speed_m_s), ("force", self.force_n), ("dedup", self.dedup_width_m)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise DeviceError(f"GN01 {name} must be finite/non-negative", code=ErrorCode.CONFIG_INVALID)
        if self.speed_m_s <= 0 or self.force_n <= 0:
            raise DeviceError("GN01 speed/force must be positive", code=ErrorCode.CONFIG_INVALID)

    def width_for_normalized(self, normalized: float) -> float:
        if isinstance(normalized, bool) or not isinstance(normalized, (int, float)):
            raise DeviceError("gripper command must be numeric", code=ErrorCode.SAFETY_GRIPPER)
        value = float(normalized)
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise DeviceError(
                "gripper command must be normalized in [0, 1]",
                code=ErrorCode.SAFETY_GRIPPER,
                details={"value": value},
            )
        return self.width_min_m + value * (self.width_max_m - self.width_min_m)


@dataclass(frozen=True)
class Gn01Status:
    connected: bool = False
    enabled: bool = False
    writes: int = 0
    dedup_skips: int = 0
    stop_calls: int = 0
    stop_failures: int = 0
    physical_stop_confirmed: bool = False
    last_error: str = ""

    def as_dict(self) -> dict:
        return {
            "connected": self.connected,
            "enabled": self.enabled,
            "writes": self.writes,
            "dedup_skips": self.dedup_skips,
            "stop_calls": self.stop_calls,
            "stop_failures": self.stop_failures,
            "physical_stop_confirmed": self.physical_stop_confirmed,
            "last_error": self.last_error,
        }


class Gn01CommandSink:
    def __init__(
        self,
        name: str,
        config: Gn01Config,
        *,
        sdk: Any = None,
        robot: Any = None,
        clock: Optional[Clock] = None,
        authorization_verifier: Optional[Callable[[Any], bool]] = None,
        capability: Optional[MotionCapability] = None,
        tool_readiness: Optional[ToolReadiness] = None,
        authority: Any = None,
        safety_gate: Any = None,
    ) -> None:
        if not isinstance(config, Gn01Config):
            raise DeviceError("GN01 config must be explicit", code=ErrorCode.CONFIG_INVALID)
        self.name = name
        self.config = config
        self._sdk = sdk
        self._robot = robot
        self._clock = clock or MonotonicClock()
        self._gripper: Any = None
        self._authorization_verifier = authorization_verifier
        self._authorization: Any = None
        self._authority_epoch: Optional[int] = None
        self._safety_gate = safety_gate
        self._capability = capability or MotionCapability()
        self._tool_readiness = tool_readiness or ToolReadiness()
        self._authority = authority
        self._guard = SingleWriterGuard(f"gn01:{name}")
        self._status = Gn01Status()
        self._last_width: Optional[float] = None
        self._closed = False

    @property
    def status(self) -> Gn01Status:
        return self._status

    def connect(self) -> None:
        if self._closed:
            raise DeviceError("GN01 sink is closed", code=ErrorCode.DEVICE_NOT_CONNECTED)
        if self._gripper is not None:
            return
        if self._robot is None:
            raise DeviceError(
                "GN01 requires the shared RDK Robot handle",
                code=ErrorCode.DEVICE_NOT_CONNECTED,
            )
        sdk = self._sdk
        if sdk is None:
            assert_real_hardware_allowed("Gn01CommandSink.connect")
            try:
                import flexivrdk as sdk  # type: ignore
            except ImportError as exc:
                raise DeviceError(
                    "flexivrdk (GN01) is not installed",
                    code=ErrorCode.DEVICE_NOT_CONNECTED,
                ) from exc
        try:
            self._gripper = sdk.Gripper(self._robot)
        except Exception as exc:  # noqa: BLE001
            raise DeviceError(
                "failed to construct GN01 interface from the shared Robot",
                code=ErrorCode.DEVICE_FAULT,
                details={"name": self.name, "reason": type(exc).__name__},
            ) from exc
        self._sdk = sdk
        self._status = replace(self._status, connected=True)

    def enable(
        self,
        authorization: Any,
        *,
        authority_epoch: int = 0,
        name: Optional[str] = None,
    ) -> None:
        """Explicit operator-gated GN01 enable; connect() never calls this."""
        if self._safety_gate is not None:
            self._safety_gate.require_usable()
        if self._gripper is None:
            raise DeviceError("GN01 sink is not connected", code=ErrorCode.DEVICE_NOT_CONNECTED)
        if self._authorization_verifier is None or not self._authorization_verifier(authorization):
            raise RealHardwareDisabledError(
                "GN01 Enable requires the shared arm preflight/ACK authorization",
                code=ErrorCode.REAL_HARDWARE_DISABLED,
            )
        if isinstance(authority_epoch, bool) or not isinstance(authority_epoch, int) or authority_epoch < 0:
            raise DeviceError("authority_epoch must be non-negative", code=ErrorCode.VALUE_OUT_OF_RANGE)
        if not self._capability.check(authorization, authority_epoch):
            raise RealHardwareDisabledError(
                "GN01 Enable requires an already-active arm/runtime capability with the same auth/epoch",
                code=ErrorCode.REAL_HARDWARE_DISABLED,
            )
        enable_name = name or self.name
        rollback_ok: Optional[bool] = None
        try:
            self._gripper.Enable(enable_name)
            params = self._gripper.params()
            self._validate_params(params)
        except Exception as exc:  # noqa: BLE001
            if hasattr(self._gripper, "Disable"):
                try:
                    self._gripper.Disable()
                    rollback_ok = True
                except Exception:
                    rollback_ok = False
            self._authorization = None
            self._authority_epoch = None
            self._capability.revoke()
            self._status = replace(
                self._status,
                enabled=False,
                last_error=f"enable_failed:{type(exc).__name__};rollback={rollback_ok}",
            )
            if isinstance(exc, DeviceError):
                raise DeviceError(
                    exc.message,
                    code=exc.code,
                    details={**exc.details, "disable_ok": rollback_ok},
                ) from exc
            raise DeviceError(
                "GN01 Enable/params validation failed",
                code=ErrorCode.DEVICE_FAULT,
                details={"reason": type(exc).__name__, "disable_ok": rollback_ok},
            ) from exc
        self._authorization = authorization
        self._authority_epoch = int(authority_epoch)
        self._status = replace(self._status, enabled=True, last_error="")

    def revoke(self) -> None:
        self._authorization = None
        self._authority_epoch = None
        self._capability.revoke()
        self._tool_readiness.reset()
        self._status = replace(self._status, enabled=False)

    def switch_tool(self) -> None:
        if self._safety_gate is not None:
            self._safety_gate.require_usable()
        """Explicit Tool.Switch; never called automatically by this sink."""
        if self._sdk is None or self._robot is None:
            raise DeviceError("GN01 sink is not connected", code=ErrorCode.DEVICE_NOT_CONNECTED)
        if (
            not self._status.enabled
            or self._authorization is None
            or self._authorization_verifier is None
            or not self._authorization_verifier(self._authorization)
            or not self._capability.check(self._authorization, self._authority_epoch)
        ):
            raise RealHardwareDisabledError(
                "Tool.Switch requires an enabled, authorized GN01 session",
                code=ErrorCode.REAL_HARDWARE_DISABLED,
            )
        try:
            tool = self._sdk.Tool(self._robot)
            tool.Switch(self.name)
        except Exception as exc:  # noqa: BLE001
            self.revoke()
            self._tool_readiness.reset()
            raise DeviceError(
                "Tool.Switch failed; capability revoked and motion blocked",
                code=ErrorCode.DEVICE_FAULT,
                details={"reason": type(exc).__name__},
            ) from exc
        self._tool_readiness.mark_switched(self.name)

    def verify_tool(self, *, name: str, tcp_hash: str, payload_hash: str) -> None:
        if self._safety_gate is not None:
            self._safety_gate.require_usable()
        if not self._status.enabled:
            raise DeviceError("GN01 is not enabled", code=ErrorCode.NOT_READY)
        if (
            self._authorization is None
            or self._authorization_verifier is None
            or not self._authorization_verifier(self._authorization)
            or not self._capability.check(self._authorization, self._authority_epoch)
        ):
            raise RealHardwareDisabledError(
                "tool verification requires an active authorized capability",
                code=ErrorCode.REAL_HARDWARE_DISABLED,
            )
        self._tool_readiness.verify(name=name, tcp_hash=tcp_hash, payload_hash=payload_hash)

    def _validate_params(self, params: Any) -> None:
        required = ("min_width", "max_width", "min_vel", "max_vel", "min_force", "max_force")
        for key in required:
            if not hasattr(params, key):
                raise DeviceError(
                    f"GN01 params missing {key!r}",
                    code=ErrorCode.CONFIG_INVALID,
                )
        if not (
            float(params.min_width) <= self.config.width_min_m
            and self.config.width_max_m <= float(params.max_width)
        ):
            raise DeviceError(
                "configured GN01 width range is not a subset of device params",
                code=ErrorCode.CONFIG_INVALID,
                details={"config": [self.config.width_min_m, self.config.width_max_m]},
            )
        if not (float(params.min_vel) <= self.config.speed_m_s <= float(params.max_vel)):
            raise DeviceError("configured GN01 speed is outside device params", code=ErrorCode.CONFIG_INVALID)
        if not (float(params.min_force) <= self.config.force_n <= float(params.max_force)):
            raise DeviceError("configured GN01 force is outside device params", code=ErrorCode.CONFIG_INVALID)

    def claim_writer(self, owner: str) -> None:
        self._guard.claim(owner)

    def release_writer(self) -> None:
        self.revoke()
        self._guard.release()

    def command_gripper(self, normalized: float, *, epoch: int, now_ns: int) -> None:
        if self._safety_gate is not None:
            self._safety_gate.require_usable()
        permit = self._authority.write_permit(epoch) if self._authority is not None else nullcontext()
        with permit:
            self._command_gripper_locked(normalized, epoch=epoch, now_ns=now_ns)

    def _command_gripper_locked(self, normalized: float, *, epoch: int, now_ns: int) -> None:
        self._guard.assert_writer(caller="command_gripper")
        if self._gripper is None or not self._status.connected:
            raise DeviceError("GN01 sink is not connected", code=ErrorCode.DEVICE_NOT_CONNECTED)
        if not self._status.enabled:
            raise DeviceError("GN01 is not enabled", code=ErrorCode.NOT_READY)
        if (
            self._authorization is None
            or self._authorization_verifier is None
            or not self._authorization_verifier(self._authorization)
            or not self._capability.check(self._authorization, epoch)
        ):
            raise RealHardwareDisabledError(
                "GN01 authorization is missing, expired or revoked",
                code=ErrorCode.REAL_HARDWARE_DISABLED,
            )
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise DeviceError("epoch must be a non-negative integer", code=ErrorCode.VALUE_OUT_OF_RANGE)
        self._tool_readiness.require_motion_ready()
        if self._authority_epoch is None or epoch != self._authority_epoch:
            raise DeviceError(
                "gripper command epoch does not match the authorized epoch",
                code=ErrorCode.EPOCH_MISMATCH,
                details={"epoch": epoch, "authorized_epoch": self._authority_epoch},
            )
        if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns <= 0:
            raise DeviceError("now_ns must be a positive integer", code=ErrorCode.VALUE_OUT_OF_RANGE)
        width = self.config.width_for_normalized(normalized)
        if self._last_width is not None and abs(width - self._last_width) <= self.config.dedup_width_m:
            self._status = replace(self._status, dedup_skips=self._status.dedup_skips + 1)
            return
        try:
            self._gripper.Move(width, self.config.speed_m_s, self.config.force_n)
        except Exception as exc:  # noqa: BLE001
            self.revoke()
            self._status = replace(
                self._status, last_error=f"move_failed:{type(exc).__name__}"
            )
            raise DeviceError(
                "GN01 Move failed",
                code=ErrorCode.DEVICE_FAULT,
                details={"reason": type(exc).__name__, "width_m": width},
            ) from exc
        self._last_width = width
        self._status = replace(self._status, writes=self._status.writes + 1)

    def command_pose(self, pose: Any, *, epoch: int, now_ns: int) -> None:
        raise DeviceError(
            "GN01 does not accept Cartesian poses; use RealRobotCommandSink",
            code=ErrorCode.CONFIG_INVALID,
        )

    def stop(self, timeout_s: float = 2.0) -> bool:
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
            raise DeviceError("stop timeout must be positive", code=ErrorCode.VALUE_OUT_OF_RANGE)
        self.revoke()
        self._status = replace(
            self._status,
            stop_calls=self._status.stop_calls + 1,
            enabled=False,
            physical_stop_confirmed=False,
        )
        if self._gripper is None:
            self._status = replace(
                self._status,
                stop_failures=self._status.stop_failures + 1,
                last_error="stop_not_connected",
            )
            return False
        try:
            self._gripper.Stop()
        except Exception as exc:  # noqa: BLE001
            self._status = replace(
                self._status,
                stop_failures=self._status.stop_failures + 1,
                physical_stop_confirmed=False,
                last_error=f"stop_failed:{type(exc).__name__}",
            )
            return False
        self._status = replace(self._status, physical_stop_confirmed=True)
        return True

    def command_gripper_status(self) -> dict:
        return self._status.as_dict()

    def close(self) -> None:
        self.revoke()
        self._gripper = None
        self._closed = True
        self._status = replace(self._status, connected=False)


class Gn01Gripper:
    """Legacy placeholder; requires the explicit M5 sink."""

    def __init__(self, *_: object, **__: object) -> None:
        assert_real_hardware_allowed("Gn01Gripper")
        raise RealHardwareDisabledError(
            "GN01 control is an M5 deliverable; use Gn01CommandSink with injected SDK",
            code=ErrorCode.REAL_HARDWARE_DISABLED,
        )


__all__ = ["Gn01CommandSink", "Gn01Config", "Gn01Gripper", "Gn01Status"]
