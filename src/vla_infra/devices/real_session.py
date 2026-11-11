"""M5 shared RDK session/factory.

One Robot handle is created here and injected into the arm NRT sink and GN01
sink.  No Enable/mode switch/Tool.Switch happens on connect.
"""

from __future__ import annotations

import threading

from typing import Any, Callable, List, Optional

import numpy as np

from ..clock import Clock, MonotonicClock
from ..errors import DeviceError, ErrorCode
from .flexiv_rdk import (
    FlexivRdkNrtCommandSink,
    FlexivRdkReadOnlyState,
    NrtLimits,
    assert_real_hardware_allowed,
)
from .gn01 import Gn01CommandSink, Gn01Config
from .m5_auth import MotionCapability
from .tool_state import ToolReadiness
from .real_robot import RealRobotCommandSink
from .stop_control import StopOutcome, run_stop_with_deadline, stop_pair_bounded
from .session_gate import SessionSafetyGate


def _lifecycle(method):
    def wrapper(self, *args, **kwargs):
        with self._lifecycle_lock:
            return method(self, *args, **kwargs)
    return wrapper


class FlexivRdkSession:
    def __init__(
        self,
        serial: str,
        *,
        sdk: Any = None,
        network_interface: Optional[str] = None,
        clock: Optional[Clock] = None,
    ) -> None:
        if not serial:
            raise DeviceError("robot serial must be non-empty", code=ErrorCode.CONFIG_INVALID)
        self.serial = serial
        self._sdk = sdk
        self._network_interface = network_interface
        self._clock = clock or MonotonicClock()
        self._robot: Any = None
        self._owned: List[Any] = []
        self._pending_owned: List[Any] = []
        self._closing = False
        self._lifecycle_lock = threading.RLock()
        self._runtime_owner_token: Any = None
        self._claim_lock = threading.Lock()
        self._quarantined = False
        self.safety_gate = SessionSafetyGate()

    @property
    def connected(self) -> bool:
        return self._robot is not None

    def claim_runtime(self) -> object:
        with self._claim_lock:
            if self._quarantined:
                raise DeviceError("session is quarantined after failed cleanup", code=ErrorCode.STOP_FAILED)
            if self._runtime_owner_token is not None:
                raise DeviceError(
                    "this RDK session already owns a production runtime",
                    code=ErrorCode.ILLEGAL_TRANSITION,
                )
            token = object()
            self._runtime_owner_token = token
            return token

    def release_runtime(self, token: object) -> None:
        with self._claim_lock:
            if token is None or token is not self._runtime_owner_token:
                raise DeviceError(
                    "runtime owner token mismatch",
                    code=ErrorCode.AUTH_FAILED,
                )
            self._runtime_owner_token = None

    @property
    def quarantined(self) -> bool:
        return self._quarantined

    @property
    def quarantine_reason(self) -> str:
        return getattr(self, "_quarantine_reason", "")

    @property
    def has_pending_resources(self) -> bool:
        return bool(self._pending_owned)

    def assert_not_quarantined(self) -> None:
        if self._quarantined:
            raise DeviceError(
                f"session is quarantined: {self.quarantine_reason}",
                code=ErrorCode.STOP_FAILED,
            )

    @_lifecycle
    def quarantine(self, reason: str) -> None:
        self._quarantine_reason = reason
        with self._claim_lock:
            self._quarantined = True
        self.safety_gate.quarantine(reason)
        for resource in list(self._owned) + list(self._pending_owned):
            try:
                if hasattr(resource, "revoke"):
                    resource.revoke()
            except Exception:
                pass
        authority = getattr(self, "_authority", None)
        if authority is not None:
            try:
                authority.revoke(f"session_quarantine:{reason}")
            except Exception:
                pass

    def _assert_owner(self, token: object) -> None:
        self.safety_gate.require_usable()
        with self._claim_lock:
            owner = self._runtime_owner_token
            if owner is None:
                raise DeviceError(
                    "writer factory requires an active runtime claim",
                    code=ErrorCode.AUTH_FAILED,
                )
            if token is None or token is not owner:
                raise DeviceError("writer factory requires the exact runtime owner token", code=ErrorCode.AUTH_FAILED)

    def _stop_pending_bounded(self, timeout_s: float) -> list:
        if len(self._pending_owned) >= 2:
            pair = stop_pair_bounded(
                lambda: self._pending_owned[0].stop_motion(timeout_s),
                lambda: self._pending_owned[1].stop(timeout_s),
                timeout_s,
            )
            return list(pair.values())
        results = []
        for resource in self._pending_owned:
            if hasattr(resource, "stop_motion"):
                results.append(run_stop_with_deadline(lambda r=resource: r.stop_motion(timeout_s), timeout_s))
            elif hasattr(resource, "stop"):
                results.append(run_stop_with_deadline(lambda r=resource: r.stop(timeout_s), timeout_s))
            else:
                results.append(StopOutcome(True, False, False))
        return results

    def _stop_owned_bounded(self, timeout_s: float) -> list:
        results = []
        for resource in list(self._owned):
            if hasattr(resource, "revoke"):
                resource.revoke()
            if hasattr(resource, "_arm") and hasattr(resource, "_gripper"):
                pair = stop_pair_bounded(
                    lambda r=resource: r._arm.stop_motion(timeout_s),
                    lambda r=resource: r._gripper.stop(timeout_s),
                    timeout_s,
                )
                results.extend(pair.values())
            elif hasattr(resource, "stop_motion"):
                results.append(run_stop_with_deadline(lambda r=resource: r.stop_motion(timeout_s), timeout_s))
            elif hasattr(resource, "stop"):
                results.append(run_stop_with_deadline(lambda r=resource: r.stop(timeout_s), timeout_s))
            else:
                results.append(StopOutcome(True, False, False))
        return results

    @_lifecycle
    def abort_owned_resources(self, timeout_s: float = 2.0) -> None:
        results = self._stop_owned_bounded(timeout_s)
        pending = self._stop_pending_bounded(timeout_s)
        results.extend(pending)
        if not all(outcome.confirmed for outcome in results):
            self.quarantine("abort_stop_failed")
            raise DeviceError(
                "compose abort could not confirm physical stop",
                code=ErrorCode.STOP_FAILED,
                details={"stop_results": [o.as_dict() for o in results]},
            )
        for resource in list(self._owned) + list(self._pending_owned):
            if hasattr(resource, "close"):
                resource.close()
        self._owned.clear()
        self._pending_owned.clear()

    @property
    def state(self) -> FlexivRdkReadOnlyState:
        """Read-only state facade sharing the session handle."""
        if self._robot is None:
            raise DeviceError("session is not connected", code=ErrorCode.DEVICE_NOT_CONNECTED)
        return FlexivRdkReadOnlyState(
            self.serial,
            sdk=self._sdk,
            robot=self._robot,
            clock=self._clock,
        )

    @_lifecycle
    def connect(self) -> None:
        self.assert_not_quarantined()
        if self._robot is not None:
            return None
        sdk = self._sdk
        if sdk is None:
            assert_real_hardware_allowed("FlexivRdkSession.connect")
            try:
                import flexivrdk as sdk  # type: ignore
            except ImportError as exc:
                raise DeviceError(
                    "flexivrdk is not installed; M5 requires the pinned station SDK",
                    code=ErrorCode.DEVICE_NOT_CONNECTED,
                ) from exc
        try:
            if self._network_interface:
                self._robot = sdk.Robot(self.serial, [self._network_interface])
            else:
                self._robot = sdk.Robot(self.serial)
        except Exception as exc:  # noqa: BLE001
            raise DeviceError(
                "failed to create the shared RDK Robot session",
                code=ErrorCode.DEVICE_FAULT,
                details={"serial": self.serial, "reason": type(exc).__name__},
            ) from exc
        self._sdk = sdk
        return None

    def create_arm(self, *args: Any, **kwargs: Any) -> FlexivRdkNrtCommandSink:
        raise DeviceError(
            "standalone arm factory is disabled; use token-gated create_composite",
            code=ErrorCode.ILLEGAL_TRANSITION,
        )

    def _create_arm_locked(
        self,
        *,
        limits: NrtLimits,
        base_to_world: np.ndarray,
        authorization_verifier: Callable[[Any], bool],
        capability: Optional[MotionCapability] = None,
        tool_readiness: Optional[ToolReadiness] = None,
        authority: Any = None,
        name: str = "flexiv_rdk_nrt",
        max_inter_command_gap_ns: int = 100_000_000,
        _track: bool = True,
        owner_token: object = None,
    ) -> FlexivRdkNrtCommandSink:
        self._assert_owner(owner_token)
        if self._robot is None:
            raise DeviceError("session is not connected", code=ErrorCode.DEVICE_NOT_CONNECTED)
        sink = FlexivRdkNrtCommandSink(
            self.serial,
            limits=limits,
            base_to_world=base_to_world,
            sdk=self._sdk,
            robot=self._robot,
            clock=self._clock,
            name=name,
            authorization_verifier=authorization_verifier,
            capability=capability,
            tool_readiness=tool_readiness,
            authority=authority,
            max_inter_command_gap_ns=max_inter_command_gap_ns,
            safety_gate=self.safety_gate,
        )
        if _track:
            self._owned.append(sink)
        return sink

    def create_gn01(self, *args: Any, **kwargs: Any) -> Gn01CommandSink:
        raise DeviceError(
            "standalone GN01 factory is disabled; use token-gated create_composite",
            code=ErrorCode.ILLEGAL_TRANSITION,
        )

    def _create_gn01_locked(
        self,
        *,
        name: str,
        config: Gn01Config,
        authorization_verifier: Callable[[Any], bool],
        capability: Optional[MotionCapability] = None,
        tool_readiness: Optional[ToolReadiness] = None,
        authority: Any = None,
        _track: bool = True,
        owner_token: object = None,
    ) -> Gn01CommandSink:
        self._assert_owner(owner_token)
        if self._robot is None:
            raise DeviceError("session is not connected", code=ErrorCode.DEVICE_NOT_CONNECTED)
        sink = Gn01CommandSink(
            name,
            config,
            sdk=self._sdk,
            robot=self._robot,
            clock=self._clock,
            authorization_verifier=authorization_verifier,
            capability=capability,
            tool_readiness=tool_readiness,
            authority=authority,
            safety_gate=self.safety_gate,
        )
        if _track:
            self._owned.append(sink)
        return sink

    @_lifecycle
    def create_composite(
        self,
        *,
        limits: NrtLimits,
        base_to_world: np.ndarray,
        gn01_name: str,
        gn01_config: Gn01Config,
        authorization_verifier: Callable[[Any], bool],
        authority: Any = None,
        tool_readiness: Optional[ToolReadiness] = None,
        cleanup_timeout_s: float = 2.0,
        owner_token: object = None,
    ) -> RealRobotCommandSink:
        self._assert_owner(owner_token)
        capability = MotionCapability()
        tool_readiness = tool_readiness or ToolReadiness()
        if authority is not None:
            self._authority = authority
        arm = None
        gripper = None
        try:
            arm = self._create_arm_locked(
                limits=limits,
                base_to_world=base_to_world,
                authorization_verifier=authorization_verifier,
                capability=capability,
                tool_readiness=tool_readiness,
                authority=authority,
                _track=False,
                owner_token=owner_token,
            )
            self._pending_owned.append(arm)
            gripper = self._create_gn01_locked(
                name=gn01_name,
                config=gn01_config,
                authorization_verifier=authorization_verifier,
                capability=capability,
                tool_readiness=tool_readiness,
                authority=authority,
                _track=False,
                owner_token=owner_token,
            )
            self._pending_owned.append(gripper)
            composite = RealRobotCommandSink(arm, gripper)
            self._owned.append(composite)
            self._pending_owned.clear()
            return composite
        except BaseException as original:
            for resource in list(self._pending_owned):
                try:
                    resource.revoke()
                except Exception:
                    pass
            outcomes = self._stop_pending_bounded(cleanup_timeout_s)
            if not all(outcome.confirmed for outcome in outcomes):
                self.quarantine("partial_construction_stop_failed")
                raise DeviceError(
                    "partial composite construction cleanup was not confirmed",
                    code=ErrorCode.STOP_FAILED,
                    details={
                        "original": type(original).__name__,
                        "stop": [o.as_dict() for o in outcomes],
                        "pending": [type(r).__name__ for r in self._pending_owned],
                    },
                ) from original
            for resource in list(self._pending_owned):
                try:
                    resource.close()
                except Exception as exc:  # noqa: BLE001
                    self.quarantine("partial_construction_close_failed")
                    raise DeviceError(
                        "partial construction close failed",
                        code=ErrorCode.STOP_FAILED,
                        details={"reason": type(exc).__name__},
                    ) from original
            self._pending_owned.clear()
            raise

    @_lifecycle
    def close(self, timeout_s: float = 2.0) -> None:
        if self._closing:
            raise DeviceError("session close is already in progress", code=ErrorCode.ILLEGAL_TRANSITION)
        if self._robot is None and not self._owned:
            return
        self._closing = True
        try:
            stopped = self._stop_owned_bounded(timeout_s)
            if not all(outcome.confirmed for outcome in stopped):
                self.quarantine("close_stop_failed")
                raise DeviceError(
                    "session close could not confirm physical stop for every sink",
                    code=ErrorCode.STOP_FAILED,
                    details={"stop_results": [o.as_dict() for o in stopped]},
                )
            for resource in self._owned:
                if hasattr(resource, "close"):
                    resource.close()
            self._owned.clear()
            self._robot = None
        finally:
            self._closing = False


__all__ = ["FlexivRdkSession"]
