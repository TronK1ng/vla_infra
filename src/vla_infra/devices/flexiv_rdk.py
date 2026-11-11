"""Flexiv RDK read-only state adapter (M2).

Safety contract:

* this module never enables the robot, switches mode, clears faults, sends a
  Cartesian/joint target, or touches the gripper;
* only ``Robot(...)`` construction and ``states()`` reads are used;
* the SDK is imported lazily and only when no injected test SDK is supplied;
* ``FlexivRdkNrtCommandSink`` is the M5 Round 01 NRT writer skeleton.

The RDK order of ``tcp_pose`` is ``[x, y, z, qw, qx, qy, qz]``; the canonical
contract uses ``xyzw``, so the adapter converts and validates explicitly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
import time
from contextlib import nullcontext
from typing import Any, Callable, Dict, Optional

import numpy as np

from ..clock import Clock, MonotonicClock
from ..contracts.constants import TCP_VELOCITY_FRAME, WRENCH_FRAME
from ..contracts.observation import RobotState, RobotStateSample
from ..contracts.pose import CartesianPose
from ..errors import DeviceError, ErrorCode, RealHardwareDisabledError
from .interfaces import SingleWriterGuard
from .m5_auth import MotionCapability
from .tool_state import ToolReadiness
from ..geometry import (
    matrix_to_rotation_6d,
    normalize_quaternion,
    quaternion_to_matrix,
)

_ALLOW_ENV = "VLA_INFRA_ALLOW_REAL_HARDWARE"

_REASON = (
    "Real hardware access requires an on-site operator, the M2 read-only "
    "preflight and explicit VLA_INFRA_ALLOW_REAL_HARDWARE=1. Automated tests "
    "must use fake devices."
)


def real_hardware_enabled() -> bool:
    """True only if an operator explicitly opted in via the environment."""
    return os.environ.get(_ALLOW_ENV, "") == "1"


def assert_real_hardware_allowed(context: str) -> None:
    if not real_hardware_enabled():
        raise RealHardwareDisabledError(
            f"{context}: real hardware access is disabled ({_REASON})",
            code=ErrorCode.REAL_HARDWARE_DISABLED,
            details={"context": context, "env": _ALLOW_ENV},
        )


@dataclass(frozen=True)
class RdkStateMapping:
    """Attribute names on the RDK ``RobotStates`` object (M2 default mapping)."""

    joint_position: str = "q"
    joint_velocity: str = "dq"
    tcp_pose: str = "tcp_pose"  # [x,y,z,qw,qx,qy,qz]
    tcp_velocity: str = "tcp_vel"  # [vx,vy,vz,wx,wy,wz], world/base
    wrench: str = "ext_wrench_in_tcp"  # [fx,fy,fz,mx,my,mz], TCP frame
    timestamp: str = "timestamp"  # [sec, nanosec]
    gripper_width: Optional[str] = None


class FlexivRdkReadOnlyState:
    """Read-only state source implementing :class:`RobotStateSource`."""

    def __init__(
        self,
        serial: str,
        *,
        clock: Optional[Clock] = None,
        sdk: Any = None,
        robot: Any = None,
        network_interface: Optional[str] = None,
        mapping: RdkStateMapping = RdkStateMapping(),
        gripper_width_provider: Optional[Callable[[], float]] = None,
        gripper_width_mapping_key: Optional[str] = None,
    ) -> None:
        if not isinstance(serial, str) or not serial:
            raise DeviceError("robot serial must be non-empty", code=ErrorCode.CONFIG_INVALID)
        if sdk is None:
            assert_real_hardware_allowed("FlexivRdkReadOnlyState")
        self.serial = serial
        self._clock = clock or MonotonicClock()
        self._sdk = sdk
        self._network_interface = network_interface
        self._mapping = mapping
        self._gripper_width_provider = gripper_width_provider
        self._gripper_width_key = gripper_width_mapping_key
        self._robot = robot
        self._sequence = -1
        self._last_host_ns: Optional[int] = None
        self._last_device_ns: Optional[int] = None
        self._started = False
        self._device_info: Dict[str, Any] = {
            "serial": serial,
            "backend": "flexivrdk_readonly",
            "write_access": False,
        }

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        if self._started:
            return
        if self._robot is not None:
            self._started = True
            return
        sdk = self._sdk
        if sdk is None:
            assert_real_hardware_allowed("FlexivRdkReadOnlyState.start")
            try:
                import flexivrdk as sdk  # type: ignore
            except ImportError as exc:
                raise DeviceError(
                    "flexivrdk is not installed; install the RDK Python package on the station",
                    code=ErrorCode.DEVICE_NOT_CONNECTED,
                ) from exc
        try:
            if self._network_interface:
                self._robot = sdk.Robot(self.serial, [self._network_interface])
            else:
                self._robot = sdk.Robot(self.serial)
        except Exception as exc:  # noqa: BLE001 - SDK boundary
            raise DeviceError(
                "failed to connect to the Flexiv robot in read-only mode",
                code=ErrorCode.DEVICE_FAULT,
                details={"serial": self.serial, "reason": type(exc).__name__},
            ) from exc
        self._device_info.update(
            {
                "sdk_version": str(getattr(sdk, "__version__", "unknown")),
                "network_interface": self._network_interface,
                "read_methods": ["states"],
            }
        )
        self._started = True

    def close(self) -> None:
        # RDK has no explicit disconnect for a read-only Robot handle in the
        # documented API; dropping the reference is sufficient and never writes.
        self._robot = None
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def device_info(self) -> Dict[str, Any]:
        return dict(self._device_info)

    # -- state --------------------------------------------------------------
    def read_state(self) -> RobotStateSample:
        if not self._started or self._robot is None:
            raise DeviceError(
                "Flexiv RDK read-only adapter is not started",
                code=ErrorCode.DEVICE_NOT_CONNECTED,
            )
        try:
            states = self._robot.states()
        except Exception as exc:  # noqa: BLE001
            raise DeviceError(
                "failed to read Flexiv robot states",
                code=ErrorCode.DEVICE_FAULT,
                details={"serial": self.serial, "reason": type(exc).__name__},
            ) from exc
        tcp_pose = np.asarray(self._get(states, self._mapping.tcp_pose), dtype=np.float64).reshape(-1)
        if tcp_pose.shape != (7,):
            raise DeviceError(
                "RDK tcp_pose must contain [x,y,z,qw,qx,qy,qz]",
                code=ErrorCode.INVALID_SHAPE,
                details={"shape": list(tcp_pose.shape)},
            )
        position = tcp_pose[:3]
        qw, qx, qy, qz = tcp_pose[3:7]
        quaternion_xyzw = normalize_quaternion([qx, qy, qz, qw])
        rotation_6d = matrix_to_rotation_6d(quaternion_to_matrix(quaternion_xyzw))
        joint_position = np.asarray(self._get(states, self._mapping.joint_position), dtype=np.float64).reshape(-1)
        joint_velocity = np.asarray(self._get(states, self._mapping.joint_velocity), dtype=np.float64).reshape(-1)
        tcp_velocity = np.asarray(self._get(states, self._mapping.tcp_velocity), dtype=np.float64).reshape(-1)
        wrench = np.asarray(self._get(states, self._mapping.wrench), dtype=np.float64).reshape(-1)
        for name, value, expected in (
            ("joint_position", joint_position, 7),
            ("joint_velocity", joint_velocity, 7),
            ("tcp_velocity", tcp_velocity, 6),
            ("wrench", wrench, 6),
        ):
            if value.shape != (expected,):
                raise DeviceError(
                    f"RDK {name} has an unexpected shape",
                    code=ErrorCode.INVALID_SHAPE,
                    details={"field": name, "shape": list(value.shape), "expected": expected},
                )
        width = self._read_gripper_width(states)
        host_ns = self._clock.now_ns()
        if self._last_host_ns is not None and host_ns <= self._last_host_ns:
            host_ns = self._last_host_ns + 1
        device_ns = self._read_device_timestamp_ns(states, host_ns)
        if self._last_device_ns is not None and device_ns <= self._last_device_ns:
            raise DeviceError(
                "RDK device timestamp is not increasing",
                code=ErrorCode.BUFFER_NOT_MONOTONIC,
                details={"device_ns": device_ns, "last": self._last_device_ns},
            )
        self._last_host_ns = host_ns
        self._last_device_ns = device_ns
        self._sequence += 1
        state = RobotState(
            joint_position=joint_position,
            joint_velocity=joint_velocity,
            tcp_position=position,
            tcp_quaternion_xyzw=quaternion_xyzw,
            tcp_rotation_6d=rotation_6d,
            tcp_velocity=tcp_velocity,
            wrench_tcp=wrench,
            gripper_width=np.array([width], dtype=np.float64),
            tcp_velocity_frame=TCP_VELOCITY_FRAME,
            wrench_frame=WRENCH_FRAME,
        )
        return RobotStateSample(
            state=state,
            device_time_ns=device_ns,
            host_time_ns=host_ns,
            sequence_id=self._sequence,
        )

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _get(states: Any, name: str) -> Any:
        if not hasattr(states, name):
            raise DeviceError(
                f"RDK RobotStates does not expose {name!r}; update the mapping after on-site verification",
                code=ErrorCode.DEVICE_FAULT,
                details={"missing_field": name},
            )
        return getattr(states, name)

    def _read_gripper_width(self, states: Any) -> float:
        key = self._gripper_width_key or self._mapping.gripper_width or "gripper_width"
        if key is not None and hasattr(states, key):
            width = float(getattr(states, key))
        elif self._gripper_width_provider is not None:
            width = float(self._gripper_width_provider())
        else:
            raise DeviceError(
                "GN01 width is unavailable: configure a read-only gripper width "
                "provider or an RDK field before collecting real data",
                code=ErrorCode.DEVICE_FAULT,
            )
        if not np.isfinite(width) or width < 0.0 or width > 0.5:
            raise DeviceError(
                "gripper width is outside the plausible range [0, 0.5] m",
                code=ErrorCode.SAFETY_GRIPPER,
                details={"width_m": width},
            )
        return width

    def _read_device_timestamp_ns(self, states: Any, host_ns: int) -> int:
        key = self._mapping.timestamp
        if key and hasattr(states, key):
            stamp = getattr(states, key)
            if isinstance(stamp, (list, tuple, np.ndarray)) and len(stamp) == 2:
                seconds, nanoseconds = int(stamp[0]), int(stamp[1])
                value = seconds * 1_000_000_000 + nanoseconds
                if value > 0:
                    return value
        # Some RDK builds omit RobotStates.timestamp.  Use host time as a
        # clearly-marked fallback rather than failing state capture.
        self._device_info["device_timestamp_fallback"] = "host_monotonic"
        return int(host_ns)


def connect(*_: object, **__: object) -> None:  # pragma: no cover - guard only
    assert_real_hardware_allowed("flexiv_rdk.connect")


__all__ = [
    "FlexivRdkReadOnlyState",
    "FlexivRdkRobot",
    "RdkStateMapping",
    "assert_real_hardware_allowed",
    "real_hardware_enabled",
]


# ---------------------------------------------------------------------------
# M5 NRT writer skeleton (injected SDK only; no real connection in tests)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NrtLimits:
    """Explicit NRT limits; never use the RDK API defaults."""

    max_linear_vel_m_s: float
    max_angular_vel_rad_s: float
    max_linear_acc_m_s2: float
    max_angular_acc_rad_s2: float

    def __post_init__(self) -> None:
        values = (
            self.max_linear_vel_m_s,
            self.max_angular_vel_rad_s,
            self.max_linear_acc_m_s2,
            self.max_angular_acc_rad_s2,
        )
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise DeviceError("NRT limits must be numbers", code=ErrorCode.CONFIG_INVALID)
            if not np.isfinite(float(value)) or float(value) <= 0:
                raise DeviceError(
                    "NRT limits must be positive and finite",
                    code=ErrorCode.CONFIG_INVALID,
                    details={"values": list(values)},
                )
        # These are hard upper bounds for the M5 NRT skeleton.  Site-specific
        # values must be lower and approved; the RDK defaults are not allowed.
        ceilings = (0.10, 0.50, 0.50, 1.00)
        for name, value, ceiling in zip(
            ("linear_vel", "angular_vel", "linear_acc", "angular_acc"), values, ceilings
        ):
            if float(value) > ceiling:
                raise DeviceError(
                    f"NRT {name} exceeds the M5 skeleton ceiling",
                    code=ErrorCode.VALUE_OUT_OF_RANGE,
                    details={"value": float(value), "ceiling": ceiling},
                )


@dataclass(frozen=True)
class RdkCommandStatus:
    connected: bool = False
    enabled: bool = False
    mode_ready: bool = False
    writes: int = 0
    stop_calls: int = 0
    stop_failures: int = 0
    physical_stop_confirmed: bool = False
    last_error: str = ""

    def as_dict(self) -> dict:
        return {
            "connected": self.connected,
            "enabled": self.enabled,
            "mode_ready": self.mode_ready,
            "writes": self.writes,
            "stop_calls": self.stop_calls,
            "stop_failures": self.stop_failures,
            "physical_stop_confirmed": self.physical_stop_confirmed,
            "last_error": self.last_error,
        }


def quaternion_multiply_xyzw(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    x0, y0, z0, w0 = normalize_quaternion(q0)
    x1, y1, z1, w1 = normalize_quaternion(q1)
    return normalize_quaternion(
        [
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        ]
    )


def base_pose_to_rdk_world(
    pose: CartesianPose,
    base_to_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a canonical base-frame pose to RDK world position + wxyz quat."""
    if not isinstance(pose, CartesianPose):
        raise DeviceError("pose must be a CartesianPose", code=ErrorCode.INVALID_SHAPE)
    if pose.frame != "base":
        raise DeviceError(
            "RDK command pose must be declared in the base frame",
            code=ErrorCode.INVALID_FRAME,
            details={"frame": pose.frame},
        )
    transform = np.asarray(base_to_world, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise DeviceError(
            "base_to_world must be a finite 4x4 transform",
            code=ErrorCode.INVALID_SHAPE,
            details={"shape": list(transform.shape)},
        )
    from ..geometry import validate_rotation_matrix

    rotation = validate_rotation_matrix(transform[:3, :3])
    translation = transform[:3, 3]
    world_position = rotation @ pose.position + translation
    world_from_base_quat = _matrix_to_quaternion_xyzw(rotation)
    world_quat_xyzw = quaternion_multiply_xyzw(world_from_base_quat, pose.quaternion_xyzw)
    # Canonicalise the sign for the first/absolute command; shortest-path is
    # applied by the writer against its previous command.
    if world_quat_xyzw[3] < 0.0:
        world_quat_xyzw = -world_quat_xyzw
    world_wxyz = np.array(
        [world_quat_xyzw[3], world_quat_xyzw[0], world_quat_xyzw[1], world_quat_xyzw[2]],
        dtype=np.float64,
    )
    return np.asarray(world_position, dtype=np.float64), world_wxyz


def _matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    from ..geometry import matrix_to_quaternion

    return matrix_to_quaternion(matrix)


class FlexivRdkNrtCommandSink:
    """M5 NRT Cartesian writer boundary (injected SDK / no auto Enable).

    Construction and ``connect()`` never call ``Enable``, ``SwitchMode`` or
    ``ClearFault``.  Those are explicit methods, and every command carries the
    approved NRT limits and an absolute world pose derived through the declared
    base->world transform.
    """

    def __init__(
        self,
        serial: str,
        *,
        limits: NrtLimits,
        base_to_world: np.ndarray,
        sdk: Any = None,
        robot: Any = None,
        network_interface: Optional[str] = None,
        clock: Optional[Clock] = None,
        name: str = "flexiv_rdk_nrt",
        authorization_verifier: Optional[Callable[[Any], bool]] = None,
        capability: Optional[MotionCapability] = None,
        tool_readiness: Optional[ToolReadiness] = None,
        authority: Any = None,
        max_inter_command_gap_ns: int = 100_000_000,
        safety_gate: Any = None,
    ) -> None:
        if not isinstance(serial, str) or not serial:
            raise DeviceError("robot serial must be non-empty", code=ErrorCode.CONFIG_INVALID)
        if not isinstance(limits, NrtLimits):
            raise DeviceError("limits must be explicit NrtLimits", code=ErrorCode.CONFIG_INVALID)
        self.serial = serial
        self.name = name
        self.limits = limits
        self._base_to_world = np.asarray(base_to_world, dtype=np.float64)
        # Validate the transform eagerly and require finite values.
        base_pose_to_rdk_world(
            CartesianPose(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])),
            self._base_to_world,
        )
        self._sdk = sdk
        self._robot = robot
        self._network_interface = network_interface
        self._clock = clock or MonotonicClock()
        self._guard = SingleWriterGuard(name)
        self._status = RdkCommandStatus()
        self._last_wxyz: Optional[np.ndarray] = None
        self._closed = False
        self._authorization_verifier = authorization_verifier
        self._authority_epoch: Optional[int] = None
        self._safety_gate = safety_gate
        self._last_command_ns: Optional[int] = None
        if (
            isinstance(max_inter_command_gap_ns, bool)
            or not isinstance(max_inter_command_gap_ns, int)
            or max_inter_command_gap_ns <= 0
        ):
            raise DeviceError("max_inter_command_gap_ns must be positive", code=ErrorCode.VALUE_OUT_OF_RANGE)
        self._max_inter_command_gap_ns = max_inter_command_gap_ns
        self._authorization: Any = None
        self._capability = capability or MotionCapability()
        self._tool_readiness = tool_readiness or ToolReadiness()
        self._authority = authority

    # -- lifecycle ----------------------------------------------------------
    @property
    def status(self) -> RdkCommandStatus:
        return self._status

    def connect(self) -> None:
        if self._closed:
            raise DeviceError("RDK writer is closed", code=ErrorCode.DEVICE_NOT_CONNECTED)
        if self._robot is not None:
            self._status = replace_status(self._status, connected=True)
            return
        sdk = self._sdk
        if sdk is None:
            assert_real_hardware_allowed("FlexivRdkNrtCommandSink.connect")
            try:
                import flexivrdk as sdk  # type: ignore
            except ImportError as exc:
                raise DeviceError(
                    "flexivrdk is not installed; M5 requires the version-locked station SDK",
                    code=ErrorCode.DEVICE_NOT_CONNECTED,
                ) from exc
        try:
            if self._network_interface:
                self._robot = sdk.Robot(self.serial, [self._network_interface])
            else:
                self._robot = sdk.Robot(self.serial)
        except Exception as exc:  # noqa: BLE001
            raise DeviceError(
                "failed to connect the RDK NRT writer",
                code=ErrorCode.DEVICE_FAULT,
                details={"serial": self.serial, "reason": type(exc).__name__},
            ) from exc
        # Connection only.  No Enable / SwitchMode / ClearFault here.
        self._status = replace_status(self._status, connected=True)
        self._sdk = sdk

    def enable_and_switch_nrt(
        self,
        authorization: Any,
        *,
        authority_epoch: int = 0,
        timeout_s: float = 5.0,
        now_ns: int = 0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Explicit Enable -> operational wait -> SwitchMode -> actual-mode verify.

        Requires a verifier-issued MotionAuthorization.  Connect alone never
        calls this method.  Failures attempt Stop and never ClearFault.
        """
        if self._safety_gate is not None:
            self._safety_gate.require_usable()
        if self._robot is None:
            raise DeviceError("RDK writer is not connected", code=ErrorCode.DEVICE_NOT_CONNECTED)
        if self._authorization_verifier is None or not self._authorization_verifier(authorization):
            raise RealHardwareDisabledError(
                "Enable requires a preflight/ACK-bound MotionAuthorization",
                code=ErrorCode.REAL_HARDWARE_DISABLED,
            )
        self._authorization = authorization
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
            raise DeviceError("enable timeout must be positive", code=ErrorCode.VALUE_OUT_OF_RANGE)
        if isinstance(authority_epoch, bool) or not isinstance(authority_epoch, int) or authority_epoch < 0:
            raise DeviceError("authority_epoch must be non-negative", code=ErrorCode.VALUE_OUT_OF_RANGE)
        try:
            if hasattr(self._robot, "estop_released") and not self._robot.estop_released():
                raise DeviceError("E-stop is not released", code=ErrorCode.SAFETY_STATE)
            if hasattr(self._robot, "fault") and self._robot.fault():
                raise DeviceError("robot is in fault before Enable", code=ErrorCode.SAFETY_STATE)
            if hasattr(self._robot, "recovery") and self._robot.recovery():
                raise DeviceError("robot is in recovery before Enable", code=ErrorCode.SAFETY_STATE)
            self._robot.Enable()
            deadline = time.monotonic() + float(timeout_s)
            while True:
                if hasattr(self._robot, "fault") and self._robot.fault():
                    raise DeviceError("robot fault during operational wait", code=ErrorCode.SAFETY_STATE)
                if self._robot.operational():
                    break
                if time.monotonic() >= deadline:
                    raise DeviceError(
                        "robot did not become operational before timeout",
                        code=ErrorCode.NOT_READY,
                        details={"timeout_s": float(timeout_s)},
                    )
                sleep(0.01)
            mode_cls = getattr(self._sdk, "Mode")
            target_mode = mode_cls.NRT_CARTESIAN_MOTION_FORCE
            self._robot.SwitchMode(target_mode)
            deadline = time.monotonic() + float(timeout_s)
            while True:
                actual_mode = self._robot.mode() if hasattr(self._robot, "mode") else target_mode
                if actual_mode == target_mode:
                    break
                if time.monotonic() >= deadline:
                    raise DeviceError(
                        "robot did not enter NRT_CARTESIAN_MOTION_FORCE",
                        code=ErrorCode.NOT_READY,
                        details={"actual_mode": str(actual_mode)},
                    )
                sleep(0.01)
            self._robot.SetForceControlAxis([False, False, False, False, False, False])
            if not self._robot.operational():
                raise DeviceError("robot is not operational after mode switch", code=ErrorCode.NOT_READY)
        except Exception as exc:  # noqa: BLE001
            stop_ok = self.stop_motion()
            self._status = replace_status(
                self._status,
                enabled=False,
                mode_ready=False,
                last_error=f"enable_failed:{type(exc).__name__}",
            )
            if isinstance(exc, DeviceError):
                raise DeviceError(
                    exc.message,
                    code=exc.code,
                    details={**exc.details, "stop_ok": stop_ok},
                ) from exc
            raise DeviceError(
                "RDK enable/mode lifecycle failed",
                code=ErrorCode.DEVICE_FAULT,
                details={"reason": type(exc).__name__, "stop_ok": stop_ok},
            ) from exc
        self._authority_epoch = int(authority_epoch)
        self._capability.grant(authorization, self._authority_epoch)
        self._status = replace_status(self._status, enabled=True, mode_ready=True, last_error="")

    def claim_writer(self, owner: str) -> None:
        self._guard.claim(owner)

    def release_writer(self) -> None:
        self.revoke()
        self._guard.release()

    def command_pose(self, pose: CartesianPose, *, epoch: int, now_ns: int) -> None:
        if self._safety_gate is not None:
            self._safety_gate.require_usable()
        permit = self._authority.write_permit(epoch) if self._authority is not None else nullcontext()
        with permit:
            self._command_pose_locked(pose, epoch=epoch, now_ns=now_ns)

    def _command_pose_locked(self, pose: CartesianPose, *, epoch: int, now_ns: int) -> None:
        self._guard.assert_writer(caller="command_pose")
        if not self._status.enabled or not self._status.mode_ready or self._robot is None:
            raise DeviceError(
                "RDK NRT writer is not enabled/mode-ready",
                code=ErrorCode.NOT_READY,
            )
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise DeviceError("epoch must be a non-negative integer", code=ErrorCode.VALUE_OUT_OF_RANGE)
        if (
            self._authorization is None
            or self._authorization_verifier is None
            or not self._authorization_verifier(self._authorization)
            or not self._capability.check(self._authorization, epoch)
        ):
            raise RealHardwareDisabledError(
                "arm authorization is missing, expired or revoked",
                code=ErrorCode.REAL_HARDWARE_DISABLED,
            )
        self._tool_readiness.require_motion_ready()
        if self._authority_epoch is None or epoch != self._authority_epoch:
            raise DeviceError(
                "command epoch does not match the authorized epoch",
                code=ErrorCode.EPOCH_MISMATCH,
                details={"epoch": epoch, "authorized_epoch": self._authority_epoch},
            )
        if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns <= 0:
            raise DeviceError("now_ns must be a positive integer", code=ErrorCode.VALUE_OUT_OF_RANGE)
        if self._last_command_ns is not None:
            if now_ns <= self._last_command_ns:
                raise DeviceError("command timestamp is not increasing", code=ErrorCode.BUFFER_NOT_MONOTONIC)
            if now_ns - self._last_command_ns > self._max_inter_command_gap_ns:
                raise DeviceError(
                    "inter-command heartbeat gap exceeded; action freshness is enforced by SafetyGate/executor",
                    code=ErrorCode.STALE_OBSERVATION,
                )
        world_position, current_wxyz = base_pose_to_rdk_world(pose, self._base_to_world)
        if self._last_wxyz is not None and float(np.dot(current_wxyz, self._last_wxyz)) < 0.0:
            current_wxyz = -current_wxyz
        command = np.concatenate([world_position, current_wxyz])
        try:
            self._robot.SendCartesianMotionForce(
                command.tolist(),
                [0.0] * 6,
                [0.0] * 6,
                self.limits.max_linear_vel_m_s,
                self.limits.max_angular_vel_rad_s,
                self.limits.max_linear_acc_m_s2,
                self.limits.max_angular_acc_rad_s2,
            )
        except Exception as exc:  # noqa: BLE001
            self.revoke()
            self._status = replace_status(
                self._status, last_error=f"command_failed:{type(exc).__name__}"
            )
            raise DeviceError(
                "RDK NRT command failed",
                code=ErrorCode.DEVICE_FAULT,
                details={"reason": type(exc).__name__},
            ) from exc
        self._last_wxyz = current_wxyz
        self._last_command_ns = now_ns
        self._status = replace_status(self._status, writes=self._status.writes + 1)

    def command_gripper(self, normalized: float, *, epoch: int, now_ns: int) -> None:
        raise DeviceError(
            "arm sink does not control the gripper; use RealRobotCommandSink/GN01",
            code=ErrorCode.CONFIG_INVALID,
        )

    def revoke(self) -> None:
        self._authorization = None
        self._authority_epoch = None
        self._capability.revoke()
        self._tool_readiness.reset()
        self._status = replace_status(self._status, enabled=False, mode_ready=False)

    def stop_motion(self, timeout_s: float = 2.0) -> bool:
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
            raise DeviceError("stop timeout must be positive", code=ErrorCode.VALUE_OUT_OF_RANGE)
        # Revoke before the possibly-blocking SDK call: if physical stop cannot
        # be confirmed, software authority is still gone.
        self.revoke()
        self._status = replace_status(
            self._status,
            stop_calls=self._status.stop_calls + 1,
            enabled=False,
            mode_ready=False,
            physical_stop_confirmed=False,
        )
        if self._robot is None:
            self._status = replace_status(
                self._status, stop_failures=self._status.stop_failures + 1, last_error="stop_not_connected"
            )
            return False
        try:
            self._robot.Stop()
        except Exception as exc:  # noqa: BLE001
            self._status = replace_status(
                self._status,
                stop_failures=self._status.stop_failures + 1,
                physical_stop_confirmed=False,
                last_error=f"stop_failed:{type(exc).__name__}",
            )
            return False
        self._status = replace_status(self._status, physical_stop_confirmed=True)
        return True

    def close(self) -> None:
        self.revoke()
        self._robot = None
        self._closed = True
        self._status = replace_status(
            self._status, connected=False, enabled=False, mode_ready=False
        )


def replace_status(status: RdkCommandStatus, **changes: Any) -> RdkCommandStatus:
    from dataclasses import replace

    return replace(status, **changes)


# Backward-compatible placeholder name used by M2 imports/tests.
class FlexivRdkRobot(FlexivRdkNrtCommandSink):
    """Compatibility alias; requires explicit limits/transform, never auto-enables."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if "limits" not in kwargs or "base_to_world" not in kwargs:
            raise RealHardwareDisabledError(
                "FlexivRdkRobot now requires explicit NRT limits and base_to_world; "
                "use FlexivRdkNrtCommandSink",
                code=ErrorCode.REAL_HARDWARE_DISABLED,
            )
        super().__init__(*args, **kwargs)


__all__ = [
    "FlexivRdkNrtCommandSink",
    "FlexivRdkReadOnlyState",
    "FlexivRdkRobot",
    "NrtLimits",
    "RdkCommandStatus",
    "base_pose_to_rdk_world",
    "quaternion_multiply_xyzw",
]
