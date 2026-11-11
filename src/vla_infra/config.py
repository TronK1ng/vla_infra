"""Strict YAML configuration loading.

Fail-closed rules:

* missing files, missing keys, unknown keys and bad types all abort startup;
* ``backend: real`` is refused in M1 (the real adapter does not exist yet);
* safety thresholds have no implicit defaults — a station must state them;
* example configs contain placeholders only; secrets are never stored, only
  the *name* of an environment variable holding a token.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml

from .contracts.policy_io import CANONICAL_STATE_LAYOUT, JOINT_STATE_LAYOUT, StateLayout
from .errors import ConfigError, ErrorCode, RealHardwareDisabledError, UnsupportedBackendError
from .execution.safety import SafetyProfile
from .sync.synchronizer import SyncConfig

STATION_SCHEMA_VERSION = "station/v1"
SAFETY_SCHEMA_VERSION = "safety/v1"
POLICY_SCHEMA_VERSION = "policy/v1"

_LAYOUTS = {
    CANONICAL_STATE_LAYOUT.name: CANONICAL_STATE_LAYOUT,
    JOINT_STATE_LAYOUT.name: JOINT_STATE_LAYOUT,
}


def _fail(message: str, **details: Any) -> "ConfigError":
    return ConfigError(message, code=ErrorCode.CONFIG_INVALID, details=details)


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise _fail("configuration file not found", path=str(path)) from exc
    except yaml.YAMLError as exc:
        raise _fail("configuration file is not valid YAML", path=str(path), error=str(exc)) from exc
    if not isinstance(raw, Mapping):
        raise _fail("configuration root must be a mapping", path=str(path))
    return dict(raw)


def _expect_keys(mapping: Mapping[str, Any], *, required: Tuple[str, ...], optional: Tuple[str, ...], context: str) -> None:
    missing = [key for key in required if key not in mapping]
    if missing:
        raise _fail("missing required configuration keys", context=context, missing=missing)
    unknown = [key for key in mapping if key not in required and key not in optional]
    if unknown:
        raise _fail("unknown configuration keys (typos are dangerous)", context=context, unknown=unknown)


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail("expected a mapping", context=context, type=type(value).__name__)
    return value


def _require_str(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail("expected a non-empty string", context=context)
    return value


def _require_positive_float(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not float(value) > 0:
        raise _fail("expected a positive number", context=context, value=value)
    return float(value)


def _require_positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _fail("expected a positive integer", context=context, value=value)
    return int(value)


def _ms_to_ns(value: Any, context: str) -> int:
    return int(round(_require_positive_float(value, context) * 1_000_000))


@dataclass(frozen=True)
class CameraConfig:
    name: str
    serial: str
    width: int
    height: int
    fps: float
    depth: bool = False
    device_latency_ms: float = 2.0


@dataclass(frozen=True)
class RobotConfig:
    serial: str
    state_rate_hz: float
    gripper: str


@dataclass(frozen=True)
class RecordingConfig:
    root: str
    min_free_gb: float
    finalize_timeout_s: float = 300.0


@dataclass(frozen=True)
class StationConfig:
    station_id: str
    backend: str
    cameras: Mapping[str, CameraConfig]
    robot: RobotConfig
    sync: SyncConfig
    recording: RecordingConfig
    calibration_version: str = "uncalibrated"

    def camera_names(self) -> Tuple[str, ...]:
        return tuple(sorted(self.cameras))


@dataclass(frozen=True)
class PolicyServerConfig:
    host: str = "127.0.0.1"
    port: int = 0
    adapter: str = "zero"
    allow_non_loopback: bool = False


@dataclass(frozen=True)
class PolicyConfig:
    endpoint: str
    timeout_s: float
    connect_timeout_s: float
    image_size: int
    state_layout: StateLayout
    expected_horizon: int
    expected_action_dt_s: float
    expected_action_frame: str
    expected_action_type: str
    expected_delta_semantics: str
    server: PolicyServerConfig
    auth_token_env: Optional[str] = None
    proxy: str = "none"

    def expected_metadata(self):
        from .contracts.constants import ACTION_DIM
        from .policy.adapters.base import PolicyMetadata

        return PolicyMetadata(
            name="local-expected",
            version="config",
            horizon=self.expected_horizon,
            action_dt_s=self.expected_action_dt_s,
            action_frame=self.expected_action_frame,
            action_type=self.expected_action_type,
            delta_semantics=self.expected_delta_semantics,
            state_layout=self.state_layout,
            image_size=self.image_size,
            action_dim=ACTION_DIM,
        )


@dataclass(frozen=True)
class AppConfig:
    station: StationConfig
    safety: SafetyProfile
    policy: PolicyConfig
    station_path: str
    safety_path: str
    policy_path: str


def load_station(path: str | Path) -> StationConfig:
    path = Path(path)
    raw = _load_yaml(path)
    _expect_keys(
        raw,
        required=("schema_version", "station_id", "backend", "cameras", "robot", "sync", "recording"),
        optional=("calibration_version", "on_site_readonly_ack"),
        context="station",
    )
    if raw["schema_version"] != STATION_SCHEMA_VERSION:
        raise _fail(
            "unsupported station schema",
            expected=STATION_SCHEMA_VERSION,
            actual=raw["schema_version"],
        )
    backend = _require_str(raw["backend"], "station.backend")
    if backend not in ("fake", "real_readonly", "real"):
        raise _fail(
            "station.backend must be 'fake', 'real_readonly' or 'real'",
            actual=backend,
        )
    if backend == "real":
        raise UnsupportedBackendError(
            "backend: real (writer) is an M5 deliverable; M2 only supports read-only "
            "collection via backend: real_readonly",
            details={"path": str(path)},
        )
    if backend == "real_readonly":
        if not bool(raw.get("on_site_readonly_ack", False)):
            raise RealHardwareDisabledError(
                "backend: real_readonly requires the explicit on-site acknowledgement "
                "'on_site_readonly_ack: true'",
                details={"path": str(path)},
            )
        if os.environ.get("VLA_INFRA_ALLOW_REAL_HARDWARE", "") != "1":
            raise RealHardwareDisabledError(
                "backend: real_readonly requires VLA_INFRA_ALLOW_REAL_HARDWARE=1; "
                "automated tests must never select it",
                details={"path": str(path)},
            )
    cameras_raw = _require_mapping(raw["cameras"], "station.cameras")
    if set(cameras_raw) != {"base", "wrist"}:
        raise _fail(
            "station must declare exactly base and wrist cameras (serial-bound)",
            actual=sorted(cameras_raw),
        )
    cameras: Dict[str, CameraConfig] = {}
    for name, camera_raw in cameras_raw.items():
        camera_raw = _require_mapping(camera_raw, f"station.cameras.{name}")
        _expect_keys(
            camera_raw,
            required=("serial", "width", "height", "fps"),
            optional=("depth", "device_latency_ms"),
            context=f"station.cameras.{name}",
        )
        cameras[name] = CameraConfig(
            name=name,
            serial=_require_str(camera_raw["serial"], f"camera.{name}.serial"),
            width=_require_positive_int(camera_raw["width"], f"camera.{name}.width"),
            height=_require_positive_int(camera_raw["height"], f"camera.{name}.height"),
            fps=_require_positive_float(camera_raw["fps"], f"camera.{name}.fps"),
            depth=bool(camera_raw.get("depth", False)),
            device_latency_ms=_require_positive_float(
                camera_raw.get("device_latency_ms", 2.0), f"camera.{name}.device_latency_ms"
            ),
        )
    serials = [camera.serial for camera in cameras.values()]
    if len(set(serials)) != len(serials):
        raise _fail("camera serials must be unique (no binding by enumeration order)")

    robot_raw = _require_mapping(raw["robot"], "station.robot")
    _expect_keys(
        robot_raw,
        required=("serial", "state_rate_hz", "gripper"),
        optional=(),
        context="station.robot",
    )
    robot = RobotConfig(
        serial=_require_str(robot_raw["serial"], "robot.serial"),
        state_rate_hz=_require_positive_float(robot_raw["state_rate_hz"], "robot.state_rate_hz"),
        gripper=_require_str(robot_raw["gripper"], "robot.gripper"),
    )
    if backend == "real_readonly":
        placeholders = [serial for serial in serials if "REPLACE_ME" in serial.upper()]
        if "REPLACE_ME" in robot.serial.upper():
            placeholders.append(robot.serial)
        if placeholders:
            raise _fail(
                "backend: real_readonly requires real camera/robot serials, not placeholders",
                placeholders=placeholders,
            )

    sync_raw = _require_mapping(raw["sync"], "station.sync")
    _expect_keys(
        sync_raw,
        required=("max_sync_error_ms", "max_state_gap_ms", "max_camera_skew_ms"),
        optional=("wrench_policy", "gripper_policy"),
        context="station.sync",
    )
    sync = SyncConfig(
        max_sync_error_ns=_ms_to_ns(sync_raw["max_sync_error_ms"], "sync.max_sync_error_ms"),
        max_state_gap_ns=_ms_to_ns(sync_raw["max_state_gap_ms"], "sync.max_state_gap_ms"),
        max_camera_skew_ns=_ms_to_ns(sync_raw["max_camera_skew_ms"], "sync.max_camera_skew_ms"),
        wrench_policy=str(sync_raw.get("wrench_policy", "zoh")),
        gripper_policy=str(sync_raw.get("gripper_policy", "zoh")),
    )
    try:
        sync.validate()
    except ValueError as exc:
        raise _fail(f"invalid sync configuration: {exc}") from exc

    recording_raw = _require_mapping(raw["recording"], "station.recording")
    _expect_keys(
        recording_raw,
        required=("root", "min_free_gb"),
        optional=("finalize_timeout_s",),
        context="station.recording",
    )
    recording = RecordingConfig(
        root=_require_str(recording_raw["root"], "recording.root"),
        min_free_gb=_require_positive_float(recording_raw["min_free_gb"], "recording.min_free_gb"),
        finalize_timeout_s=_require_positive_float(
            recording_raw.get("finalize_timeout_s", 300.0),
            "recording.finalize_timeout_s",
        ),
    )
    return StationConfig(
        station_id=_require_str(raw["station_id"], "station.station_id"),
        backend=backend,
        cameras=cameras,
        robot=robot,
        sync=sync,
        recording=recording,
        calibration_version=_require_str(
            raw.get("calibration_version", "uncalibrated"),
            "station.calibration_version",
        ),
    )


def load_safety(path: str | Path) -> SafetyProfile:
    path = Path(path)
    raw = _load_yaml(path)
    _expect_keys(
        raw,
        required=(
            "schema_version",
            "workspace",
            "tcp_height",
            "motion",
            "staleness",
            "force",
            "gripper",
            "action",
        ),
        optional=("joint_limits",),
        context="safety",
    )
    if raw["schema_version"] != SAFETY_SCHEMA_VERSION:
        raise _fail("unsupported safety schema", expected=SAFETY_SCHEMA_VERSION, actual=raw["schema_version"])

    def section(name: str, required: Tuple[str, ...], optional: Tuple[str, ...] = ()) -> Mapping[str, Any]:
        value = _require_mapping(raw[name], f"safety.{name}")
        _expect_keys(value, required=required, optional=optional, context=f"safety.{name}")
        return value

    workspace = section("workspace", ("min", "max"))
    lo = tuple(float(v) for v in workspace["min"])
    hi = tuple(float(v) for v in workspace["max"])
    if len(lo) != 3 or len(hi) != 3:
        raise _fail("safety.workspace.min/max must have three components")
    height = section("tcp_height", ("min_m", "max_m"))
    motion = section(
        "motion",
        (
            "max_step_translation_m",
            "max_step_rotation_rad",
            "max_linear_velocity_m_s",
            "max_angular_velocity_rad_s",
            "max_first_target_translation_m",
            "max_first_target_rotation_rad",
        ),
        optional=(
            "min_segment_dt_ms",
            "max_linear_acceleration_m_s2",
            "max_angular_acceleration_rad_s2",
        ),
    )
    staleness = section("staleness", ("max_observation_age_ms", "max_chunk_age_ms"))
    force = section("force", ("max_force_n", "max_torque_nm"))
    gripper = section("gripper", ("width_min_m", "width_max_m", "max_speed_m_s"))
    action = section("action", ("allowed_frames", "allowed_types"))
    frames = tuple(str(v) for v in action["allowed_frames"])
    types = tuple(str(v) for v in action["allowed_types"])

    joint_min: Optional[Tuple[float, ...]] = None
    joint_max: Optional[Tuple[float, ...]] = None
    joint_velocity_max: Optional[Tuple[float, ...]] = None
    if "joint_limits" in raw:
        limits = section("joint_limits", ("min_rad", "max_rad"), optional=("max_velocity_rad_s",))
        joint_min = tuple(float(v) for v in limits["min_rad"])
        joint_max = tuple(float(v) for v in limits["max_rad"])
        if "max_velocity_rad_s" in limits:
            joint_velocity_max = tuple(float(v) for v in limits["max_velocity_rad_s"])

    try:
        return SafetyProfile(
            workspace_min=(lo[0], lo[1], lo[2]),
            workspace_max=(hi[0], hi[1], hi[2]),
            tcp_height_min_m=float(height["min_m"]),
            tcp_height_max_m=float(height["max_m"]),
            max_step_translation_m=float(motion["max_step_translation_m"]),
            max_step_rotation_rad=float(motion["max_step_rotation_rad"]),
            max_linear_velocity_m_s=float(motion["max_linear_velocity_m_s"]),
            max_angular_velocity_rad_s=float(motion["max_angular_velocity_rad_s"]),
            max_first_target_translation_m=float(motion["max_first_target_translation_m"]),
            max_first_target_rotation_rad=float(motion["max_first_target_rotation_rad"]),
            max_observation_age_ns=_ms_to_ns(staleness["max_observation_age_ms"], "staleness.max_observation_age_ms"),
            max_chunk_age_ns=_ms_to_ns(staleness["max_chunk_age_ms"], "staleness.max_chunk_age_ms"),
            max_force_n=float(force["max_force_n"]),
            max_torque_nm=float(force["max_torque_nm"]),
            gripper_width_min_m=float(gripper["width_min_m"]),
            gripper_width_max_m=float(gripper["width_max_m"]),
            gripper_max_speed_m_s=float(gripper["max_speed_m_s"]),
            allowed_action_frames=frames,
            allowed_action_types=types,
            joint_position_min_rad=joint_min,
            joint_position_max_rad=joint_max,
            min_segment_dt_ns=_ms_to_ns(motion.get("min_segment_dt_ms", 1.0), "motion.min_segment_dt_ms"),
            max_linear_acceleration_m_s2=(
                None
                if "max_linear_acceleration_m_s2" not in motion
                else float(motion["max_linear_acceleration_m_s2"])
            ),
            max_angular_acceleration_rad_s2=(
                None
                if "max_angular_acceleration_rad_s2" not in motion
                else float(motion["max_angular_acceleration_rad_s2"])
            ),
            joint_velocity_max_rad_s=joint_velocity_max,
        )
    except Exception as exc:
        if isinstance(exc, ConfigError):
            raise
        raise _fail(f"invalid safety configuration: {exc}") from exc


def _require_proxy(value: Any) -> str:
    proxy = _require_str(value, "policy.proxy")
    lowered = proxy.lower()
    if lowered in ("none", "off", "false", "direct", "env", "auto", "true"):
        return proxy
    if "://" in proxy:
        return proxy
    raise _fail(
        "policy.proxy must be 'none', 'env' or an explicit proxy URL",
        value=proxy,
    )


def load_policy(path: str | Path) -> PolicyConfig:
    path = Path(path)
    raw = _load_yaml(path)
    _expect_keys(
        raw,
        required=(
            "schema_version",
            "endpoint",
            "timeout_s",
            "connect_timeout_s",
            "image_size",
            "state_layout",
            "expected_horizon",
            "expected_action_dt_s",
            "expected_action_frame",
            "expected_action_type",
            "server",
        ),
        optional=("auth_token_env", "proxy", "expected_delta_semantics"),
        context="policy",
    )
    if raw["schema_version"] != POLICY_SCHEMA_VERSION:
        raise _fail("unsupported policy schema", expected=POLICY_SCHEMA_VERSION, actual=raw["schema_version"])
    endpoint = _require_str(raw["endpoint"], "policy.endpoint")
    if not endpoint.startswith(("ws://", "wss://")):
        raise _fail("policy.endpoint must use ws:// or wss://", endpoint=endpoint)
    layout_name = _require_str(raw["state_layout"], "policy.state_layout")
    if layout_name not in _LAYOUTS:
        raise _fail(
            "unknown policy.state_layout",
            actual=layout_name,
            supported=sorted(_LAYOUTS),
        )
    server_raw = _require_mapping(raw["server"], "policy.server")
    _expect_keys(
        server_raw,
        required=("host", "port", "adapter"),
        optional=("allow_non_loopback",),
        context="policy.server",
    )
    server = PolicyServerConfig(
        host=_require_str(server_raw["host"], "policy.server.host"),
        port=int(server_raw["port"]) if not isinstance(server_raw["port"], bool) else 0,
        adapter=_require_str(server_raw["adapter"], "policy.server.adapter"),
        allow_non_loopback=bool(server_raw.get("allow_non_loopback", False)),
    )
    if server.port < 0 or server.port > 65535:
        raise _fail("policy.server.port must be in [0, 65535]", port=server.port)
    if server.adapter not in ("zero", "fixed", "sine"):
        raise _fail("policy.server.adapter must be one of zero/fixed/sine", actual=server.adapter)
    frame = _require_str(raw["expected_action_frame"], "policy.expected_action_frame")
    action_type = _require_str(raw["expected_action_type"], "policy.expected_action_type")
    if frame not in ("base", "tcp"):
        raise _fail("policy.expected_action_frame must be base or tcp", actual=frame)
    if action_type not in ("delta_tcp", "absolute_tcp"):
        raise _fail("policy.expected_action_type is invalid", actual=action_type)
    delta_semantics = _require_str(
        raw.get("expected_delta_semantics", "absolute" if action_type == "absolute_tcp" else "cumulative"),
        "policy.expected_delta_semantics",
    )
    if delta_semantics not in ("cumulative", "absolute"):
        raise _fail(
            "policy.expected_delta_semantics must be cumulative or absolute",
            actual=delta_semantics,
        )
    expected_semantics = "absolute" if action_type == "absolute_tcp" else "cumulative"
    if delta_semantics != expected_semantics:
        raise _fail(
            "policy.expected_delta_semantics does not match expected_action_type",
            action_type=action_type,
            delta_semantics=delta_semantics,
            expected=expected_semantics,
        )
    return PolicyConfig(
        endpoint=endpoint,
        timeout_s=_require_positive_float(raw["timeout_s"], "policy.timeout_s"),
        connect_timeout_s=_require_positive_float(raw["connect_timeout_s"], "policy.connect_timeout_s"),
        image_size=_require_positive_int(raw["image_size"], "policy.image_size"),
        state_layout=_LAYOUTS[layout_name],
        expected_horizon=_require_positive_int(raw["expected_horizon"], "policy.expected_horizon"),
        expected_action_dt_s=_require_positive_float(raw["expected_action_dt_s"], "policy.expected_action_dt_s"),
        expected_action_frame=frame,
        expected_action_type=action_type,
        expected_delta_semantics=delta_semantics,
        server=server,
        auth_token_env=(
            _require_str(raw["auth_token_env"], "policy.auth_token_env")
            if raw.get("auth_token_env") is not None
            else None
        ),
        proxy=_require_proxy(raw.get("proxy", "none")),
    )


def load_app_config(
    *,
    station_path: str | Path,
    safety_path: str | Path,
    policy_path: str | Path,
) -> AppConfig:
    return AppConfig(
        station=load_station(station_path),
        safety=load_safety(safety_path),
        policy=load_policy(policy_path),
        station_path=str(station_path),
        safety_path=str(safety_path),
        policy_path=str(policy_path),
    )


__all__ = [
    "AppConfig",
    "CameraConfig",
    "PolicyConfig",
    "PolicyServerConfig",
    "RecordingConfig",
    "RobotConfig",
    "StationConfig",
    "load_app_config",
    "load_policy",
    "load_safety",
    "load_station",
]
