"""Strict M5 real-station configuration (Round 01 skeleton).

This loader is deliberately separate from the fake/real-readonly M1 loader.
It does not connect, enable or command any device.  ``allow_real_motion`` is
parsed but still requires the runtime gates/operator ACK; the template ships
with ``false``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple

import numpy as np
import yaml

from .devices.flexiv_rdk import NrtLimits
from .devices.gn01 import Gn01Config
from .errors import ConfigError, ErrorCode, RealHardwareDisabledError
from .execution.safety import SafetyProfile

M5_STATION_SCHEMA = "m5-station/v1"
M5_SAFETY_SCHEMA = "m5-safety/v1"
_ALLOW_ENV = "VLA_INFRA_ALLOW_REAL_HARDWARE"


def _fail(message: str, **details: Any) -> ConfigError:
    return ConfigError(message, code=ErrorCode.CONFIG_INVALID, details=details)


def _load(path: str | Path) -> dict:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise _fail("M5 config file not found", path=str(path)) from exc
    if not isinstance(raw, Mapping):
        raise _fail("M5 config root must be a mapping", path=str(path))
    return dict(raw)


def _keys(mapping: Mapping[str, Any], required: Tuple[str, ...], optional: Tuple[str, ...], context: str) -> None:
    missing = [key for key in required if key not in mapping]
    if missing:
        raise _fail("missing M5 config keys", context=context, missing=missing)
    unknown = [key for key in mapping if key not in required and key not in optional]
    if unknown:
        raise _fail("unknown M5 config keys", context=context, unknown=unknown)


def _str(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail("expected non-empty string", context=context)
    return value


def _num(value: Any, context: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(float(value)):
        raise _fail("expected finite number", context=context, value=value)
    if positive and float(value) <= 0:
        raise _fail("expected positive number", context=context, value=value)
    return float(value)


@dataclass(frozen=True)
class M5RobotConfig:
    serial: str
    network_interface: str
    rdk_version_min: str
    state_rate_hz: float


@dataclass(frozen=True)
class M5StationConfig:
    schema_version: str
    station_id: str
    backend: str
    allow_real_motion: bool
    robot: M5RobotConfig
    gn01: Gn01Config
    gn01_name: str
    base_to_world: np.ndarray
    nrt_limits: NrtLimits
    tool_name: str
    tool_tcp_hash: str
    tool_payload_hash: str


@dataclass(frozen=True)
class M5SafetyConfig:
    schema_version: str
    safety: SafetyProfile
    stop_timeout_s: float
    preflight_timeout_s: float
    enable_timeout_s: float
    mode_switch_timeout_s: float


@dataclass(frozen=True)
class M5Config:
    station: M5StationConfig
    safety: M5SafetyConfig
    station_path: str
    safety_path: str

    def config_hash(self) -> str:
        payload = {
            "station": {
                "schema_version": self.station.schema_version,
                "station_id": self.station.station_id,
                "backend": self.station.backend,
                "allow_real_motion": self.station.allow_real_motion,
                "robot": self.station.robot.__dict__,
                "gn01": self.station.gn01.__dict__,
                "gn01_name": self.station.gn01_name,
                "base_to_world": self.station.base_to_world.tolist(),
                "nrt_limits": self.station.nrt_limits.__dict__,
                "tool_name": self.station.tool_name,
                "tool_tcp_hash": self.station.tool_tcp_hash,
                "tool_payload_hash": self.station.tool_payload_hash,
            },
            "safety": {
                "schema_version": self.safety.schema_version,
                "stop_timeout_s": self.safety.stop_timeout_s,
                "preflight_timeout_s": self.safety.preflight_timeout_s,
                "enable_timeout_s": self.safety.enable_timeout_s,
                "mode_switch_timeout_s": self.safety.mode_switch_timeout_s,
                "profile": dataclasses.asdict(self.safety.safety),
            },
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def validate_tool_profile(self) -> None:
        values = (
            self.station.tool_name,
            self.station.tool_tcp_hash,
            self.station.tool_payload_hash,
        )
        if any("REPLACE_ME" in value for value in values):
            raise ConfigError("M5 tool profile still contains placeholder values")
        for label, value in (("tcp", values[1]), ("payload", values[2])):
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ConfigError(f"M5 {label} hash must be lowercase SHA-256 hex")

    def assert_real_motion_gate(self) -> None:
        if not self.station.allow_real_motion:
            raise RealHardwareDisabledError(
                "M5 station has allow_real_motion: false",
                details={"station_id": self.station.station_id},
            )
        if os.environ.get(_ALLOW_ENV, "") != "1":
            raise RealHardwareDisabledError(
                "M5 real motion requires VLA_INFRA_ALLOW_REAL_HARDWARE=1",
                details={"station_id": self.station.station_id},
            )


def _safety_profile(raw: Mapping[str, Any]) -> SafetyProfile:
    def section(name: str, required: Tuple[str, ...], optional: Tuple[str, ...] = ()) -> Mapping[str, Any]:
        value = raw[name]
        if not isinstance(value, Mapping):
            raise _fail(f"safety.{name} must be a mapping")
        _keys(value, required, optional, f"m5.safety.{name}")
        return value

    workspace = section("workspace", ("min", "max"))
    lo = tuple(float(v) for v in workspace["min"])
    hi = tuple(float(v) for v in workspace["max"])
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
        (
            "min_segment_dt_ms",
            "max_linear_acceleration_m_s2",
            "max_angular_acceleration_rad_s2",
        ),
    )
    staleness = section("staleness", ("max_observation_age_ms", "max_chunk_age_ms"))
    force = section("force", ("max_force_n", "max_torque_nm"))
    gripper = section("gripper", ("width_min_m", "width_max_m", "max_speed_m_s"))
    action = section("action", ("allowed_frames", "allowed_types"))
    joint_min = joint_max = joint_vel = None
    if "joint_limits" in raw:
        limits = section("joint_limits", ("min_rad", "max_rad"), ("max_velocity_rad_s",))
        joint_min = tuple(float(v) for v in limits["min_rad"])
        joint_max = tuple(float(v) for v in limits["max_rad"])
        if "max_velocity_rad_s" in limits:
            joint_vel = tuple(float(v) for v in limits["max_velocity_rad_s"])

    def ms(value: Any, context: str) -> int:
        return int(round(_num(value, context, positive=True) * 1_000_000))

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
        max_observation_age_ns=ms(staleness["max_observation_age_ms"], "staleness.observation"),
        max_chunk_age_ns=ms(staleness["max_chunk_age_ms"], "staleness.chunk"),
        max_force_n=float(force["max_force_n"]),
        max_torque_nm=float(force["max_torque_nm"]),
        gripper_width_min_m=float(gripper["width_min_m"]),
        gripper_width_max_m=float(gripper["width_max_m"]),
        gripper_max_speed_m_s=float(gripper["max_speed_m_s"]),
        allowed_action_frames=tuple(str(v) for v in action["allowed_frames"]),
        allowed_action_types=tuple(str(v) for v in action["allowed_types"]),
        joint_position_min_rad=joint_min,
        joint_position_max_rad=joint_max,
        min_segment_dt_ns=ms(motion.get("min_segment_dt_ms", 1.0), "motion.segment_dt"),
        max_linear_acceleration_m_s2=(
            None if "max_linear_acceleration_m_s2" not in motion else float(motion["max_linear_acceleration_m_s2"])
        ),
        max_angular_acceleration_rad_s2=(
            None if "max_angular_acceleration_rad_s2" not in motion else float(motion["max_angular_acceleration_rad_s2"])
        ),
        joint_velocity_max_rad_s=joint_vel,
    )


def load_m5_config(station_path: str | Path, safety_path: str | Path) -> M5Config:
    station_raw = _load(station_path)
    _keys(
        station_raw,
        required=(
            "schema_version", "station_id", "backend", "allow_real_motion",
            "robot", "gn01", "base_to_world", "nrt_limits",
            "tool_name", "tool_tcp_hash", "tool_payload_hash",
        ),
        optional=(),
        context="m5.station",
    )
    if station_raw["schema_version"] != M5_STATION_SCHEMA:
        raise _fail("unsupported M5 station schema", actual=station_raw["schema_version"])
    backend = _str(station_raw["backend"], "m5.station.backend")
    if backend != "real":
        raise _fail("M5 station backend must be 'real'", actual=backend)
    if not isinstance(station_raw["allow_real_motion"], bool):
        raise _fail("allow_real_motion must be boolean")
    robot_raw = station_raw["robot"]
    if not isinstance(robot_raw, Mapping):
        raise _fail("robot must be a mapping")
    _keys(robot_raw, ("serial", "network_interface", "rdk_version_min", "state_rate_hz"), (), "m5.robot")
    robot = M5RobotConfig(
        serial=_str(robot_raw["serial"], "m5.robot.serial"),
        network_interface=_str(robot_raw["network_interface"], "m5.robot.network_interface"),
        rdk_version_min=_str(robot_raw["rdk_version_min"], "m5.robot.rdk_version_min"),
        state_rate_hz=_num(robot_raw["state_rate_hz"], "m5.robot.state_rate_hz", positive=True),
    )
    gn01_raw = station_raw["gn01"]
    if not isinstance(gn01_raw, Mapping):
        raise _fail("gn01 must be a mapping")
    _keys(gn01_raw, ("name", "width_min_m", "width_max_m", "speed_m_s", "force_n"), ("dedup_width_m",), "m5.gn01")
    gn01 = Gn01Config(
        width_min_m=_num(gn01_raw["width_min_m"], "m5.gn01.width_min_m"),
        width_max_m=_num(gn01_raw["width_max_m"], "m5.gn01.width_max_m"),
        speed_m_s=_num(gn01_raw["speed_m_s"], "m5.gn01.speed_m_s", positive=True),
        force_n=_num(gn01_raw["force_n"], "m5.gn01.force_n", positive=True),
        dedup_width_m=_num(gn01_raw.get("dedup_width_m", 0.0001), "m5.gn01.dedup_width_m"),
    )
    base = np.asarray(station_raw["base_to_world"], dtype=np.float64)
    if base.shape != (4, 4) or not np.all(np.isfinite(base)):
        raise _fail("base_to_world must be a finite 4x4 matrix")
    limits_raw = station_raw["nrt_limits"]
    if not isinstance(limits_raw, Mapping):
        raise _fail("nrt_limits must be a mapping")
    _keys(
        limits_raw,
        ("max_linear_vel_m_s", "max_angular_vel_rad_s", "max_linear_acc_m_s2", "max_angular_acc_rad_s2"),
        (),
        "m5.nrt_limits",
    )
    limits = NrtLimits(
        max_linear_vel_m_s=_num(limits_raw["max_linear_vel_m_s"], "nrt.max_linear_vel", positive=True),
        max_angular_vel_rad_s=_num(limits_raw["max_angular_vel_rad_s"], "nrt.max_angular_vel", positive=True),
        max_linear_acc_m_s2=_num(limits_raw["max_linear_acc_m_s2"], "nrt.max_linear_acc", positive=True),
        max_angular_acc_rad_s2=_num(limits_raw["max_angular_acc_rad_s2"], "nrt.max_angular_acc", positive=True),
    )
    station = M5StationConfig(
        schema_version=M5_STATION_SCHEMA,
        station_id=_str(station_raw["station_id"], "m5.station_id"),
        backend=backend,
        allow_real_motion=bool(station_raw["allow_real_motion"]),
        robot=robot,
        gn01=gn01,
        gn01_name=_str(gn01_raw["name"], "m5.gn01.name"),
        base_to_world=base,
        nrt_limits=limits,
        tool_name=_str(station_raw["tool_name"], "m5.station.tool_name"),
        tool_tcp_hash=_str(station_raw["tool_tcp_hash"], "m5.station.tool_tcp_hash"),
        tool_payload_hash=_str(station_raw["tool_payload_hash"], "m5.station.tool_payload_hash"),
    )

    safety_raw = _load(safety_path)
    _keys(
        safety_raw,
        required=(
            "schema_version", "workspace", "tcp_height", "motion", "staleness",
            "force", "gripper", "action", "stop_timeout_s", "preflight_timeout_s",
            "enable_timeout_s", "mode_switch_timeout_s",
        ),
        optional=("joint_limits",),
        context="m5.safety",
    )
    if safety_raw["schema_version"] != M5_SAFETY_SCHEMA:
        raise _fail("unsupported M5 safety schema", actual=safety_raw["schema_version"])
    safety = _safety_profile(safety_raw)
    safety_cfg = M5SafetyConfig(
        schema_version=M5_SAFETY_SCHEMA,
        safety=safety,
        stop_timeout_s=_num(safety_raw["stop_timeout_s"], "m5.safety.stop_timeout_s", positive=True),
        preflight_timeout_s=_num(safety_raw["preflight_timeout_s"], "m5.safety.preflight_timeout_s", positive=True),
        enable_timeout_s=_num(safety_raw["enable_timeout_s"], "m5.safety.enable_timeout_s", positive=True),
        mode_switch_timeout_s=_num(safety_raw["mode_switch_timeout_s"], "m5.safety.mode_switch_timeout_s", positive=True),
    )
    return M5Config(
        station=station,
        safety=safety_cfg,
        station_path=str(station_path),
        safety_path=str(safety_path),
    )


__all__ = [
    "M5Config",
    "M5RobotConfig",
    "M5SafetyConfig",
    "M5StationConfig",
    "load_m5_config",
]
