"""Provisional M1 safety gate (full M3 profile is still pending).

Implemented gates (default: reject, never silently clip):

* workspace AABB and TCP height bounds on every target;
* per-step translation / rotation limits and first-target deviation from the
  measured TCP pose;
* implied linear / angular velocity from ``action_dt_s``;
* observation and chunk staleness;
* action frame / type allow-list, absolute-in-TCP rejection (ambiguous);
* measured wrench force/torque limits and gripper width bounds;
* optional joint position limits supplied by configuration;
* per-tick re-checks in the executor (cheap, no allocation-heavy logic).

Not yet implemented (tracked as M3 work): acceleration limits, velocity
profiles with time stretching instead of rejection, full RDK-specific joint
limit integration, contact-force prediction, and hardware safety certification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..contracts.action import ActionChunk
from ..contracts.constants import ACTION_FRAMES, ACTION_TYPES
from ..contracts.observation import RobotState
from ..contracts.pose import CartesianPose, pose_from_rotvec
from ..errors import ErrorCode, SafetyViolation
from ..geometry import rotation_angle_between, rotation_vector_between

_EPS = 1e-9
#: Action chunks are float32; a boundary value can be ~1e-7 below a limit.
_POSITION_EPS = 1e-6


@dataclass(frozen=True)
class SafetyProfile:
    workspace_min: Tuple[float, float, float]
    workspace_max: Tuple[float, float, float]
    tcp_height_min_m: float
    tcp_height_max_m: float
    max_step_translation_m: float
    max_step_rotation_rad: float
    max_linear_velocity_m_s: float
    max_angular_velocity_rad_s: float
    max_first_target_translation_m: float
    max_first_target_rotation_rad: float
    max_observation_age_ns: int
    max_chunk_age_ns: int
    max_force_n: float
    max_torque_nm: float
    gripper_width_min_m: float
    gripper_width_max_m: float
    gripper_max_speed_m_s: float
    allowed_action_frames: Tuple[str, ...] = ("base", "tcp")
    allowed_action_types: Tuple[str, ...] = ("delta_tcp", "absolute_tcp")
    joint_position_min_rad: Optional[Tuple[float, ...]] = None
    joint_position_max_rad: Optional[Tuple[float, ...]] = None
    min_segment_dt_ns: int = 1_000_000
    velocity_tolerance: float = 1e-6
    max_linear_acceleration_m_s2: Optional[float] = None
    max_angular_acceleration_rad_s2: Optional[float] = None
    joint_velocity_max_rad_s: Optional[Tuple[float, ...]] = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        lo = tuple(float(v) for v in self.workspace_min)
        hi = tuple(float(v) for v in self.workspace_max)
        if len(lo) != 3 or len(hi) != 3:
            raise SafetyViolation(
                "workspace_min/max must have three components",
                code=ErrorCode.CONFIG_INVALID,
            )
        if any(a >= b for a, b in zip(lo, hi)):
            raise SafetyViolation(
                "workspace_min must be strictly below workspace_max on every axis",
                code=ErrorCode.CONFIG_INVALID,
                details={"min": lo, "max": hi},
            )
        if not all(np.isfinite(v) for v in lo + hi):
            raise SafetyViolation("workspace bounds must be finite", code=ErrorCode.CONFIG_INVALID)
        if not (lo[2] <= self.tcp_height_min_m < self.tcp_height_max_m <= hi[2]):
            raise SafetyViolation(
                "TCP height bounds must be ordered and contained in the workspace Z range",
                code=ErrorCode.CONFIG_INVALID,
                details={
                    "workspace_z": [lo[2], hi[2]],
                    "height": [self.tcp_height_min_m, self.tcp_height_max_m],
                },
            )
        positive_fields = (
            "max_step_translation_m",
            "max_step_rotation_rad",
            "max_linear_velocity_m_s",
            "max_angular_velocity_rad_s",
            "max_first_target_translation_m",
            "max_first_target_rotation_rad",
            "max_force_n",
            "max_torque_nm",
            "gripper_max_speed_m_s",
        )
        for name in positive_fields:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise SafetyViolation(
                    f"{name} must be finite and positive",
                    code=ErrorCode.CONFIG_INVALID,
                    details={"field": name, "value": value},
                )
        for name in ("max_observation_age_ns", "max_chunk_age_ns", "min_segment_dt_ns"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise SafetyViolation(
                    f"{name} must be a positive integer",
                    code=ErrorCode.CONFIG_INVALID,
                    details={"field": name, "value": value},
                )
        if not (0.0 <= self.gripper_width_min_m < self.gripper_width_max_m):
            raise SafetyViolation(
                "gripper width range is invalid",
                code=ErrorCode.CONFIG_INVALID,
                details={
                    "min": self.gripper_width_min_m,
                    "max": self.gripper_width_max_m,
                },
            )
        if not self.allowed_action_frames or not set(self.allowed_action_frames) <= ACTION_FRAMES:
            raise SafetyViolation(
                "allowed_action_frames must be a non-empty subset of the contract frames",
                code=ErrorCode.CONFIG_INVALID,
                details={"frames": list(self.allowed_action_frames)},
            )
        if not self.allowed_action_types or not set(self.allowed_action_types) <= ACTION_TYPES:
            raise SafetyViolation(
                "allowed_action_types must be a non-empty subset of the contract types",
                code=ErrorCode.CONFIG_INVALID,
                details={"types": list(self.allowed_action_types)},
            )
        if (self.joint_position_min_rad is None) != (self.joint_position_max_rad is None):
            raise SafetyViolation(
                "joint limits must be configured for all seven joints or not at all",
                code=ErrorCode.CONFIG_INVALID,
            )
        for name in ("max_linear_acceleration_m_s2", "max_angular_acceleration_rad_s2"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or float(value) <= 0.0):
                raise SafetyViolation(
                    f"{name} must be finite and positive when configured",
                    code=ErrorCode.CONFIG_INVALID,
                    details={"field": name, "value": value},
                )
        if self.joint_velocity_max_rad_s is not None:
            limits = tuple(float(v) for v in self.joint_velocity_max_rad_s)
            if len(limits) != 7 or any(not np.isfinite(v) or v <= 0 for v in limits):
                raise SafetyViolation(
                    "joint_velocity_max_rad_s must contain seven finite positive values",
                    code=ErrorCode.CONFIG_INVALID,
                )
            object.__setattr__(self, "joint_velocity_max_rad_s", limits)
        if self.joint_position_min_rad is not None:
            lo_j = tuple(float(v) for v in self.joint_position_min_rad)
            hi_j = tuple(float(v) for v in self.joint_position_max_rad or ())
            if len(lo_j) != 7 or len(hi_j) != 7:
                raise SafetyViolation(
                    "joint limit arrays must have seven entries",
                    code=ErrorCode.CONFIG_INVALID,
                )
            if any(a >= b for a, b in zip(lo_j, hi_j)):
                raise SafetyViolation(
                    "joint_position_min_rad must be below joint_position_max_rad",
                    code=ErrorCode.CONFIG_INVALID,
                )

    @property
    def workspace_low(self) -> np.ndarray:
        return np.asarray(self.workspace_min, dtype=np.float64)

    @property
    def workspace_high(self) -> np.ndarray:
        return np.asarray(self.workspace_max, dtype=np.float64)


@dataclass(frozen=True)
class CommandTarget:
    pose: CartesianPose
    gripper_normalized: float
    step_index: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.pose, CartesianPose):
            raise SafetyViolation(
                "command pose must be a CartesianPose",
                code=ErrorCode.SAFETY_STATE,
            )
        value = float(self.gripper_normalized)
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise SafetyViolation(
                "command gripper must be normalized to [0, 1]",
                code=ErrorCode.SAFETY_GRIPPER,
                details={"value": value},
            )
        object.__setattr__(self, "gripper_normalized", value)


def _rotation_distance(a: CartesianPose, b: CartesianPose) -> float:
    return rotation_angle_between(a.quaternion_xyzw, b.quaternion_xyzw)


class SafetyGate:
    def __init__(self, profile: SafetyProfile) -> None:
        self.profile = profile

    # -- measured state -----------------------------------------------------
    def check_state(self, state: RobotState) -> None:
        p = self.profile
        force = float(np.linalg.norm(state.wrench_tcp[:3]))
        torque = float(np.linalg.norm(state.wrench_tcp[3:]))
        if force > p.max_force_n + _EPS:
            raise SafetyViolation(
                f"measured TCP force {force:.3f} N exceeds limit {p.max_force_n} N",
                code=ErrorCode.SAFETY_FORCE,
                details={"force_n": force, "limit_n": p.max_force_n},
            )
        if torque > p.max_torque_nm + _EPS:
            raise SafetyViolation(
                f"measured TCP torque {torque:.3f} Nm exceeds limit {p.max_torque_nm} Nm",
                code=ErrorCode.SAFETY_TORQUE,
                details={"torque_nm": torque, "limit_nm": p.max_torque_nm},
            )
        width = state.gripper_width_m
        if width < p.gripper_width_min_m - 1e-6 or width > p.gripper_width_max_m + 1e-6:
            raise SafetyViolation(
                f"measured gripper width {width:.4f} m is outside configured limits",
                code=ErrorCode.SAFETY_GRIPPER,
                details={
                    "width_m": width,
                    "min_m": p.gripper_width_min_m,
                    "max_m": p.gripper_width_max_m,
                },
            )
        if p.joint_position_min_rad is not None and p.joint_position_max_rad is not None:
            lo = np.asarray(p.joint_position_min_rad)
            hi = np.asarray(p.joint_position_max_rad)
            if np.any(state.joint_position < lo - 1e-6) or np.any(state.joint_position > hi + 1e-6):
                raise SafetyViolation(
                    "measured joint position violates configured limits",
                    code=ErrorCode.SAFETY_STATE,
                    details={"joint_position": state.joint_position.tolist()},
                )
        if p.joint_velocity_max_rad_s is not None:
            limits = np.asarray(p.joint_velocity_max_rad_s)
            if np.any(np.abs(state.joint_velocity) > limits + 1e-6):
                raise SafetyViolation(
                    "measured joint velocity violates configured limits",
                    code=ErrorCode.SAFETY_VELOCITY,
                    details={"joint_velocity": state.joint_velocity.tolist()},
                )

    # -- chunk-level --------------------------------------------------------
    def check_chunk(
        self,
        chunk: ActionChunk,
        measured_state: RobotState,
        *,
        received_time_ns: int,
        now_ns: int,
    ) -> Tuple[CommandTarget, ...]:
        p = self.profile
        if chunk.action_frame not in p.allowed_action_frames:
            raise SafetyViolation(
                f"action frame {chunk.action_frame!r} is not allowed",
                code=ErrorCode.INVALID_FRAME,
                details={"frame": chunk.action_frame, "allowed": list(p.allowed_action_frames)},
            )
        if chunk.action_type not in p.allowed_action_types:
            raise SafetyViolation(
                f"action type {chunk.action_type!r} is not allowed",
                code=ErrorCode.INVALID_ACTION,
                details={"type": chunk.action_type, "allowed": list(p.allowed_action_types)},
            )
        if chunk.action_type == "absolute_tcp" and chunk.action_frame != "base":
            raise SafetyViolation(
                "absolute_tcp actions require action_frame='base'",
                code=ErrorCode.INVALID_FRAME,
                details={"frame": chunk.action_frame},
            )
        obs_age = now_ns - chunk.observation_capture_time_ns
        if obs_age < -p.max_observation_age_ns:
            raise SafetyViolation(
                "observation capture time is far in the future (clock domain error?)",
                code=ErrorCode.STALE_OBSERVATION,
                details={"obs_age_ns": obs_age},
            )
        if obs_age > p.max_observation_age_ns:
            raise SafetyViolation(
                f"observation is {obs_age / 1e6:.1f} ms old (limit {p.max_observation_age_ns / 1e6:.1f} ms)",
                code=ErrorCode.STALE_OBSERVATION,
                details={"obs_age_ns": obs_age, "limit_ns": p.max_observation_age_ns},
            )
        chunk_age = now_ns - received_time_ns
        if chunk_age > p.max_chunk_age_ns:
            raise SafetyViolation(
                f"action chunk is {chunk_age / 1e6:.1f} ms old (limit {p.max_chunk_age_ns / 1e6:.1f} ms)",
                code=ErrorCode.CHUNK_EXPIRED,
                details={"chunk_age_ns": chunk_age, "limit_ns": p.max_chunk_age_ns},
            )
        self.check_state(measured_state)
        current = measured_state.pose
        targets = self._targets_from_chunk(chunk, current)
        if not targets:
            raise SafetyViolation("action chunk produced no targets", code=ErrorCode.INVALID_ACTION)
        first = targets[0]
        first_translation = current.translation_distance(first.pose)
        first_rotation = _rotation_distance(current, first.pose)
        if first_translation > p.max_first_target_translation_m + _EPS:
            raise SafetyViolation(
                f"first target is {first_translation:.4f} m from the measured TCP "
                f"(limit {p.max_first_target_translation_m} m)",
                code=ErrorCode.SAFETY_FIRST_TARGET,
                details={"distance_m": first_translation, "limit_m": p.max_first_target_translation_m},
            )
        if first_rotation > p.max_first_target_rotation_rad + _EPS:
            raise SafetyViolation(
                f"first target rotates {first_rotation:.4f} rad from the measured TCP "
                f"(limit {p.max_first_target_rotation_rad} rad)",
                code=ErrorCode.SAFETY_FIRST_TARGET,
                details={"angle_rad": first_rotation, "limit_rad": p.max_first_target_rotation_rad},
            )
        dt_s = chunk.action_dt_s
        previous = current
        previous_linear_velocity = np.asarray(measured_state.tcp_velocity[:3], dtype=np.float64)
        previous_angular_velocity = np.asarray(measured_state.tcp_velocity[3:], dtype=np.float64)
        for index, target in enumerate(targets):
            translation = previous.translation_distance(target.pose)
            rotation = _rotation_distance(previous, target.pose)
            if translation > p.max_step_translation_m + _EPS:
                raise SafetyViolation(
                    f"step {index} translates {translation:.4f} m (limit {p.max_step_translation_m} m)",
                    code=ErrorCode.SAFETY_STEP,
                    details={"step": index, "distance_m": translation},
                )
            if rotation > p.max_step_rotation_rad + _EPS:
                raise SafetyViolation(
                    f"step {index} rotates {rotation:.4f} rad (limit {p.max_step_rotation_rad} rad)",
                    code=ErrorCode.SAFETY_STEP,
                    details={"step": index, "angle_rad": rotation},
                )
            if translation / dt_s > p.max_linear_velocity_m_s + p.velocity_tolerance:
                raise SafetyViolation(
                    f"step {index} implies {translation / dt_s:.3f} m/s "
                    f"(limit {p.max_linear_velocity_m_s} m/s)",
                    code=ErrorCode.SAFETY_VELOCITY,
                    details={"step": index, "velocity_m_s": translation / dt_s},
                )
            if rotation / dt_s > p.max_angular_velocity_rad_s + p.velocity_tolerance:
                raise SafetyViolation(
                    f"step {index} implies {rotation / dt_s:.3f} rad/s "
                    f"(limit {p.max_angular_velocity_rad_s} rad/s)",
                    code=ErrorCode.SAFETY_VELOCITY,
                    details={"step": index, "velocity_rad_s": rotation / dt_s},
                )
            current_linear_velocity = (
                np.asarray(target.pose.position, dtype=np.float64)
                - np.asarray(previous.position, dtype=np.float64)
            ) / dt_s
            current_angular_velocity = (
                rotation_vector_between(previous.quaternion_xyzw, target.pose.quaternion_xyzw)
                / dt_s
            )
            if p.max_linear_acceleration_m_s2 is not None:
                acceleration = float(
                    np.linalg.norm(current_linear_velocity - previous_linear_velocity) / dt_s
                )
                if acceleration > p.max_linear_acceleration_m_s2 + p.velocity_tolerance:
                    raise SafetyViolation(
                        f"step {index} implies linear acceleration {acceleration:.3f} m/s^2 "
                        f"(limit {p.max_linear_acceleration_m_s2} m/s^2)",
                        code=ErrorCode.SAFETY_ACCELERATION,
                        details={"step": index, "acceleration_m_s2": acceleration},
                    )
            if p.max_angular_acceleration_rad_s2 is not None:
                acceleration = float(
                    np.linalg.norm(current_angular_velocity - previous_angular_velocity) / dt_s
                )
                if acceleration > p.max_angular_acceleration_rad_s2 + p.velocity_tolerance:
                    raise SafetyViolation(
                        f"step {index} implies angular acceleration {acceleration:.3f} rad/s^2 "
                        f"(limit {p.max_angular_acceleration_rad_s2} rad/s^2)",
                        code=ErrorCode.SAFETY_ACCELERATION,
                        details={"step": index, "acceleration_rad_s2": acceleration},
                    )
            previous_linear_velocity = current_linear_velocity
            previous_angular_velocity = current_angular_velocity
            self._check_workspace(target.pose, step=index)
            previous = target.pose
        return tuple(targets)

    def _targets_from_chunk(
        self, chunk: ActionChunk, current_pose: CartesianPose
    ) -> list[CommandTarget]:
        targets: list[CommandTarget] = []
        previous_pose = current_pose
        for index, row in enumerate(np.asarray(chunk.actions, dtype=np.float64)):
            if not np.all(np.isfinite(row)):
                raise SafetyViolation(
                    f"action row {index} contains NaN/Inf",
                    code=ErrorCode.NON_FINITE,
                    details={"step": index},
                )
            translation = row[:3]
            rotation = row[3:6]
            gripper = float(row[6])
            if chunk.action_type == "delta_tcp":
                target_pose = previous_pose.apply_delta(
                    translation, rotation, frame=chunk.action_frame
                )
            else:
                target_pose = pose_from_rotvec(translation, rotation)
            targets.append(
                CommandTarget(pose=target_pose, gripper_normalized=gripper, step_index=index)
            )
            previous_pose = target_pose
        return targets

    # -- per-tick -----------------------------------------------------------
    def check_tick(
        self,
        previous: CommandTarget,
        target: CommandTarget,
        dt_ns: int,
        *,
        previous_linear_velocity_m_s: Optional[float] = None,
        previous_angular_velocity_rad_s: Optional[float] = None,
        previous_linear_velocity: Optional[np.ndarray] = None,
        previous_angular_velocity: Optional[np.ndarray] = None,
    ) -> None:
        if not isinstance(dt_ns, int) or dt_ns <= 0:
            raise SafetyViolation(
                "per-tick dt must be positive",
                code=ErrorCode.SAFETY_STATE,
                details={"dt_ns": dt_ns},
            )
        p = self.profile
        if not isinstance(target.pose, CartesianPose):  # defensive against fake targets
            raise SafetyViolation("target pose is not a CartesianPose", code=ErrorCode.SAFETY_STATE)
        dt_s = dt_ns / 1_000_000_000.0
        translation = previous.pose.translation_distance(target.pose)
        rotation = _rotation_distance(previous.pose, target.pose)
        if translation > p.max_step_translation_m + _EPS:
            raise SafetyViolation(
                f"per-tick translation {translation:.4f} m exceeds step limit",
                code=ErrorCode.SAFETY_STEP,
                details={"distance_m": translation, "limit_m": p.max_step_translation_m},
            )
        if rotation > p.max_step_rotation_rad + _EPS:
            raise SafetyViolation(
                f"per-tick rotation {rotation:.4f} rad exceeds step limit",
                code=ErrorCode.SAFETY_STEP,
                details={"angle_rad": rotation, "limit_rad": p.max_step_rotation_rad},
            )
        if translation / dt_s > p.max_linear_velocity_m_s + p.velocity_tolerance:
            raise SafetyViolation(
                f"per-tick velocity {translation / dt_s:.3f} m/s exceeds limit",
                code=ErrorCode.SAFETY_VELOCITY,
                details={"velocity_m_s": translation / dt_s, "dt_ns": dt_ns},
            )
        if rotation / dt_s > p.max_angular_velocity_rad_s + p.velocity_tolerance:
            raise SafetyViolation(
                f"per-tick angular velocity {rotation / dt_s:.3f} rad/s exceeds limit",
                code=ErrorCode.SAFETY_VELOCITY,
                details={"velocity_rad_s": rotation / dt_s, "dt_ns": dt_ns},
            )
        if not np.isfinite(target.gripper_normalized) or not (0.0 <= target.gripper_normalized <= 1.0):
            raise SafetyViolation(
                "per-tick gripper command is outside [0, 1]",
                code=ErrorCode.SAFETY_GRIPPER,
                details={"value": target.gripper_normalized},
            )
        linear_velocity = translation / dt_s
        angular_velocity = rotation / dt_s
        current_linear_vector = (
            np.asarray(target.pose.position, dtype=np.float64)
            - np.asarray(previous.pose.position, dtype=np.float64)
        ) / dt_s
        current_angular_vector = (
            rotation_vector_between(previous.pose.quaternion_xyzw, target.pose.quaternion_xyzw)
            / dt_s
        )
        if p.max_linear_acceleration_m_s2 is not None:
            if previous_linear_velocity is not None:
                acceleration = float(
                    np.linalg.norm(
                        current_linear_vector - np.asarray(previous_linear_velocity, dtype=np.float64)
                    )
                    / dt_s
                )
            elif previous_linear_velocity_m_s is not None:
                acceleration = abs(linear_velocity - float(previous_linear_velocity_m_s)) / dt_s
            else:
                acceleration = None
            if acceleration is not None and acceleration > p.max_linear_acceleration_m_s2 + p.velocity_tolerance:
                raise SafetyViolation(
                    f"per-tick linear acceleration {acceleration:.3f} m/s^2 exceeds limit",
                    code=ErrorCode.SAFETY_ACCELERATION,
                    details={"acceleration_m_s2": acceleration, "dt_ns": dt_ns},
                )
        if p.max_angular_acceleration_rad_s2 is not None:
            if previous_angular_velocity is not None:
                acceleration = float(
                    np.linalg.norm(
                        current_angular_vector - np.asarray(previous_angular_velocity, dtype=np.float64)
                    )
                    / dt_s
                )
            elif previous_angular_velocity_rad_s is not None:
                acceleration = abs(angular_velocity - float(previous_angular_velocity_rad_s)) / dt_s
            else:
                acceleration = None
            if acceleration is not None and acceleration > p.max_angular_acceleration_rad_s2 + p.velocity_tolerance:
                raise SafetyViolation(
                    f"per-tick angular acceleration {acceleration:.3f} rad/s^2 exceeds limit",
                    code=ErrorCode.SAFETY_ACCELERATION,
                    details={"acceleration_rad_s2": acceleration, "dt_ns": dt_ns},
                )
        self._check_workspace(target.pose, step=target.step_index)

    def _check_workspace(self, pose: CartesianPose, *, step: int) -> None:
        p = self.profile
        position = np.asarray(pose.position, dtype=np.float64)
        if not np.all(np.isfinite(position)):
            raise SafetyViolation(
                "target position contains NaN/Inf",
                code=ErrorCode.NON_FINITE,
                details={"step": step},
            )
        low = p.workspace_low
        high = p.workspace_high
        if np.any(position < low - _POSITION_EPS) or np.any(position > high + _POSITION_EPS):
            raise SafetyViolation(
                f"target step {step} is outside the workspace AABB",
                code=ErrorCode.SAFETY_WORKSPACE,
                details={
                    "step": step,
                    "position": position.tolist(),
                    "min": low.tolist(),
                    "max": high.tolist(),
                },
            )
        z = float(position[2])
        if z < p.tcp_height_min_m - _POSITION_EPS or z > p.tcp_height_max_m + _POSITION_EPS:
            raise SafetyViolation(
                f"target step {step} height {z:.4f} m is outside [{p.tcp_height_min_m}, "
                f"{p.tcp_height_max_m}] m",
                code=ErrorCode.SAFETY_HEIGHT,
                details={"step": step, "z": z},
            )


__all__ = ["CommandTarget", "SafetyGate", "SafetyProfile"]
