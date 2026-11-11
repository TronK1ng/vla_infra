"""Contract validators.

Constructors call these functions, so an invalid contract object cannot exist
in the process.  Validators raise :class:`ValidationError` with a stable code
and machine-readable details; they never clip or repair physical values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional

import numpy as np

from ..errors import ErrorCode, ValidationError
from ..geometry import quaternion_to_matrix
from .constants import (
    ACTION_DIM,
    ACTION_FRAMES,
    ACTION_TYPES,
    IMAGE_ENCODING_DEPTH16,
    IMAGE_ENCODING_RGB8,
    MAX_ACTION_DT_S,
    MAX_ACTION_HORIZON,
    MIN_ACTION_DT_S,
    OPTIONAL_DEPTH_IMAGES,
    POLICY_OBSERVATION_SCHEMA_VERSION,
    POLICY_KEY_CAPTURE_TIME,
    POLICY_KEY_IMAGE,
    POLICY_KEY_IMAGE_ENCODING,
    POLICY_KEY_PROMPT,
    POLICY_KEY_SCHEMA,
    POLICY_KEY_SEQUENCE,
    POLICY_KEY_STATE,
    POLICY_KEY_STATE_DIM,
    POLICY_KEY_STATE_LAYOUT,
    POLICY_KEY_WRIST_IMAGE,
    REQUIRED_RGB_IMAGES,
    SUPPORTED_ACTION_SCHEMAS,
    SUPPORTED_OBSERVATION_SCHEMAS,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .action import ActionChunk
    from .observation import CanonicalObservation, RobotState

_ROT6D_TOL = 1e-3
_QUAT_NORM_TOL = 1e-3
_GRIPPER_WIDTH_SANITY_M = 0.5
_ALPHA_TOL = 1e-6
_MAX_WRENCH_SANITY = 10_000.0


def _check_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValidationError(
            f"{name} must be an integer",
            code=ErrorCode.INVALID_DTYPE,
            details={"field": name, "type": type(value).__name__},
        )
    if int(value) < minimum:
        raise ValidationError(
            f"{name} must be >= {minimum}",
            code=ErrorCode.VALUE_OUT_OF_RANGE,
            details={"field": name, "value": int(value)},
        )
    return int(value)


def _require_schema(schema: str, supported: frozenset[str], name: str) -> None:
    if not isinstance(schema, str) or schema not in supported:
        raise ValidationError(
            f"unsupported {name} schema {schema!r}",
            code=ErrorCode.SCHEMA_MISMATCH,
            details={"schema": schema, "supported": sorted(supported)},
        )


def validate_quaternion_and_rotation(quaternion: np.ndarray, rotation_6d: np.ndarray) -> None:
    """Check unit norm and quaternion/6D consistency."""
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or abs(norm - 1.0) > _QUAT_NORM_TOL:
        raise ValidationError(
            "tcp_quaternion_xyzw must be unit length",
            code=ErrorCode.INVALID_QUATERNION,
            details={"norm": norm},
        )
    expected = np.array(quaternion_to_matrix(quaternion), dtype=np.float64)
    expected_6d = np.concatenate([expected[:, 0], expected[:, 1]])
    if not np.allclose(rotation_6d, expected_6d, atol=_ROT6D_TOL):
        raise ValidationError(
            "tcp_rotation_6d is inconsistent with tcp_quaternion_xyzw",
            code=ErrorCode.INVALID_QUATERNION,
            details={
                "rotation_6d": np.asarray(rotation_6d).tolist(),
                "expected_6d": expected_6d.tolist(),
            },
        )


def validate_robot_state(state: "RobotState") -> None:
    for name, value in (
        ("joint_position", state.joint_position),
        ("joint_velocity", state.joint_velocity),
        ("tcp_position", state.tcp_position),
        ("tcp_quaternion_xyzw", state.tcp_quaternion_xyzw),
        ("tcp_rotation_6d", state.tcp_rotation_6d),
        ("tcp_velocity", state.tcp_velocity),
        ("wrench_tcp", state.wrench_tcp),
        ("gripper_width", state.gripper_width),
    ):
        arr = np.asarray(value)
        if not np.all(np.isfinite(arr)):
            raise ValidationError(
                f"{name} contains NaN/Inf",
                code=ErrorCode.NON_FINITE,
                details={"field": name},
            )
    validate_quaternion_and_rotation(state.tcp_quaternion_xyzw, state.tcp_rotation_6d)
    width = float(state.gripper_width[0])
    if width < 0.0 or width > _GRIPPER_WIDTH_SANITY_M:
        raise ValidationError(
            "gripper_width is outside the plausible physical range [0, 0.5] m",
            code=ErrorCode.SAFETY_GRIPPER,
            details={"gripper_width_m": width},
        )
    if float(np.max(np.abs(state.wrench_tcp))) > _MAX_WRENCH_SANITY:
        raise ValidationError(
            "wrench_tcp magnitude is implausible (unit error?)",
            code=ErrorCode.SAFETY_FORCE,
            details={"max_abs": float(np.max(np.abs(state.wrench_tcp)))},
        )


def validate_sync_info(obs: "CanonicalObservation") -> None:
    sync = obs.sync
    before = _check_int(sync.robot_sample_before_ns, "robot_sample_before_ns")
    after = _check_int(sync.robot_sample_after_ns, "robot_sample_after_ns")
    capture = _check_int(obs.capture_time_ns, "capture_time_ns", minimum=1)
    if before > after:
        raise ValidationError(
            "robot_sample_before_ns is later than robot_sample_after_ns",
            code=ErrorCode.INVALID_SYNC,
            details={"before": before, "after": after},
        )
    if not (before <= capture <= after):
        raise ValidationError(
            "capture_time_ns is not bracketed by the robot samples",
            code=ErrorCode.INVALID_SYNC,
            details={"before": before, "capture": capture, "after": after},
        )
    alpha = float(sync.interpolation_alpha)
    if not np.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        raise ValidationError(
            "interpolation_alpha must lie in [0, 1]",
            code=ErrorCode.INVALID_SYNC,
            details={"alpha": alpha},
        )
    if sync.interpolation_applied:
        gap = after - before
        expected_alpha = 0.0 if gap == 0 else (capture - before) / float(gap)
        if abs(alpha - expected_alpha) > _ALPHA_TOL:
            raise ValidationError(
                "interpolation_alpha is inconsistent with the sample times",
                code=ErrorCode.INVALID_SYNC,
                details={"alpha": alpha, "expected_alpha": expected_alpha},
            )
    max_offset = max(capture - before, after - capture)
    age = _check_int(sync.max_age_ns, "max_age_ns")
    if age != max_offset:
        raise ValidationError(
            "max_age_ns must equal the largest bracketing offset",
            code=ErrorCode.INVALID_SYNC,
            details={"max_age_ns": age, "expected": max_offset},
        )
    if not isinstance(sync.state_gap_ns, int) or sync.state_gap_ns < 0:
        raise ValidationError("state_gap_ns must be a non-negative int", code=ErrorCode.INVALID_SYNC)
    if not isinstance(sync.camera_skew_ns, int) or sync.camera_skew_ns < 0:
        raise ValidationError("camera_skew_ns must be a non-negative int", code=ErrorCode.INVALID_SYNC)


def validate_canonical_observation(obs: "CanonicalObservation") -> None:
    _require_schema(obs.schema_version, SUPPORTED_OBSERVATION_SCHEMAS, "observation")
    _check_int(obs.sequence_id, "sequence_id")
    _check_int(obs.capture_time_ns, "capture_time_ns", minimum=1)
    if not isinstance(obs.task, str) or not obs.task.strip():
        raise ValidationError(
            "task must be a non-empty string",
            code=ErrorCode.VALUE_OUT_OF_RANGE,
        )
    if not isinstance(obs.images, Mapping):
        raise ValidationError("images must be a mapping", code=ErrorCode.INVALID_SHAPE)
    for key in REQUIRED_RGB_IMAGES:
        if key not in obs.images:
            raise ValidationError(
                f"observation is missing required image {key!r}",
                code=ErrorCode.INVALID_SHAPE,
                details={"missing": key, "present": sorted(obs.images)},
            )
    for key, frame in obs.images.items():
        if key in REQUIRED_RGB_IMAGES:
            expected = IMAGE_ENCODING_RGB8
        elif key in OPTIONAL_DEPTH_IMAGES:
            expected = IMAGE_ENCODING_DEPTH16
        else:
            expected = None
        if expected is not None and frame.encoding != expected:
            raise ValidationError(
                f"image {key!r} must use encoding {expected!r}",
                code=ErrorCode.INVALID_ENCODING,
                details={"key": key, "encoding": frame.encoding},
            )
        if frame.host_time_ns > obs.capture_time_ns:
            # The target time is the base frame's host time; wrist may be a few
            # ms older/newer but never in the future relative to the tick.
            raise ValidationError(
                f"image {key!r} host_time_ns is in the future relative to capture_time_ns",
                code=ErrorCode.INVALID_SYNC,
                details={"key": key, "host_time_ns": frame.host_time_ns},
            )
    skew = max(abs(obs.images[key].host_time_ns - obs.capture_time_ns) for key in REQUIRED_RGB_IMAGES)
    if obs.sync.camera_skew_ns != skew:
        raise ValidationError(
            "sync.camera_skew_ns does not match the RGB frame timestamps",
            code=ErrorCode.INVALID_SYNC,
            details={"camera_skew_ns": obs.sync.camera_skew_ns, "expected": skew},
        )
    validate_robot_state(obs.robot)
    validate_sync_info(obs)
    if not isinstance(obs.robot, object):  # pragma: no cover - defensive
        raise ValidationError("robot state is missing", code=ErrorCode.INVALID_SHAPE)


def validate_action_chunk(chunk: "ActionChunk") -> None:
    _require_schema(chunk.schema_version, SUPPORTED_ACTION_SCHEMAS, "action")
    _check_int(chunk.request_sequence_id, "request_sequence_id")
    _check_int(chunk.observation_capture_time_ns, "observation_capture_time_ns", minimum=1)
    dt = float(chunk.action_dt_s)
    if not np.isfinite(dt) or dt < MIN_ACTION_DT_S or dt > MAX_ACTION_DT_S:
        raise ValidationError(
            f"action_dt_s must be in [{MIN_ACTION_DT_S}, {MAX_ACTION_DT_S}] s",
            code=ErrorCode.VALUE_OUT_OF_RANGE,
            details={"action_dt_s": dt},
        )
    if chunk.action_frame not in ACTION_FRAMES:
        raise ValidationError(
            f"action_frame must be one of {sorted(ACTION_FRAMES)}, got {chunk.action_frame!r}",
            code=ErrorCode.INVALID_FRAME,
            details={"action_frame": chunk.action_frame},
        )
    if chunk.action_type not in ACTION_TYPES:
        raise ValidationError(
            f"action_type must be one of {sorted(ACTION_TYPES)}, got {chunk.action_type!r}",
            code=ErrorCode.INVALID_ACTION,
            details={"action_type": chunk.action_type},
        )
    actions = np.asarray(chunk.actions)
    if actions.shape != (actions.shape[0], ACTION_DIM) or actions.shape[0] < 1:
        raise ValidationError(
            f"actions must be (H, {ACTION_DIM}) with H>=1",
            code=ErrorCode.INVALID_SHAPE,
        )
    if actions.shape[0] > MAX_ACTION_HORIZON:
        raise ValidationError(
            f"action horizon {actions.shape[0]} exceeds {MAX_ACTION_HORIZON}",
            code=ErrorCode.VALUE_OUT_OF_RANGE,
        )
    if actions.dtype != np.float32:
        raise ValidationError(
            "actions dtype must be float32",
            code=ErrorCode.INVALID_DTYPE,
            details={"dtype": str(actions.dtype)},
        )
    if not np.all(np.isfinite(actions)):
        raise ValidationError(
            "actions contain NaN/Inf",
            code=ErrorCode.NON_FINITE,
        )
    gripper = actions[:, 6]
    if np.any(gripper < 0.0) or np.any(gripper > 1.0):
        raise ValidationError(
            "gripper channel must lie in [0, 1]",
            code=ErrorCode.SAFETY_GRIPPER,
            details={"min": float(gripper.min()), "max": float(gripper.max())},
        )
    if not isinstance(chunk.extras, Mapping):
        raise ValidationError("extras must be a mapping", code=ErrorCode.INVALID_SHAPE)
    for key in ("server_receive_time_ns", "server_send_time_ns"):
        value = getattr(chunk, key)
        if value is not None:
            _check_int(value, key, minimum=0)


def validate_policy_observation(
    observation: Mapping[str, Any],
    *,
    expected_layout: Optional[Any] = None,
    expected_image_size: Optional[int] = None,
    require_metadata: bool = True,
) -> None:
    """Validate the networked policy observation dictionary.

    ``expected_layout`` is a :class:`~vla_infra.contracts.policy_io.StateLayout`;
    when supplied the incoming state axis order must match exactly.
    """
    if not isinstance(observation, Mapping):
        raise ValidationError(
            "policy observation must be a mapping",
            code=ErrorCode.INVALID_SHAPE,
            details={"type": type(observation).__name__},
        )
    required = [
        POLICY_KEY_IMAGE,
        POLICY_KEY_WRIST_IMAGE,
        POLICY_KEY_STATE,
        POLICY_KEY_PROMPT,
        POLICY_KEY_SEQUENCE,
        POLICY_KEY_CAPTURE_TIME,
        POLICY_KEY_SCHEMA,
        POLICY_KEY_IMAGE_ENCODING,
    ]
    if require_metadata:
        required += [POLICY_KEY_STATE_LAYOUT, POLICY_KEY_STATE_DIM]
    for key in required:
        if key not in observation:
            raise ValidationError(
                f"policy observation is missing key {key!r}",
                code=ErrorCode.SCHEMA_MISMATCH,
                details={"missing": key, "present": sorted(map(str, observation))},
            )
    prompt = observation[POLICY_KEY_PROMPT]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValidationError(
            "prompt must be a non-empty string",
            code=ErrorCode.VALUE_OUT_OF_RANGE,
        )
    if observation[POLICY_KEY_IMAGE_ENCODING] != IMAGE_ENCODING_RGB8:
        raise ValidationError(
            "observation images must be declared as RGB8; silent BGR is forbidden",
            code=ErrorCode.INVALID_ENCODING,
            details={"encoding": observation[POLICY_KEY_IMAGE_ENCODING]},
        )
    for key in (POLICY_KEY_IMAGE, POLICY_KEY_WRIST_IMAGE):
        image = np.asarray(observation[key])
        if image.dtype != np.uint8:
            raise ValidationError(
                f"{key} dtype must be uint8",
                code=ErrorCode.INVALID_DTYPE,
                details={"key": key, "dtype": str(image.dtype)},
            )
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValidationError(
                f"{key} must have shape (H, W, 3)",
                code=ErrorCode.INVALID_SHAPE,
                details={"key": key, "shape": list(image.shape)},
            )
        if image.shape[0] != image.shape[1]:
            raise ValidationError(
                f"{key} must be square",
                code=ErrorCode.INVALID_SHAPE,
                details={"key": key, "shape": list(image.shape)},
            )
        if expected_image_size is not None and image.shape[0] != expected_image_size:
            raise ValidationError(
                f"{key} must be resized to {expected_image_size}",
                code=ErrorCode.INVALID_SHAPE,
                details={"key": key, "shape": list(image.shape)},
            )
    state = np.asarray(observation[POLICY_KEY_STATE])
    if state.dtype != np.float32:
        raise ValidationError(
            "observation/state dtype must be float32",
            code=ErrorCode.INVALID_DTYPE,
            details={"dtype": str(state.dtype)},
        )
    if state.ndim != 1:
        raise ValidationError(
            "observation/state must be 1-D",
            code=ErrorCode.INVALID_SHAPE,
            details={"shape": list(state.shape)},
        )
    if not np.all(np.isfinite(state)):
        raise ValidationError("observation/state contains NaN/Inf", code=ErrorCode.NON_FINITE)
    if require_metadata:
        layout_meta = observation[POLICY_KEY_STATE_LAYOUT]
        if not isinstance(layout_meta, Mapping):
            raise ValidationError(
                "meta/state_layout must be a mapping",
                code=ErrorCode.INVALID_SHAPE,
            )
        dim = observation[POLICY_KEY_STATE_DIM]
        if isinstance(dim, bool) or not isinstance(dim, (int, np.integer)) or int(dim) != state.shape[0]:
            raise ValidationError(
                "meta/state_dim does not match the state vector length",
                code=ErrorCode.INVALID_SHAPE,
                details={"state_dim": dim, "actual": int(state.shape[0])},
            )
        if expected_layout is not None:
            expected_meta = expected_layout.to_dict()
            for key, value in expected_meta.items():
                if layout_meta.get(key) != value:
                    raise ValidationError(
                        f"state layout mismatch for {key!r}",
                        code=ErrorCode.SCHEMA_MISMATCH,
                        details={
                            "key": key,
                            "expected": value,
                            "received": layout_meta.get(key),
                        },
                    )
    _check_int(observation[POLICY_KEY_SEQUENCE], POLICY_KEY_SEQUENCE)
    _check_int(observation[POLICY_KEY_CAPTURE_TIME], POLICY_KEY_CAPTURE_TIME, minimum=1)
    if observation[POLICY_KEY_SCHEMA] != POLICY_OBSERVATION_SCHEMA_VERSION:
        raise ValidationError(
            f"unsupported policy observation schema {observation[POLICY_KEY_SCHEMA]!r}",
            code=ErrorCode.SCHEMA_MISMATCH,
            details={"supported": POLICY_OBSERVATION_SCHEMA_VERSION},
        )


__all__ = [
    "validate_action_chunk",
    "validate_canonical_observation",
    "validate_policy_observation",
    "validate_robot_state",
    "validate_sync_info",
    "validate_quaternion_and_rotation",
]
