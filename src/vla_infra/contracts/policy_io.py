"""Encoding from canonical state to the networked policy observation.

The state vector layout is explicit metadata, never inferred from array length.
Two layouts are provided for M1; robot-side adapters may add more, but they must
declare name, axes, units, frame and normalisation so the server can reject a
mismatch instead of silently reinterpreting axes.

Image handling: RGB frames are letterboxed to a square uint8 image on the
client, because the network should not carry full-resolution frames.  The
padding is deterministic (centred, zero-filled) and the source size is recorded
in metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from ..errors import ErrorCode, ValidationError
from .constants import (
    ACTION_DIM,
    DEFAULT_POLICY_IMAGE_SIZE,
    IMAGE_ENCODING_RGB8,
    JOINT_COUNT,
    POLICY_KEY_CAPTURE_TIME,
    POLICY_KEY_IMAGE,
    POLICY_KEY_IMAGE_ENCODING,
    POLICY_KEY_PROMPT,
    POLICY_KEY_SCHEMA,
    POLICY_KEY_SEQUENCE,
    POLICY_KEY_SOURCE_IMAGE_SIZE,
    POLICY_KEY_STATE,
    POLICY_KEY_STATE_DIM,
    POLICY_KEY_STATE_LAYOUT,
    POLICY_KEY_WRIST_IMAGE,
    POLICY_OBSERVATION_SCHEMA_VERSION,
    ROTATION_6D_DIM,
    TCP_POSITION_AXES,
    TCP_VELOCITY_AXES,
    WRENCH_AXES,
)
from .observation import CanonicalObservation, RobotState

STATE_FRAME_BASE = "base"


@dataclass(frozen=True)
class StateLayout:
    """Explicit declaration of a policy state vector."""

    name: str
    axes: Tuple[str, ...]
    units: Tuple[str, ...]
    frame: str = STATE_FRAME_BASE
    normalization: str = "none"

    def __post_init__(self) -> None:
        axes = tuple(str(a) for a in self.axes)
        units = tuple(str(u) for u in self.units)
        if not self.name or not isinstance(self.name, str):
            raise ValidationError("state layout name must be a non-empty string")
        if not axes:
            raise ValidationError("state layout must declare at least one axis")
        if len(set(axes)) != len(axes):
            raise ValidationError(
                "state layout axes must be unique",
                code=ErrorCode.SCHEMA_MISMATCH,
                details={"axes": list(axes)},
            )
        if len(units) != len(axes):
            raise ValidationError(
                "state layout units must have one entry per axis",
                code=ErrorCode.INVALID_SHAPE,
                details={"axes": len(axes), "units": len(units)},
            )
        if self.normalization != "none":
            raise ValidationError(
                f"unsupported normalization mode {self.normalization!r}; "
                "M1 only supports 'none' (physical units on the wire)",
                code=ErrorCode.SCHEMA_MISMATCH,
            )
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "units", units)

    @property
    def dim(self) -> int:
        return len(self.axes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "axes": list(self.axes),
            "units": list(self.units),
            "frame": self.frame,
            "normalization": self.normalization,
            "dim": self.dim,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StateLayout":
        if not isinstance(value, Mapping):
            raise ValidationError("state layout metadata must be a mapping")
        try:
            layout = cls(
                name=str(value["name"]),
                axes=tuple(value["axes"]),
                units=tuple(value["units"]),
                frame=str(value.get("frame", STATE_FRAME_BASE)),
                normalization=str(value.get("normalization", "none")),
            )
        except KeyError as exc:
            raise ValidationError(
                f"state layout metadata is missing {exc.args[0]!r}",
                code=ErrorCode.SCHEMA_MISMATCH,
            ) from exc
        if "dim" in value and int(value["dim"]) != layout.dim:
            raise ValidationError(
                "state layout dim field disagrees with axes",
                code=ErrorCode.SCHEMA_MISMATCH,
            )
        return layout

    def encode(self, state: RobotState) -> np.ndarray:
        """Encode a canonical robot state as float32 in the declared order."""
        values = state_scalar_values(state)
        try:
            vector = np.array([values[axis] for axis in self.axes], dtype=np.float32)
        except KeyError as exc:
            raise ValidationError(
                f"state layout axis {exc.args[0]!r} is not produced by the canonical state",
                code=ErrorCode.SCHEMA_MISMATCH,
                details={"axes": list(self.axes), "available": sorted(values)},
            ) from exc
        if not np.all(np.isfinite(vector)):
            raise ValidationError("encoded state contains NaN/Inf", code=ErrorCode.NON_FINITE)
        vector.setflags(write=False)
        return vector


def state_scalar_values(state: RobotState) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for axis, value in zip(TCP_POSITION_AXES, state.tcp_position):
        values[axis] = float(value)
    for index, value in enumerate(state.tcp_rotation_6d):
        values[f"rot6d_{index}"] = float(value)
    for axis, value in zip(TCP_VELOCITY_AXES, state.tcp_velocity):
        values[axis] = float(value)
    for axis, value in zip(WRENCH_AXES, state.wrench_tcp):
        values[axis] = float(value)
    values["gripper_width"] = float(state.gripper_width[0])
    for index, value in enumerate(state.joint_position):
        values[f"q{index}"] = float(value)
    for index, value in enumerate(state.joint_velocity):
        values[f"dq{index}"] = float(value)
    for axis, value in zip(("qx", "qy", "qz", "qw"), state.tcp_quaternion_xyzw):
        values[axis] = float(value)
    values["joint_count"] = float(JOINT_COUNT)
    return values


CANONICAL_STATE_LAYOUT = StateLayout(
    name="canonical_v1",
    axes=TCP_POSITION_AXES
    + tuple(f"rot6d_{i}" for i in range(ROTATION_6D_DIM))
    + TCP_VELOCITY_AXES
    + WRENCH_AXES
    + ("gripper_width",),
    units=("m", "m", "m")
    + ("unit",) * ROTATION_6D_DIM
    + ("m/s", "m/s", "m/s", "rad/s", "rad/s", "rad/s")
    + ("N", "N", "N", "Nm", "Nm", "Nm")
    + ("m",),
)

JOINT_STATE_LAYOUT = StateLayout(
    name="joint_v1",
    axes=tuple(f"q{i}" for i in range(JOINT_COUNT)) + ("gripper_width",),
    units=("rad",) * JOINT_COUNT + ("m",),
)


def letterbox_rgb(image: np.ndarray, size: int) -> np.ndarray:
    """Deterministic centred letterbox of an RGB uint8 image to ``(size,size,3)``."""
    if size <= 0:
        raise ValidationError("image size must be positive", code=ErrorCode.VALUE_OUT_OF_RANGE)
    arr = np.asarray(image)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValidationError(
            "letterbox expects uint8 HxWx3 RGB data",
            code=ErrorCode.INVALID_DTYPE,
            details={"dtype": str(arr.dtype), "shape": list(arr.shape)},
        )
    h, w = arr.shape[0], arr.shape[1]
    scale = min(size / float(w), size / float(h))
    new_w = max(1, min(size, int(round(w * scale))))
    new_h = max(1, min(size, int(round(h * scale))))
    y_idx = np.minimum((np.arange(new_h) * (h / new_h)).astype(np.int64), h - 1)
    x_idx = np.minimum((np.arange(new_w) * (w / new_w)).astype(np.int64), w - 1)
    resized = arr[y_idx][:, x_idx]
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def letterbox_geometry(width: int, height: int, size: int) -> Tuple[int, int, int, int]:
    """Return ``(x0, y0, new_w, new_h)`` used by :func:`letterbox_rgb`."""
    scale = min(size / float(width), size / float(height))
    new_w = max(1, min(size, int(round(width * scale))))
    new_h = max(1, min(size, int(round(height * scale))))
    return (size - new_w) // 2, (size - new_h) // 2, new_w, new_h


def build_policy_observation(
    observation: CanonicalObservation,
    layout: StateLayout,
    *,
    prompt: Optional[str] = None,
    image_size: int = DEFAULT_POLICY_IMAGE_SIZE,
) -> Dict[str, Any]:
    """Build the OpenPI-shaped, JSON/msgpack-encodable policy observation."""
    task = prompt if prompt is not None else observation.task
    if not isinstance(task, str) or not task.strip():
        raise ValidationError("prompt must be a non-empty string")
    base = letterbox_rgb(observation.base_rgb.data, image_size)
    wrist = letterbox_rgb(observation.wrist_rgb.data, image_size)
    state = layout.encode(observation.robot)
    return {
        POLICY_KEY_IMAGE: base,
        POLICY_KEY_WRIST_IMAGE: wrist,
        POLICY_KEY_STATE: state,
        POLICY_KEY_PROMPT: task,
        POLICY_KEY_SEQUENCE: int(observation.sequence_id),
        POLICY_KEY_CAPTURE_TIME: int(observation.capture_time_ns),
        POLICY_KEY_SCHEMA: POLICY_OBSERVATION_SCHEMA_VERSION,
        POLICY_KEY_IMAGE_ENCODING: IMAGE_ENCODING_RGB8,
        POLICY_KEY_STATE_LAYOUT: layout.to_dict(),
        POLICY_KEY_STATE_DIM: int(state.shape[0]),
        POLICY_KEY_SOURCE_IMAGE_SIZE: [
            int(observation.base_rgb.height),
            int(observation.base_rgb.width),
        ],
    }


__all__ = [
    "StateLayout",
    "CANONICAL_STATE_LAYOUT",
    "JOINT_STATE_LAYOUT",
    "ACTION_DIM",
    "build_policy_observation",
    "letterbox_rgb",
    "letterbox_geometry",
    "state_scalar_values",
]
