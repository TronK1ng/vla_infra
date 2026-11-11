"""Canonical observation contract (master plan §5.2).

The dataclasses are frozen and copy every array on construction; arrays are
made read-only.  This keeps a single observation safe to hand to the recorder,
the policy encoder and any diagnostic consumer without accidental mutation.

Timestamps:

* ``host_time_ns`` values come from the process monotonic clock and are the only
  values used for scheduling / age calculations.
* ``device_time_ns`` values are raw device timestamps, preserved unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional

import numpy as np

from ..errors import ErrorCode, ValidationError
from .constants import (
    IMAGE_ENCODING_DEPTH16,
    IMAGE_ENCODING_RGB8,
    IMAGE_KEY_BASE_DEPTH,
    IMAGE_KEY_BASE_RGB,
    IMAGE_KEY_WRIST_DEPTH,
    IMAGE_KEY_WRIST_RGB,
    ROBOT_STATE_SHAPES,
    TCP_VELOCITY_FRAME,
    WRENCH_FRAME,
)
from .pose import CartesianPose


def _freeze_array(value: Any, dtype: np.dtype, shape: Optional[tuple], name: str) -> np.ndarray:
    arr = np.array(value, dtype=dtype, copy=True)
    if shape is not None and arr.shape != shape:
        raise ValidationError(
            f"{name} must have shape {shape}, got {arr.shape}",
            code=ErrorCode.INVALID_SHAPE,
            details={"field": name, "expected_shape": list(shape), "actual_shape": list(arr.shape)},
        )
    if not np.all(np.isfinite(arr)):
        raise ValidationError(
            f"{name} contains NaN/Inf",
            code=ErrorCode.NON_FINITE,
            details={"field": name},
        )
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class ImageFrame:
    """One camera frame with its raw device timestamp and host receive time."""

    name: str
    data: np.ndarray
    device_time_ns: int
    host_time_ns: int
    frame_number: int
    encoding: str = IMAGE_ENCODING_RGB8

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValidationError("image name must be a non-empty string")
        if self.encoding not in (IMAGE_ENCODING_RGB8, IMAGE_ENCODING_DEPTH16):
            raise ValidationError(
                f"unsupported image encoding {self.encoding!r}",
                code=ErrorCode.INVALID_ENCODING,
                details={"encoding": self.encoding},
            )
        if not isinstance(self.device_time_ns, int) or self.device_time_ns <= 0:
            raise ValidationError(
                "device_time_ns must be a positive integer",
                code=ErrorCode.VALUE_OUT_OF_RANGE,
                details={"field": "device_time_ns"},
            )
        if not isinstance(self.host_time_ns, int) or self.host_time_ns <= 0:
            raise ValidationError(
                "host_time_ns must be a positive integer",
                code=ErrorCode.VALUE_OUT_OF_RANGE,
                details={"field": "host_time_ns"},
            )
        if not isinstance(self.frame_number, int) or self.frame_number < 0:
            raise ValidationError(
                "frame_number must be a non-negative integer",
                code=ErrorCode.VALUE_OUT_OF_RANGE,
                details={"field": "frame_number"},
            )
        arr = np.asarray(self.data)
        if self.encoding == IMAGE_ENCODING_RGB8:
            if arr.dtype != np.uint8:
                raise ValidationError(
                    "RGB image dtype must be uint8",
                    code=ErrorCode.INVALID_DTYPE,
                    details={"dtype": str(arr.dtype)},
                )
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValidationError(
                    "RGB image must have shape (H, W, 3)",
                    code=ErrorCode.INVALID_SHAPE,
                    details={"shape": list(arr.shape)},
                )
            if arr.shape[0] <= 0 or arr.shape[1] <= 0:
                raise ValidationError("RGB image must be non-empty", code=ErrorCode.INVALID_SHAPE)
        else:
            if arr.dtype != np.uint16:
                raise ValidationError(
                    "depth image dtype must be uint16",
                    code=ErrorCode.INVALID_DTYPE,
                    details={"dtype": str(arr.dtype)},
                )
            if arr.ndim != 2:
                raise ValidationError(
                    "depth image must have shape (H, W)",
                    code=ErrorCode.INVALID_SHAPE,
                    details={"shape": list(arr.shape)},
                )
        object.__setattr__(self, "data", np.array(arr, copy=True, order="C"))

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    def as_readonly(self) -> "ImageFrame":
        """Return a copy whose pixel buffer is read-only (used by observation)."""
        frozen = ImageFrame(
            name=self.name,
            data=self.data,
            device_time_ns=self.device_time_ns,
            host_time_ns=self.host_time_ns,
            frame_number=self.frame_number,
            encoding=self.encoding,
        )
        frozen.data.setflags(write=False)
        return frozen


@dataclass(frozen=True)
class RobotState:
    """One robot state sample in physical units.

    Frame/order metadata is part of the contract, not inferred from array
    lengths:

    * ``joint_position`` order is the RDK joint order ``J1..J7`` (rad);
    * ``tcp_position`` is in metres in the base frame;
    * ``tcp_quaternion_xyzw`` is ``R_base_tcp``;
    * ``tcp_rotation_6d`` is derived from the same rotation (first two columns);
    * ``tcp_velocity`` is ``[vx,vy,vz,wx,wy,wz]`` in :data:`TCP_VELOCITY_FRAME`;
    * ``wrench_tcp`` is ``[Fx,Fy,Fz,Tx,Ty,Tz]`` at the TCP in :data:`WRENCH_FRAME`;
    * ``gripper_width`` is metres.
    """

    joint_position: np.ndarray
    joint_velocity: np.ndarray
    tcp_position: np.ndarray
    tcp_quaternion_xyzw: np.ndarray
    tcp_rotation_6d: np.ndarray
    tcp_velocity: np.ndarray
    wrench_tcp: np.ndarray
    gripper_width: np.ndarray
    tcp_velocity_frame: str = TCP_VELOCITY_FRAME
    wrench_frame: str = WRENCH_FRAME

    def __post_init__(self) -> None:
        for name, shape in ROBOT_STATE_SHAPES.items():
            value = getattr(self, name)
            object.__setattr__(self, name, _freeze_array(value, np.float64, shape, name))
        if self.tcp_velocity_frame != TCP_VELOCITY_FRAME:
            raise ValidationError(
                f"tcp_velocity_frame must be {TCP_VELOCITY_FRAME!r} in v1",
                code=ErrorCode.INVALID_FRAME,
                details={"frame": self.tcp_velocity_frame},
            )
        if self.wrench_frame != WRENCH_FRAME:
            raise ValidationError(
                f"wrench_frame must be {WRENCH_FRAME!r} in v1",
                code=ErrorCode.INVALID_FRAME,
                details={"frame": self.wrench_frame},
            )
        from .validation import validate_robot_state

        validate_robot_state(self)

    @property
    def pose(self) -> CartesianPose:
        return CartesianPose(self.tcp_position, self.tcp_quaternion_xyzw)

    @property
    def gripper_width_m(self) -> float:
        return float(self.gripper_width[0])


@dataclass(frozen=True)
class RobotStateSample:
    """Timestamped state stored in the ring buffer.

    ``host_time_ns`` is the monotonic receive time; ``device_time_ns`` is the
    raw controller timestamp.  Samples must be appended in strictly increasing
    host time order.
    """

    state: RobotState
    device_time_ns: int
    host_time_ns: int
    sequence_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.state, RobotState):
            raise ValidationError(
                "RobotStateSample.state must be a RobotState",
                code=ErrorCode.INVALID_SHAPE,
                details={"type": type(self.state).__name__},
            )
        if not isinstance(self.device_time_ns, int) or self.device_time_ns <= 0:
            raise ValidationError(
                "device_time_ns must be a positive integer",
                code=ErrorCode.VALUE_OUT_OF_RANGE,
            )
        if not isinstance(self.host_time_ns, int) or self.host_time_ns <= 0:
            raise ValidationError(
                "host_time_ns must be a positive integer",
                code=ErrorCode.VALUE_OUT_OF_RANGE,
            )
        if not isinstance(self.sequence_id, int) or self.sequence_id < 0:
            raise ValidationError(
                "sequence_id must be a non-negative integer",
                code=ErrorCode.VALUE_OUT_OF_RANGE,
            )


@dataclass(frozen=True)
class SyncInfo:
    """Provenance of an interpolated observation.

    ``robot_sample_before_ns`` / ``_after_ns`` are *host* times of the two
    samples used; ``interpolation_alpha`` is 0 when an exact sample matched.
    ``max_age_ns`` is the largest distance between the observation target time
    and the bracketing samples (a symmetric look-ahead/behind measure, as
    required by the plan's "maximum synchronisation error").
    """

    robot_sample_before_ns: int
    robot_sample_after_ns: int
    interpolation_alpha: float
    max_age_ns: int
    dropped_frames: Mapping[str, int] = field(default_factory=dict)
    interpolation_applied: bool = True
    camera_skew_ns: int = 0
    state_gap_ns: int = 0

    def __post_init__(self) -> None:
        before = self.robot_sample_before_ns
        after = self.robot_sample_after_ns
        if (
            not isinstance(before, int)
            or not isinstance(after, int)
            or before <= 0
            or after <= 0
            or before > after
        ):
            raise ValidationError(
                "sync sample times must be positive integers with before <= after",
                code=ErrorCode.INVALID_SYNC,
                details={"before": before, "after": after},
            )
        alpha = float(self.interpolation_alpha)
        if not np.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
            raise ValidationError(
                "interpolation_alpha must lie in [0, 1]",
                code=ErrorCode.INVALID_SYNC,
                details={"alpha": alpha},
            )
        if not isinstance(self.max_age_ns, int) or self.max_age_ns < 0:
            raise ValidationError(
                "max_age_ns must be a non-negative integer",
                code=ErrorCode.INVALID_SYNC,
            )
        dropped = {str(k): int(v) for k, v in dict(self.dropped_frames).items()}
        if any(v < 0 for v in dropped.values()):
            raise ValidationError(
                "dropped frame counters must be non-negative",
                code=ErrorCode.VALUE_OUT_OF_RANGE,
                details={"dropped_frames": dropped},
            )
        object.__setattr__(self, "dropped_frames", MappingProxyType(dropped))


@dataclass(frozen=True)
class CanonicalObservation:
    """Canonical, immutable observation handed to recorder and policy layers."""

    schema_version: str
    sequence_id: int
    capture_time_ns: int
    task: str
    images: Mapping[str, ImageFrame]
    robot: RobotState
    sync: SyncInfo

    def __post_init__(self) -> None:
        images = dict(self.images)
        for key, frame in images.items():
            if not isinstance(frame, ImageFrame):
                raise ValidationError(
                    f"images[{key!r}] must be an ImageFrame",
                    code=ErrorCode.INVALID_SHAPE,
                )
            if frame.name != key:
                raise ValidationError(
                    f"image key {key!r} does not match frame.name {frame.name!r}",
                    code=ErrorCode.INVALID_SYNC,
                )
            images[key] = frame.as_readonly()
        object.__setattr__(self, "images", MappingProxyType(images))
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValidationError(
                "task must be a non-empty string",
                code=ErrorCode.VALUE_OUT_OF_RANGE,
            )
        from .validation import validate_canonical_observation

        validate_canonical_observation(self)

    # -- convenience accessors ---------------------------------------------
    def image(self, key: str) -> ImageFrame:
        try:
            return self.images[key]
        except KeyError as exc:
            raise ValidationError(
                f"observation has no image {key!r}",
                code=ErrorCode.INVALID_SHAPE,
                details={"available": sorted(self.images)},
            ) from exc

    @property
    def base_rgb(self) -> ImageFrame:
        return self.image(IMAGE_KEY_BASE_RGB)

    @property
    def wrist_rgb(self) -> ImageFrame:
        return self.image(IMAGE_KEY_WRIST_RGB)

    @property
    def base_depth(self) -> Optional[ImageFrame]:
        return self.images.get(IMAGE_KEY_BASE_DEPTH)

    @property
    def wrist_depth(self) -> Optional[ImageFrame]:
        return self.images.get(IMAGE_KEY_WRIST_DEPTH)

    @property
    def schema(self) -> str:
        return self.schema_version
