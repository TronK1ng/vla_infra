"""Cartesian pose value type shared by robot state, safety and execution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import (
    FRAMES,
    compose_delta_pose,
    matrix_to_rotation_6d,
    quaternion_to_matrix,
    slerp,
)
from ..errors import ErrorCode, ValidationError


def _freeze(value: np.ndarray) -> np.ndarray:
    arr = np.array(value, dtype=np.float64, copy=True)
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class CartesianPose:
    """TCP pose in the base frame.

    ``position`` is metres; ``quaternion_xyzw`` is the orientation of the TCP in
    the base frame.  Input quaternions are normalised, but a norm error larger
    than 1e-3 is rejected rather than silently corrected.
    """

    position: np.ndarray
    quaternion_xyzw: np.ndarray
    frame: str = "base"

    def __post_init__(self) -> None:
        pos = np.asarray(self.position, dtype=np.float64)
        quat = np.asarray(self.quaternion_xyzw, dtype=np.float64)
        if pos.shape != (3,):
            raise ValidationError(
                f"position must be (3,), got {pos.shape}",
                code=ErrorCode.INVALID_SHAPE,
                details={"field": "position", "shape": list(pos.shape)},
            )
        if quat.shape != (4,):
            raise ValidationError(
                f"quaternion must be (4,), got {quat.shape}",
                code=ErrorCode.INVALID_SHAPE,
                details={"field": "quaternion_xyzw", "shape": list(quat.shape)},
            )
        if self.frame not in FRAMES:
            raise ValidationError(
                f"unsupported pose frame {self.frame!r}",
                code=ErrorCode.INVALID_FRAME,
                details={"frame": self.frame},
            )
        if not np.all(np.isfinite(pos)):
            raise ValidationError("position contains NaN/Inf", code=ErrorCode.NON_FINITE)
        if not np.all(np.isfinite(quat)):
            raise ValidationError("quaternion contains NaN/Inf", code=ErrorCode.NON_FINITE)
        norm = float(np.linalg.norm(quat))
        if norm < 1e-8 or abs(norm - 1.0) > 1e-3:
            raise ValidationError(
                "quaternion must be unit length (tolerance 1e-3)",
                code=ErrorCode.INVALID_QUATERNION,
                details={"norm": norm},
            )
        object.__setattr__(self, "position", _freeze(pos))
        object.__setattr__(self, "quaternion_xyzw", _freeze(quat / norm))

    # -- derived values -----------------------------------------------------
    @property
    def rotation_matrix(self) -> np.ndarray:
        return quaternion_to_matrix(self.quaternion_xyzw)

    @property
    def rotation_6d(self) -> np.ndarray:
        return matrix_to_rotation_6d(self.rotation_matrix)

    def with_position(self, position: np.ndarray) -> "CartesianPose":
        return CartesianPose(np.asarray(position, dtype=np.float64), self.quaternion_xyzw, self.frame)

    def with_quaternion(self, quaternion_xyzw: np.ndarray) -> "CartesianPose":
        return CartesianPose(self.position, quaternion_xyzw, self.frame)

    def apply_delta(
        self,
        delta_position_m: np.ndarray,
        delta_rotation_rad: np.ndarray,
        *,
        frame: str,
    ) -> "CartesianPose":
        pos, quat = compose_delta_pose(
            self.position,
            self.quaternion_xyzw,
            delta_position_m,
            delta_rotation_rad,
            frame=frame,
        )
        return CartesianPose(pos, quat, frame=self.frame)

    def interpolate(self, other: "CartesianPose", alpha: float) -> "CartesianPose":
        if other.frame != self.frame:
            raise ValidationError(
                "cannot interpolate poses in different frames",
                code=ErrorCode.INVALID_FRAME,
                details={"from": self.frame, "to": other.frame},
            )
        alpha = float(np.clip(alpha, 0.0, 1.0))
        pos = (1.0 - alpha) * self.position + alpha * other.position
        quat = slerp(self.quaternion_xyzw, other.quaternion_xyzw, alpha)
        return CartesianPose(pos, quat, frame=self.frame)

    def translation_distance(self, other: "CartesianPose") -> float:
        return float(np.linalg.norm(other.position - self.position))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        p = np.round(self.position, 6).tolist()
        q = np.round(self.quaternion_xyzw, 6).tolist()
        return f"CartesianPose(position={p}, quaternion_xyzw={q}, frame={self.frame!r})"


def pose_from_rotvec(position: np.ndarray, rotvec: np.ndarray, *, frame: str = "base") -> CartesianPose:
    """Build a pose from an absolute rotation vector (axis-angle)."""
    from ..geometry import axis_angle_to_matrix, matrix_to_quaternion

    r = axis_angle_to_matrix(rotvec)
    return CartesianPose(np.asarray(position), matrix_to_quaternion(r), frame=frame)


def pose_to_rotvec(pose: CartesianPose) -> np.ndarray:
    from ..geometry import matrix_to_axis_angle

    return matrix_to_axis_angle(pose.rotation_matrix)
