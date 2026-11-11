"""Rigid-body geometry with explicitly documented conventions.

Conventions (also mirrored in ``docs/CONVENTIONS.md``):

* Quaternions are ``xyzw``, unit length, representing the orientation of the
  TCP *in the base frame* (``R_base_tcp``).  ``q`` and ``-q`` are the same
  rotation; :func:`normalize_quaternion` preserves the input sign, helpers that
  produce a rotation choose ``w >= 0`` canonically.
* ``rotation_6d`` is the first two **columns** of ``R_base_tcp`` concatenated:
  ``[R00, R10, R20, R01, R11, R21]``.  This is the convention used by common
  VLA code bases; it is tested, and it deliberately avoids quaternion sign
  flips.
* Rotation deltas are axis-angle rotation vectors in radians.
* ``frame="base"`` deltas are expressed in base axes; ``frame="tcp"`` deltas
  are expressed in the tool frame.
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

from .errors import ErrorCode, ValidationError

Frame = Literal["base", "tcp"]

_ROTATION_TOL = 1e-3
_QUATERNION_NORM_TOL = 1e-3
_EPS = 1e-12

__all__ = [
    "FRAME_BASE",
    "FRAME_TCP",
    "skew",
    "normalize_quaternion",
    "quaternion_to_matrix",
    "matrix_to_quaternion",
    "matrix_to_rotation_6d",
    "rotation_6d_to_matrix",
    "axis_angle_to_matrix",
    "matrix_to_axis_angle",
    "slerp",
    "rotation_angle_between",
    "rotation_vector_between",
    "compose_delta_pose",
    "validate_rotation_matrix",
]

FRAME_BASE: Frame = "base"
FRAME_TCP: Frame = "tcp"
FRAMES: Tuple[str, ...] = (FRAME_BASE, FRAME_TCP)


def _as_array(value: object, shape: Tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape:
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
    return arr


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def validate_rotation_matrix(matrix: object, *, tol: float = _ROTATION_TOL) -> np.ndarray:
    arr = _as_array(matrix, (3, 3), "rotation_matrix")
    if not np.allclose(arr.T @ arr, np.eye(3), atol=tol):
        raise ValidationError(
            "matrix is not orthonormal",
            code=ErrorCode.INVALID_QUATERNION,
            details={"reason": "not_orthonormal"},
        )
    if not np.isclose(np.linalg.det(arr), 1.0, atol=tol):
        raise ValidationError(
            "matrix determinant must be +1",
            code=ErrorCode.INVALID_QUATERNION,
            details={"det": float(np.linalg.det(arr))},
        )
    return arr


def normalize_quaternion(quaternion: object) -> np.ndarray:
    """Return a unit quaternion (xyzw); rejects zero / non-finite input."""
    arr = _as_array(quaternion, (4,), "quaternion_xyzw")
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        raise ValidationError(
            "quaternion norm is ~0",
            code=ErrorCode.INVALID_QUATERNION,
            details={"norm": norm},
        )
    return arr / norm


def quaternion_to_matrix(quaternion: object) -> np.ndarray:
    x, y, z, w = normalize_quaternion(quaternion)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion(matrix: object) -> np.ndarray:
    """Convert a rotation matrix to a unit xyzw quaternion with ``w >= 0``."""
    m = validate_rotation_matrix(matrix)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
        s = np.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] >= m[2, 2]:
        s = np.sqrt(max(0.0, 1.0 - m[0, 0] + m[1, 1] - m[2, 2])) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(max(0.0, 1.0 - m[0, 0] - m[1, 1] + m[2, 2])) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = normalize_quaternion(np.array([x, y, z, w], dtype=np.float64))
    if q[3] < 0:
        q = -q
    return q


def matrix_to_rotation_6d(matrix: object) -> np.ndarray:
    """First two **columns** of the rotation matrix, concatenated."""
    m = validate_rotation_matrix(matrix)
    return np.concatenate([m[:, 0], m[:, 1]])


def rotation_6d_to_matrix(value: object) -> np.ndarray:
    vec = _as_array(value, (6,), "rotation_6d")
    a1, a2 = vec[:3], vec[3:]
    n1 = float(np.linalg.norm(a1))
    if n1 < 1e-8:
        raise ValidationError(
            "rotation_6d first column is degenerate",
            code=ErrorCode.INVALID_QUATERNION,
            details={"reason": "degenerate_first_column"},
        )
    b1 = a1 / n1
    a2_orth = a2 - np.dot(b1, a2) * b1
    n2 = float(np.linalg.norm(a2_orth))
    if n2 < 1e-8:
        raise ValidationError(
            "rotation_6d second column is collinear with the first",
            code=ErrorCode.INVALID_QUATERNION,
            details={"reason": "degenerate_second_column"},
        )
    b2 = a2_orth / n2
    b3 = np.cross(b1, b2)
    return np.column_stack([b1, b2, b3])


def axis_angle_to_matrix(rotvec: object) -> np.ndarray:
    vec = _as_array(rotvec, (3,), "rotation_vector_rad")
    theta = float(np.linalg.norm(vec))
    if theta < 1e-10:
        # First-order approximation: R ~= I + skew(v).
        return np.eye(3) + skew(vec)
    axis = vec / theta
    k = skew(axis)
    r = np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)
    return r


def matrix_to_axis_angle(matrix: object) -> np.ndarray:
    m = validate_rotation_matrix(matrix)
    q = matrix_to_quaternion(m)
    w = float(np.clip(q[3], -1.0, 1.0))
    v_norm = float(np.linalg.norm(q[:3]))
    if v_norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(v_norm, w)
    return q[:3] / v_norm * angle


def slerp(q0: object, q1: object, t: float) -> np.ndarray:
    """Shortest-path spherical interpolation; ``t`` must be in ``[0, 1]``."""
    if not np.isfinite(t) or t < 0.0 or t > 1.0:
        raise ValidationError(
            "slerp t must lie in [0, 1]",
            code=ErrorCode.VALUE_OUT_OF_RANGE,
            details={"t": t},
        )
    a = normalize_quaternion(q0)
    b = normalize_quaternion(q1)
    dot = float(np.dot(a, b))
    if dot < 0.0:  # q and -q are the same rotation: walk the short arc.
        b = -b
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = a + t * (b - a)
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        s0 = np.sin(theta_0 - theta) / sin_theta_0
        s1 = np.sin(theta) / sin_theta_0
        result = s0 * a + s1 * b
    # Preserve the starting quaternion's sign for continuity, then normalize.
    result = normalize_quaternion(result)
    if np.dot(result, a) < 0.0:
        result = -result
    return result


def rotation_vector_between(q0: object, q1: object) -> np.ndarray:
    """Return the world/base-frame rotation vector taking q0 to q1.

    The result is the axis-angle vector ``axis * angle`` in radians.  It keeps
    direction/sign information, unlike ``rotation_angle_between``'s scalar
    magnitude, and is suitable for angular acceleration checks.
    """
    a = normalize_quaternion(q0)
    b = normalize_quaternion(q1)
    if float(np.dot(a, b)) < 0.0:
        b = -b
    ax, ay, az, aw = (float(v) for v in a)
    bx, by, bz, bw = (float(v) for v in b)
    # q_rel = b * conjugate(a) (xyzw convention)
    w = bw * aw + bx * ax + by * ay + bz * az
    x = -bw * ax + aw * bx - by * az + bz * ay
    y = -bw * ay + aw * by - bz * ax + bx * az
    z = -bw * az + aw * bz - bx * ay + by * ax
    norm = float(np.linalg.norm([w, x, y, z]))
    if norm < 1e-12:
        return np.zeros(3)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    angle = 2.0 * float(np.arccos(float(np.clip(w, -1.0, 1.0))))
    sin_half = float(np.sin(angle / 2.0))
    if angle < 1e-12 or abs(sin_half) < 1e-12:
        return np.zeros(3)
    axis = np.array([x, y, z], dtype=np.float64) / sin_half
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-12:
        return np.zeros(3)
    return axis / axis_norm * angle


def rotation_angle_between(q0: object, q1: object) -> float:
    a = normalize_quaternion(q0)
    b = normalize_quaternion(q1)
    dot = float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))
    return float(2.0 * np.arccos(dot))


def compose_delta_pose(
    position: object,
    quaternion_xyzw: object,
    delta_position_m: object,
    delta_rotation_rad: object,
    *,
    frame: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a 6-DoF delta to a pose expressed in the base frame.

    ``frame="base"``: translation in base axes, rotation left-multiplied.
    ``frame="tcp"``: translation in tool axes, rotation right-multiplied.
    The returned quaternion is canonicalized to ``w >= 0``.
    """
    if frame not in FRAMES:
        raise ValidationError(
            f"unknown action frame {frame!r}",
            code=ErrorCode.INVALID_FRAME,
            details={"frame": frame, "allowed": list(FRAMES)},
        )
    pos = _as_array(position, (3,), "tcp_position")
    quat = normalize_quaternion(quaternion_xyzw)
    dp = _as_array(delta_position_m, (3,), "delta_position_m")
    dr = _as_array(delta_rotation_rad, (3,), "delta_rotation_rad")
    r_current = quaternion_to_matrix(quat)
    r_delta = axis_angle_to_matrix(dr)
    if frame == FRAME_BASE:
        new_pos = pos + dp
        new_rot = r_delta @ r_current
    else:
        new_pos = pos + r_current @ dp
        new_rot = r_current @ r_delta
    new_quat = matrix_to_quaternion(new_rot)
    if not np.all(np.isfinite(new_pos)):  # pragma: no cover - defensive
        raise ValidationError("composed position is not finite", code=ErrorCode.NON_FINITE)
    return new_pos, new_quat
