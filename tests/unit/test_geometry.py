"""Geometry conventions and round-trip tests."""

from __future__ import annotations

import numpy as np
import pytest

from vla_infra.contracts.pose import CartesianPose, pose_from_rotvec, pose_to_rotvec
from vla_infra.errors import ErrorCode, ValidationError
from vla_infra.geometry import (
    axis_angle_to_matrix,
    compose_delta_pose,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    normalize_quaternion,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
    rotation_angle_between,
    slerp,
)


def test_quaternion_matrix_roundtrip() -> None:
    rng = np.random.default_rng(7)
    for _ in range(50):
        rotvec = rng.normal(size=3)
        rotvec = rotvec / np.linalg.norm(rotvec) * rng.uniform(0.0, np.pi - 1e-3)
        matrix = axis_angle_to_matrix(rotvec)
        quat = matrix_to_quaternion(matrix)
        assert quat[3] >= 0.0
        assert np.isclose(np.linalg.norm(quat), 1.0)
        assert np.allclose(quaternion_to_matrix(quat), matrix, atol=1e-10)


def test_q_and_minus_q_are_same_rotation() -> None:
    quat = normalize_quaternion([0.1, -0.2, 0.3, 0.9])
    assert np.allclose(quaternion_to_matrix(quat), quaternion_to_matrix(-quat))
    assert rotation_angle_between(quat, -quat) == pytest.approx(0.0, abs=1e-7)


def test_matrix_to_quaternion_canonical_sign() -> None:
    quat = matrix_to_quaternion(quaternion_to_matrix([0.0, 0.0, 1.0, 0.0]))
    assert quat[3] >= 0.0


def test_rotation_6d_uses_first_two_columns() -> None:
    angle = 0.3
    matrix = axis_angle_to_matrix([0.0, 0.0, angle])
    six = matrix_to_rotation_6d(matrix)
    assert np.allclose(six, [matrix[0, 0], matrix[1, 0], matrix[2, 0], matrix[0, 1], matrix[1, 1], matrix[2, 1]])
    assert np.allclose(rotation_6d_to_matrix(six), matrix, atol=1e-12)


def test_rotation_6d_rejects_degenerate_and_nonfinite() -> None:
    with pytest.raises(ValidationError) as exc:
        rotation_6d_to_matrix([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    assert exc.value.code is ErrorCode.INVALID_QUATERNION
    with pytest.raises(ValidationError):
        rotation_6d_to_matrix([1.0, 0.0, 0.0, np.nan, 0.0, 0.0])
    with pytest.raises(ValidationError):
        rotation_6d_to_matrix([1.0, 0.0, 0.0, 2.0, 0.0, 0.0])  # collinear


def test_axis_angle_roundtrip() -> None:
    for rotvec in (
        np.array([0.0, 0.0, 0.0]),
        np.array([1e-9, 0.0, 0.0]),
        np.array([0.2, -0.4, 0.1]),
        np.array([1.0, 0.0, 0.0]) * (np.pi - 1e-6),
    ):
        matrix = axis_angle_to_matrix(rotvec)
        recovered = matrix_to_axis_angle(matrix)
        assert np.allclose(axis_angle_to_matrix(recovered), matrix, atol=1e-9)


def test_slerp_endpoints_and_midpoint() -> None:
    q0 = [0.0, 0.0, 0.0, 1.0]
    half = np.deg2rad(45.0)
    q1 = [0.0, 0.0, np.sin(half), np.cos(half)]
    assert np.allclose(slerp(q0, q1, 0.0), q0)
    assert np.allclose(slerp(q0, q1, 1.0), q1)
    middle = slerp(q0, q1, 0.5)
    expected = [0.0, 0.0, np.sin(np.deg2rad(22.5)), np.cos(np.deg2rad(22.5))]
    assert np.allclose(np.abs(middle), np.abs(expected), atol=1e-12)


def test_slerp_takes_short_arc_and_is_normalized() -> None:
    q0 = [0.0, 0.0, 0.0, 1.0]
    q1 = -np.array([0.0, 0.0, np.sin(0.1), np.cos(0.1)])
    middle = slerp(q0, q1, 0.5)
    assert np.linalg.norm(middle) == pytest.approx(1.0)
    angle = rotation_angle_between(q0, q1)
    assert angle == pytest.approx(0.2, abs=1e-9)
    assert rotation_angle_between(q0, middle) == pytest.approx(0.1, abs=1e-9)


def test_slerp_rejects_out_of_range_t() -> None:
    for t in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValidationError):
            slerp([0, 0, 0, 1], [0, 0, 0, 1], t)


def test_invalid_quaternions_rejected() -> None:
    for bad in ([0.0, 0.0, 0.0, 0.0], [np.nan, 0.0, 0.0, 1.0], [1.0, 2.0]):
        with pytest.raises(ValidationError):
            normalize_quaternion(bad)


def test_compose_delta_pose_frame_semantics() -> None:
    # 90 degrees about Z: tool X points along base +Y.
    quat = [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]
    position = np.array([0.5, 0.0, 0.3])

    pos_base, _ = compose_delta_pose(position, quat, [0.1, 0.0, 0.0], [0.0, 0.0, 0.0], frame="base")
    assert np.allclose(pos_base, [0.6, 0.0, 0.3])

    pos_tcp, _ = compose_delta_pose(position, quat, [0.1, 0.0, 0.0], [0.0, 0.0, 0.0], frame="tcp")
    assert np.allclose(pos_tcp, [0.5, 0.1, 0.3], atol=1e-12)

    with pytest.raises(ValidationError) as exc:
        compose_delta_pose(position, quat, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], frame="world")
    assert exc.value.code is ErrorCode.INVALID_FRAME


def test_pose_validation_and_interpolation() -> None:
    pose = CartesianPose(np.array([0.5, 0.0, 0.3]), np.array([0.0, 0.0, 0.0, 1.0]))
    other = pose.apply_delta([0.1, 0.0, 0.0], [0.0, 0.0, np.pi / 2], frame="base")
    middle = pose.interpolate(other, 0.5)
    assert np.allclose(middle.position, [0.55, 0.0, 0.3])
    assert rotation_angle_between(pose.quaternion_xyzw, middle.quaternion_xyzw) == pytest.approx(np.pi / 4, abs=1e-9)
    assert pose.translation_distance(other) == pytest.approx(0.1)

    with pytest.raises(ValidationError):
        CartesianPose(np.array([0.5, 0.0, 0.3]), np.array([0.0, 0.0, 0.0, 0.0]))
    with pytest.raises(ValidationError):
        CartesianPose(np.array([0.5, 0.0, 0.3]), np.array([0.0, 0.0, 0.0, 1.2]))
    with pytest.raises(ValidationError):
        CartesianPose(np.array([np.nan, 0.0, 0.3]), np.array([0.0, 0.0, 0.0, 1.0]))


def test_pose_rotvec_roundtrip() -> None:
    pose = pose_from_rotvec([0.4, 0.1, 0.2], [0.2, -0.1, 0.3])
    recovered = pose_to_rotvec(pose)
    assert np.allclose(axis_angle_to_matrix(recovered), pose.rotation_matrix, atol=1e-12)
    assert pose.position.flags.writeable is False
