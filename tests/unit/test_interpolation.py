"""Cartesian interpolation: SLERP, velocity stretching, no start jump."""

from __future__ import annotations

import numpy as np
import pytest

from vla_infra.contracts.pose import CartesianPose
from vla_infra.errors import SafetyViolation
from vla_infra.execution.interpolation import build_plan
from vla_infra.execution.safety import CommandTarget, SafetyGate
from vla_infra.testing import make_chunk, make_pose, make_robot_state, make_safety_profile

START = 1_000_000_000


def _targets(positions, grippers=None):
    grippers = grippers or [0.5] * len(positions)
    return [
        CommandTarget(pose=make_pose(position=position), gripper_normalized=gripper)
        for position, gripper in zip(positions, grippers)
    ]


def test_plan_starts_at_measured_pose() -> None:
    start_pose = make_pose()
    targets = _targets([(0.51, 0.0, 0.3), (0.52, 0.0, 0.3)])
    plan = build_plan(
        targets,
        start_pose=start_pose,
        start_gripper=0.5,
        start_time_ns=START,
        action_dt_s=0.05,
        profile=make_safety_profile(),
    )
    first = plan.sample(START)
    assert np.allclose(first.target.pose.position, start_pose.position)
    assert first.alpha_in_segment == 0.0
    middle = plan.sample(START + 25_000_000)
    assert np.allclose(middle.target.pose.position, [0.505, 0.0, 0.3], atol=1e-12)
    end = plan.sample(plan.end_time_ns)
    assert np.allclose(end.target.pose.position, [0.52, 0.0, 0.3], atol=1e-12)
    assert plan.finished(plan.end_time_ns)
    assert not plan.finished(plan.end_time_ns - 1)


def test_orientation_is_slerped() -> None:
    start_pose = CartesianPose(np.array([0.5, 0.0, 0.3]), np.array([0.0, 0.0, 0.0, 1.0]))
    target_pose = start_pose.apply_delta([0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2], frame="base")
    plan = build_plan(
        [CommandTarget(pose=target_pose, gripper_normalized=0.5)],
        start_pose=start_pose,
        start_gripper=0.5,
        start_time_ns=START,
        action_dt_s=0.05,
        profile=make_safety_profile(),
    )
    middle = plan.sample(START + plan.durations_ns[0] // 2)
    assert np.allclose(
        np.abs(middle.target.pose.quaternion_xyzw),
        np.abs([0.0, 0.0, np.sin(np.pi / 8), np.cos(np.pi / 8)]),
        atol=1e-9,
    )


def test_segment_duration_is_not_silently_retimed() -> None:
    profile = make_safety_profile(max_linear_velocity_m_s=0.01)
    plan = build_plan(
        _targets([(0.51, 0.0, 0.3)]),
        start_pose=make_pose(),
        start_gripper=0.5,
        start_time_ns=START,
        action_dt_s=0.05,
        profile=profile,
    )
    # Leader decision: policy macro timing is not silently stretched.  The
    # safety gate rejects the infeasible nominal speed instead.
    assert plan.durations_ns[0] == 50_000_000
    assert plan.total_duration_ns == 50_000_000
    with pytest.raises(SafetyViolation):
        SafetyGate(profile).check_chunk(
            make_chunk(observation_capture_time_ns=START),
            make_robot_state(),
            received_time_ns=START,
            now_ns=START,
        )


def test_gripper_is_interpolated() -> None:
    plan = build_plan(
        _targets([(0.51, 0.0, 0.3)], grippers=[1.0]),
        start_pose=make_pose(),
        start_gripper=0.0,
        start_time_ns=START,
        action_dt_s=0.05,
        profile=make_safety_profile(),
    )
    assert plan.sample(START).target.gripper_normalized == pytest.approx(0.0)
    assert plan.sample(START + 25_000_000).target.gripper_normalized == pytest.approx(0.5)
    assert plan.sample(plan.end_time_ns).target.gripper_normalized == pytest.approx(1.0)


def test_empty_plan_rejected() -> None:
    with pytest.raises(ValueError):
        build_plan(
            [],
            start_pose=make_pose(),
            start_gripper=0.5,
            start_time_ns=START,
            action_dt_s=0.05,
            profile=make_safety_profile(),
        )
