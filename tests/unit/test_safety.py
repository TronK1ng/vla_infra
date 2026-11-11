"""Provisional safety gate: every dangerous condition must be rejected."""

from __future__ import annotations

import numpy as np
import pytest

from vla_infra.errors import ErrorCode, SafetyViolation
from vla_infra.execution.safety import CommandTarget, SafetyGate
from vla_infra.testing import make_chunk, make_pose, make_robot_state, make_safety_profile

NOW = 1_000_000_000


def _gate(**overrides) -> SafetyGate:
    return SafetyGate(make_safety_profile(**overrides))


def _check(gate: SafetyGate, chunk, *, now=NOW, received=None, state=None):
    return gate.check_chunk(
        chunk,
        state or make_robot_state(),
        received_time_ns=received if received is not None else now,
        now_ns=now,
    )


def test_profile_validation_rejects_dangerous_config() -> None:
    with pytest.raises(SafetyViolation):
        make_safety_profile(workspace_min=(0.5, -0.5, 0.0), workspace_max=(0.4, 0.5, 0.8))
    with pytest.raises(SafetyViolation):
        make_safety_profile(tcp_height_max_m=0.9)  # outside workspace z (0.8)
    with pytest.raises(SafetyViolation):
        make_safety_profile(max_step_translation_m=-0.01)
    with pytest.raises(SafetyViolation):
        make_safety_profile(max_linear_velocity_m_s=0.0)
    with pytest.raises(SafetyViolation):
        make_safety_profile(allowed_action_frames=("world",))
    with pytest.raises(SafetyViolation):
        make_safety_profile(joint_position_min_rad=(-1.0,) * 7)  # max missing
    with pytest.raises(SafetyViolation):
        make_safety_profile(gripper_width_min_m=0.1, gripper_width_max_m=0.1)


def test_valid_chunk_produces_targets() -> None:
    gate = _gate()
    targets = _check(gate, make_chunk())
    assert len(targets) == 4
    assert targets[0].step_index == 0
    assert targets[0].pose.position[0] == pytest.approx(0.501)


def test_measured_state_force_and_torque_limits() -> None:
    gate = _gate(max_force_n=5.0, max_torque_nm=1.0)
    with pytest.raises(SafetyViolation) as exc:
        gate.check_state(make_robot_state(wrench=(6, 0, 0, 0, 0, 0)))
    assert exc.value.code is ErrorCode.SAFETY_FORCE
    with pytest.raises(SafetyViolation) as exc:
        gate.check_state(make_robot_state(wrench=(0, 0, 0, 0, 2, 0)))
    assert exc.value.code is ErrorCode.SAFETY_TORQUE


def test_measured_gripper_width_bounds() -> None:
    gate = _gate()
    with pytest.raises(SafetyViolation) as exc:
        gate.check_state(make_robot_state(gripper_width=0.2))
    assert exc.value.code is ErrorCode.SAFETY_GRIPPER


def test_joint_limits_when_configured() -> None:
    gate = _gate(joint_position_min_rad=(-0.1,) * 7, joint_position_max_rad=(0.1,) * 7)
    gate.check_state(make_robot_state())
    bad = make_robot_state(joints=[0.5, 0, 0, 0, 0, 0, 0])
    with pytest.raises(SafetyViolation):
        gate.check_state(bad)


def test_frame_and_type_allow_list() -> None:
    gate = _gate(allowed_action_frames=("base",), allowed_action_types=("delta_tcp",))
    with pytest.raises(SafetyViolation) as exc:
        _check(gate, make_chunk(action_frame="tcp"))
    assert exc.value.code is ErrorCode.INVALID_FRAME
    with pytest.raises(SafetyViolation):
        _check(gate, make_chunk(action_type="absolute_tcp"))


def test_absolute_actions_in_tcp_frame_are_rejected() -> None:
    gate = _gate()
    with pytest.raises(SafetyViolation) as exc:
        _check(gate, make_chunk(action_type="absolute_tcp", action_frame="tcp"))
    assert exc.value.code is ErrorCode.INVALID_FRAME


def test_stale_observation_rejected() -> None:
    gate = _gate(max_observation_age_ns=100_000_000)
    chunk = make_chunk(observation_capture_time_ns=NOW)
    with pytest.raises(SafetyViolation) as exc:
        _check(gate, chunk, now=NOW + 200_000_000)
    assert exc.value.code is ErrorCode.STALE_OBSERVATION
    with pytest.raises(SafetyViolation) as exc:
        _check(gate, chunk, now=NOW - 10_000_000_000)
    assert "future" in exc.value.message.lower()


def test_stale_chunk_rejected() -> None:
    gate = _gate(max_chunk_age_ns=100_000_000)
    with pytest.raises(SafetyViolation) as exc:
        _check(gate, make_chunk(), now=NOW, received=NOW - 200_000_000)
    assert exc.value.code is ErrorCode.CHUNK_EXPIRED


def test_first_target_jump_rejected() -> None:
    gate = _gate(max_step_translation_m=1.0, max_first_target_translation_m=0.05)
    with pytest.raises(SafetyViolation) as exc:
        _check(gate, make_chunk(delta_m=(0.1, 0.0, 0.0)))
    assert exc.value.code is ErrorCode.SAFETY_FIRST_TARGET


def test_first_target_rotation_rejected() -> None:
    gate = _gate(
        max_step_rotation_rad=1.0,
        max_first_target_translation_m=1.0,
        max_first_target_rotation_rad=0.05,
    )
    actions = np.zeros((2, 7), dtype=np.float32)
    actions[:, 3] = 0.1
    with pytest.raises(SafetyViolation) as exc:
        _check(gate, make_chunk(actions=actions))
    assert exc.value.code is ErrorCode.SAFETY_FIRST_TARGET


def test_per_step_translation_and_rotation_limits() -> None:
    gate = _gate(max_step_translation_m=0.005, max_first_target_translation_m=1.0)
    with pytest.raises(SafetyViolation) as exc:
        _check(gate, make_chunk(delta_m=(0.02, 0.0, 0.0)))
    assert exc.value.code is ErrorCode.SAFETY_STEP
    gate = _gate(max_step_rotation_rad=0.01, max_first_target_rotation_rad=1.0)
    actions = np.zeros((2, 7), dtype=np.float32)
    actions[:, 3] = 0.02
    with pytest.raises(SafetyViolation) as exc:
        _check(gate, make_chunk(actions=actions))
    assert exc.value.code is ErrorCode.SAFETY_STEP


def test_implied_velocity_limits() -> None:
    gate = _gate(max_step_translation_m=1.0, max_first_target_translation_m=1.0, max_linear_velocity_m_s=0.1)
    with pytest.raises(SafetyViolation) as exc:
        _check(gate, make_chunk(delta_m=(0.02, 0.0, 0.0), action_dt_s=0.05))
    assert exc.value.code is ErrorCode.SAFETY_VELOCITY

    actions = np.zeros((2, 7), dtype=np.float32)
    actions[:, 3] = 0.1
    gate = _gate(
        max_step_rotation_rad=1.0,
        max_first_target_rotation_rad=1.0,
        max_angular_velocity_rad_s=1.0,
    )
    with pytest.raises(SafetyViolation) as exc:
        _check(gate, make_chunk(actions=actions, action_dt_s=0.05))
    assert exc.value.code is ErrorCode.SAFETY_VELOCITY


def test_workspace_and_height_limits() -> None:
    gate = _gate(
        max_step_translation_m=1.0,
        max_first_target_translation_m=1.0,
        max_linear_velocity_m_s=10.0,
        workspace_max=(0.51, 0.5, 0.8),
        tcp_height_min_m=0.02,
        tcp_height_max_m=0.75,
    )
    with pytest.raises(SafetyViolation) as exc:
        _check(gate, make_chunk(delta_m=(0.05, 0.0, 0.0)))
    assert exc.value.code is ErrorCode.SAFETY_WORKSPACE

    gate = _gate(
        max_step_translation_m=1.0,
        max_first_target_translation_m=1.0,
        max_linear_velocity_m_s=10.0,
    )
    actions = np.zeros((1, 7), dtype=np.float32)
    actions[0, 2] = -0.3
    with pytest.raises(SafetyViolation) as exc:
        _check(gate, make_chunk(actions=actions))
    assert exc.value.code is ErrorCode.SAFETY_HEIGHT


def test_check_tick_limits() -> None:
    gate = _gate(max_step_translation_m=0.05)
    previous = CommandTarget(pose=make_pose(), gripper_normalized=0.5)
    target = CommandTarget(pose=make_pose(position=(0.6, 0.0, 0.3)), gripper_normalized=0.5)
    with pytest.raises(SafetyViolation) as exc:
        gate.check_tick(previous, target, 10_000_000)
    assert exc.value.code is ErrorCode.SAFETY_STEP

    gate = _gate(max_step_translation_m=1.0, max_linear_velocity_m_s=0.1)
    near = CommandTarget(pose=make_pose(position=(0.55, 0.0, 0.3)), gripper_normalized=0.5)
    with pytest.raises(SafetyViolation) as exc:
        gate.check_tick(previous, near, 10_000_000)
    assert exc.value.code is ErrorCode.SAFETY_VELOCITY

    with pytest.raises(SafetyViolation):
        gate.check_tick(previous, near, 0)

    with pytest.raises(SafetyViolation):
        CommandTarget(pose=make_pose(), gripper_normalized=1.5)


def test_absolute_actions_map_to_absolute_targets() -> None:
    actions = np.zeros((2, 7), dtype=np.float32)
    actions[:, 0] = [0.501, 0.502]
    actions[:, 2] = 0.31
    targets = _check(_gate(), make_chunk(actions=actions, action_type="absolute_tcp"))
    assert targets[0].pose.position.tolist() == pytest.approx([0.501, 0.0, 0.31])
    assert targets[1].pose.position.tolist() == pytest.approx([0.502, 0.0, 0.31])


def test_gripper_values_are_not_clipped() -> None:
    actions = np.zeros((2, 7), dtype=np.float32)
    actions[:, 6] = 0.9
    # The chunk itself is valid; the safety gate preserves the value unchanged.
    targets = _check(_gate(), make_chunk(actions=actions))
    assert all(target.gripper_normalized == pytest.approx(0.9) for target in targets)


def test_chunk_acceleration_limit() -> None:
    gate = _gate(max_linear_acceleration_m_s2=0.1, max_angular_acceleration_rad_s2=0.1)
    chunk = make_chunk(delta_m=(0.01, 0.0, 0.0), action_dt_s=0.05)
    with pytest.raises(SafetyViolation) as exc:
        _check(gate, chunk)
    assert exc.value.code is ErrorCode.SAFETY_ACCELERATION


def test_tick_acceleration_limit() -> None:
    gate = _gate(max_linear_acceleration_m_s2=0.5)
    previous = CommandTarget(pose=make_pose(position=(0.5, 0.0, 0.3)), gripper_normalized=0.5)
    target = CommandTarget(pose=make_pose(position=(0.501, 0.0, 0.3)), gripper_normalized=0.5)
    with pytest.raises(SafetyViolation) as exc:
        gate.check_tick(
            previous,
            target,
            1_000_000,
            previous_linear_velocity_m_s=0.0,
        )
    assert exc.value.code is ErrorCode.SAFETY_ACCELERATION


def test_measured_joint_velocity_limit() -> None:
    gate = _gate(joint_velocity_max_rad_s=(0.1,) * 7)
    state = make_robot_state()
    object.__setattr__(state, "joint_velocity", np.array([0.2, 0, 0, 0, 0, 0, 0]))
    with pytest.raises(SafetyViolation) as exc:
        gate.check_state(state)
    assert exc.value.code is ErrorCode.SAFETY_VELOCITY


def _custom_chunk(rows, *, dt=0.05):
    return make_chunk(actions=np.asarray(rows, dtype=np.float32), action_dt_s=dt)


def test_chunk_acceleration_keeps_vector_direction() -> None:
    gate = _gate(
        max_linear_acceleration_m_s2=1.0,
        max_step_translation_m=0.05,
        max_first_target_translation_m=0.1,
        max_linear_velocity_m_s=1.0,
    )
    reverse = np.zeros((2, 7), dtype=np.float32)
    reverse[0, 0] = 0.01
    reverse[1, 0] = -0.01
    with pytest.raises(SafetyViolation) as exc:
        _check(gate, _custom_chunk(reverse))
    assert exc.value.code is ErrorCode.SAFETY_ACCELERATION

    axis = np.zeros((2, 7), dtype=np.float32)
    axis[0, 0] = 0.01
    axis[1, 1] = 0.01
    with pytest.raises(SafetyViolation) as exc:
        _check(gate, _custom_chunk(axis))
    assert exc.value.code is ErrorCode.SAFETY_ACCELERATION


def test_chunk_angular_acceleration_keeps_axis_and_sign() -> None:
    gate = _gate(
        max_angular_acceleration_rad_s2=0.5,
        max_step_rotation_rad=0.05,
        max_first_target_rotation_rad=0.1,
        max_angular_velocity_rad_s=1.0,
    )
    reverse = np.zeros((2, 7), dtype=np.float32)
    reverse[0, 3] = 0.01
    reverse[1, 3] = -0.01
    with pytest.raises(SafetyViolation) as exc:
        _check(gate, _custom_chunk(reverse))
    assert exc.value.code is ErrorCode.SAFETY_ACCELERATION

    axis = np.zeros((2, 7), dtype=np.float32)
    axis[0, 3] = 0.01
    axis[1, 4] = 0.01
    with pytest.raises(SafetyViolation) as exc:
        _check(gate, _custom_chunk(axis))
    assert exc.value.code is ErrorCode.SAFETY_ACCELERATION


def test_tick_vector_acceleration_direction_and_axis() -> None:
    gate = _gate(max_linear_acceleration_m_s2=1.0)
    previous = CommandTarget(pose=make_pose(position=(0.5, 0.0, 0.3)), gripper_normalized=0.5)
    target = CommandTarget(pose=make_pose(position=(0.501, 0.0, 0.3)), gripper_normalized=0.5)
    with pytest.raises(SafetyViolation) as exc:
        gate.check_tick(
            previous,
            target,
            1_000_000,
            previous_linear_velocity=np.array([-1.0, 0.0, 0.0]),
        )
    assert exc.value.code is ErrorCode.SAFETY_ACCELERATION
    with pytest.raises(SafetyViolation) as exc:
        gate.check_tick(
            previous,
            target,
            1_000_000,
            previous_linear_velocity=np.array([0.0, 1.0, 0.0]),
        )
    assert exc.value.code is ErrorCode.SAFETY_ACCELERATION


def test_tick_angular_vector_acceleration_direction() -> None:
    gate = _gate(max_angular_acceleration_rad_s2=0.5, max_angular_velocity_rad_s=100.0)
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    q1 = np.array([0.0, 0.0, np.sin(0.005), np.cos(0.005)])
    previous = CommandTarget(pose=make_pose(quaternion=q0), gripper_normalized=0.5)
    target = CommandTarget(pose=make_pose(quaternion=q1), gripper_normalized=0.5)
    with pytest.raises(SafetyViolation) as exc:
        gate.check_tick(
            previous,
            target,
            1_000_000,
            previous_angular_velocity=np.array([0.0, 0.0, -1.0]),
        )
    assert exc.value.code is ErrorCode.SAFETY_ACCELERATION
