"""Fake device behaviour and single-writer enforcement."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from vla_infra.clock import FakeClock
from vla_infra.contracts.pose import CartesianPose
from vla_infra.devices.fake import FakeCamera, FakeGripper, FakeRobotArm
from vla_infra.errors import DeviceError, ErrorCode, SingleWriterViolationError, ValidationError


def test_camera_pacing_and_frame_number() -> None:
    clock = FakeClock(start_ns=1_000_000_000)
    camera = FakeCamera("base_rgb", clock, width=8, height=6, fps=15)
    camera.start()
    first = camera.read()
    assert first is not None
    assert first.frame_number == 0
    assert first.data.shape == (6, 8, 3)
    assert camera.read() is None  # no new frame yet
    clock.advance(66_000_000)
    assert camera.read() is None
    clock.advance(1_000_000)
    second = camera.read()
    assert second is not None and second.frame_number == 1
    assert second.device_time_ns > first.device_time_ns
    assert camera.stats.frames_produced == 2
    assert camera.stats.duplicate_reads >= 1


def test_camera_fault_injection() -> None:
    clock = FakeClock(start_ns=1_000_000_000)
    camera = FakeCamera("wrist_rgb", clock)
    camera.start()
    camera.inject_failures(2)
    assert camera.read() is None
    assert camera.read() is None
    assert camera.read() is not None
    assert camera.stats.failures == 2


def test_camera_jitter_keeps_device_time_monotonic() -> None:
    clock = FakeClock(start_ns=1_000_000_000)
    camera = FakeCamera("base_rgb", clock, fps=30, jitter_ns=5_000_000)
    camera.start()
    timestamps = []
    for _ in range(10):
        frame = camera.read()
        if frame is not None:
            timestamps.append(frame.device_time_ns)
        clock.advance(34_000_000)
    assert len(timestamps) == 10
    assert all(b > a for a, b in zip(timestamps, timestamps[1:]))


def test_camera_depth_encoding() -> None:
    clock = FakeClock(start_ns=1_000_000_000)
    camera = FakeCamera("base_depth", clock, width=4, height=4, encoding="depth16")
    camera.start()
    frame = camera.read()
    assert frame is not None
    assert frame.encoding == "depth16"
    assert frame.data.dtype == np.uint16


def test_robot_state_is_monotonic_and_timestamped() -> None:
    clock = FakeClock(start_ns=1_000_000_000)
    robot = FakeRobotArm(clock)
    first = robot.read_state()
    second = robot.read_state()  # same clock tick: host time must still advance
    assert second.host_time_ns > first.host_time_ns
    assert second.device_time_ns > first.device_time_ns
    assert second.sequence_id == first.sequence_id + 1
    assert first.state.gripper_width.flags.writeable is False


def test_command_requires_claim() -> None:
    clock = FakeClock(start_ns=1_000_000_000)
    robot = FakeRobotArm(clock)
    pose = CartesianPose(np.array([0.5, 0.0, 0.3]), np.array([0.0, 0.0, 0.0, 1.0]))
    with pytest.raises(DeviceError) as exc:
        robot.command_pose(pose, epoch=1, now_ns=clock.now_ns())
    assert exc.value.code is ErrorCode.DEVICE_NOT_CONNECTED


def test_single_writer_violation_from_other_thread() -> None:
    clock = FakeClock(start_ns=1_000_000_000)
    robot = FakeRobotArm(clock)
    robot.claim_writer("executor")
    pose = CartesianPose(np.array([0.5, 0.0, 0.3]), np.array([0.0, 0.0, 0.0, 1.0]))
    robot.command_pose(pose, epoch=1, now_ns=clock.now_ns())
    errors: list[BaseException] = []

    def rogue() -> None:
        try:
            robot.command_pose(pose, epoch=1, now_ns=clock.now_ns())
        except BaseException as exc:  # noqa: BLE001 - captured for assertion
            errors.append(exc)

    thread = threading.Thread(target=rogue)
    thread.start()
    thread.join(timeout=5)
    assert len(errors) == 1
    assert isinstance(errors[0], SingleWriterViolationError)
    assert errors[0].code is ErrorCode.SINGLE_WRITER_VIOLATION


def test_other_thread_cannot_steal_claim() -> None:
    robot = FakeRobotArm(FakeClock(start_ns=1_000_000_000))
    robot.claim_writer("executor")
    errors: list[BaseException] = []

    def rogue() -> None:
        try:
            robot.claim_writer("thief")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=rogue)
    thread.start()
    thread.join(timeout=5)
    assert len(errors) == 1 and isinstance(errors[0], SingleWriterViolationError)


def test_claim_is_idempotent_for_same_thread() -> None:
    robot = FakeRobotArm(FakeClock(start_ns=1_000_000_000))
    robot.claim_writer("executor")
    robot.claim_writer("executor")
    robot.release_writer()
    robot.claim_writer("executor")


def test_gripper_normalized_mapping_and_limits() -> None:
    gripper = FakeGripper(width_min_m=0.0, width_max_m=0.1, initial_width_m=0.05)
    assert gripper.command_normalized(0.0) == pytest.approx(0.0)
    assert gripper.command_normalized(1.0) == pytest.approx(0.1)
    assert gripper.command_normalized(0.5) == pytest.approx(0.05)
    with pytest.raises(ValidationError) as exc:
        gripper.command_normalized(1.2)
    assert exc.value.code is ErrorCode.SAFETY_GRIPPER
    with pytest.raises(ValidationError):
        gripper.command_normalized(float("nan"))


def test_robot_command_updates_pose_and_logs() -> None:
    clock = FakeClock(start_ns=1_000_000_000)
    robot = FakeRobotArm(clock)
    robot.claim_writer("executor")
    target = robot.pose.apply_delta([0.01, 0.0, 0.0], [0.0, 0.0, 0.0], frame="base")
    robot.command_pose(target, epoch=7, now_ns=clock.now_ns())
    robot.command_gripper(0.25, epoch=7, now_ns=clock.now_ns())
    assert np.allclose(robot.pose.position, target.position)
    assert robot.write_count == 2
    assert robot.last_command is not None
    assert robot.last_command.kind == "gripper"
    assert robot.gripper.last_normalized_command == pytest.approx(0.25)
