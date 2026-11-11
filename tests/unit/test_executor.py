"""Executor: single writer, HOLD gating, safety rejection, plan expiry."""

from __future__ import annotations

import threading

import pytest

from vla_infra.clock import FakeClock
from vla_infra.devices.fake import FakeRobotArm
from vla_infra.errors import ErrorCode, NotArmedError, SingleWriterViolationError
from vla_infra.execution.authority import ControlAuthority
from vla_infra.execution.executor import Executor, ExecutorEvent, ExecutorEventKind
from vla_infra.execution.inbox import ActionInbox
from vla_infra.execution.safety import SafetyGate
from vla_infra.execution.supervisor import Supervisor, SystemState
from vla_infra.testing import make_chunk, make_robot_sample, make_robot_state, make_safety_profile

NOW = 1_000_000_000


class Harness:
    def __init__(self, **profile_overrides):
        self.clock = FakeClock(start_ns=NOW)
        self.robot = FakeRobotArm(self.clock)
        self.profile = make_safety_profile(**profile_overrides)
        self.authority = ControlAuthority()
        self.inbox = ActionInbox(self.authority)
        self.events = []
        self.executor = Executor(
            clock=self.clock,
            robot=self.robot,
            state_provider=self.robot.read_state,
            authority=self.authority,
            inbox=self.inbox,
            safety=SafetyGate(self.profile),
            tick_period_ns=10_000_000,
            plan_grace_ns=250_000_000,
            on_event=self.events.append,
        )
        self.executor.start()

    def arm(self, *, running: bool = True) -> int:
        epoch = self.authority.grant()
        if running:
            self.authority.activate()
        return epoch

    def submit(self, chunk, epoch):
        return self.inbox.submit(chunk, epoch=epoch, received_time_ns=self.clock.now_ns())

    def step(self, count: int = 1):
        report = None
        for _ in range(count):
            report = self.executor.tick(self.clock.now_ns())
            self.clock.advance(10_000_000)
        return report


def test_unarmed_executor_never_writes() -> None:
    harness = Harness()
    harness.step(5)
    assert harness.robot.write_count == 0
    assert harness.executor.stats.held_ticks == 5


def test_armed_but_not_running_never_writes() -> None:
    harness = Harness()
    epoch = harness.arm(running=False)
    harness.submit(make_chunk(observation_capture_time_ns=NOW), epoch)
    harness.step(5)
    assert harness.robot.write_count == 0


def test_running_executor_writes_and_completes_plan() -> None:
    harness = Harness()
    epoch = harness.arm()
    chunk = make_chunk(observation_capture_time_ns=NOW, delta_m=(0.002, 0.0, 0.0), horizon=4)
    assert harness.submit(chunk, epoch)
    harness.step(30)
    assert harness.robot.write_count > 0
    assert harness.executor.stats.plans_started == 1
    assert harness.executor.stats.plans_completed == 1
    assert harness.robot.pose.position[0] > 0.5


def test_hold_stops_writes_mid_plan() -> None:
    harness = Harness()
    epoch = harness.arm()
    harness.submit(make_chunk(observation_capture_time_ns=NOW, delta_m=(0.002, 0.0, 0.0), horizon=20), epoch)
    harness.step(3)
    writes_before = harness.robot.write_count
    assert writes_before > 0
    harness.authority.revoke("hold")
    harness.step(5)
    assert harness.robot.write_count == writes_before
    assert harness.executor.stats.held_ticks >= 1


def test_old_epoch_inbox_entry_is_not_written() -> None:
    harness = Harness()
    epoch = harness.arm()
    harness.submit(make_chunk(observation_capture_time_ns=NOW), epoch)
    harness.authority.revoke("hold")
    harness.step(3)
    assert harness.robot.write_count == 0


def test_safety_rejection_produces_event_and_no_write() -> None:
    harness = Harness(max_first_target_translation_m=0.01, max_step_translation_m=0.5)
    epoch = harness.arm()
    harness.submit(make_chunk(observation_capture_time_ns=NOW, delta_m=(0.2, 0.0, 0.0)), epoch)
    report = harness.executor.tick(harness.clock.now_ns())
    assert report.kind == "rejected"
    assert harness.robot.write_count == 0
    assert any(event.kind is ExecutorEventKind.PLAN_REJECTED for event in harness.events)
    rejected = [event for event in harness.events if event.kind is ExecutorEventKind.PLAN_REJECTED]
    assert rejected[0].error_code is ErrorCode.SAFETY_FIRST_TARGET


def test_stale_chunk_is_never_consumed() -> None:
    harness = Harness(max_chunk_age_ns=100_000_000)
    epoch = harness.arm()
    stale_chunk = make_chunk(observation_capture_time_ns=NOW)
    harness.inbox.submit(stale_chunk, epoch=epoch, received_time_ns=1)
    report = harness.executor.tick(harness.clock.now_ns())
    assert report.kind == "idle"
    assert harness.robot.write_count == 0


def test_plan_expiry_when_loop_stalls() -> None:
    harness = Harness()
    epoch = harness.arm()
    harness.submit(
        make_chunk(observation_capture_time_ns=NOW, delta_m=(0.002, 0.0, 0.0), horizon=1),
        epoch,
    )
    harness.executor.tick(harness.clock.now_ns())
    writes = harness.robot.write_count
    harness.clock.advance(2_000_000_000)
    report = harness.executor.tick(harness.clock.now_ns())
    assert report.kind == "expired"
    assert harness.robot.write_count == writes
    assert any(event.kind is ExecutorEventKind.PLAN_EXPIRED for event in harness.events)


def test_single_writer_violation_from_foreign_thread() -> None:
    harness = Harness()
    pose = harness.robot.pose
    errors: list[BaseException] = []

    def rogue() -> None:
        try:
            harness.robot.command_pose(pose, epoch=0, now_ns=NOW)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=rogue)
    thread.start()
    thread.join(timeout=5)
    assert len(errors) == 1
    assert isinstance(errors[0], SingleWriterViolationError)


def test_stale_state_blocks_plan_start() -> None:
    harness = Harness()
    frozen = harness.robot.read_state()
    harness.clock.advance(5_000_000_000)
    harness.arm()
    # The fake robot always returns a fresh sample; simulate a frozen provider.
    harness.executor._state_provider = lambda: frozen
    harness.submit(make_chunk(observation_capture_time_ns=harness.clock.now_ns()), harness.authority.epoch)
    report = harness.executor.tick(harness.clock.now_ns())
    assert report.kind == "stale_state"
    assert harness.robot.write_count == 0


class _ExplodingPoseRobot(FakeRobotArm):
    def __init__(self, clock):
        super().__init__(clock)
        self.pose_attempts = 0

    def command_pose(self, pose, *, epoch, now_ns):
        self.pose_attempts += 1
        raise RuntimeError("simulated pose backend failure")


class _ExplodingGripperRobot(FakeRobotArm):
    def __init__(self, clock):
        super().__init__(clock)
        self.gripper_attempts = 0

    def command_gripper(self, normalized, *, epoch, now_ns):
        self.gripper_attempts += 1
        raise RuntimeError("simulated gripper backend failure")


def _executor_with(robot, *, on_event):
    clock = FakeClock(start_ns=NOW)
    authority = ControlAuthority()
    inbox = ActionInbox(authority)
    executor = Executor(
        clock=clock,
        robot=robot,
        state_provider=robot.read_state,
        authority=authority,
        inbox=inbox,
        safety=SafetyGate(make_safety_profile()),
        tick_period_ns=10_000_000,
        on_event=on_event,
    )
    executor.start()
    return clock, authority, inbox, executor


def test_unexpected_state_provider_exception_is_fail_closed() -> None:
    clock = FakeClock(start_ns=NOW)
    robot = FakeRobotArm(clock)
    events = []
    authority = ControlAuthority()
    inbox = ActionInbox(authority)
    executor = Executor(
        clock=clock,
        robot=robot,
        state_provider=lambda: (_ for _ in ()).throw(RuntimeError("state bug")),
        authority=authority,
        inbox=inbox,
        safety=SafetyGate(make_safety_profile()),
        on_event=events.append,
    )
    executor.start()
    authority.grant()
    authority.activate()
    with pytest.raises(Exception) as exc:
        executor.tick(clock.now_ns())
    assert type(exc.value).__name__ == "VlaInfraError"
    assert not authority.valid
    assert executor.active_plan is None
    assert executor.stats.internal_errors == 1
    assert any(event.kind is ExecutorEventKind.INTERNAL_ERROR for event in events)
    assert executor.tick(clock.now_ns()).kind == "faulted"
    assert robot.write_count == 0


def test_pose_write_failure_revokes_authority_and_faults() -> None:
    clock = FakeClock(start_ns=NOW)
    robot = _ExplodingPoseRobot(clock)
    events = []
    authority = ControlAuthority()
    inbox = ActionInbox(authority)
    executor = Executor(
        clock=clock,
        robot=robot,
        state_provider=robot.read_state,
        authority=authority,
        inbox=inbox,
        safety=SafetyGate(make_safety_profile()),
        on_event=events.append,
    )
    executor.start()
    epoch = authority.grant()
    authority.activate()
    inbox.submit(make_chunk(observation_capture_time_ns=NOW), epoch=epoch, received_time_ns=NOW)
    report = executor.tick(clock.now_ns())
    assert report.kind == "write_failed"
    assert report.wrote is False
    assert report.target is not None
    assert report.command_status == "attempted"
    assert report.pose_attempted is True
    assert report.pose_ok is False
    assert report.gripper_attempted is False
    assert robot.pose_attempts == 1
    assert not authority.valid
    assert executor.active_plan is None
    assert executor.stats.write_failures == 1
    assert executor.tick(clock.now_ns()).kind == "faulted"
    assert robot.pose_attempts == 1  # no further writes


def test_partial_pose_gripper_failure_is_audited_and_stops() -> None:
    clock = FakeClock(start_ns=NOW)
    robot = _ExplodingGripperRobot(clock)
    events = []
    authority = ControlAuthority()
    inbox = ActionInbox(authority)
    executor = Executor(
        clock=clock,
        robot=robot,
        state_provider=robot.read_state,
        authority=authority,
        inbox=inbox,
        safety=SafetyGate(make_safety_profile()),
        on_event=events.append,
    )
    executor.start()
    epoch = authority.grant()
    authority.activate()
    inbox.submit(make_chunk(observation_capture_time_ns=NOW), epoch=epoch, received_time_ns=NOW)
    report = executor.tick(clock.now_ns())
    assert report.kind == "write_partial"
    assert report.wrote is True
    assert report.target is not None
    assert report.command_status == "partial"
    assert report.pose_attempted is True
    assert report.pose_ok is True
    assert report.gripper_attempted is True
    assert report.gripper_ok is False
    assert robot.gripper_attempts == 1
    assert not authority.valid
    assert executor.active_plan is None
    assert executor.stats.partial_writes == 1
    failure = [event for event in events if event.kind is ExecutorEventKind.WRITE_PARTIAL][-1]
    assert failure.pose_ok is True
    assert failure.gripper_ok is False
    assert failure.gripper_sent is True
    assert executor.tick(clock.now_ns()).kind == "faulted"
    assert robot.gripper_attempts == 1


def test_gripper_is_not_resent_when_command_is_unchanged() -> None:
    harness = Harness()
    epoch = harness.arm()
    harness.submit(
        make_chunk(observation_capture_time_ns=NOW, delta_m=(0.001, 0.0, 0.0), horizon=10),
        epoch,
    )
    harness.step(40)
    gripper_commands = [command for command in harness.robot.commands if command.kind == "gripper"]
    pose_commands = [command for command in harness.robot.commands if command.kind == "pose"]
    assert len(pose_commands) > 5
    assert len(gripper_commands) == 1
    assert harness.executor.stats.gripper_writes == 1


def test_internal_error_event_faults_supervisor() -> None:
    clock = FakeClock(start_ns=NOW)
    authority = ControlAuthority()
    inbox = ActionInbox(authority)
    supervisor = Supervisor(clock=clock, authority=authority, inbox=inbox)
    robot = FakeRobotArm(clock)

    def on_event(event: ExecutorEvent) -> None:
        if event.kind in (
            ExecutorEventKind.WRITE_FAILED,
            ExecutorEventKind.WRITE_PARTIAL,
            ExecutorEventKind.INTERNAL_ERROR,
        ):
            supervisor.fault(f"executor_{event.kind.value}", now_ns=event.time_ns)

    executor = Executor(
        clock=clock,
        robot=robot,
        state_provider=lambda: (_ for _ in ()).throw(RuntimeError("state bug")),
        authority=authority,
        inbox=inbox,
        safety=SafetyGate(make_safety_profile()),
        on_event=on_event,
    )
    executor.start()
    supervisor.connect_devices(now_ns=clock.now_ns())
    supervisor.devices_ready(now_ns=clock.now_ns())
    supervisor.on_policy_connected(now_ns=clock.now_ns())
    now = clock.now_ns()
    for source in ("robot", "camera", "executor", "policy"):
        supervisor.heartbeat(source, now_ns=now)
    supervisor.arm(operator_ack=True, now_ns=now)
    supervisor.start_running(now_ns=now)
    epoch = authority.epoch
    inbox.submit(make_chunk(observation_capture_time_ns=now), epoch=epoch, received_time_ns=now)
    with pytest.raises(Exception) as exc:
        executor.tick(now)
    assert type(exc.value).__name__ == "VlaInfraError"
    assert supervisor.state is SystemState.FAULT
    assert supervisor.latched_fault == "executor_internal_error"
    assert not authority.valid
    assert robot.write_count == 0


def test_run_forever_stops_after_unexpected_exception() -> None:
    clock = FakeClock(start_ns=NOW)
    robot = FakeRobotArm(clock)
    authority = ControlAuthority()
    inbox = ActionInbox(authority)
    executor = Executor(
        clock=clock,
        robot=robot,
        state_provider=lambda: (_ for _ in ()).throw(RuntimeError("loop bug")),
        authority=authority,
        inbox=inbox,
        safety=SafetyGate(make_safety_profile()),
    )
    executor.start()
    authority.grant()
    authority.activate()
    stop = threading.Event()
    executor.run_forever(stop)
    assert not authority.valid
    assert executor.stats.internal_errors == 1
    assert robot.write_count == 0


class _NotArmedPoseRobot(FakeRobotArm):
    def command_pose(self, pose, *, epoch, now_ns):
        raise NotArmedError("device-level not armed", code=ErrorCode.NOT_ARMED)


class _NotArmedGripperRobot(FakeRobotArm):
    def command_gripper(self, normalized, *, epoch, now_ns):
        raise NotArmedError("device-level not armed", code=ErrorCode.NOT_ARMED)


def test_device_level_not_armed_from_pose_is_a_write_failure() -> None:
    clock = FakeClock(start_ns=NOW)
    robot = _NotArmedPoseRobot(clock)
    events = []
    authority = ControlAuthority()
    inbox = ActionInbox(authority)
    executor = Executor(
        clock=clock,
        robot=robot,
        state_provider=robot.read_state,
        authority=authority,
        inbox=inbox,
        safety=SafetyGate(make_safety_profile()),
        on_event=events.append,
    )
    executor.start()
    epoch = authority.grant()
    authority.activate()
    inbox.submit(make_chunk(observation_capture_time_ns=NOW), epoch=epoch, received_time_ns=NOW)
    report = executor.tick(clock.now_ns())
    assert report.kind == "write_failed"
    assert report.wrote is False
    assert not authority.valid
    failure = [event for event in events if event.kind is ExecutorEventKind.WRITE_FAILED][-1]
    assert failure.error_code is ErrorCode.DEVICE_FAULT
    assert executor.tick(clock.now_ns()).kind == "faulted"
    assert robot.write_count == 0


def test_device_level_not_armed_from_gripper_after_pose_is_partial() -> None:
    clock = FakeClock(start_ns=NOW)
    robot = _NotArmedGripperRobot(clock)
    events = []
    authority = ControlAuthority()
    inbox = ActionInbox(authority)
    executor = Executor(
        clock=clock,
        robot=robot,
        state_provider=robot.read_state,
        authority=authority,
        inbox=inbox,
        safety=SafetyGate(make_safety_profile()),
        on_event=events.append,
    )
    executor.start()
    epoch = authority.grant()
    authority.activate()
    inbox.submit(make_chunk(observation_capture_time_ns=NOW), epoch=epoch, received_time_ns=NOW)
    report = executor.tick(clock.now_ns())
    assert report.kind == "write_partial"
    assert report.wrote is True
    assert not authority.valid
    partial = [event for event in events if event.kind is ExecutorEventKind.WRITE_PARTIAL][-1]
    assert partial.pose_ok is True
    assert partial.gripper_ok is False
    assert partial.gripper_sent is True
    assert executor.tick(clock.now_ns()).kind == "faulted"
    assert robot.write_count == 1  # pose only


def test_gripper_dedup_cache_is_epoch_bound() -> None:
    harness = Harness()
    first_epoch = harness.arm()
    assert harness.submit(
        make_chunk(observation_capture_time_ns=NOW, horizon=10, request_sequence_id=1),
        first_epoch,
    )
    harness.step(2)
    first_gripper_count = len(
        [command for command in harness.robot.commands if command.kind == "gripper"]
    )
    assert first_gripper_count == 1

    harness.authority.revoke("hold")
    second_epoch = harness.authority.grant()
    harness.authority.activate()
    assert harness.submit(
        make_chunk(observation_capture_time_ns=harness.clock.now_ns(), horizon=10, request_sequence_id=2),
        second_epoch,
    )
    harness.step(2)
    second_gripper_count = len(
        [command for command in harness.robot.commands if command.kind == "gripper"]
    )
    assert second_gripper_count == first_gripper_count + 1


def test_first_write_runs_full_per_tick_gate() -> None:
    harness = Harness()
    epoch = harness.arm()
    chunk = make_chunk(observation_capture_time_ns=NOW, delta_m=(0.001, 0.0, 0.0), horizon=4)
    assert harness.submit(chunk, epoch)
    calls = []
    original = harness.executor._safety.check_tick

    def recording_check(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    harness.executor._safety.check_tick = recording_check
    harness.step(2)
    assert harness.robot.write_count > 0
    assert len(calls) >= 1
    # First call's previous target is the measured pose, not a skipped check.
    assert calls[0][0][0].pose.position.tolist() == pytest.approx([0.5, 0.0, 0.3])


def test_first_tick_uses_real_control_dt_with_nonzero_measured_twist() -> None:
    harness = Harness(max_linear_acceleration_m_s2=1.0)
    sample = make_robot_sample(
        harness.clock,
        host_time_ns=NOW,
        state=make_robot_state(tcp_velocity=(0.001, 0.0, 0.0, 0.0, 0.0, 0.0)),
    )
    harness.executor._state_provider = lambda: sample
    epoch = harness.arm()
    assert harness.submit(make_chunk(observation_capture_time_ns=NOW, delta_m=(0.0, 0.0, 0.0)), epoch)
    harness.step(2)
    assert harness.robot.write_count > 0
    assert harness.executor.stats.plans_rejected == 0
