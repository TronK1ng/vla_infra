"""Supervisor state machine, watchdogs, HOLD/FAULT arbitration."""

from __future__ import annotations

import pytest

from vla_infra.clock import FakeClock
from vla_infra.errors import ErrorCode, IllegalTransitionError, VlaInfraError
from vla_infra.execution.authority import ControlAuthority
from vla_infra.execution.inbox import ActionInbox
from vla_infra.execution.supervisor import (
    Supervisor,
    SupervisorConfig,
    SystemState,
    WatchdogConfig,
)
from vla_infra.testing import make_chunk

SOURCES = ("robot", "camera", "executor", "policy")


def _supervisor(**config_overrides) -> tuple[FakeClock, ControlAuthority, ActionInbox, Supervisor]:
    clock = FakeClock(start_ns=1_000_000_000)
    authority = ControlAuthority()
    inbox = ActionInbox(authority)
    config = SupervisorConfig(**config_overrides)
    supervisor = Supervisor(clock=clock, config=config, authority=authority, inbox=inbox)
    return clock, authority, inbox, supervisor


def _heartbeat_all(supervisor: Supervisor, now: int) -> None:
    for source in SOURCES:
        supervisor.heartbeat(source, now_ns=now)


def _ready(supervisor: Supervisor, now: int, *, policy: bool = True) -> None:
    supervisor.connect_devices(now_ns=now)
    supervisor.devices_ready(now_ns=now)
    if policy:
        supervisor.on_policy_connected(now_ns=now)
    _heartbeat_all(supervisor, now)


def _arm(supervisor: Supervisor, now: int) -> None:
    _heartbeat_all(supervisor, now)
    supervisor.arm(operator_ack=True, now_ns=now)
    supervisor.start_running(now_ns=now)


def test_full_lifecycle() -> None:
    clock, authority, inbox, supervisor = _supervisor()
    now = clock.now_ns()
    assert supervisor.state is SystemState.OFFLINE
    _ready(supervisor, now)
    assert supervisor.state is SystemState.HOLD
    assert supervisor.is_devices_ready
    assert not authority.valid
    supervisor.arm(operator_ack=True, now_ns=now)
    assert supervisor.state is SystemState.ARMED
    assert authority.valid and not authority.running
    supervisor.start_running(now_ns=now)
    assert supervisor.state is SystemState.RUNNING
    assert authority.running
    supervisor.hold("operator_hold", now_ns=now)
    assert supervisor.state is SystemState.HOLD
    assert not authority.valid
    assert inbox.stats.cleared >= 0


def test_arm_requires_acknowledgement() -> None:
    clock, _, _, supervisor = _supervisor()
    now = clock.now_ns()
    _ready(supervisor, now)
    with pytest.raises(VlaInfraError) as exc:
        supervisor.arm(operator_ack=False, now_ns=now)
    assert exc.value.code is ErrorCode.NOT_READY
    supervisor.arm(operator_ack=True, now_ns=now)


def test_arm_requires_policy_connection_and_preflight() -> None:
    clock, _, _, supervisor = _supervisor()
    now = clock.now_ns()
    supervisor.connect_devices(now_ns=now)
    with pytest.raises(VlaInfraError) as exc:
        supervisor.arm(operator_ack=True, now_ns=now)
    assert exc.value.code is ErrorCode.ILLEGAL_TRANSITION
    supervisor.devices_ready(now_ns=now)
    with pytest.raises(VlaInfraError) as exc:
        supervisor.arm(operator_ack=True, now_ns=now)
    assert exc.value.code is ErrorCode.NOT_READY
    supervisor.on_policy_connected(now_ns=now)
    with pytest.raises(VlaInfraError) as exc:
        supervisor.arm(operator_ack=True, now_ns=now)  # watchdogs never fed
    assert exc.value.code is ErrorCode.NOT_READY
    assert "stale" in exc.value.message


def test_hold_clears_inbox_and_bumps_epoch() -> None:
    clock, authority, inbox, supervisor = _supervisor()
    now = clock.now_ns()
    _ready(supervisor, now)
    _arm(supervisor, now)
    old_epoch = authority.epoch
    assert inbox.submit(make_chunk(), epoch=old_epoch, received_time_ns=now)
    supervisor.hold("test", now_ns=now)
    assert inbox.depth() == 0
    assert authority.epoch > old_epoch
    assert not inbox.submit(make_chunk(request_sequence_id=2), epoch=old_epoch, received_time_ns=now)


def test_policy_disconnect_forces_hold_and_rearm_required() -> None:
    clock, authority, inbox, supervisor = _supervisor()
    now = clock.now_ns()
    _ready(supervisor, now)
    _arm(supervisor, now)
    supervisor.on_policy_disconnected("timeout", now_ns=now)
    assert supervisor.state is SystemState.HOLD
    assert not authority.valid
    # A reconnect must not auto-resume motion.
    supervisor.on_policy_connected(now_ns=now)
    _heartbeat_all(supervisor, now)
    supervisor.tick(now_ns=now)
    assert supervisor.state is SystemState.HOLD
    supervisor.arm(operator_ack=True, now_ns=now)
    assert supervisor.state is SystemState.ARMED


def test_policy_watchdog_holds() -> None:
    clock, _, _, supervisor = _supervisor()
    now = clock.now_ns()
    _ready(supervisor, now)
    _arm(supervisor, now)
    later = now + 1_100_000_000
    for source in ("robot", "camera", "executor"):
        supervisor.heartbeat(source, now_ns=later)
    supervisor.tick(now_ns=later)
    assert supervisor.state is SystemState.HOLD
    assert "policy" in supervisor.history()[-1].reason


def test_robot_watchdog_latches_fault() -> None:
    clock, _, _, supervisor = _supervisor()
    now = clock.now_ns()
    _ready(supervisor, now)
    _arm(supervisor, now)
    later = now + 300_000_000
    for source in ("camera", "executor", "policy"):
        supervisor.heartbeat(source, now_ns=later)
    supervisor.tick(now_ns=later)
    assert supervisor.state is SystemState.FAULT
    assert supervisor.latched_fault == "watchdog_expired:robot"
    assert supervisor.authority.valid is False


def test_fault_does_not_auto_recover_and_reset_requires_ack() -> None:
    clock, _, _, supervisor = _supervisor()
    now = clock.now_ns()
    _ready(supervisor, now)
    _arm(supervisor, now)
    supervisor.fault("test_fault", now_ns=now)
    assert supervisor.state is SystemState.FAULT
    supervisor.tick(now_ns=now + 10_000_000_000)
    assert supervisor.state is SystemState.FAULT
    with pytest.raises(VlaInfraError):
        supervisor.reset(operator_ack=False, now_ns=now)
    with pytest.raises(IllegalTransitionError):
        supervisor.hold("no", now_ns=now)
    supervisor.reset(operator_ack=True, now_ns=now)
    assert supervisor.state is SystemState.OFFLINE
    assert supervisor.latched_fault is None
    assert not supervisor.is_devices_ready


def test_chunk_starvation_holds() -> None:
    clock, _, _, supervisor = _supervisor(chunk_starvation_ns=500_000_000)
    now = clock.now_ns()
    _ready(supervisor, now)
    _arm(supervisor, now)
    supervisor.on_executor_idle(now_ns=now)
    later = now + 600_000_000
    _heartbeat_all(supervisor, later)
    supervisor.tick(now_ns=later)
    assert supervisor.state is SystemState.HOLD
    assert supervisor.history()[-1].reason == "chunk_starvation"


def test_chunk_accepted_clears_idle_timer() -> None:
    clock, _, _, supervisor = _supervisor(chunk_starvation_ns=500_000_000)
    now = clock.now_ns()
    _ready(supervisor, now)
    _arm(supervisor, now)
    supervisor.on_executor_idle(now_ns=now)
    supervisor.on_chunk_accepted(now_ns=now + 100_000_000)
    later = now + 600_000_000
    _heartbeat_all(supervisor, later)
    supervisor.tick(now_ns=later)
    assert supervisor.state is SystemState.RUNNING


def test_illegal_transitions_raise() -> None:
    clock, _, _, supervisor = _supervisor()
    with pytest.raises(IllegalTransitionError):
        supervisor.hold("x", now_ns=clock.now_ns())
    with pytest.raises(IllegalTransitionError):
        supervisor.start_running(now_ns=clock.now_ns())
    supervisor.fault("offline_fault", now_ns=clock.now_ns())
    with pytest.raises(IllegalTransitionError):
        supervisor.connect_devices(now_ns=clock.now_ns())


def test_history_is_bounded_and_audited() -> None:
    clock, _, _, supervisor = _supervisor(history_limit=8)
    now = clock.now_ns()
    _ready(supervisor, now)
    for i in range(20):
        now += 10_000_000
        _heartbeat_all(supervisor, now)
        supervisor.arm(operator_ack=True, now_ns=now)
        supervisor.start_running(now_ns=now)
        supervisor.hold("cycle", now_ns=now)
    assert len(supervisor.history()) == 8
    assert all(record.reason for record in supervisor.history())


def test_status_is_json_friendly() -> None:
    clock, _, _, supervisor = _supervisor()
    now = clock.now_ns()
    _ready(supervisor, now)
    status = supervisor.status(now_ns=now)
    assert status["state"] == "HOLD"
    assert len(status["transitions"]) >= 3
    assert {w["name"] for w in status["watchdogs"]} == set(SOURCES)
    assert status["inbox"]["submitted"] == 0


def test_unknown_heartbeat_source_rejected() -> None:
    clock, _, _, supervisor = _supervisor()
    with pytest.raises(VlaInfraError):
        supervisor.heartbeat("lidar", now_ns=clock.now_ns())


def test_custom_watchdog_config() -> None:
    clock, _, _, supervisor = _supervisor(
        watchdogs={
            "robot": WatchdogConfig(100_000_000, "hold"),
            "camera": WatchdogConfig(100_000_000, "hold"),
            "executor": WatchdogConfig(100_000_000, "hold"),
            "policy": WatchdogConfig(100_000_000, "hold"),
        }
    )
    now = clock.now_ns()
    _ready(supervisor, now)
    _arm(supervisor, now)
    supervisor.tick(now_ns=now + 200_000_000)
    assert supervisor.state is SystemState.HOLD
