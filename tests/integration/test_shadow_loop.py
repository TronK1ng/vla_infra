"""End-to-end M1 chain on fake devices and a loopback dummy policy server.

These tests exercise the same wiring as the ``vla-shadow`` CLI but assert on
internal state: chunks flow over a real WebSocket, the executor writes only
through the fake backend, disconnect forces HOLD, and reconnect never resumes
motion without a new arm.
"""

from __future__ import annotations

import time
from pathlib import Path

from vla_infra.config import load_app_config
from vla_infra.execution.supervisor import SystemState
from vla_infra.policy.adapters.base import FixedDeltaAdapter
from vla_infra.policy.dummy_server import DummyPolicyServer
from vla_infra.runtime.shadow import ShadowHarness

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def _app():
    return load_app_config(
        station_path=CONFIGS / "station.example.yaml",
        safety_path=CONFIGS / "safety.example.yaml",
        policy_path=CONFIGS / "policy.example.yaml",
    )


def _fixed_adapter() -> FixedDeltaAdapter:
    return FixedDeltaAdapter(
        delta_m=(0.001, 0.0, 0.0),
        delta_rotvec_rad=(0.0, 0.0, 0.0),
        gripper_normalized=0.5,
        horizon=10,
        action_dt_s=0.05,
        action_frame="base",
        action_type="delta_tcp",
    )


def wait_until(predicate, timeout_s: float = 4.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def step_until(harness: ShadowHarness, predicate, timeout_s: float = 4.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        harness._step()
        harness._clock.sleep(10_000_000)
    return predicate()


def test_shadow_harness_executes_on_fake_backend() -> None:
    harness = ShadowHarness(_app(), adapter=_fixed_adapter())
    result = harness.run(duration_s=0.8, operator_ack=True)
    assert result.final_state == "RUNNING"
    assert result.armed is True
    assert result.observations > 0
    assert result.policy_chunks > 0
    assert result.writes > 0
    assert result.supervisor["latched_fault"] is None
    assert result.executor["plans_rejected"] == 0
    assert harness._robot is not None
    assert harness._robot.pose.position[0] > 0.5
    # All commands came from the executor thread through the fake sink.
    assert all(command.epoch > 0 for command in harness._robot.commands)


def test_shadow_mode_without_ack_never_writes() -> None:
    harness = ShadowHarness(_app(), adapter=_fixed_adapter())
    result = harness.run(duration_s=0.4, operator_ack=False)
    assert result.final_state == "HOLD"
    assert result.armed is False
    assert result.writes == 0
    assert result.policy_chunks > 0  # shadow inference still ran
    assert harness._robot is not None
    assert harness._robot.write_count == 0


def test_disconnect_holds_and_reconnect_does_not_resume() -> None:
    harness = ShadowHarness(_app(), adapter=_fixed_adapter())
    harness.start(operator_ack=True)
    old_server = harness._server
    assert old_server is not None
    endpoint = old_server.endpoint
    port = int(endpoint.rsplit(":", 1)[1])
    try:
        for _ in range(30):
            harness._step()
            harness._clock.sleep(10_000_000)
        assert harness.supervisor.state is SystemState.RUNNING
        writes_before = harness._writes
        assert writes_before > 0

        old_server.stop()
        assert step_until(
            harness, lambda: harness.supervisor.state is SystemState.HOLD, timeout_s=5.0
        )
        assert not harness.supervisor.authority.valid
        writes_at_hold = harness._writes
        for _ in range(15):
            harness._step()
            harness._clock.sleep(10_000_000)
        assert harness.supervisor.state is SystemState.HOLD
        assert harness._writes == writes_at_hold

        # Bring a new server up on the same port; the worker may reconnect, but
        # motion must stay in HOLD until an explicit new arm.
        new_server = DummyPolicyServer(harness._adapter, host="127.0.0.1", port=port)
        new_server.start()
        try:
            for _ in range(200):
                harness._step()
                harness._clock.sleep(10_000_000)
                if harness.worker.connected and harness.worker.stats.results >= 2:
                    break
            assert harness.worker.connected
            assert harness.supervisor.state is SystemState.HOLD
            assert not harness.supervisor.authority.valid
            assert harness._writes == writes_at_hold
            now = harness._clock.now_ns()
            for source in ("robot", "camera", "executor", "policy"):
                harness.supervisor.heartbeat(source, now_ns=now)
            harness.supervisor.arm(operator_ack=True, now_ns=now)
            assert harness.supervisor.state is SystemState.ARMED
        finally:
            new_server.stop()
    finally:
        harness.stop()
