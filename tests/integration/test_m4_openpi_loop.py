"""M4 fake closed loop: OpenPI-shaped dummy server + zero action + fake robot."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

VENDOR = Path(__file__).resolve().parents[2] / ".openpi_vendor"
if VENDOR.exists() and str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
pytest.importorskip("openpi_client")

from vla_infra.config import load_app_config
from vla_infra.policy.adapters.base import ZeroActionAdapter
from vla_infra.runtime.shadow import ShadowHarness

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def _app():
    return load_app_config(
        station_path=CONFIGS / "station.example.yaml",
        safety_path=CONFIGS / "safety.example.yaml",
        policy_path=CONFIGS / "policy.example.yaml",
    )


def test_m4_zero_action_fake_closed_loop() -> None:
    app = _app()
    adapter = ZeroActionAdapter(
        horizon=app.policy.expected_horizon,
        action_dt_s=0.05,
        gripper_normalized=0.5,
        image_size=app.policy.image_size,
    )
    harness = ShadowHarness(app, adapter=adapter)
    result = harness.run(duration_s=0.5, operator_ack=True)
    assert result.final_state == "RUNNING"
    assert result.armed is True
    assert result.observations > 0
    assert result.policy_chunks > 0
    assert result.writes > 0
    assert result.client["successes"] > 0
    assert result.client["last_rtt_ns"] is not None
    assert result.server["successes"] > 0
    assert result.supervisor["latched_fault"] is None
    assert harness._robot is not None
    # Zero action means the fake TCP pose never moves.
    assert abs(float(harness._robot.pose.position[0]) - 0.5) < 1e-9


def test_m4_official_openpi_zero_action_fake_closed_loop() -> None:
    app = _app()
    adapter = ZeroActionAdapter(
        horizon=app.policy.expected_horizon,
        action_dt_s=app.policy.expected_action_dt_s,
        gripper_normalized=0.5,
        image_size=app.policy.image_size,
    )
    harness = ShadowHarness(app, adapter=adapter, transport_kind="official")
    result = harness.run(duration_s=0.5, operator_ack=True)
    assert result.final_state == "RUNNING"
    assert result.writes > 0
    assert result.policy_chunks > 0
    assert result.client["successes"] > 0
    assert result.server["successes"] > 0
    assert result.transport["transport_kind"] == "official_openpi"
    assert result.transport["package"] == "openpi-client"
    assert result.transport["package_version"] == "0.1.2"
    assert result.transport["client_class"].endswith("WebsocketClientPolicy")
    assert result.transport["server_class"].endswith("OfficialOpenPIServer")
    assert result.transport["endpoint"].startswith("ws://127.0.0.1:")


def test_m4_official_disconnect_holds_and_reconnect_does_not_resume() -> None:
    from vla_infra.execution.supervisor import SystemState
    from vla_infra.policy.openpi_official import OfficialOpenPIServer

    app = _app()
    adapter = ZeroActionAdapter(
        horizon=app.policy.expected_horizon,
        action_dt_s=app.policy.expected_action_dt_s,
        gripper_normalized=0.5,
        image_size=app.policy.image_size,
    )
    harness = ShadowHarness(app, adapter=adapter, transport_kind="official")
    harness.start(operator_ack=True)
    old_server = harness._server
    assert isinstance(old_server, OfficialOpenPIServer)
    port = int(old_server.endpoint.rsplit(":", 1)[1])
    try:
        for _ in range(40):
            harness._step()
            harness._clock.sleep(10_000_000)
        assert harness.supervisor.state is SystemState.RUNNING
        writes_before = harness._writes
        assert writes_before > 0
        old_server.stop()
        deadline = 200
        for _ in range(deadline):
            harness._step()
            harness._clock.sleep(10_000_000)
            if harness.supervisor.state is SystemState.HOLD:
                break
        assert harness.supervisor.state is SystemState.HOLD
        writes_at_hold = harness._writes
        for _ in range(20):
            harness._step()
            harness._clock.sleep(10_000_000)
        assert harness._writes == writes_at_hold
        new_server = OfficialOpenPIServer(harness._adapter, port=port)
        new_server.start()
        try:
            for _ in range(300):
                harness._step()
                harness._clock.sleep(10_000_000)
                if harness.worker.connected and harness.worker.stats.results >= 2:
                    break
            assert harness.worker.connected
            assert harness.supervisor.state is SystemState.HOLD
            assert not harness.supervisor.authority.valid
            assert harness._writes == writes_at_hold
        finally:
            new_server.stop()
    finally:
        harness.stop()
