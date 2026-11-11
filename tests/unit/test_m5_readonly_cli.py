from __future__ import annotations

from types import SimpleNamespace

import pytest

from vla_infra.cli.m5_readonly import run_probe


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _Robot:
    def __init__(self, serial, interfaces=None) -> None:
        self.serial = serial
        self.interfaces = interfaces
        self.calls: list[str] = []

    def states(self):
        self.calls.append("states")
        return SimpleNamespace(
            q=[0.0] * 7,
            dq=[0.0] * 7,
            tcp_pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            tcp_vel=[0.0] * 6,
            ext_wrench_in_tcp=[0.0] * 6,
            timestamp=[1, 2],
        )

    def fault(self):
        self.calls.append("fault")
        return False

    def operational(self):
        self.calls.append("operational")
        return False

    def __getattr__(self, name):
        if name in {"Enable", "ClearFault", "SwitchMode", "Stop", "SendCartesianMotionForce"}:
            raise AssertionError(f"write API accessed: {name}")
        raise AttributeError(name)


class _Sdk:
    __version__ = "1.9.3"

    def __init__(self) -> None:
        self.robot = None

    def Robot(self, serial, interfaces=None):
        self.robot = _Robot(serial, interfaces)
        return self.robot


def test_readonly_probe_uses_only_read_calls():
    sdk = _Sdk()
    clock = _Clock()
    result = run_probe(
        sdk=sdk,
        serial="Rizon4s-123456",
        network_interface="eth0",
        duration_s=0.3,
        rate_hz=10.0,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    assert result["samples"] == 3
    assert result["write_calls"] == 0
    assert set(sdk.robot.calls) == {"states", "fault", "operational"}


@pytest.mark.parametrize("serial", ["", "REPLACE_ME_ROBOT_SERIAL"])
def test_readonly_probe_rejects_placeholder_serial(serial):
    with pytest.raises(ValueError):
        run_probe(sdk=_Sdk(), serial=serial, network_interface=None, duration_s=1, rate_hz=1)


def test_readonly_probe_fails_closed_on_shape_mismatch():
    sdk = _Sdk()
    clock = _Clock()
    robot = sdk.Robot("Rizon4s-123456")
    original = robot.states

    def bad_states():
        state = original()
        state.q = [0.0] * 6
        return state

    robot.states = bad_states
    sdk.Robot = lambda *_args, **_kwargs: robot
    with pytest.raises(RuntimeError, match="unexpected q length"):
        run_probe(
            sdk=sdk,
            serial="Rizon4s-123456",
            network_interface=None,
            duration_s=1,
            rate_hz=1,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
