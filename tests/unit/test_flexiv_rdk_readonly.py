"""Flexiv RDK read-only adapter tested with a fake SDK; no SDK import/hardware."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vla_infra.clock import FakeClock
from vla_infra.devices.flexiv_rdk import FlexivRdkReadOnlyState
from vla_infra.errors import DeviceError, ErrorCode


class _States:
    def __init__(self, **overrides):
        self.q = np.zeros(7)
        self.dq = np.zeros(7)
        angle = 0.2
        self.tcp_pose = np.array([0.5, 0.0, 0.3, np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
        self.tcp_vel = np.zeros(6)
        self.ext_wrench_in_tcp = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        self.timestamp = [10, 500]
        self.gripper_width = 0.04
        for key, value in overrides.items():
            setattr(self, key, value)


class _Robot:
    calls: list = []

    def __init__(self, serial, interfaces=None):
        type(self).calls = [("init", serial, interfaces)]

    def states(self):
        type(self).calls.append(("states",))
        return _States()


def _sdk(states=None):
    class Robot(_Robot):
        def states(self):
            type(self).calls.append(("states",))
            return states if states is not None else _States()

    return SimpleNamespace(Robot=Robot, __version__="fake-rdk")


def test_read_only_state_converts_quaternion_and_never_writes() -> None:
    clock = FakeClock(start_ns=1_000_000_000)
    sdk = _sdk()
    reader = FlexivRdkReadOnlyState("SN-1", clock=clock, sdk=sdk)
    reader.start()
    try:
        sample = reader.read_state()
        state = sample.state
        assert state.tcp_position.tolist() == pytest.approx([0.5, 0.0, 0.3])
        assert state.tcp_quaternion_xyzw.tolist() == pytest.approx(
            [0.0, 0.0, np.sin(0.1), np.cos(0.1)], abs=1e-9
        )
        assert state.gripper_width_m == pytest.approx(0.04)
        assert sample.device_time_ns == 10_000_000_500
        assert sample.host_time_ns == 1_000_000_000
        # Only construction + states() are allowed.
        assert all(call[0] in ("init", "states") for call in sdk.Robot.calls)
    finally:
        reader.close()


def test_missing_wrench_field_fails_closed() -> None:
    states = _States()
    del states.ext_wrench_in_tcp
    reader = FlexivRdkReadOnlyState("SN", sdk=_sdk(states))
    reader.start()
    with pytest.raises(DeviceError) as exc:
        reader.read_state()
    assert "ext_wrench_in_tcp" in exc.value.message
    reader.close()


def test_missing_gripper_width_requires_provider() -> None:
    states = _States()
    del states.gripper_width
    reader = FlexivRdkReadOnlyState("SN", sdk=_sdk(states))
    reader.start()
    with pytest.raises(DeviceError) as exc:
        reader.read_state()
    assert "GN01" in exc.value.message
    reader.close()


def test_gripper_width_provider_is_used_read_only() -> None:
    states = _States()
    del states.gripper_width
    reader = FlexivRdkReadOnlyState(
        "SN", sdk=_sdk(states), gripper_width_provider=lambda: 0.07
    )
    reader.start()
    try:
        assert reader.read_state().state.gripper_width_m == pytest.approx(0.07)
    finally:
        reader.close()


def test_gripper_width_range_checked() -> None:
    states = _States(gripper_width=0.9)
    reader = FlexivRdkReadOnlyState("SN", sdk=_sdk(states))
    reader.start()
    with pytest.raises(DeviceError) as exc:
        reader.read_state()
    assert exc.value.code is ErrorCode.SAFETY_GRIPPER
    reader.close()


def test_timestamp_fallback_to_host() -> None:
    states = _States()
    del states.timestamp
    clock = FakeClock(start_ns=1_000_000_000)
    reader = FlexivRdkReadOnlyState("SN", sdk=_sdk(states), clock=clock)
    reader.start()
    try:
        sample = reader.read_state()
        assert sample.device_time_ns == 1_000_000_000
        assert reader.device_info["device_timestamp_fallback"] == "host_monotonic"
    finally:
        reader.close()
