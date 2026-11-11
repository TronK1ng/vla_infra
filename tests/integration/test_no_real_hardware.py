"""Real hardware must be unreachable from the automated suite."""

from __future__ import annotations

import sys

import pytest

from vla_infra.devices.flexiv_rdk import (
    FlexivRdkReadOnlyState,
    FlexivRdkRobot,
    assert_real_hardware_allowed,
    real_hardware_enabled,
)
from vla_infra.devices.gn01 import Gn01Gripper
from vla_infra.devices.realsense import RealSenseCamera, enumerate_devices
from vla_infra.errors import DeviceError, RealHardwareDisabledError


def test_m5_writer_and_gn01_always_raise() -> None:
    with pytest.raises(RealHardwareDisabledError):
        FlexivRdkRobot()
    with pytest.raises(RealHardwareDisabledError):
        Gn01Gripper()


def test_readonly_adapters_require_explicit_gate() -> None:
    with pytest.raises(RealHardwareDisabledError):
        FlexivRdkReadOnlyState("SN")
    camera = RealSenseCamera("base", "SN", enable_depth=False)
    with pytest.raises(RealHardwareDisabledError):
        camera.start()
    with pytest.raises(RealHardwareDisabledError):
        enumerate_devices()


def test_hardware_sdk_is_never_imported() -> None:
    assert "flexivrdk" not in sys.modules
    assert "pyrealsense2" not in sys.modules
    with pytest.raises(RealHardwareDisabledError):
        FlexivRdkReadOnlyState("SN")
    with pytest.raises(RealHardwareDisabledError):
        RealSenseCamera("base", "SN", enable_depth=False).start()
    assert "flexivrdk" not in sys.modules
    assert "pyrealsense2" not in sys.modules


def test_environment_opt_in_does_not_implement_writer_or_fake_sdk() -> None:
    monkeypatch_env = pytest.MonkeyPatch()
    try:
        monkeypatch_env.setenv("VLA_INFRA_ALLOW_REAL_HARDWARE", "1")
        assert real_hardware_enabled()
        assert_real_hardware_allowed("test")
        with pytest.raises(RealHardwareDisabledError):
            FlexivRdkRobot()  # writer remains M5
        camera = RealSenseCamera("base", "SN", enable_depth=False)
        with pytest.raises(DeviceError) as exc:
            camera.start()  # SDK is absent; error must be a device error, not a crash
        assert "pyrealsense2" in exc.value.message
    finally:
        monkeypatch_env.undo()


def test_gate_is_closed_by_default(monkeypatch) -> None:
    monkeypatch.delenv("VLA_INFRA_ALLOW_REAL_HARDWARE", raising=False)
    assert not real_hardware_enabled()
    with pytest.raises(RealHardwareDisabledError):
        assert_real_hardware_allowed("test")
