"""Orbbec adapter tests through an injected SDK (no USB access)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vla_infra.clock import FakeClock
from vla_infra.devices.orbbec import OrbbecCamera, enumerate_devices
from vla_infra.errors import DeviceError, ErrorCode


class _Info:
    def __init__(self, serial="ORBBEC-SN", connection="USB3.0"):
        self.serial = serial
        self.connection = connection

    def get_name(self): return "Orbbec Gemini 336L"
    def get_serial_number(self): return self.serial
    def get_firmware_version(self): return "1.6.00"
    def get_connection_type(self): return self.connection


class _Device:
    def __init__(self, info): self.info = info
    def get_device_info(self): return self.info


class _DeviceList:
    def __init__(self, devices): self.devices = devices
    def get_count(self): return len(self.devices)
    def get_device_by_index(self, index): return self.devices[index]


class _Profile:
    def __init__(self, kind): self.kind = kind
    def get_intrinsic(self):
        return SimpleNamespace(width=4, height=3, fx=100.0, fy=101.0, cx=2.0, cy=1.5)
    def get_distortion(self):
        return SimpleNamespace(
            model=SimpleNamespace(name="BROWN_CONRADY"),
            k1=0.1, k2=0.2, p1=0.0, p2=0.0, k3=0.0, k4=0.0, k5=0.0, k6=0.0,
        )


class _Profiles:
    def __init__(self, kind): self.kind = kind
    def get_video_stream_profile(self, width, height, fmt, fps):
        assert (width, height, fps) == (4, 3, 15)
        return _Profile(self.kind)


class _Frame:
    def __init__(self, data, index=7, timestamp_us=1_000_500, depth_scale=None):
        self.data = data
        self.index = index
        self.timestamp_us = timestamp_us
        self.depth_scale = depth_scale
    def get_data(self): return self.data
    def get_index(self): return self.index
    def get_timestamp_us(self): return self.timestamp_us
    def get_width(self): return 4
    def get_height(self): return 3
    def get_depth_scale(self): return self.depth_scale


class _Frames:
    def __init__(self, color, depth): self.color, self.depth = color, depth
    def get_color_frame(self): return self.color
    def get_depth_frame(self): return self.depth


class _Config:
    def __init__(self): self.streams = []
    def enable_stream(self, profile): self.streams.append(profile)
    def set_frame_aggregate_output_mode(self, mode): self.aggregate = mode


def _sdk(*, serial="ORBBEC-SN", connection="USB3.0", frames=True):
    info = _Info(serial, connection)
    device = _Device(info)
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    rgb[..., :] = (11, 22, 33)
    depth = np.full((3, 4), 1200, dtype=np.uint16)
    frame_set = _Frames(_Frame(rgb), _Frame(depth, depth_scale=1.0)) if frames else None

    class Pipeline:
        def __init__(self, actual):
            assert actual is device
            self.stopped = False
        def get_stream_profile_list(self, sensor): return _Profiles(sensor)
        def start(self, config): self.config = config
        def wait_for_frames(self, timeout):
            assert timeout == 1
            return frame_set
        def stop(self): self.stopped = True

    return SimpleNamespace(
        Context=lambda: SimpleNamespace(query_devices=lambda: _DeviceList([device])),
        Pipeline=Pipeline,
        Config=_Config,
        AlignFilter=lambda **kwargs: SimpleNamespace(process=lambda value: value),
        OBSensorType=SimpleNamespace(COLOR_SENSOR="color", DEPTH_SENSOR="depth"),
        OBFormat=SimpleNamespace(RGB="rgb", Y16="y16"),
        OBFrameAggregateOutputMode=SimpleNamespace(FULL_FRAME_REQUIRE="full"),
        OBStreamType=SimpleNamespace(COLOR_STREAM="color"),
    )


def test_orbbec_binds_serial_and_preserves_rgb_depth() -> None:
    camera = OrbbecCamera(
        "wrist", "ORBBEC-SN", width=4, height=3, fps=15,
        clock=FakeClock(start_ns=2_000_000_000), sdk=_sdk(),
    )
    camera.start()
    try:
        pair = camera.read_pair()
        assert pair is not None
        assert pair.color.data.tolist()[0][0] == [11, 22, 33]
        assert pair.color.device_time_ns == 1_000_500_000
        assert pair.color.frame_number == 7
        assert pair.depth is not None and pair.depth.data.dtype == np.uint16
        assert camera.depth_scale == pytest.approx(1.0)
        assert camera.device_info["serial"] == "ORBBEC-SN"
        assert camera.device_info["driver"] == "orbbec"
        assert camera.intrinsics is not None and camera.intrinsics.sha256
    finally:
        camera.stop()


def test_orbbec_usb2_refused() -> None:
    camera = OrbbecCamera("wrist", "ORBBEC-SN", width=4, height=3, sdk=_sdk(connection="USB2.1"))
    with pytest.raises(DeviceError) as exc:
        camera.start()
    assert "USB 3" in exc.value.message


def test_orbbec_serial_mismatch_refused() -> None:
    camera = OrbbecCamera("wrist", "EXPECTED", width=4, height=3, sdk=_sdk(serial="OTHER"))
    with pytest.raises(DeviceError) as exc:
        camera.start()
    assert exc.value.code is ErrorCode.DEVICE_NOT_CONNECTED


def test_orbbec_no_new_frame_returns_none() -> None:
    camera = OrbbecCamera("wrist", "ORBBEC-SN", width=4, height=3, sdk=_sdk(frames=False))
    camera.start()
    try:
        assert camera.read_pair() is None
        assert camera.stats.duplicate_reads == 1
    finally:
        camera.stop()


def test_enumerate_orbbec_devices() -> None:
    result = enumerate_devices(sdk=_sdk())
    assert result == [{
        "name": "Orbbec Gemini 336L",
        "serial": "ORBBEC-SN",
        "firmware_version": "1.6.00",
        "connection_type": "USB3.0",
    }]
