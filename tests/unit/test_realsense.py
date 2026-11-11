"""RealSense adapter tested through an injected SDK (no USB, no pyrealsense2)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vla_infra.clock import FakeClock
from vla_infra.devices.realsense import RealSenseCamera, enumerate_devices
from vla_infra.errors import DeviceError, ErrorCode


class _Intrinsics:
    width = 4
    height = 3
    fx = 100.0
    fy = 101.0
    ppx = 2.0
    ppy = 1.5
    model = "pinhole"
    coeffs = (0.0, 0.0, 0.0, 0.0, 0.0)


class _VideoProfile:
    def as_video_stream_profile(self):
        return self

    def get_intrinsics(self):
        return _Intrinsics()


class _DepthSensor:
    def get_depth_scale(self):
        return 0.001


class _Device:
    def __init__(self, serial: str, usb: str = "3.2") -> None:
        self._serial = serial
        self._usb = usb

    def get_info(self, key):
        return {
            "serial_number": self._serial,
            "name": "Intel RealSense D415",
            "firmware_version": "5.13.0.50",
            "usb_type_descriptor": self._usb,
        }[key]

    def first_depth_sensor(self):
        return _DepthSensor()


class _Profile:
    def __init__(self, device: _Device) -> None:
        self._device = device

    def get_device(self):
        return self._device

    def get_stream(self, stream):
        return _VideoProfile()


class _ColorFrame:
    def __init__(self, data, timestamp_ms: float, frame_number: int) -> None:
        self._data = data
        self._timestamp = timestamp_ms
        self._frame_number = frame_number

    def get_data(self):
        return self._data

    def get_timestamp(self):
        return self._timestamp

    def get_frame_number(self):
        return self._frame_number


class _DepthFrame:
    def __init__(self, data) -> None:
        self._data = data

    def get_data(self):
        return self._data


class _Frames:
    def __init__(self, color, depth=None) -> None:
        self._color = color
        self._depth = depth

    def get_color_frame(self):
        return self._color

    def get_depth_frame(self):
        return self._depth


class _Align:
    def process(self, frames):
        return frames


class _Pipeline:
    def __init__(self, device: _Device, frames) -> None:
        self._device = device
        self._frames = frames
        self.stopped = False

    def start(self, config):
        return _Profile(self._device)

    def poll_for_frames(self):
        return self._frames

    def stop(self):
        self.stopped = True


class _Config:
    def __init__(self) -> None:
        self.enabled_device = None
        self.streams = []

    def enable_device(self, serial):
        self.enabled_device = serial

    def enable_stream(self, stream, *args):
        self.streams.append((stream, args))


def _sdk(device: _Device, pipeline_holder: dict):
    def pipeline_factory():
        pipeline = _Pipeline(device, pipeline_holder.get("frames"))
        pipeline_holder["pipeline"] = pipeline
        return pipeline

    return SimpleNamespace(
        config=_Config,
        stream=SimpleNamespace(color="color", depth="depth"),
        format=SimpleNamespace(rgb8="rgb8", z16="z16"),
        pipeline=pipeline_factory,
        align=lambda _stream: _Align(),
        camera_info=SimpleNamespace(
            serial_number="serial_number",
            name="name",
            firmware_version="firmware_version",
            usb_type_descriptor="usb_type_descriptor",
        ),
        context=lambda: SimpleNamespace(query_devices=lambda: [device]),
    )


def _frames(frame_number: int = 0, timestamp_ms: float = 1000.5):
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    rgb[..., 0] = 11
    rgb[..., 1] = 22
    rgb[..., 2] = 33
    depth = np.full((3, 4), 1500, dtype=np.uint16)
    return _Frames(_ColorFrame(rgb, timestamp_ms, frame_number), _DepthFrame(depth))


def test_realsense_binds_serial_and_preserves_rgb_depth() -> None:
    clock = FakeClock(start_ns=1_000_000_000)
    holder = {"frames": _frames()}
    device = _Device("BASE-SERIAL")
    camera = RealSenseCamera(
        "base",
        "BASE-SERIAL",
        width=4,
        height=3,
        enable_depth=True,
        clock=clock,
        sdk=_sdk(device, holder),
    )
    camera.start()
    try:
        pair = camera.read_pair()
        assert pair is not None
        assert pair.color.data.tolist()[0][0] == [11, 22, 33]
        assert pair.color.device_time_ns == 1_000_500_000
        assert pair.color.host_time_ns == 1_000_000_000
        assert pair.color.frame_number == 0
        assert pair.depth is not None and pair.depth.data.dtype == np.uint16
        assert camera.depth_scale == pytest.approx(0.001)
        assert camera.intrinsics is not None
        assert camera.intrinsics.sha256
        assert camera.device_info["serial"] == "BASE-SERIAL"
        assert camera.device_info["intrinsics_sha256"] == camera.intrinsics.sha256
    finally:
        camera.stop()


def test_realsense_serial_mismatch_refused() -> None:
    holder = {"frames": _frames()}
    camera = RealSenseCamera("base", "EXPECTED", width=4, height=3, sdk=_sdk(_Device("OTHER"), holder))
    with pytest.raises(DeviceError) as exc:
        camera.start()
    assert "serial" in exc.value.message.lower()


def test_realsense_usb2_refused() -> None:
    holder = {"frames": _frames()}
    camera = RealSenseCamera("base", "S", width=4, height=3, sdk=_sdk(_Device("S", usb="2.1"), holder))
    with pytest.raises(DeviceError) as exc:
        camera.start()
    assert "USB" in exc.value.message


def test_realsense_poll_returns_none_without_new_frames() -> None:
    holder = {"frames": None}
    camera = RealSenseCamera("base", "S", width=4, height=3, sdk=_sdk(_Device("S"), holder))
    camera.start()
    try:
        assert camera.read_pair() is None
    finally:
        camera.stop()


def test_realsense_duplicate_frame_number_rejected() -> None:
    clock = FakeClock(start_ns=1_000_000_000)
    holder = {"frames": _frames(5)}
    camera = RealSenseCamera("base", "S", width=4, height=3, sdk=_sdk(_Device("S"), holder), clock=clock)
    camera.start()
    try:
        assert camera.read_pair() is not None
        holder["frames"] = _frames(5)
        with pytest.raises(DeviceError) as exc:
            camera.read_pair()
        assert exc.value.code is ErrorCode.BUFFER_NOT_MONOTONIC
    finally:
        camera.stop()


def test_enumerate_devices_with_injected_sdk() -> None:
    fake = _sdk(_Device("S"), {"frames": _frames()})
    assert enumerate_devices(sdk=fake)[0]["serial"] == "S"
