"""Orbbec SDK v2 RGB-D camera adapter (Gemini 336L read-only).

The vendor SDK is imported lazily.  The adapter binds by serial number, uses
RGB8 plus Z16 depth, aligns depth to color, and preserves Orbbec's device
timestamp (microseconds) as nanoseconds.  No camera property or firmware is
modified.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional

import numpy as np

from ..clock import Clock, MonotonicClock
from ..contracts.constants import IMAGE_ENCODING_DEPTH16, IMAGE_ENCODING_RGB8
from ..contracts.observation import ImageFrame
from ..errors import DeviceError, ErrorCode
from .flexiv_rdk import assert_real_hardware_allowed
from .interfaces import CameraFramePair, CameraStats
from .realsense import CameraIntrinsics


class OrbbecCamera:
    """Read-only Orbbec SDK v2 source, validated for Gemini 330 series."""

    def __init__(
        self,
        name: str,
        serial: str,
        *,
        width: int = 640,
        height: int = 480,
        fps: float = 15.0,
        enable_depth: bool = True,
        clock: Optional[Clock] = None,
        sdk: Any = None,
        require_usb3: bool = True,
    ) -> None:
        if not name or not serial:
            raise DeviceError("camera name/serial must be non-empty", code=ErrorCode.CONFIG_INVALID)
        self.name = name
        self.serial = str(serial)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(round(float(fps)))
        self.enable_depth = bool(enable_depth)
        self._clock = clock or MonotonicClock()
        self._sdk = sdk
        self._require_usb3 = bool(require_usb3)
        self._pipeline = None
        self._align = None
        self._device_info: Dict[str, Any] = {}
        self._intrinsics: Optional[CameraIntrinsics] = None
        self._depth_scale: Optional[float] = None
        self._stats = CameraStats()
        self._last_frame_number = -1
        self._last_device_ns = -1
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        sdk = self._sdk
        if sdk is None:
            assert_real_hardware_allowed("OrbbecCamera.start")
            try:
                import pyorbbecsdk as sdk  # type: ignore
            except ImportError as exc:
                raise DeviceError(
                    "pyorbbecsdk is not installed; install pyorbbecsdk2 on the station",
                    code=ErrorCode.DEVICE_NOT_CONNECTED,
                ) from exc
        pipeline = None
        try:
            device = self._find_device(sdk)
            info = device.get_device_info()
            connection_type = str(info.get_connection_type())
            if self._require_usb3 and "USB2" in connection_type.upper().replace(" ", ""):
                raise DeviceError(
                    "Orbbec is not on a USB 3 port; RGB-D bandwidth is not guaranteed",
                    code=ErrorCode.DEVICE_FAULT,
                    details={"camera": self.name, "connection_type": connection_type},
                )
            pipeline = sdk.Pipeline(device)
            config = sdk.Config()
            color_profiles = pipeline.get_stream_profile_list(sdk.OBSensorType.COLOR_SENSOR)
            color_profile = color_profiles.get_video_stream_profile(
                self.width, self.height, sdk.OBFormat.RGB, self.fps
            )
            config.enable_stream(color_profile)
            if self.enable_depth:
                depth_profiles = pipeline.get_stream_profile_list(sdk.OBSensorType.DEPTH_SENSOR)
                depth_profile = depth_profiles.get_video_stream_profile(
                    self.width, self.height, sdk.OBFormat.Y16, self.fps
                )
                config.enable_stream(depth_profile)
                config.set_frame_aggregate_output_mode(sdk.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
                self._align = sdk.AlignFilter(align_to_stream=sdk.OBStreamType.COLOR_STREAM)
            pipeline.start(config)

            intr = color_profile.get_intrinsic()
            distortion = color_profile.get_distortion()
            coeffs = tuple(
                float(getattr(distortion, key, 0.0))
                for key in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
            )
            self._intrinsics = CameraIntrinsics(
                width=int(intr.width),
                height=int(intr.height),
                fx=float(intr.fx),
                fy=float(intr.fy),
                ppx=float(intr.cx),
                ppy=float(intr.cy),
                model=str(getattr(getattr(distortion, "model", "unknown"), "name", "unknown")),
                coeffs=coeffs,
            )
            self._device_info = {
                "driver": "orbbec",
                "name": str(info.get_name()),
                "serial": str(info.get_serial_number()),
                "firmware_version": str(info.get_firmware_version()),
                "connection_type": connection_type,
                "intrinsics": self._intrinsics.as_dict(),
                "intrinsics_sha256": self._intrinsics.sha256,
                "depth_scale": None,
            }
        except DeviceError:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass
            raise
        except Exception as exc:  # noqa: BLE001 - SDK boundary
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass
            raise DeviceError(
                f"failed to start Orbbec {self.name} ({self.serial})",
                code=ErrorCode.DEVICE_FAULT,
                details={"camera": self.name, "reason": type(exc).__name__},
            ) from exc
        self._pipeline = pipeline
        self._started = True

    def _find_device(self, sdk: Any) -> Any:
        devices = sdk.Context().query_devices()
        for index in range(int(devices.get_count())):
            device = devices.get_device_by_index(index)
            actual = str(device.get_device_info().get_serial_number())
            if actual == self.serial:
                return device
        raise DeviceError(
            "Orbbec serial not found",
            code=ErrorCode.DEVICE_NOT_CONNECTED,
            details={"camera": self.name, "serial": self.serial},
        )

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        self._pipeline = None
        self._align = None
        self._started = False

    def read(self) -> Optional[ImageFrame]:
        pair = self.read_pair()
        return None if pair is None else pair.color

    def read_pair(self) -> Optional[CameraFramePair]:
        if not self._started or self._pipeline is None:
            raise DeviceError(f"Orbbec {self.name} is not started", code=ErrorCode.DEVICE_NOT_CONNECTED)
        self._stats = replace(self._stats, read_calls=self._stats.read_calls + 1)
        try:
            frames = self._pipeline.wait_for_frames(1)
            if not frames:
                self._stats = replace(self._stats, duplicate_reads=self._stats.duplicate_reads + 1)
                return None
            if self._align is not None:
                frames = self._align.process(frames)
            if not frames:
                return None
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame() if self.enable_depth else None
            if not color_frame or (self.enable_depth and not depth_frame):
                self._stats = replace(self._stats, failures=self._stats.failures + 1)
                return None
        except Exception as exc:  # noqa: BLE001 - SDK boundary
            self._stats = replace(self._stats, failures=self._stats.failures + 1)
            raise DeviceError(
                f"Orbbec {self.name} frame read failed",
                code=ErrorCode.DEVICE_FAULT,
                details={"camera": self.name, "reason": type(exc).__name__},
            ) from exc

        host_ns = self._clock.now_ns()
        device_ns = int(color_frame.get_timestamp_us()) * 1_000
        frame_number = int(color_frame.get_index())
        if device_ns <= 0:
            raise DeviceError("Orbbec reported a non-positive device timestamp", code=ErrorCode.INVALID_SYNC)
        if device_ns <= self._last_device_ns or (
            self._last_frame_number >= 0 and frame_number <= self._last_frame_number
        ):
            raise DeviceError(
                "Orbbec frame timestamp/index is not increasing",
                code=ErrorCode.BUFFER_NOT_MONOTONIC,
                details={"camera": self.name, "device_ns": device_ns, "frame_number": frame_number},
            )
        gap = max(0, frame_number - self._last_frame_number - 1) if self._last_frame_number >= 0 else 0
        self._last_device_ns = device_ns
        self._last_frame_number = frame_number

        color = np.frombuffer(color_frame.get_data(), dtype=np.uint8).reshape(
            int(color_frame.get_height()), int(color_frame.get_width()), 3
        )
        if color.shape != (self.height, self.width, 3):
            raise DeviceError(
                "Orbbec color stream shape mismatch",
                code=ErrorCode.INVALID_DTYPE,
                details={"camera": self.name, "shape": list(color.shape)},
            )
        color_image = ImageFrame(
            name=self.name,
            data=np.array(color, copy=True, order="C"),
            device_time_ns=device_ns,
            host_time_ns=host_ns,
            frame_number=frame_number,
            encoding=IMAGE_ENCODING_RGB8,
        )
        depth_image = None
        if depth_frame:
            depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
                int(depth_frame.get_height()), int(depth_frame.get_width())
            )
            self._depth_scale = float(depth_frame.get_depth_scale())
            self._device_info["depth_scale"] = self._depth_scale
            depth_image = ImageFrame(
                name=f"{self.name}_depth",
                data=np.array(depth, copy=True, order="C"),
                device_time_ns=device_ns,
                host_time_ns=host_ns,
                frame_number=frame_number,
                encoding=IMAGE_ENCODING_DEPTH16,
            )
        self._stats = CameraStats(
            read_calls=self._stats.read_calls,
            frames_produced=self._stats.frames_produced + 1,
            failures=self._stats.failures,
            duplicate_reads=self._stats.duplicate_reads,
            dropped_frames=self._stats.dropped_frames + gap,
            last_frame_number=frame_number,
        )
        return CameraFramePair(color=color_image, depth=depth_image)

    @property
    def stats(self) -> CameraStats:
        return self._stats

    @property
    def intrinsics(self) -> Optional[CameraIntrinsics]:
        return self._intrinsics

    @property
    def depth_scale(self) -> Optional[float]:
        return self._depth_scale

    @property
    def device_info(self) -> Dict[str, Any]:
        return dict(self._device_info)


def enumerate_devices(*, sdk: Any = None) -> list[dict]:
    if sdk is None:
        assert_real_hardware_allowed("Orbbec.enumerate_devices")
        import pyorbbecsdk as sdk  # type: ignore
    devices = sdk.Context().query_devices()
    result = []
    for index in range(int(devices.get_count())):
        info = devices.get_device_by_index(index).get_device_info()
        result.append(
            {
                "name": str(info.get_name()),
                "serial": str(info.get_serial_number()),
                "firmware_version": str(info.get_firmware_version()),
                "connection_type": str(info.get_connection_type()),
            }
        )
    return result


__all__ = ["OrbbecCamera", "enumerate_devices"]
