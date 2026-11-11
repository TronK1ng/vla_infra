"""Intel RealSense camera adapter (M2 read-only).

The SDK is imported lazily in :meth:`start`; importing this module (and running
the test suite) never touches USB or the SDK.  Frames are required to be RGB8
(no BGR conversion), depth is Z16 and aligned to color, and the configured serial
must match the device returned by the pipeline.

Device timestamps from RealSense are reported in milliseconds; they are preserved
as nanoseconds in :class:`ImageFrame` and never used for host-side scheduling.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

import numpy as np

from ..clock import Clock, MonotonicClock
from ..contracts.constants import IMAGE_ENCODING_DEPTH16, IMAGE_ENCODING_RGB8
from ..contracts.observation import ImageFrame
from ..errors import DeviceError, ErrorCode
from .flexiv_rdk import assert_real_hardware_allowed
from .interfaces import CameraFramePair, CameraStats


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    model: str = "unknown"
    coeffs: tuple = ()

    def as_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "ppx": self.ppx,
            "ppy": self.ppy,
            "model": self.model,
            "coeffs": list(self.coeffs),
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RealSenseCamera:
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
        depth_scale_expected: Optional[float] = None,
        depth_scale_tolerance: float = 1e-3,
    ) -> None:
        if not name or not serial:
            raise DeviceError("camera name/serial must be non-empty", code=ErrorCode.CONFIG_INVALID)
        self.name = name
        self.serial = str(serial)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.enable_depth = bool(enable_depth)
        self._clock = clock or MonotonicClock()
        self._sdk = sdk
        self._require_usb3 = bool(require_usb3)
        self._depth_scale_expected = depth_scale_expected
        self._depth_scale_tolerance = float(depth_scale_tolerance)
        self._pipeline = None
        self._align = None
        self._profile = None
        self._device_info: Dict[str, Any] = {}
        self._intrinsics: Optional[CameraIntrinsics] = None
        self._depth_scale: Optional[float] = None
        self._stats = CameraStats()
        self._last_frame_number = -1
        self._last_device_ns = -1
        self._started = False

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        if self._started:
            return
        sdk = self._sdk
        if sdk is None:
            assert_real_hardware_allowed("RealSenseCamera.start")
            try:
                import pyrealsense2 as sdk  # type: ignore
            except ImportError as exc:
                raise DeviceError(
                    "pyrealsense2 is not installed; install the SDK on the station",
                    code=ErrorCode.DEVICE_NOT_CONNECTED,
                ) from exc
        try:
            config = sdk.config()
            config.enable_device(self.serial)
            config.enable_stream(
                sdk.stream.color,
                self.width,
                self.height,
                sdk.format.rgb8,
                self.fps,
            )
            if self.enable_depth:
                config.enable_stream(
                    sdk.stream.depth,
                    self.width,
                    self.height,
                    sdk.format.z16,
                    self.fps,
                )
            pipeline = sdk.pipeline()
            profile = pipeline.start(config)
        except Exception as exc:  # noqa: BLE001 - SDK boundary
            raise DeviceError(
                f"failed to start RealSense {self.name} ({self.serial})",
                code=ErrorCode.DEVICE_FAULT,
                details={"camera": self.name, "reason": type(exc).__name__},
            ) from exc
        try:
            device = profile.get_device()
            actual_serial = str(device.get_info(sdk.camera_info.serial_number))
            if actual_serial != self.serial:
                raise DeviceError(
                    "RealSense serial binding mismatch",
                    code=ErrorCode.DEVICE_FAULT,
                    details={
                        "camera": self.name,
                        "configured_serial": self.serial,
                        "actual_serial": actual_serial,
                    },
                )
            usb_type = ""
            try:
                usb_type = str(device.get_info(sdk.camera_info.usb_type_descriptor))
            except Exception:  # pragma: no cover - some devices do not expose it
                usb_type = ""
            if self._require_usb3 and usb_type and not usb_type.startswith("3"):
                raise DeviceError(
                    "RealSense is not on a USB 3 port; bandwidth/rate is not guaranteed",
                    code=ErrorCode.DEVICE_FAULT,
                    details={"camera": self.name, "usb_type": usb_type},
                )
            color_profile = profile.get_stream(sdk.stream.color).as_video_stream_profile()
            intr = color_profile.get_intrinsics()
            self._intrinsics = CameraIntrinsics(
                width=int(intr.width),
                height=int(intr.height),
                fx=float(intr.fx),
                fy=float(intr.fy),
                ppx=float(intr.ppx),
                ppy=float(intr.ppy),
                model=str(getattr(intr, "model", "unknown")),
                coeffs=tuple(float(value) for value in getattr(intr, "coeffs", ())),
            )
            if self.enable_depth:
                depth_sensor = device.first_depth_sensor()
                self._depth_scale = float(depth_sensor.get_depth_scale())
                if self._depth_scale_expected is not None:
                    if abs(self._depth_scale - self._depth_scale_expected) > self._depth_scale_tolerance:
                        raise DeviceError(
                            "RealSense depth scale does not match configuration",
                            code=ErrorCode.DEVICE_FAULT,
                            details={
                                "camera": self.name,
                                "actual": self._depth_scale,
                                "expected": self._depth_scale_expected,
                            },
                        )
                self._align = sdk.align(sdk.stream.color)
            self._device_info = {
                "name": str(device.get_info(sdk.camera_info.name)),
                "serial": actual_serial,
                "firmware_version": str(device.get_info(sdk.camera_info.firmware_version)),
                "usb_type_descriptor": usb_type,
                "depth_scale": self._depth_scale,
                "intrinsics": self._intrinsics.as_dict() if self._intrinsics else None,
                "intrinsics_sha256": self._intrinsics.sha256 if self._intrinsics else None,
            }
        except DeviceError:
            try:
                pipeline.stop()
            except Exception:  # pragma: no cover
                pass
            raise
        except Exception as exc:  # noqa: BLE001
            try:
                pipeline.stop()
            except Exception:  # pragma: no cover
                pass
            raise DeviceError(
                f"failed to inspect RealSense {self.name}",
                code=ErrorCode.DEVICE_FAULT,
                details={"camera": self.name, "reason": type(exc).__name__},
            ) from exc
        self._pipeline = pipeline
        self._profile = profile
        self._started = True

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:  # pragma: no cover - best effort
                pass
        self._pipeline = None
        self._profile = None
        self._align = None
        self._started = False

    # -- data ---------------------------------------------------------------
    def read(self) -> Optional[ImageFrame]:
        pair = self.read_pair()
        return None if pair is None else pair.color

    def read_pair(self) -> Optional[CameraFramePair]:
        if not self._started or self._pipeline is None:
            raise DeviceError(
                f"RealSense {self.name} is not started",
                code=ErrorCode.DEVICE_NOT_CONNECTED,
            )
        self._stats = replace(self._stats, read_calls=self._stats.read_calls + 1)
        try:
            frames = self._pipeline.poll_for_frames()
        except Exception as exc:  # noqa: BLE001
            self._stats = replace(self._stats, failures=self._stats.failures + 1)
            raise DeviceError(
                f"RealSense {self.name} poll failed",
                code=ErrorCode.DEVICE_FAULT,
                details={"camera": self.name, "reason": type(exc).__name__},
            ) from exc
        if not frames:
            self._stats = replace(self._stats, duplicate_reads=self._stats.duplicate_reads + 1)
            return None
        if self._align is not None:
            frames = self._align.process(frames)
        color_frame = frames.get_color_frame()
        if not color_frame:
            self._stats = replace(self._stats, failures=self._stats.failures + 1)
            return None
        host_ns = self._clock.now_ns()
        device_ns = int(round(float(color_frame.get_timestamp()) * 1_000_000))
        if device_ns <= 0:
            raise DeviceError(
                "RealSense reported a non-positive device timestamp",
                code=ErrorCode.INVALID_SYNC,
                details={"camera": self.name, "timestamp": color_frame.get_timestamp()},
            )
        if device_ns <= self._last_device_ns:
            raise DeviceError(
                "RealSense device timestamp is not increasing",
                code=ErrorCode.BUFFER_NOT_MONOTONIC,
                details={"camera": self.name, "device_ns": device_ns, "last": self._last_device_ns},
            )
        frame_number = int(color_frame.get_frame_number())
        if self._last_frame_number >= 0 and frame_number <= self._last_frame_number:
            raise DeviceError(
                "RealSense frame number is not increasing",
                code=ErrorCode.BUFFER_NOT_MONOTONIC,
                details={"camera": self.name, "frame_number": frame_number, "last": self._last_frame_number},
            )
        gap = 0
        if self._last_frame_number >= 0:
            gap = max(0, frame_number - self._last_frame_number - 1)
        self._last_frame_number = frame_number
        self._last_device_ns = device_ns
        color = np.asanyarray(color_frame.get_data())
        if color.dtype != np.uint8 or color.ndim != 3 or color.shape[2] != 3:
            raise DeviceError(
                "RealSense color stream is not RGB8",
                code=ErrorCode.INVALID_DTYPE,
                details={"camera": self.name, "dtype": str(color.dtype), "shape": list(color.shape)},
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
        if self.enable_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth = np.asanyarray(depth_frame.get_data())
                if depth.dtype != np.uint16 or depth.ndim != 2:
                    raise DeviceError(
                        "RealSense depth stream is not Z16",
                        code=ErrorCode.INVALID_DTYPE,
                        details={"camera": self.name, "dtype": str(depth.dtype)},
                    )
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
    """List connected RealSense serials without opening a pipeline (read-only)."""
    if sdk is None:
        assert_real_hardware_allowed("RealSense.enumerate_devices")
        import pyrealsense2 as sdk  # type: ignore
    devices = []
    for device in sdk.context().query_devices():
        devices.append(
            {
                "name": str(device.get_info(sdk.camera_info.name)),
                "serial": str(device.get_info(sdk.camera_info.serial_number)),
                "firmware_version": str(device.get_info(sdk.camera_info.firmware_version)),
            }
        )
    return devices


__all__ = ["CameraIntrinsics", "RealSenseCamera", "enumerate_devices"]
