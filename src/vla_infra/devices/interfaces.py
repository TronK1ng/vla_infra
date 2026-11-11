"""Device interfaces shared by fake and real backends.

Every backend implements exactly these protocols, so the synchronizer,
executor and CLI do not branch on fake/real.  Command sinks enforce a
single-writer contract: only the thread that claimed the device may write to
it, which makes accidental "network callback writes RDK" bugs fail loudly.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

from ..contracts.observation import ImageFrame, RobotStateSample
from ..contracts.pose import CartesianPose
from ..errors import DeviceError, ErrorCode, SingleWriterViolationError


@dataclass(frozen=True)
class CameraFramePair:
    color: ImageFrame
    depth: Optional[ImageFrame] = None


@dataclass(frozen=True)
class CameraStats:
    read_calls: int = 0
    frames_produced: int = 0
    failures: int = 0
    duplicate_reads: int = 0
    dropped_frames: int = 0
    last_frame_number: int = -1

    def as_dict(self) -> dict:
        return {
            "read_calls": self.read_calls,
            "frames_produced": self.frames_produced,
            "failures": self.failures,
            "duplicate_reads": self.duplicate_reads,
            "dropped_frames": self.dropped_frames,
            "last_frame_number": self.last_frame_number,
        }


@runtime_checkable
class CameraSource(Protocol):
    name: str

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def read(self) -> Optional[ImageFrame]:
        """Return the next color frame, or ``None`` if no new frame is available.

        Must never return a stale frame twice as if it were new.  Must not
        block for longer than one frame period.
        """

    def read_pair(self) -> Optional[CameraFramePair]:
        """Return color (+ optional aligned depth) from one frameset."""

    @property
    def stats(self) -> CameraStats: ...


@runtime_checkable
class RobotStateSource(Protocol):
    def read_state(self) -> RobotStateSample: ...


@runtime_checkable
class RobotCommandSink(Protocol):
    def claim_writer(self, owner: str) -> None: ...

    def command_pose(self, pose: CartesianPose, *, epoch: int, now_ns: int) -> None: ...

    def command_gripper(self, normalized: float, *, epoch: int, now_ns: int) -> None: ...


class SingleWriterGuard:
    """Thread-affine write ownership check (one device, one writer)."""

    def __init__(self, device_name: str) -> None:
        self._device_name = device_name
        self._lock = threading.Lock()
        self._owner_thread: Optional[int] = None
        self._owner_name: Optional[str] = None

    def claim(self, owner: str) -> None:
        tid = threading.get_ident()
        with self._lock:
            if self._owner_thread is None:
                self._owner_thread = tid
                self._owner_name = owner
                return
            if self._owner_thread != tid:
                raise SingleWriterViolationError(
                    f"{self._device_name}: command attempted from thread {tid} "
                    f"but the writer is thread {self._owner_thread} ({self._owner_name})",
                    details={
                        "device": self._device_name,
                        "owner_thread": self._owner_thread,
                        "owner": self._owner_name,
                        "caller_thread": tid,
                    },
                )
            if self._owner_name != owner:
                raise SingleWriterViolationError(
                    f"{self._device_name}: thread already claimed by {self._owner_name!r}",
                    details={"device": self._device_name, "owner": self._owner_name, "caller": owner},
                )

    def assert_writer(self, *, caller: str = "") -> None:
        tid = threading.get_ident()
        with self._lock:
            if self._owner_thread is None:
                raise DeviceError(
                    f"{self._device_name}: command sink was not claimed by any writer",
                    code=ErrorCode.DEVICE_NOT_CONNECTED,
                    details={"device": self._device_name, "caller": caller},
                )
            if self._owner_thread != tid:
                raise SingleWriterViolationError(
                    f"{self._device_name}: command from non-owner thread {tid} "
                    f"(owner {self._owner_thread})",
                    details={
                        "device": self._device_name,
                        "owner_thread": self._owner_thread,
                        "caller_thread": tid,
                    },
                )

    def release(self) -> None:
        tid = threading.get_ident()
        with self._lock:
            if self._owner_thread is not None and self._owner_thread != tid:
                raise SingleWriterViolationError(
                    f"{self._device_name}: release attempted from thread {tid} "
                    f"but the writer is thread {self._owner_thread} ({self._owner_name})",
                    details={
                        "device": self._device_name,
                        "owner_thread": self._owner_thread,
                        "caller_thread": tid,
                    },
                )
            self._owner_thread = None
            self._owner_name = None

    @property
    def owner_thread(self) -> Optional[int]:
        with self._lock:
            return self._owner_thread

    @property
    def owner_name(self) -> Optional[str]:
        with self._lock:
            return self._owner_name


__all__ = [
    "CameraFramePair",
    "CameraSource",
    "CameraStats",
    "RobotCommandSink",
    "RobotStateSource",
    "SingleWriterGuard",
]
