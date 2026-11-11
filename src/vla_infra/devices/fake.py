"""Deterministic fake devices for M1.

The fake backend implements the same protocols as the future RDK/RealSense
adapters and adds two properties tests rely on:

* all time comes from an injected clock, so frame pacing and sample timestamps
  are deterministic;
* command sinks refuse writes from any thread that did not claim the device.

No module here imports a hardware SDK or opens a network socket.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Deque, Optional, Sequence

import numpy as np

from ..clock import Clock
from ..contracts.observation import ImageFrame, RobotState, RobotStateSample
from ..contracts.pose import CartesianPose
from ..errors import DeviceError, ErrorCode, ValidationError
from ..geometry import quaternion_to_matrix
from .interfaces import CameraFramePair, CameraStats, SingleWriterGuard


class FakeCamera:
    """Frame-paced fake camera.

    ``read()`` returns a new :class:`ImageFrame` only when at least one frame
    period has elapsed since the previous frame.  Repeated calls in between
    return ``None`` and are counted as duplicate reads; the same frame is never
    returned twice as new.
    """

    def __init__(
        self,
        name: str,
        clock: Clock,
        *,
        width: int = 640,
        height: int = 480,
        fps: float = 15.0,
        encoding: str = "rgb8",
        device_latency_ns: int = 2_000_000,
        jitter_ns: int = 0,
        seed: int = 0,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width/height must be positive")
        if fps <= 0:
            raise ValueError("fps must be positive")
        if encoding not in ("rgb8", "depth16"):
            raise ValueError(f"unsupported fake camera encoding {encoding!r}")
        if device_latency_ns < 0 or jitter_ns < 0:
            raise ValueError("latencies must be non-negative")
        self.name = name
        self._clock = clock
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.encoding = encoding
        self._period_ns = int(round(1_000_000_000 / fps))
        self._latency_ns = int(device_latency_ns)
        self._jitter_ns = int(jitter_ns)
        self._seed = int(seed)
        self._started = False
        self._last_host_ns: Optional[int] = None
        self._last_device_ns: Optional[int] = None
        self._frame_number = -1
        self._failures_remaining = 0
        self._stats = CameraStats()

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        self._started = True
        self._last_host_ns = None
        self._last_device_ns = None

    def stop(self) -> None:
        self._started = False

    def inject_failures(self, count: int) -> None:
        """Make the next ``count`` reads fail (simulated USB dropout)."""
        self._failures_remaining = max(0, int(count))

    # -- data ---------------------------------------------------------------
    def read(self) -> Optional[ImageFrame]:
        now = self._clock.now_ns()
        self._stats = _bump(self._stats, read_calls=1)
        if not self._started:
            self._stats = _bump(self._stats, failures=1)
            return None
        if self._failures_remaining > 0:
            self._failures_remaining -= 1
            self._stats = _bump(self._stats, failures=1)
            return None
        if self._last_host_ns is not None:
            if now < self._last_host_ns:
                raise DeviceError(
                    f"camera {self.name}: host clock moved backwards",
                    code=ErrorCode.BUFFER_NOT_MONOTONIC,
                    details={"camera": self.name, "now_ns": now, "last_ns": self._last_host_ns},
                )
            if now - self._last_host_ns < self._period_ns:
                self._stats = _bump(self._stats, duplicate_reads=1)
                return None
        self._frame_number += 1
        frame_number = self._frame_number
        host_ns = now
        device_ns = now - self._latency_ns + self._jitter(frame_number)
        if self._last_device_ns is not None:
            device_ns = max(device_ns, self._last_device_ns + 1)
        device_ns = max(1, device_ns)
        data = self._pattern(frame_number)
        self._last_host_ns = host_ns
        self._last_device_ns = device_ns
        self._stats = CameraStats(
            read_calls=self._stats.read_calls,
            frames_produced=self._stats.frames_produced + 1,
            failures=self._stats.failures,
            duplicate_reads=self._stats.duplicate_reads,
            dropped_frames=self._stats.dropped_frames,
            last_frame_number=frame_number,
        )
        return ImageFrame(
            name=self.name,
            data=data,
            device_time_ns=device_ns,
            host_time_ns=host_ns,
            frame_number=frame_number,
            encoding=self.encoding,
        )

    def read_pair(self) -> Optional[CameraFramePair]:
        frame = self.read()
        if frame is None:
            return None
        return CameraFramePair(color=frame, depth=None)

    @property
    def stats(self) -> CameraStats:
        return self._stats

    # -- helpers ------------------------------------------------------------
    def _jitter(self, frame_number: int) -> int:
        if self._jitter_ns == 0:
            return 0
        span = 2 * self._jitter_ns + 1
        return int((frame_number * 2_654_435_761 + self._seed * 97) % span) - self._jitter_ns

    def _pattern(self, frame_number: int) -> np.ndarray:
        yy, xx = np.indices((self.height, self.width))
        if self.encoding == "depth16":
            depth = (500 + xx * 2 + yy * 3 + frame_number) % 65535
            return depth.astype(np.uint16)
        image = np.empty((self.height, self.width, 3), dtype=np.uint8)
        image[..., 0] = (xx * 3 + frame_number) % 256
        image[..., 1] = (yy * 5 + frame_number) % 256
        image[..., 2] = (xx + yy + frame_number * 7) % 256
        return image


def _bump(stats: CameraStats, **increments: int) -> CameraStats:
    return replace(
        stats,
        **{name: getattr(stats, name) + int(delta) for name, delta in increments.items()},
    )


class FakeGripper:
    """Width-limited fake parallel gripper (metres)."""

    def __init__(
        self,
        *,
        width_min_m: float = 0.0,
        width_max_m: float = 0.1,
        initial_width_m: float = 0.05,
    ) -> None:
        if not (0.0 <= width_min_m < width_max_m):
            raise ValueError("gripper width range is invalid")
        if not (width_min_m <= initial_width_m <= width_max_m):
            raise ValueError("initial gripper width outside limits")
        self.width_min_m = float(width_min_m)
        self.width_max_m = float(width_max_m)
        self._width_m = float(initial_width_m)
        self._last_command: Optional[float] = None

    @property
    def width_m(self) -> float:
        return self._width_m

    @property
    def last_normalized_command(self) -> Optional[float]:
        return self._last_command

    def command_normalized(self, value: float) -> float:
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValidationError(
                "gripper command must be a normalized scalar in [0, 1]",
                code=ErrorCode.SAFETY_GRIPPER,
                details={"value": float(value)},
            )
        self._last_command = float(value)
        self._width_m = self.width_min_m + float(value) * (self.width_max_m - self.width_min_m)
        return self._width_m


@dataclass
class FakeCommand:
    kind: str
    epoch: int
    now_ns: int
    pose: Optional[CartesianPose] = None
    gripper_normalized: Optional[float] = None


class FakeRobotArm:
    """Fake robot implementing both state source and command sink.

    Commands are applied immediately (there is no dynamics model); the executor
    and safety layers are responsible for pacing and limiting.  Command writes
    are guarded by :class:`SingleWriterGuard`.
    """

    def __init__(
        self,
        clock: Clock,
        *,
        name: str = "fake_robot",
        initial_pose: Optional[CartesianPose] = None,
        gripper: Optional[FakeGripper] = None,
        wrench: Optional[Sequence[float]] = None,
        joint_position: Optional[Sequence[float]] = None,
        command_history: int = 2048,
    ) -> None:
        self.name = name
        self._clock = clock
        self._pose = initial_pose or CartesianPose(
            np.array([0.5, 0.0, 0.3]), np.array([0.0, 0.0, 0.0, 1.0])
        )
        self._gripper = gripper or FakeGripper()
        self._wrench = np.zeros(6) if wrench is None else np.asarray(wrench, dtype=np.float64)
        self._joints = (
            np.zeros(7) if joint_position is None else np.asarray(joint_position, dtype=np.float64)
        )
        self._sequence = -1
        self._guard = SingleWriterGuard(name)
        self._commands: Deque[FakeCommand] = deque(maxlen=command_history)
        self._write_count = 0
        self._last_device_ns: Optional[int] = None
        self._last_host_ns: Optional[int] = None

    # -- state source -------------------------------------------------------
    def read_state(self) -> RobotStateSample:
        now = self._clock.now_ns()
        if self._last_host_ns is not None:
            now = max(now, self._last_host_ns + 1)
        self._last_host_ns = now
        self._sequence += 1
        device_ns = max(1, now - 1_000_000)
        if self._last_device_ns is not None:
            device_ns = max(device_ns, self._last_device_ns + 1)
        self._last_device_ns = device_ns
        rot6d = np.concatenate(
            [quaternion_to_matrix(self._pose.quaternion_xyzw)[:, 0], quaternion_to_matrix(self._pose.quaternion_xyzw)[:, 1]]
        )
        state = RobotState(
            joint_position=self._joints,
            joint_velocity=np.zeros(7),
            tcp_position=self._pose.position,
            tcp_quaternion_xyzw=self._pose.quaternion_xyzw,
            tcp_rotation_6d=rot6d,
            tcp_velocity=np.zeros(6),
            wrench_tcp=self._wrench,
            gripper_width=np.array([self._gripper.width_m]),
        )
        return RobotStateSample(
            state=state,
            device_time_ns=device_ns,
            host_time_ns=now,
            sequence_id=self._sequence,
        )

    # -- command sink -------------------------------------------------------
    def claim_writer(self, owner: str) -> None:
        self._guard.claim(owner)

    def release_writer(self) -> None:
        self._guard.release()

    def command_pose(self, pose: CartesianPose, *, epoch: int, now_ns: int) -> None:
        self._guard.assert_writer(caller="command_pose")
        if not isinstance(pose, CartesianPose):
            raise ValidationError("command_pose expects a CartesianPose")
        self._pose = pose
        self._write_count += 1
        self._commands.append(FakeCommand(kind="pose", epoch=int(epoch), now_ns=int(now_ns), pose=pose))

    def command_gripper(self, normalized: float, *, epoch: int, now_ns: int) -> None:
        self._guard.assert_writer(caller="command_gripper")
        self._gripper.command_normalized(normalized)
        self._write_count += 1
        self._commands.append(
            FakeCommand(
                kind="gripper",
                epoch=int(epoch),
                now_ns=int(now_ns),
                gripper_normalized=float(normalized),
            )
        )

    # -- test helpers -------------------------------------------------------
    @property
    def pose(self) -> CartesianPose:
        return self._pose

    @property
    def gripper(self) -> FakeGripper:
        return self._gripper

    @property
    def write_count(self) -> int:
        return self._write_count

    @property
    def commands(self) -> Sequence[FakeCommand]:
        return tuple(self._commands)

    @property
    def last_command(self) -> Optional[FakeCommand]:
        return self._commands[-1] if self._commands else None

    @property
    def wrench(self) -> np.ndarray:
        return self._wrench

    def set_wrench(self, wrench: Sequence[float]) -> None:
        self._wrench = np.asarray(wrench, dtype=np.float64)

    def set_joints(self, joints: Sequence[float]) -> None:
        self._joints = np.asarray(joints, dtype=np.float64)

    def clear_commands(self) -> None:
        self._commands.clear()
        self._write_count = 0
