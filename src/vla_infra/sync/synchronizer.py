"""Observation synchroniser (master plan §6.3).

* target time is a host monotonic timestamp (the newest required RGB frame in
  the tick); device timestamps are preserved but never used for scheduling;
* robot state is linearly interpolated (position/velocity/joints) and SLERPed
  (orientation), never extrapolated;
* wrench and gripper width use zero-order hold by default (contact signals are
  not smeared across samples);
* an observation whose synchronisation error exceeds the configured bound is
  dropped and counted, never silently built from stale state.
"""

from __future__ import annotations

import threading
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np

from ..clock import Clock
from ..contracts.constants import OBSERVATION_SCHEMA_VERSION
from ..contracts.observation import (
    CanonicalObservation,
    ImageFrame,
    RobotState,
    RobotStateSample,
    SyncInfo,
)
from ..errors import ErrorCode, SyncError
from ..geometry import matrix_to_rotation_6d, quaternion_to_matrix, slerp
from .ring_buffer import TimestampRingBuffer

_INTERPOLATION_POLICIES = ("zoh", "linear")


@dataclass(frozen=True)
class SyncConfig:
    max_sync_error_ns: int = 50_000_000
    max_state_gap_ns: int = 100_000_000
    max_camera_skew_ns: int = 50_000_000
    wrench_policy: str = "zoh"
    gripper_policy: str = "zoh"

    def validate(self) -> None:
        for name in ("max_sync_error_ns", "max_state_gap_ns", "max_camera_skew_ns"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_state_gap_ns < self.max_sync_error_ns:
            raise ValueError("max_state_gap_ns must be >= max_sync_error_ns")
        for name in ("wrench_policy", "gripper_policy"):
            if getattr(self, name) not in _INTERPOLATION_POLICIES:
                raise ValueError(f"{name} must be one of {_INTERPOLATION_POLICIES}")


@dataclass
class SyncStats:
    built: int = 0
    sequence: int = -1
    dropped: Counter = field(default_factory=Counter)
    max_observed_age_ns: int = 0
    max_observed_skew_ns: int = 0

    def as_dict(self) -> dict:
        return {
            "built": self.built,
            "sequence": self.sequence,
            "dropped": dict(self.dropped),
            "max_observed_age_ns": self.max_observed_age_ns,
            "max_observed_skew_ns": self.max_observed_skew_ns,
        }


def _zoh_vec(before: np.ndarray, after: np.ndarray, alpha: float, policy: str) -> np.ndarray:
    if policy == "linear":
        return (1.0 - alpha) * before + alpha * after
    return before.copy()


def interpolate_robot_state(
    before: RobotStateSample,
    after: RobotStateSample,
    alpha: float,
    *,
    wrench_policy: str = "zoh",
    gripper_policy: str = "zoh",
) -> RobotState:
    """Interpolate two samples; ``alpha`` must be in ``[0, 1]`` and never extrapolate."""
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    a_state = before.state
    b_state = after.state
    quat = slerp(a_state.tcp_quaternion_xyzw, b_state.tcp_quaternion_xyzw, alpha)
    rotation_6d = matrix_to_rotation_6d(quaternion_to_matrix(quat))
    return RobotState(
        joint_position=(1.0 - alpha) * a_state.joint_position + alpha * b_state.joint_position,
        joint_velocity=(1.0 - alpha) * a_state.joint_velocity + alpha * b_state.joint_velocity,
        tcp_position=(1.0 - alpha) * a_state.tcp_position + alpha * b_state.tcp_position,
        tcp_quaternion_xyzw=quat,
        tcp_rotation_6d=rotation_6d,
        tcp_velocity=(1.0 - alpha) * a_state.tcp_velocity + alpha * b_state.tcp_velocity,
        wrench_tcp=_zoh_vec(a_state.wrench_tcp, b_state.wrench_tcp, alpha, wrench_policy),
        gripper_width=_zoh_vec(a_state.gripper_width, b_state.gripper_width, alpha, gripper_policy),
    )


class ObservationSynchronizer:
    """Builds canonical observations from camera frames and buffered robot state.

    Threading: :meth:`add_robot_sample` may be called by the robot-state worker;
    :meth:`build` is intended for a single observation thread.
    """

    def __init__(
        self,
        clock: Clock,
        config: Optional[SyncConfig] = None,
        *,
        buffer_capacity: int = 2048,
    ) -> None:
        self._clock = clock
        self.config = config or SyncConfig()
        self.config.validate()
        self.buffer = TimestampRingBuffer(buffer_capacity, name="observation_sync")
        self._stats = SyncStats()
        self._lock = threading.Lock()
        self._age_samples = deque(maxlen=4096)
        self._skew_samples = deque(maxlen=4096)

    # -- inputs -------------------------------------------------------------
    def add_robot_sample(self, sample: RobotStateSample) -> None:
        self.buffer.append(sample)

    # -- build --------------------------------------------------------------
    def build(
        self,
        images: Mapping[str, ImageFrame],
        task: str,
        *,
        target_time_ns: Optional[int] = None,
        extra_dropped: Optional[Mapping[str, int]] = None,
    ) -> CanonicalObservation:
        from ..contracts.constants import REQUIRED_RGB_IMAGES

        for key in REQUIRED_RGB_IMAGES:
            if key not in images:
                self._drop("missing_image", f"missing image {key}")
        required_hosts = [images[key].host_time_ns for key in REQUIRED_RGB_IMAGES]
        if target_time_ns is None:
            target_ns = max(required_hosts)
        else:
            if not isinstance(target_time_ns, int) or target_time_ns <= 0:
                raise ValueError("target_time_ns must be a positive integer")
            target_ns = int(target_time_ns)
            if target_ns < max(required_hosts):
                self._drop("future_image", "target time precedes a required image")
        skew = max(abs(host - target_ns) for host in required_hosts)
        if any(image.host_time_ns > target_ns for image in images.values()):
            self._drop("future_image", "an image is newer than the observation target")
        if skew > self.config.max_camera_skew_ns:
            self._drop(
                "camera_skew",
                f"camera skew {skew} ns exceeds {self.config.max_camera_skew_ns} ns",
            )
        bracket = self.buffer.bracket(target_ns)
        if bracket is None:
            self._drop("robot_stale", "no robot samples bracket the target time")
        assert bracket is not None
        if bracket.gap_ns > self.config.max_state_gap_ns:
            self._drop(
                "state_gap",
                f"state gap {bracket.gap_ns} ns exceeds {self.config.max_state_gap_ns} ns",
            )
        max_age = max(target_ns - bracket.before.host_time_ns, bracket.after.host_time_ns - target_ns)
        if max_age > self.config.max_sync_error_ns:
            self._drop(
                "sync_error",
                f"sync error {max_age} ns exceeds {self.config.max_sync_error_ns} ns",
            )
        alpha = 0.0 if bracket.gap_ns == 0 else (target_ns - bracket.before.host_time_ns) / float(bracket.gap_ns)
        robot = interpolate_robot_state(
            bracket.before,
            bracket.after,
            alpha,
            wrench_policy=self.config.wrench_policy,
            gripper_policy=self.config.gripper_policy,
        )
        dropped = {
            "camera_dropouts": int(dict(extra_dropped or {}).get("camera_dropouts", 0)),
            "sync_camera_skew": int(self._stats.dropped.get("camera_skew", 0)),
            "sync_robot_stale": int(self._stats.dropped.get("robot_stale", 0)),
            "sync_state_gap": int(self._stats.dropped.get("state_gap", 0)),
            "sync_error": int(self._stats.dropped.get("sync_error", 0)),
            "sync_future_image": int(self._stats.dropped.get("future_image", 0)),
        }
        with self._lock:
            self._stats.built += 1
            self._stats.sequence += 1
            sequence_id = self._stats.sequence
            self._stats.max_observed_age_ns = max(self._stats.max_observed_age_ns, max_age)
            self._stats.max_observed_skew_ns = max(self._stats.max_observed_skew_ns, skew)
            self._age_samples.append(int(max_age))
            self._skew_samples.append(int(skew))
        sync = SyncInfo(
            robot_sample_before_ns=bracket.before.host_time_ns,
            robot_sample_after_ns=bracket.after.host_time_ns,
            interpolation_alpha=alpha,
            max_age_ns=max_age,
            dropped_frames=dropped,
            interpolation_applied=True,
            camera_skew_ns=skew,
            state_gap_ns=bracket.gap_ns,
        )
        return CanonicalObservation(
            schema_version=OBSERVATION_SCHEMA_VERSION,
            sequence_id=sequence_id,
            capture_time_ns=target_ns,
            task=task,
            images=images,
            robot=robot,
            sync=sync,
        )

    # -- stats --------------------------------------------------------------
    @property
    def stats(self) -> SyncStats:
        with self._lock:
            return SyncStats(
                built=self._stats.built,
                sequence=self._stats.sequence,
                dropped=Counter(self._stats.dropped),
                max_observed_age_ns=self._stats.max_observed_age_ns,
                max_observed_skew_ns=self._stats.max_observed_skew_ns,
            )

    def sync_report(self) -> dict:
        """Distribution report used by the M2 recorder/preflight."""
        with self._lock:
            ages = tuple(self._age_samples)
            skews = tuple(self._skew_samples)
            stats = self._stats
            dropped = dict(stats.dropped)

        def percentile(samples, fraction):
            if not samples:
                return None
            ordered = sorted(samples)
            return ordered[min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))] 

        def summary(samples):
            if not samples:
                return {
                    "count": 0,
                    "min_ns": None,
                    "p50_ns": None,
                    "p95_ns": None,
                    "p99_ns": None,
                    "max_ns": None,
                    "mean_ns": None,
                }
            return {
                "count": len(samples),
                "min_ns": min(samples),
                "p50_ns": percentile(samples, 0.50),
                "p95_ns": percentile(samples, 0.95),
                "p99_ns": percentile(samples, 0.99),
                "max_ns": max(samples),
                "mean_ns": int(round(sum(samples) / len(samples))),
            }

        return {
            "schema_version": "sync-report/v1",
            "built": stats.built,
            "dropped": dropped,
            "sync_error_ns": summary(ages),
            "camera_skew_ns": summary(skews),
            "config": {
                "max_sync_error_ns": self.config.max_sync_error_ns,
                "max_state_gap_ns": self.config.max_state_gap_ns,
                "max_camera_skew_ns": self.config.max_camera_skew_ns,
                "wrench_policy": self.config.wrench_policy,
                "gripper_policy": self.config.gripper_policy,
            },
        }

    def _drop(self, reason: str, message: str) -> None:
        with self._lock:
            self._stats.dropped[reason] += 1
        raise SyncError(
            f"dropped observation: {message}",
            code=ErrorCode.STALE_OBSERVATION,
            details={"reason": reason},
        )


__all__ = [
    "ObservationSynchronizer",
    "SyncConfig",
    "SyncStats",
    "interpolate_robot_state",
]
