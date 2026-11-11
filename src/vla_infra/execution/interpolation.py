"""Cartesian trajectory interpolation.

* position: linear interpolation;
* orientation: SLERP;
* gripper: linear in normalised width;
* each segment uses the declared ``action_dt_s`` (only raised to the configured
  minimum segment duration).  Velocity feasibility is enforced by the safety
  gate at the nominal timing; the interpolator never silently retimes the
  policy's macro timing.
* **M3 baseline decision:** reject-only.  Velocity-based automatic retiming is
  deliberately not implemented in M3; it is deferred to an independent,
  explicitly audited milestone.  M3 acceptance must not claim retiming support.

The interpolator only ever receives targets that already passed the safety
gate; it performs no clipping of workspace or step violations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from ..contracts.pose import CartesianPose
from .safety import CommandTarget, SafetyProfile

_NS = 1_000_000_000


@dataclass(frozen=True)
class PlanSample:
    target: CommandTarget
    segment_index: int
    alpha_in_segment: float


@dataclass(frozen=True)
class CartesianPlan:
    start_pose: CartesianPose
    start_gripper: float
    start_time_ns: int
    targets: Tuple[CommandTarget, ...]
    durations_ns: Tuple[int, ...]
    segment_starts_ns: Tuple[int, ...]
    end_time_ns: int

    def sample(self, now_ns: int) -> PlanSample:
        """Sample the plan at ``now_ns``, clamped to ``[start_time, end_time]``."""
        if not self.targets:
            raise ValueError("plan has no targets")
        if now_ns <= self.start_time_ns:
            return PlanSample(
                CommandTarget(
                    pose=self.start_pose,
                    gripper_normalized=self.start_gripper,
                    step_index=0,
                ),
                0,
                0.0,
            )
        if now_ns >= self.end_time_ns:
            last = len(self.targets) - 1
            return PlanSample(self.targets[last], last, 1.0)
        index = len(self.targets) - 1
        for i, segment_start in enumerate(self.segment_starts_ns):
            if now_ns < segment_start + self.durations_ns[i]:
                index = i
                break
        segment_start = self.segment_starts_ns[index]
        duration = self.durations_ns[index]
        alpha = 0.0 if duration <= 0 else (now_ns - segment_start) / float(duration)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        from_pose = self.start_pose if index == 0 else self.targets[index - 1].pose
        from_gripper = self.start_gripper if index == 0 else self.targets[index - 1].gripper_normalized
        target = self.targets[index]
        pose = from_pose.interpolate(target.pose, alpha)
        gripper = from_gripper + alpha * (target.gripper_normalized - from_gripper)
        return PlanSample(
            target=CommandTarget(pose=pose, gripper_normalized=float(gripper), step_index=target.step_index),
            segment_index=index,
            alpha_in_segment=alpha,
        )

    def finished(self, now_ns: int) -> bool:
        return now_ns >= self.end_time_ns

    @property
    def total_duration_ns(self) -> int:
        return self.end_time_ns - self.start_time_ns


def _segment_duration_ns(
    action_dt_s: float,
    profile: SafetyProfile,
) -> int:
    """Nominal policy timing; no velocity-based retiming is performed in M1."""
    nominal = int(round(action_dt_s * _NS))
    return max(profile.min_segment_dt_ns, nominal)


def build_plan(
    targets: Sequence[CommandTarget],
    *,
    start_pose: CartesianPose,
    start_gripper: float,
    start_time_ns: int,
    action_dt_s: float,
    profile: SafetyProfile,
) -> CartesianPlan:
    if not targets:
        raise ValueError("cannot build a plan without targets")
    if not isinstance(start_time_ns, int) or start_time_ns <= 0:
        raise ValueError("start_time_ns must be a positive integer")
    durations = []
    segment_starts = []
    cursor = start_time_ns
    for target in targets:
        duration = _segment_duration_ns(action_dt_s, profile)
        segment_starts.append(cursor)
        durations.append(duration)
        cursor += duration
    return CartesianPlan(
        start_pose=start_pose,
        start_gripper=float(start_gripper),
        start_time_ns=start_time_ns,
        targets=tuple(targets),
        durations_ns=tuple(durations),
        segment_starts_ns=tuple(segment_starts),
        end_time_ns=cursor,
    )


__all__ = ["CartesianPlan", "PlanSample", "build_plan"]
