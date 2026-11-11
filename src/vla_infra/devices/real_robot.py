"""Composite real robot command sink (arm + gripper), M5 Round 01.

The composite never makes two writes atomic.  Pose and gripper writes are
separate; a failure is surfaced so the executor can audit partial writes and
revoke authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..contracts.pose import CartesianPose
from .flexiv_rdk import FlexivRdkNrtCommandSink
from .gn01 import Gn01CommandSink


@dataclass(frozen=True)
class CompositeStatus:
    pose_writes: int = 0
    gripper_writes: int = 0
    partial_writes: int = 0
    last_error: str = ""

    def as_dict(self) -> dict:
        return {
            "pose_writes": self.pose_writes,
            "gripper_writes": self.gripper_writes,
            "partial_writes": self.partial_writes,
            "last_error": self.last_error,
        }


class RealRobotCommandSink:
    def __init__(self, arm: FlexivRdkNrtCommandSink, gripper: Gn01CommandSink) -> None:
        if not isinstance(arm, FlexivRdkNrtCommandSink):
            raise TypeError("arm must be FlexivRdkNrtCommandSink")
        if not isinstance(gripper, Gn01CommandSink):
            raise TypeError("gripper must be Gn01CommandSink")
        self._arm = arm
        self._gripper = gripper
        self._status = CompositeStatus()

    @property
    def status(self) -> CompositeStatus:
        return self._status

    def connect(self) -> None:
        self._arm.connect()
        try:
            self._gripper.connect()
        except Exception:
            # The arm connection is read/write only after explicit enable; keep
            # the failure visible and do not enable anything here.
            raise

    def claim_writer(self, owner: str) -> None:
        self._arm.claim_writer(owner)
        try:
            self._gripper.claim_writer(owner)
        except Exception:
            self._arm.release_writer()
            raise

    def release_writer(self) -> None:
        self.revoke()
        try:
            self._gripper.release_writer()
        finally:
            self._arm.release_writer()

    def command_pose(self, pose: CartesianPose, *, epoch: int, now_ns: int) -> None:
        try:
            self._arm.command_pose(pose, epoch=epoch, now_ns=now_ns)
        except Exception as exc:
            self._status = _replace(self._status, last_error=f"pose_failed:{type(exc).__name__}")
            raise
        self._status = _replace(self._status, pose_writes=self._status.pose_writes + 1)

    def command_gripper(self, normalized: float, *, epoch: int, now_ns: int) -> None:
        try:
            self._gripper.command_gripper(normalized, epoch=epoch, now_ns=now_ns)
        except Exception as exc:
            self._status = _replace(
                self._status,
                partial_writes=self._status.partial_writes + 1,
                last_error=f"gripper_failed:{type(exc).__name__}",
            )
            raise
        self._status = _replace(self._status, gripper_writes=self._status.gripper_writes + 1)

    def revoke(self) -> None:
        self._arm.revoke()
        self._gripper.revoke()

    def stop_motion(self, timeout_s: float = 2.0) -> bool:
        arm_ok = self._arm.stop_motion(timeout_s)
        gripper_ok = self._gripper.stop(timeout_s)
        self.revoke()
        if not (arm_ok and gripper_ok):
            self._status = _replace(
                self._status,
                last_error=f"stop_failed:arm={arm_ok},gripper={gripper_ok}",
            )
            return False
        return True

    def close(self) -> None:
        self._gripper.close()
        self._arm.close()


def _replace(status: CompositeStatus, **changes: Any) -> CompositeStatus:
    from dataclasses import replace

    return replace(status, **changes)


__all__ = ["CompositeStatus", "RealRobotCommandSink"]
