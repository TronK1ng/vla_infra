"""Single-writer Cartesian executor.

The executor is the only component allowed to call a robot command sink.  It
owns no network or camera code; actions arrive through a bounded
:class:`~vla_infra.execution.inbox.ActionInbox`, are validated by the safety
gate and interpolated before every write.  Every write happens inside
``ControlAuthority.write_permit`` so HOLD/FAULT revocation is synchronous with
respect to the device.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Optional

import numpy as np

from ..clock import Clock
from ..contracts.action import ActionChunk
from ..contracts.observation import RobotStateSample
from ..devices.interfaces import RobotCommandSink
from ..errors import ErrorCode, NotArmedError, SafetyViolation, VlaInfraError
from ..geometry import rotation_vector_between
from .authority import ControlAuthority
from .inbox import ActionInbox, InboxEntry
from .interpolation import CartesianPlan, build_plan
from .safety import CommandTarget, SafetyGate

#: Normalised gripper values closer than this are treated as unchanged *within
#: one authority epoch*.  The unit is normalised gripper command (0..1).
GRIPPER_DEDUP_TOLERANCE = 1e-4


class ExecutorEventKind(str, Enum):
    PLAN_STARTED = "plan_started"
    PLAN_COMPLETED = "plan_completed"
    PLAN_REJECTED = "plan_rejected"
    PLAN_EXPIRED = "plan_expired"
    IDLE = "idle"
    HELD = "held"
    STALE_STATE = "stale_state"
    WRITE_FAILED = "write_failed"
    WRITE_PARTIAL = "write_partial"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True)
class ExecutorEvent:
    kind: ExecutorEventKind
    time_ns: int
    message: str = ""
    error_code: Optional[ErrorCode] = None
    epoch: int = 0
    #: Per-write audit: pose may have succeeded while gripper failed.
    pose_ok: bool = False
    gripper_ok: bool = False
    gripper_sent: bool = False


@dataclass(frozen=True)
class ExecutionReport:
    kind: str
    wrote: bool
    target: Optional[CommandTarget] = None
    message: str = ""
    # Four-stage audit correlation (M3):
    # raw chunk -> SafetyGate filtered target -> interpolated commanded target
    # -> measured state.  ``command_status`` is not_attempted/attempted/partial/
    # succeeded, so a HOLD/reject never claims a command was sent.
    chunk: Optional[ActionChunk] = None
    chunk_epoch: Optional[int] = None
    chunk_received_time_ns: Optional[int] = None
    filtered_target: Optional[CommandTarget] = None
    command_status: str = "not_attempted"
    #: Exact state sample this tick read and submitted to the safety gate.
    state_sample: Optional[RobotStateSample] = None
    pose_attempted: bool = False
    gripper_attempted: bool = False
    pose_ok: bool = False
    gripper_ok: bool = False


@dataclass(frozen=True)
class ExecutorStats:
    ticks: int = 0
    writes: int = 0
    plans_started: int = 0
    plans_completed: int = 0
    plans_rejected: int = 0
    plans_expired: int = 0
    idle_ticks: int = 0
    held_ticks: int = 0
    pose_writes: int = 0
    gripper_writes: int = 0
    partial_writes: int = 0
    write_failures: int = 0
    internal_errors: int = 0
    last_epoch: int = 0

    def as_dict(self) -> dict:
        return {
            "ticks": self.ticks,
            "writes": self.writes,
            "plans_started": self.plans_started,
            "plans_completed": self.plans_completed,
            "plans_rejected": self.plans_rejected,
            "plans_expired": self.plans_expired,
            "idle_ticks": self.idle_ticks,
            "held_ticks": self.held_ticks,
            "pose_writes": self.pose_writes,
            "gripper_writes": self.gripper_writes,
            "partial_writes": self.partial_writes,
            "write_failures": self.write_failures,
            "internal_errors": self.internal_errors,
            "last_epoch": self.last_epoch,
        }


@dataclass
class _ActivePlan:
    plan: CartesianPlan
    entry: InboxEntry
    previous_target: CommandTarget
    last_write_ns: int
    first_write: bool = True
    previous_linear_velocity: Optional[np.ndarray] = None
    previous_angular_velocity: Optional[np.ndarray] = None


class Executor:
    def __init__(
        self,
        *,
        clock: Clock,
        robot: RobotCommandSink,
        state_provider: Callable[[], Optional[RobotStateSample]],
        authority: ControlAuthority,
        inbox: ActionInbox,
        safety: SafetyGate,
        name: str = "executor",
        tick_period_ns: int = 10_000_000,
        plan_grace_ns: int = 250_000_000,
        max_state_age_ns: int = 200_000_000,
        on_event: Optional[Callable[[ExecutorEvent], None]] = None,
    ) -> None:
        if tick_period_ns <= 0:
            raise ValueError("tick_period_ns must be positive")
        if plan_grace_ns < 0 or max_state_age_ns <= 0:
            raise ValueError("plan grace / state age values are invalid")
        self._clock = clock
        self._robot = robot
        self._state_provider = state_provider
        self._authority = authority
        self._inbox = inbox
        self._safety = safety
        self.name = name
        self.tick_period_ns = int(tick_period_ns)
        self.plan_grace_ns = int(plan_grace_ns)
        self.max_state_age_ns = int(max_state_age_ns)
        self._on_event = on_event
        self._active: Optional[_ActivePlan] = None
        self._stats = ExecutorStats()
        self._idle_emitted = False
        self._started = False
        self._last_gripper_command: Optional[float] = None
        self._last_gripper_epoch: Optional[int] = None
        self._hard_faulted = False
        self._last_tick_ns: Optional[int] = None

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        self._robot.claim_writer(self.name)
        self._started = True

    @property
    def started(self) -> bool:
        return self._started

    @property
    def stats(self) -> ExecutorStats:
        return self._stats

    @property
    def active_plan(self) -> Optional[CartesianPlan]:
        return None if self._active is None else self._active.plan

    # -- main step ----------------------------------------------------------
    def tick(self, now_ns: Optional[int] = None) -> ExecutionReport:
        """Run one tick; any unexpected exception is fail-closed before re-raising."""
        if self._hard_faulted:
            return ExecutionReport(kind="faulted", wrote=False, message="executor is faulted")
        try:
            return self._tick_impl(now_ns)
        except Exception as exc:
            now = self._clock.now_ns() if now_ns is None else int(now_ns)
            self._stats = _bump(self._stats, internal_errors=1)
            self._fail_closed(
                ExecutorEventKind.INTERNAL_ERROR,
                f"unhandled executor exception: {type(exc).__name__}: {exc}",
                ErrorCode.INTERNAL_ERROR,
                now_ns=now,
                hard=True,
            )
            raise VlaInfraError(
                "executor internal error; authority revoked",
                code=ErrorCode.INTERNAL_ERROR,
                details={"reason": type(exc).__name__},
            ) from exc

    def _tick_impl(self, now_ns: Optional[int]) -> ExecutionReport:
        now = self._clock.now_ns() if now_ns is None else int(now_ns)
        if not self._started:
            raise RuntimeError("Executor.start() must be called before tick()")
        self._stats = _bump(self._stats, ticks=1)
        previous_tick_ns = self._last_tick_ns
        self._last_tick_ns = now

        if not self._authority.running:
            if self._active is not None:
                self._drop_active(ExecutorEventKind.HELD, "control authority is no longer running")
            self._idle_emitted = False
            self._stats = _bump(self._stats, held_ticks=1)
            return ExecutionReport(kind="unarmed", wrote=False)

        state = self._state_provider()
        if state is None:
            self._fail_closed(
                ExecutorEventKind.STALE_STATE,
                "no robot state sample available",
                ErrorCode.STALE_OBSERVATION,
                now_ns=now,
                hard=False,
            )
            return ExecutionReport(kind="no_state", wrote=False)
        age = now - state.host_time_ns
        if age < -self.max_state_age_ns or age > self.max_state_age_ns:
            self._fail_closed(
                ExecutorEventKind.STALE_STATE,
                f"robot state age {age / 1e6:.1f} ms",
                ErrorCode.STALE_OBSERVATION,
                now_ns=now,
                hard=False,
            )
            return ExecutionReport(kind="stale_state", wrote=False, message=f"state age {age} ns")
        try:
            self._safety.check_state(state.state)
        except SafetyViolation as exc:
            self._stats = _bump(self._stats, plans_rejected=1)
            self._fail_closed(
                ExecutorEventKind.PLAN_REJECTED,
                exc.message,
                exc.code,
                now_ns=now,
                hard=False,
            )
            return ExecutionReport(
                kind="rejected",
                wrote=False,
                message=exc.message,
                state_sample=state,
            )

        if self._active is None:
            entry = self._inbox.take(
                now_ns=now, epoch=self._authority.epoch, max_age_ns=self._safety.profile.max_chunk_age_ns
            )
            if entry is None:
                if not self._idle_emitted:
                    self._emit(ExecutorEventKind.IDLE, now, "no chunk pending")
                    self._idle_emitted = True
                self._stats = _bump(self._stats, idle_ticks=1)
                return ExecutionReport(kind="idle", wrote=False, state_sample=state)
            try:
                targets = self._safety.check_chunk(
                    entry.chunk,
                    state.state,
                    received_time_ns=entry.received_time_ns,
                    now_ns=now,
                )
            except SafetyViolation as exc:
                self._stats = _bump(self._stats, plans_rejected=1)
                self._fail_closed(
                    ExecutorEventKind.PLAN_REJECTED,
                    exc.message,
                    exc.code,
                    now_ns=now,
                    hard=False,
                )
                return ExecutionReport(
                    kind="rejected",
                    wrote=False,
                    message=exc.message,
                    chunk=entry.chunk,
                    chunk_epoch=entry.epoch,
                    chunk_received_time_ns=entry.received_time_ns,
                    command_status="not_attempted",
                    state_sample=state,
                )
            profile = self._safety.profile
            width = state.state.gripper_width_m
            span = profile.gripper_width_max_m - profile.gripper_width_min_m
            start_gripper = 0.0 if span <= 0 else float(np.clip((width - profile.gripper_width_min_m) / span, 0.0, 1.0))
            plan = build_plan(
                targets,
                start_pose=state.state.pose,
                start_gripper=start_gripper,
                start_time_ns=now,
                action_dt_s=entry.chunk.action_dt_s,
                profile=profile,
            )
            self._active = _ActivePlan(
                plan=plan,
                entry=entry,
                previous_target=CommandTarget(pose=state.state.pose, gripper_normalized=start_gripper),
                last_write_ns=(
                    previous_tick_ns
                    if previous_tick_ns is not None and previous_tick_ns < now
                    else now - self.tick_period_ns
                ),
                previous_linear_velocity=np.asarray(state.state.tcp_velocity[:3], dtype=np.float64),
                previous_angular_velocity=np.asarray(state.state.tcp_velocity[3:], dtype=np.float64),
            )
            self._idle_emitted = False
            self._stats = replace(
                _bump(self._stats, plans_started=1), last_epoch=entry.epoch
            )
            self._emit(ExecutorEventKind.PLAN_STARTED, now, f"horizon={entry.chunk.horizon}", None, entry.epoch)

        active = self._active
        assert active is not None
        if now > active.plan.end_time_ns + self.plan_grace_ns:
            self._stats = _bump(self._stats, plans_expired=1)
            self._fail_closed(
                ExecutorEventKind.PLAN_EXPIRED,
                "plan deadline passed before it finished",
                ErrorCode.STALE_PLAN,
                now_ns=now,
                hard=False,
            )
            return ExecutionReport(
                kind="expired",
                wrote=False,
                chunk=active.entry.chunk,
                chunk_epoch=active.entry.epoch,
                chunk_received_time_ns=active.entry.received_time_ns,
                command_status="not_attempted",
                state_sample=state,
            )

        sample = active.plan.sample(now)
        # Even the first write of a plan runs the full per-tick gate.  The
        # previous target is the measured pose and the previous twist is the
        # measured TCP twist, so rearm/new-plan boundaries cannot bypass it.
        dt_ns = max(1, now - active.last_write_ns)
        filtered_target = active.plan.targets[sample.segment_index]
        try:
            self._safety.check_tick(
                active.previous_target,
                sample.target,
                dt_ns,
                previous_linear_velocity=active.previous_linear_velocity,
                previous_angular_velocity=active.previous_angular_velocity,
            )
        except SafetyViolation as exc:
            self._stats = _bump(self._stats, plans_rejected=1)
            self._fail_closed(
                ExecutorEventKind.PLAN_REJECTED,
                exc.message,
                exc.code,
                now_ns=now,
                hard=False,
            )
            return ExecutionReport(
                kind="rejected",
                wrote=False,
                message=exc.message,
                chunk=active.entry.chunk,
                chunk_epoch=active.entry.epoch,
                chunk_received_time_ns=active.entry.received_time_ns,
                filtered_target=filtered_target,
                command_status="not_attempted",
                state_sample=state,
            )
        send_gripper = self._should_send_gripper(
            sample.target.gripper_normalized, active.entry.epoch
        )
        pose_ok = False
        gripper_ok = False
        gripper_sent = False
        pose_attempted = False
        gripper_attempted = False
        permit_acquired = False
        try:
            with self._authority.write_permit(active.entry.epoch):
                permit_acquired = True
                # pose and gripper are separate device calls; the audit fields on
                # the failure event make a partial success explicit.
                pose_attempted = True
                self._robot.command_pose(sample.target.pose, epoch=active.entry.epoch, now_ns=now)
                pose_ok = True
                if send_gripper:
                    gripper_attempted = True
                    gripper_sent = True
                    self._robot.command_gripper(
                        sample.target.gripper_normalized, epoch=active.entry.epoch, now_ns=now
                    )
                    gripper_ok = True
        except Exception as exc:
            if isinstance(exc, NotArmedError) and not permit_acquired:
                # The authority was revoked between plan selection and permit
                # acquisition.  This is the only benign NotArmedError: no device
                # call was attempted.
                self._drop_active(ExecutorEventKind.HELD, exc.message, exc.code)
                return ExecutionReport(
                    kind="held",
                    wrote=False,
                    message=exc.message,
                    chunk=active.entry.chunk,
                    chunk_epoch=active.entry.epoch,
                    chunk_received_time_ns=active.entry.received_time_ns,
                    filtered_target=filtered_target,
                    command_status="not_attempted",
                )
            # Any exception raised *inside* the permit (including a
            # device-level NotArmedError) is a write failure, because the
            # authority lock makes a concurrent HOLD impossible here.
            if isinstance(exc, NotArmedError):
                code = ErrorCode.DEVICE_FAULT
            elif isinstance(exc, VlaInfraError):
                code = exc.code
            else:
                code = ErrorCode.INTERNAL_ERROR
            if pose_ok and send_gripper and not gripper_ok:
                self._stats = _bump(self._stats, partial_writes=1)
                kind = ExecutorEventKind.WRITE_PARTIAL
                message = f"pose succeeded but gripper failed: {exc}"
            else:
                self._stats = _bump(self._stats, write_failures=1)
                kind = ExecutorEventKind.WRITE_FAILED
                message = f"pose/gripper write failed: {exc}"
            self._fail_closed(
                kind,
                message,
                code,
                now_ns=now,
                hard=True,
                pose_ok=pose_ok,
                gripper_ok=gripper_ok,
                gripper_sent=gripper_sent,
                epoch=active.entry.epoch,
            )
            return ExecutionReport(
                kind=kind.value,
                wrote=pose_ok,
                message=message,
                target=sample.target,
                chunk=active.entry.chunk,
                chunk_epoch=active.entry.epoch,
                chunk_received_time_ns=active.entry.received_time_ns,
                filtered_target=filtered_target,
                command_status="partial" if (pose_ok and send_gripper and not gripper_ok) else "attempted",
                state_sample=state,
                pose_attempted=pose_attempted,
                gripper_attempted=gripper_attempted,
                pose_ok=pose_ok,
                gripper_ok=gripper_ok,
            )
        self._stats = _bump(
            self._stats,
            writes=1,
            pose_writes=1,
            gripper_writes=1 if gripper_sent else 0,
        )
        if gripper_sent:
            self._last_gripper_command = sample.target.gripper_normalized
            self._last_gripper_epoch = active.entry.epoch
        write_dt_ns = max(1, now - active.last_write_ns)
        active.previous_linear_velocity = (
            np.asarray(sample.target.pose.position, dtype=np.float64)
            - np.asarray(active.previous_target.pose.position, dtype=np.float64)
        ) / (write_dt_ns / 1_000_000_000.0)
        active.previous_angular_velocity = (
            rotation_vector_between(
                active.previous_target.pose.quaternion_xyzw,
                sample.target.pose.quaternion_xyzw,
            )
            / (write_dt_ns / 1_000_000_000.0)
        )
        active.previous_target = sample.target
        active.last_write_ns = now
        active.first_write = False
        if active.plan.finished(now):
            self._stats = _bump(self._stats, plans_completed=1)
            self._emit(ExecutorEventKind.PLAN_COMPLETED, now, f"steps={len(active.plan.targets)}")
            self._active = None
            return ExecutionReport(
                kind="completed",
                wrote=True,
                target=sample.target,
                chunk=active.entry.chunk,
                chunk_epoch=active.entry.epoch,
                chunk_received_time_ns=active.entry.received_time_ns,
                filtered_target=filtered_target,
                command_status="succeeded",
                state_sample=state,
                pose_attempted=pose_attempted,
                gripper_attempted=gripper_attempted,
                pose_ok=pose_ok,
                gripper_ok=gripper_ok,
            )
        return ExecutionReport(
            kind="wrote",
            wrote=True,
            target=sample.target,
            chunk=active.entry.chunk,
            chunk_epoch=active.entry.epoch,
            chunk_received_time_ns=active.entry.received_time_ns,
            filtered_target=filtered_target,
            command_status="succeeded",
            state_sample=state,
            pose_attempted=pose_attempted,
            gripper_attempted=gripper_attempted,
            pose_ok=pose_ok,
            gripper_ok=gripper_ok,
        )

    # -- loop ---------------------------------------------------------------
    def run_forever(
        self,
        stop_event: threading.Event,
        *,
        on_tick: Optional[Callable[[int], None]] = None,
    ) -> None:
        next_tick = self._clock.now_ns()
        while not stop_event.is_set():
            now = self._clock.now_ns()
            try:
                self.tick(now)
            except Exception as exc:
                if not self._hard_faulted:
                    self._stats = _bump(self._stats, internal_errors=1)
                    self._fail_closed(
                        ExecutorEventKind.INTERNAL_ERROR,
                        f"executor loop error: {type(exc).__name__}: {exc}",
                        ErrorCode.INTERNAL_ERROR,
                        now_ns=now,
                        hard=True,
                    )
                break
            if self._hard_faulted:
                break
            if on_tick is not None:
                try:
                    on_tick(now)
                except Exception as exc:
                    self._stats = _bump(self._stats, internal_errors=1)
                    self._fail_closed(
                        ExecutorEventKind.INTERNAL_ERROR,
                        f"executor tick callback failed: {type(exc).__name__}",
                        ErrorCode.INTERNAL_ERROR,
                        now_ns=now,
                        hard=True,
                    )
                    break
            next_tick += self.tick_period_ns
            current = self._clock.now_ns()
            if next_tick <= current:
                next_tick = current + self.tick_period_ns
            self._clock.sleep(max(1, next_tick - current))

    # -- helpers ------------------------------------------------------------
    def _should_send_gripper(self, normalized: float, epoch: int) -> bool:
        """Suppress redundant gripper commands only within one authority epoch."""
        if self._last_gripper_command is None or self._last_gripper_epoch != epoch:
            return True
        return abs(float(normalized) - self._last_gripper_command) > GRIPPER_DEDUP_TOLERANCE

    def _fail_closed(
        self,
        kind: ExecutorEventKind,
        message: str,
        code: Optional[ErrorCode],
        *,
        now_ns: Optional[int] = None,
        hard: bool,
        pose_ok: bool = False,
        gripper_ok: bool = False,
        gripper_sent: bool = False,
        epoch: Optional[int] = None,
    ) -> None:
        """Revoke the authority *before* recording an execution failure.

        This local barrier does not depend on the event callback: even if a
        runtime forgets to map an event to HOLD/FAULT, no further command can be
        issued with the current epoch.
        """
        now = self._clock.now_ns() if now_ns is None else int(now_ns)
        if epoch is None:
            epoch = self._active.entry.epoch if self._active is not None else self._authority.epoch
        self._active = None
        self._idle_emitted = False
        self._last_gripper_command = None
        self._last_gripper_epoch = None
        if hard:
            self._hard_faulted = True
        try:
            self._authority.revoke(f"executor:{kind.value}")
        except Exception:  # pragma: no cover - revocation must not mask the failure
            pass
        self._emit(
            kind,
            now,
            message,
            code,
            epoch,
            pose_ok=pose_ok,
            gripper_ok=gripper_ok,
            gripper_sent=gripper_sent,
        )

    def _drop_active(
        self,
        kind: ExecutorEventKind,
        message: str,
        code: Optional[ErrorCode] = None,
    ) -> None:
        if self._active is None:
            return
        epoch = self._active.entry.epoch
        self._active = None
        # A HOLD boundary invalidates the gripper de-duplication cache too.
        self._last_gripper_command = None
        self._last_gripper_epoch = None
        self._emit(kind, self._clock.now_ns(), message, code, epoch)

    def _emit(
        self,
        kind: ExecutorEventKind,
        now_ns: int,
        message: str = "",
        code: Optional[ErrorCode] = None,
        epoch: int = 0,
        *,
        pose_ok: bool = False,
        gripper_ok: bool = False,
        gripper_sent: bool = False,
    ) -> None:
        if self._on_event is None:
            return
        event = ExecutorEvent(
            kind=kind,
            time_ns=int(now_ns),
            message=message,
            error_code=code,
            epoch=int(epoch),
            pose_ok=pose_ok,
            gripper_ok=gripper_ok,
            gripper_sent=gripper_sent,
        )
        try:
            self._on_event(event)
        except Exception:  # pragma: no cover - callback must not break control
            pass


def _bump(stats: ExecutorStats, **increments: int) -> ExecutorStats:
    """Additive counter update (never overwrite counters with 1)."""
    return replace(
        stats,
        **{name: getattr(stats, name) + int(delta) for name, delta in increments.items()},
    )


__all__ = [
    "ExecutionReport",
    "Executor",
    "ExecutorEvent",
    "ExecutorEventKind",
    "ExecutorStats",
]
