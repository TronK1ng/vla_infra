"""M3 fake/simulation execution runtime.

Scope: fake devices only.  There is no RDK, RealSense, network, or policy
rollout here.  The runtime demonstrates the M3 shape:

* a dedicated 100 Hz executor thread;
* a separate 20 Hz policy/chunk producer and 500 Hz fake state acquisition;
* supervisor watchdogs for robot/camera/policy/executor;
* a bounded audit log that never blocks the control loop;
* deterministic fault-injection entry points for tests and the M3 CLI.

The real 100 Hz RDK runtime is intentionally out of scope for M3 automation.
"""

from __future__ import annotations

import platform
import threading
import time
from collections import deque
from queue import Empty, Full, Queue
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Mapping, Optional

import numpy as np

from ..clock import Clock, MonotonicClock
from ..config import AppConfig
from ..contracts.action import ActionChunk
from ..contracts.constants import IMAGE_KEY_BASE_RGB, IMAGE_KEY_WRIST_RGB
from ..contracts.observation import RobotStateSample
from ..devices.fake import FakeCamera, FakeGripper, FakeRobotArm
from ..errors import VlaInfraError
from ..execution.authority import ControlAuthority
from ..execution.executor import ExecutionReport, Executor, ExecutorEvent
from ..execution.inbox import ActionInbox
from ..execution.safety import CommandTarget, SafetyGate
from ..execution.supervisor import Supervisor, SupervisorConfig, SystemState, WatchdogConfig
from .execution_log import ExecutionLog


class M3StartupError(VlaInfraError):
    """Startup readiness failed before authority was granted."""


@dataclass(frozen=True)
class M3Config:
    control_period_ns: int = 10_000_000       # 100 Hz
    policy_period_ns: int = 50_000_000        # 20 Hz
    state_period_ns: int = 2_000_000          # 500 Hz
    horizon: int = 5
    action_dt_s: float = 0.05
    delta_m: tuple = (0.001, 0.0, 0.0)
    gripper: float = 0.5
    control_thread_join_s: float = 5.0
    startup_timeout_s: float = 2.0
    startup_cleanup_timeout_s: float = 1.0

    def __post_init__(self) -> None:
        for name in ("control_period_ns", "policy_period_ns", "state_period_ns"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 1 <= self.horizon <= 100:
            raise ValueError("horizon must be in [1, 100]")
        if self.action_dt_s <= 0:
            raise ValueError("action_dt_s must be positive")
        if self.startup_timeout_s <= 0:
            raise ValueError("startup_timeout_s must be positive")
        if self.startup_cleanup_timeout_s <= 0:
            raise ValueError("startup_cleanup_timeout_s must be positive")
        if not (0.0 <= self.gripper <= 1.0):
            raise ValueError("gripper must be in [0, 1]")


@dataclass
class _TickMetric:
    tick: int
    host_time_ns: int
    period_ns: Optional[int]
    runtime_ns: int
    jitter_ns: Optional[int]
    overrun: bool
    wrote: bool


@dataclass(frozen=True)
class M3Result:
    final_state: str
    armed: bool
    ticks: int
    writes: int
    policy_chunks: int
    log_path: str
    log_records: int
    log_dropped: int
    metrics: Mapping[str, Any]
    executor: Mapping[str, Any]
    supervisor: Mapping[str, Any]
    error: str = ""
    stop_error: str = ""
    threads: tuple = ()
    lifecycle: str = "NEW"

    def as_dict(self) -> dict:
        return {
            "final_state": self.final_state,
            "armed": self.armed,
            "ticks": self.ticks,
            "writes": self.writes,
            "policy_chunks": self.policy_chunks,
            "log_path": self.log_path,
            "log_records": self.log_records,
            "log_dropped": self.log_dropped,
            "metrics": dict(self.metrics),
            "executor": dict(self.executor),
            "supervisor": {
                "state": self.supervisor.get("state"),
                "epoch": self.supervisor.get("epoch"),
                "latched_fault": self.supervisor.get("latched_fault"),
                "transitions": self.supervisor.get("transitions", []),
            },
            "error": self.error,
            "stop_error": self.stop_error,
            "threads": list(self.threads),
            "lifecycle": self.lifecycle,
        }


class M3Runtime:
    def __init__(
        self,
        app: AppConfig,
        *,
        output: str | Path,
        config: Optional[M3Config] = None,
        clock: Optional[Clock] = None,
        task: str = "m3 fake execution",
        on_tick: Optional[Callable[[ExecutionReport, Mapping[str, Any]], None]] = None,
    ) -> None:
        if app.station.backend != "fake":
            raise VlaInfraError("M3Runtime only supports backend: fake; no real device path")
        self.app = app
        self.config = config or M3Config()
        self._clock = clock or MonotonicClock()
        self._task = task
        self._output = Path(output)
        self._on_tick = on_tick
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._callback_queue: Queue = Queue(maxsize=1)
        self._latest_state: Optional[RobotStateSample] = None
        self._cameras: dict[str, FakeCamera] = {}
        self._last_camera_frames: dict[str, int] = {}
        self._last_camera_ns: dict[str, int] = {}
        self._last_robot_ns: Optional[int] = None
        self._last_policy_ns: Optional[int] = None
        self._last_executor_ns: Optional[int] = None
        self._camera_errors: dict[str, int] = {}
        self._callback_enqueued = 0
        self._callback_processed = 0
        self._callback_dropped = 0
        self._callback_errors = 0
        self._callback_last_drop_tick: Optional[int] = None
        self._callback_last_error_tick: Optional[int] = None
        self._callback_last_error_type: Optional[str] = None
        self._robot: Optional[FakeRobotArm] = None
        self._supervisor: Optional[Supervisor] = None
        self._inbox: Optional[ActionInbox] = None
        self._executor: Optional[Executor] = None
        self._authority: Optional[ControlAuthority] = None
        self._log: Optional[ExecutionLog] = None
        self._metrics: Deque[_TickMetric] = deque(maxlen=200_000)
        self._policy_chunks = 0
        self._sequence = 0
        self._policy_enabled = True
        self._state_enabled = True
        self._camera_enabled = True
        self._started = False
        self._stop_error = ""
        self._lifecycle = "NEW"

    # -- accessors ----------------------------------------------------------
    @property
    def supervisor(self) -> Supervisor:
        assert self._supervisor is not None
        return self._supervisor

    @property
    def executor(self) -> Executor:
        assert self._executor is not None
        return self._executor

    @property
    def log_path(self) -> Path:
        return self._output

    def set_policy_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._policy_enabled = bool(enabled)

    def set_state_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._state_enabled = bool(enabled)

    def set_camera_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._camera_enabled = bool(enabled)

    def submit_actions(
        self,
        actions,
        *,
        capture_offset_ns: int = 0,
        action_frame: str = "base",
        action_type: str = "delta_tcp",
    ) -> bool:
        """Test/fault-injection hook: submit an explicit chunk to the inbox."""
        with self._lock:
            state = self._latest_state
        if state is None:
            return False
        self._sequence += 1
        chunk = ActionChunk(
            request_sequence_id=self._sequence,
            observation_capture_time_ns=max(1, state.host_time_ns + int(capture_offset_ns)),
            action_dt_s=self.config.action_dt_s,
            action_frame=action_frame,
            action_type=action_type,
            actions=np.asarray(actions, dtype=np.float32),
        )
        return self._inbox.submit(  # type: ignore[union-attr]
            chunk,
            epoch=self._authority.epoch,  # type: ignore[union-attr]
            received_time_ns=self._clock.now_ns(),
        )

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._lifecycle != "NEW":
                raise RuntimeError(f"M3Runtime cannot start from {self._lifecycle}; restart is not supported")
            self._lifecycle = "STARTING"
            self._last_robot_ns = None
            self._last_policy_ns = None
            self._last_executor_ns = None
            self._last_camera_ns = {}
            self._last_camera_frames = {}
        try:
            self._build_components()
            self._log = ExecutionLog(self._output)
            self._log.start()
            self._start_threads()
            self._wait_for_readiness()
            self._arm_after_readiness()
        except BaseException:
            self._startup_cleanup()
            raise
        self._started = True
        with self._lock:
            self._lifecycle = "RUNNING"

    def _build_components(self) -> None:
        station = self.app.station
        safety = self.app.safety
        image_keys = {"base": IMAGE_KEY_BASE_RGB, "wrist": IMAGE_KEY_WRIST_RGB}
        for index, (name, camera) in enumerate(station.cameras.items()):
            device = FakeCamera(
                image_keys[name],
                self._clock,
                width=camera.width,
                height=camera.height,
                fps=camera.fps,
                device_latency_ns=int(camera.device_latency_ms * 1_000_000),
                seed=index,
            )
            device.start()
            self._cameras[name] = device
        self._robot = FakeRobotArm(
            self._clock,
            gripper=FakeGripper(
                width_min_m=safety.gripper_width_min_m,
                width_max_m=safety.gripper_width_max_m,
                initial_width_m=(safety.gripper_width_min_m + safety.gripper_width_max_m) / 2.0,
            ),
        )
        self._authority = ControlAuthority(owner="supervisor")
        self._inbox = ActionInbox(self._authority)
        watchdogs = {
            "robot": WatchdogConfig(250_000_000, "fault"),
            "executor": WatchdogConfig(250_000_000, "fault"),
            "policy": WatchdogConfig(1_000_000_000, "hold"),
        }
        for name in station.cameras:
            watchdogs[f"camera:{name}"] = WatchdogConfig(500_000_000, "hold")
        self._supervisor = Supervisor(
            clock=self._clock,
            authority=self._authority,
            inbox=self._inbox,
            config=SupervisorConfig(watchdogs=watchdogs),
        )
        self._executor = Executor(
            clock=self._clock,
            robot=self._robot,
            state_provider=self._state_provider,
            authority=self._authority,
            inbox=self._inbox,
            safety=SafetyGate(safety),
            tick_period_ns=self.config.control_period_ns,
            on_event=self._on_executor_event,
        )

    def _start_threads(self) -> None:
        self._threads = [
            threading.Thread(target=self._state_loop, name="m3-state", daemon=True),
            threading.Thread(target=self._policy_loop, name="m3-policy", daemon=True),
            threading.Thread(target=self._executor_loop, name="m3-executor-100hz", daemon=True),
        ]
        if self._on_tick is not None:
            self._threads.append(
                threading.Thread(target=self._callback_loop, name="m3-callback", daemon=True)
            )
        for thread in self._threads:
            thread.start()

    def _wait_for_readiness(self) -> None:
        assert self._supervisor is not None
        deadline = time.monotonic() + self.config.startup_timeout_s
        while True:
            with self._lock:
                missing = []
                if self._last_robot_ns is None:
                    missing.append("robot")
                for name in self._cameras:
                    if self._last_camera_ns.get(name) is None:
                        missing.append(f"camera:{name}")
                if self._last_policy_ns is None:
                    missing.append("policy")
                if self._last_executor_ns is None:
                    missing.append("executor")
            if not missing:
                return
            if time.monotonic() >= deadline:
                raise M3StartupError(
                    "startup readiness timed out before authority was granted",
                    details={"missing": missing, "timeout_s": self.config.startup_timeout_s},
                )
            self._clock.sleep(5_000_000)

    def _arm_after_readiness(self) -> None:
        assert self._supervisor is not None
        self._supervisor.connect_devices()
        self._supervisor.devices_ready()
        self._supervisor.on_policy_connected()
        now = self._clock.now_ns()
        # No heartbeat is fabricated here.  The supervisor's arm() validates
        # the real per-source watchdog timestamps produced by the threads.
        self._supervisor.arm(operator_ack=True, now_ns=now)
        self._supervisor.start_running(now_ns=now)

    def _startup_cleanup(self) -> None:
        deadline = time.monotonic() + self.config.startup_cleanup_timeout_s
        self._stop.set()
        for thread in self._threads:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(remaining)
        alive = [thread.name for thread in self._threads if thread.is_alive()]
        if alive:
            self._stop_error = "m3_startup_cleanup_failed: " + ",".join(alive)
            with self._lock:
                # Do not close the logger/cameras or claim cleanup while a live
                # thread may still access them.
                self._lifecycle = "START_CLEANUP_FAILED"
                self._started = False
            return
        if self._log is not None:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                closed = self._log.close(timeout_s=remaining)
            except Exception:
                closed = False
            if not closed or self._log.thread_alive():
                self._stop_error = "m3_startup_cleanup_failed: m3-execution-log"
                with self._lock:
                    self._lifecycle = "START_CLEANUP_FAILED"
                    self._started = False
                return
        for camera in self._cameras.values():
            camera.stop()
        if self._supervisor is not None and self._supervisor.state in (SystemState.RUNNING, SystemState.ARMED):
            try:
                self._supervisor.hold("startup_failed")
            except VlaInfraError:
                pass
        with self._lock:
            self._lifecycle = "START_FAILED"
            self._started = False

    def rearm(self) -> None:
        """Explicit HOLD -> ARMED -> RUNNING recovery with real freshness.

        No heartbeat is fabricated.  Every required source must have produced
        new real data after the latest HOLD transition; otherwise rearm is
        refused and writes remain blocked.
        """
        with self._lock:
            if self._lifecycle != "RUNNING":
                raise VlaInfraError(f"rearm is not allowed from lifecycle {self._lifecycle}")
            supervisor = self._supervisor
        if supervisor is None or supervisor.state is not SystemState.HOLD:
            raise VlaInfraError("rearm is only valid from HOLD")
        transitions = supervisor.history()
        hold_times = [t.time_ns for t in transitions if t.to_state is SystemState.HOLD]
        if not hold_times:
            raise VlaInfraError("rearm has no HOLD transition to anchor freshness")
        hold_time = max(hold_times)
        with self._lock:
            robot_ns = self._last_robot_ns
            camera_ns = dict(self._last_camera_ns)
            policy_ns = self._last_policy_ns
            executor_ns = self._last_executor_ns
        missing = []
        if robot_ns is None or robot_ns <= hold_time:
            missing.append("robot")
        for name in self._cameras:
            if camera_ns.get(name) is None or camera_ns[name] <= hold_time:
                missing.append(f"camera:{name}")
        if policy_ns is None or policy_ns <= hold_time:
            missing.append("policy")
        if executor_ns is None or executor_ns <= hold_time:
            missing.append("executor")
        if missing:
            raise VlaInfraError(
                "rearm refused: no fresh real heartbeat after HOLD",
                details={"missing": missing, "hold_time_ns": hold_time},
            )
        now = self._clock.now_ns()
        supervisor.arm(operator_ack=True, now_ns=now)
        supervisor.start_running(now_ns=now)

    def run(self, *, duration_s: float = 1.0) -> M3Result:
        self.start()
        deadline = self._clock.now_ns() + int(duration_s * 1_000_000_000)
        try:
            while self._clock.now_ns() < deadline:
                self._clock.sleep(min(10_000_000, max(1, deadline - self._clock.now_ns())))
        finally:
            return self.stop()

    def stop(self) -> M3Result:
        with self._lock:
            if self._lifecycle == "STOPPED":
                return self._result()
            if self._lifecycle not in ("RUNNING", "STOPPING", "STOP_FAILED"):
                # NEW/STARTING has no live control loop to stop.
                if self._lifecycle == "NEW":
                    self._lifecycle = "STOPPED"
                    return self._result()
            self._lifecycle = "STOPPING"
        deadline = time.monotonic() + self.config.control_thread_join_s
        self._stop.set()

        def remaining() -> float:
            return max(0.0, deadline - time.monotonic())

        # HOLD/revoke runs in a separate thread: if a command sink blocks while
        # holding the authority lock, ``hold`` itself blocks.  The stop total
        # deadline must still be enforceable, so we never call it inline.
        revoke_thread = threading.Thread(
            target=self._graceful_hold,
            name="m3-stop-revoke",
            daemon=True,
        )
        revoke_thread.start()
        revoke_thread.join(remaining())
        for thread in self._threads:
            if remaining() <= 0:
                break
            thread.join(remaining())
        alive = [
            thread.name
            for thread in self._threads
            if thread.is_alive()
        ] + (["m3-stop-revoke"] if revoke_thread.is_alive() else [])
        if alive:
            self._stop_error = "m3_stop_timeout: " + ",".join(alive)
            with self._lock:
                self._lifecycle = "STOP_FAILED"
                self._started = True
            # Do not close the log, stop cameras, or release the writer while a
            # thread may still be using them.  Do not call supervisor.fault()
            # or supervisor.status() here: either may block on the authority
            # lock held by a stuck write.  Return an explicit terminal-timeout
            # snapshot instead.
            return self._result(supervisor_status={
                "state": "STOP_TIMEOUT",
                "epoch": None,
                "latched_fault": None,
                "transitions": [],
            })
        if self._log is not None:
            if not self._log.close(timeout_s=max(0.05, remaining())):
                self._stop_error = "m3_stop_timeout:m3-execution-log"
                with self._lock:
                    self._lifecycle = "STOP_FAILED"
                    self._started = True
                # Keep cameras/live resources untouched; the logger thread is
                # still draining a blocked write/fsync.
                return self._result()
        for camera in self._cameras.values():
            camera.stop()
        with self._lock:
            self._lifecycle = "STOPPED"
            self._started = False
            self._stop_error = ""
        return self._result()

    def _graceful_hold(self) -> None:
        supervisor = self._supervisor
        if supervisor is None:
            return
        if supervisor.state not in (SystemState.RUNNING, SystemState.ARMED):
            return
        try:
            supervisor.hold("m3_stop")
        except VlaInfraError:  # pragma: no cover - defensive
            pass

    # -- provider/loops -----------------------------------------------------
    def _state_provider(self) -> Optional[RobotStateSample]:
        with self._lock:
            return self._latest_state

    def _state_loop(self) -> None:
        assert self._robot is not None
        assert self._supervisor is not None
        next_tick = self._clock.now_ns()
        while not self._stop.is_set():
            with self._lock:
                enabled = self._state_enabled
                cameras_enabled = self._camera_enabled
            if enabled:
                try:
                    sample = self._robot.read_state()
                except Exception:
                    with self._lock:
                        self._camera_errors["state"] = self._camera_errors.get("state", 0) + 1
                    next_tick += self.config.state_period_ns
                    now = self._clock.now_ns()
                    if next_tick > now:
                        self._clock.sleep(next_tick - now)
                    else:
                        next_tick = now
                    continue
                now_state = self._clock.now_ns()
                with self._lock:
                    self._latest_state = sample
                    self._last_robot_ns = now_state
                self._supervisor.heartbeat("robot")
            if cameras_enabled:
                for name, device in self._cameras.items():
                    try:
                        frame = device.read()
                    except Exception:
                        with self._lock:
                            self._camera_errors[name] = self._camera_errors.get(name, 0) + 1
                        continue
                    if frame is None:
                        continue
                    previous = self._last_camera_frames.get(name, -1)
                    if frame.frame_number <= previous:
                        # duplicate/old frame is not a health heartbeat
                        continue
                    self._last_camera_frames[name] = frame.frame_number
                    with self._lock:
                        self._last_camera_ns[name] = self._clock.now_ns()
                    self._supervisor.heartbeat(f"camera:{name}")
            next_tick += self.config.state_period_ns
            now = self._clock.now_ns()
            if next_tick > now:
                self._clock.sleep(next_tick - now)
            else:
                next_tick = now

    def _policy_loop(self) -> None:
        assert self._supervisor is not None
        assert self._inbox is not None
        assert self._authority is not None
        next_tick = self._clock.now_ns()
        while not self._stop.is_set():
            with self._lock:
                enabled = self._policy_enabled
                state = self._latest_state
            if enabled and state is not None:
                try:
                    chunk = self._make_chunk(state)
                except Exception:
                    chunk = None
                if chunk is not None:
                    self._inbox.submit(
                        chunk,
                        epoch=self._authority.epoch,
                        received_time_ns=self._clock.now_ns(),
                    )
                    self._policy_chunks += 1
                    with self._lock:
                        self._last_policy_ns = self._clock.now_ns()
                    self._supervisor.heartbeat("policy")
            next_tick += self.config.policy_period_ns
            now = self._clock.now_ns()
            if next_tick > now:
                self._clock.sleep(next_tick - now)
            else:
                next_tick = now

    def _make_chunk(self, state: RobotStateSample) -> ActionChunk:
        self._sequence += 1
        actions = np.zeros((self.config.horizon, 7), dtype=np.float32)
        actions[:, 0:3] = np.asarray(self.config.delta_m, dtype=np.float32)
        actions[:, 6] = np.float32(self.config.gripper)
        return ActionChunk(
            request_sequence_id=self._sequence,
            observation_capture_time_ns=state.host_time_ns,
            action_dt_s=self.config.action_dt_s,
            action_frame="base",
            action_type="delta_tcp",
            actions=actions,
        )

    def _executor_loop(self) -> None:
        assert self._executor is not None
        assert self._supervisor is not None
        last_tick: Optional[int] = None
        next_tick = self._clock.now_ns()
        tick_index = 0
        try:
            self._executor.start()
        except Exception as exc:  # pragma: no cover - fake device should not fail here
            self._append_log(0, self._clock.now_ns(), self._clock.now_ns(), None, None, None, False, f"start:{type(exc).__name__}")
            return
        while not self._stop.is_set():
            now = self._clock.now_ns()
            start = self._clock.now_ns()
            try:
                report = self._executor.tick(now)
            except Exception as exc:  # hard fault -> stop control loop
                self._append_log(tick_index, now, start, None, None, None, False, f"{type(exc).__name__}")
                break
            runtime_ns = self._clock.now_ns() - start
            period_ns = None if last_tick is None else now - last_tick
            jitter_ns = None if period_ns is None else abs(period_ns - self.config.control_period_ns)
            overrun = runtime_ns > self.config.control_period_ns
            metric = _TickMetric(
                tick=tick_index,
                host_time_ns=now,
                period_ns=period_ns,
                runtime_ns=runtime_ns,
                jitter_ns=jitter_ns,
                overrun=overrun,
                wrote=report.wrote,
            )
            self._metrics.append(metric)
            measured = None
            if self._robot is not None:
                try:
                    measured = self._robot.read_state()
                except Exception:
                    measured = None
            self._append_log(tick_index, now, start, report, metric, measured, report.wrote, report.kind)
            with self._lock:
                self._last_executor_ns = self._clock.now_ns()
            self._supervisor.heartbeat("executor")
            try:
                self._supervisor.tick(now_ns=now)
            except Exception:
                pass
            if self._on_tick is not None:
                try:
                    self._callback_queue.put_nowait((report, {"tick": tick_index, "runtime_ns": runtime_ns}))
                    with self._lock:
                        self._callback_enqueued += 1
                except Full:
                    with self._lock:
                        self._callback_dropped += 1
                        self._callback_last_drop_tick = tick_index
            tick_index += 1
            last_tick = now
            next_tick += self.config.control_period_ns
            current = self._clock.now_ns()
            if next_tick <= current:
                next_tick = current + self.config.control_period_ns
            self._clock.sleep(max(1, next_tick - current))
        try:
            if self._robot is not None:
                self._robot.release_writer()
        except Exception:
            pass

    def _callback_loop(self) -> None:
        assert self._on_tick is not None
        while not self._stop.is_set():
            try:
                report, info = self._callback_queue.get(timeout=0.05)
            except Empty:
                continue
            try:
                self._on_tick(report, info)
            except Exception as exc:
                with self._lock:
                    self._callback_errors += 1
                    self._callback_last_error_tick = int(info.get("tick", -1))
                    self._callback_last_error_type = type(exc).__name__
            finally:
                with self._lock:
                    self._callback_processed += 1

    # -- logging ------------------------------------------------------------
    def _append_log(
        self,
        tick: int,
        now_ns: int,
        start_ns: int,
        report: Optional[ExecutionReport],
        metric: Optional[_TickMetric],
        measured: Optional[RobotStateSample],
        wrote: bool,
        kind: str,
    ) -> None:
        if self._log is None:
            return
        raw = report.state_sample if report is not None else None
        chunk = report.chunk if report is not None else None
        raw_action = None
        if chunk is not None:
            segment = (
                None
                if report is None or report.filtered_target is None
                else int(report.filtered_target.step_index)
            )
            row = None
            if segment is not None and 0 <= segment < chunk.horizon:
                row = chunk.actions[segment].tolist()
            raw_action = {
                "request_sequence_id": int(chunk.request_sequence_id),
                "observation_capture_time_ns": int(chunk.observation_capture_time_ns),
                "received_time_ns": None if report is None else report.chunk_received_time_ns,
                "epoch": None if report is None else report.chunk_epoch,
                "action_frame": chunk.action_frame,
                "action_type": chunk.action_type,
                "action_dt_s": float(chunk.action_dt_s),
                "horizon": int(chunk.horizon),
                "segment_index": segment,
                "action_row": row,
            }
        filtered = None if report is None or report.filtered_target is None else _target_dict(report.filtered_target)
        commanded = None if report is None or report.target is None else _target_dict(report.target)
        record = {
            "schema_version": "m3-execution/v1",
            "tick": tick,
            "start_time_ns": now_ns,
            "host_time_ns": now_ns,
            "period_ns": None if metric is None else metric.period_ns,
            "runtime_ns": None if metric is None else metric.runtime_ns,
            "jitter_ns": None if metric is None else metric.jitter_ns,
            "overrun": False if metric is None else metric.overrun,
            "kind": kind,
            "wrote": bool(wrote),
            "raw_action": raw_action,
            "raw_state": None if raw is None else _state_dict(raw),
            "filtered_target": filtered,
            "commanded_target": commanded,
            "commanded_status": "not_attempted" if report is None else report.command_status,
            "pose_attempted": False if report is None else bool(report.pose_attempted),
            "gripper_attempted": False if report is None else bool(report.gripper_attempted),
            "pose_ok": False if report is None else bool(report.pose_ok),
            "gripper_ok": False if report is None else bool(report.gripper_ok),
            "epoch": None if report is None else report.chunk_epoch,
            "measured_state": None if measured is None else _state_dict(measured),
        }
        self._log.append(record)

    # -- events/result ------------------------------------------------------
    def _on_executor_event(self, event: ExecutorEvent) -> None:
        if self._supervisor is None:
            return
        try:
            if event.kind.value in ("write_failed", "write_partial", "internal_error"):
                self._supervisor.fault(f"executor_{event.kind.value}", now_ns=event.time_ns)
            elif event.kind.value in ("plan_rejected", "plan_expired", "stale_state"):
                self._supervisor.hold(f"executor_{event.kind.value}", now_ns=event.time_ns)
            elif event.kind.value == "idle":
                self._supervisor.on_executor_idle(now_ns=event.time_ns)
            elif event.kind.value == "plan_started":
                self._supervisor.on_executor_active(now_ns=event.time_ns)
        except VlaInfraError:
            pass

    def _result(self, supervisor_status: Optional[Mapping[str, Any]] = None) -> M3Result:
        assert self._supervisor is not None
        status = self._supervisor.status() if supervisor_status is None else dict(supervisor_status)
        periods = [m.period_ns for m in self._metrics if m.period_ns is not None]
        runtimes = [m.runtime_ns for m in self._metrics]
        jitters = [m.jitter_ns for m in self._metrics if m.jitter_ns is not None]
        metrics = {
            "tick_period_ns": _summary(periods),
            "runtime_ns": _summary(runtimes),
            "jitter_ns": _summary(jitters),
            "ticks": len(self._metrics),
            "writes": sum(1 for m in self._metrics if m.wrote),
            "overruns": sum(1 for m in self._metrics if m.overrun),
            "target_period_ns": self.config.control_period_ns,
            "policy_period_ns": self.config.policy_period_ns,
            "state_period_ns": self.config.state_period_ns,
            "machine": {"platform": platform.platform(), "python": platform.python_version()},
            "log_dropped": 0 if self._log is None else self._log.dropped,
            "log_records": 0 if self._log is None else self._log.written,
        }
        with self._lock:
            metrics["callback"] = {
                "submitted": self._callback_enqueued + self._callback_dropped,
                "enqueued": self._callback_enqueued,
                "processed": self._callback_processed,
                "dropped": self._callback_dropped,
                "errors": self._callback_errors,
                "last_drop_tick": self._callback_last_drop_tick,
                "last_error_tick": self._callback_last_error_tick,
                "last_error_type": self._callback_last_error_type,
                "semantics": "latest_only_notification; never safety/command/audit critical",
            }
        return M3Result(
            final_state=status["state"],
            armed=any(t["to"] == "ARMED" for t in status["transitions"]),
            ticks=len(self._metrics),
            writes=metrics["writes"],
            policy_chunks=self._policy_chunks,
            log_path=str(self._output),
            log_records=metrics["log_records"],
            log_dropped=metrics["log_dropped"],
            metrics=metrics,
            executor=self._executor.stats.as_dict() if self._executor else {},
            supervisor=status,
            error=self._stop_error,
            stop_error=self._stop_error,
            threads=tuple(thread.name for thread in self._threads),
            lifecycle=self._lifecycle,
        )


def _state_dict(sample: RobotStateSample) -> dict:
    state = sample.state
    return {
        "host_time_ns": sample.host_time_ns,
        "device_time_ns": sample.device_time_ns,
        "sequence_id": sample.sequence_id,
        "joint_position": state.joint_position.tolist(),
        "joint_velocity": state.joint_velocity.tolist(),
        "tcp_position": state.tcp_position.tolist(),
        "tcp_quaternion_xyzw": state.tcp_quaternion_xyzw.tolist(),
        "tcp_velocity": state.tcp_velocity.tolist(),
        "wrench_tcp": state.wrench_tcp.tolist(),
        "gripper_width": state.gripper_width.tolist(),
    }


def _target_dict(target: CommandTarget) -> dict:
    pose = target.pose
    return {
        "position": pose.position.tolist(),
        "quaternion_xyzw": pose.quaternion_xyzw.tolist(),
        "gripper_normalized": float(target.gripper_normalized),
        "step_index": int(target.step_index),
    }


def _summary(samples: list[int]) -> dict:
    if not samples:
        return {"count": 0, "p50_ns": None, "p95_ns": None, "p99_ns": None, "max_ns": None, "mean_ns": None}
    ordered = sorted(samples)
    def pct(f: float) -> int:
        return ordered[min(len(ordered) - 1, max(0, int(round(f * (len(ordered) - 1)))))]
    return {
        "count": len(ordered),
        "p50_ns": pct(0.50),
        "p95_ns": pct(0.95),
        "p99_ns": pct(0.99),
        "max_ns": ordered[-1],
        "mean_ns": int(sum(ordered) / len(ordered)),
    }


__all__ = ["M3Config", "M3Result", "M3Runtime", "_summary"]
