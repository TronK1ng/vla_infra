"""Asynchronous episode recorder.

Design constraints (master plan §7.3 and M2 acceptance):

* capture/control threads only append to bounded buffers and never perform disk
  I/O or image encoding;
* a dedicated writer thread owns the raw episode writer;
* low-frequency image/action events use a bounded queue; high-rate lowdim samples
  use a bounded in-memory buffer, so a slow disk cannot block the caller;
* disk free space is checked before the episode and periodically while writing;
* a write failure marks the recorder failed and leaves the temporary episode
  identifiable, never blocking the capture loop.
"""

from __future__ import annotations

import threading
import time

import numpy as np
from dataclasses import dataclass, field, replace
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Callable, Mapping, Optional

from ..clock import Clock, MonotonicClock
from ..contracts.observation import CanonicalObservation, RobotStateSample
from ..contracts.policy_io import CANONICAL_STATE_LAYOUT
from ..errors import ErrorCode
from .raw_writer import RawEpisodeWriter, RecordingError

_CAMERA_KEY_SUFFIXES = ("_rgb", "_depth")


def camera_name_from_observation_key(key: str) -> str:
    for suffix in _CAMERA_KEY_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


@dataclass(frozen=True)
class RecorderConfig:
    min_free_gb: float = 20.0
    queue_size: int = 256
    max_lowdim_rows: int = 16_384
    lowdim_batch_rows: int = 512
    disk_check_interval_s: float = 1.0
    stop_timeout_s: float = 10.0
    finalize_timeout_s: float = 300.0
    record_actions: bool = True

    def __post_init__(self) -> None:
        if self.min_free_gb < 0:
            raise ValueError("min_free_gb must be non-negative")
        if self.queue_size < 1 or self.max_lowdim_rows < 1:
            raise ValueError("queue/buffer limits must be positive")
        if self.lowdim_batch_rows < 1:
            raise ValueError("lowdim_batch_rows must be positive")
        if self.disk_check_interval_s <= 0 or self.stop_timeout_s <= 0:
            raise ValueError("disk/stop interval must be positive")
        if self.finalize_timeout_s <= 0:
            raise ValueError("finalize_timeout_s must be positive")


@dataclass(frozen=True)
class RecorderResult:
    committed: bool
    path: Optional[Path]
    lowdim_rows: int
    camera_frames: int
    observations: int
    action_records: int
    dropped_lowdim: int
    dropped_observations: int
    dropped_actions: int
    error: str = ""
    sync_report: Mapping[str, Any] = field(default_factory=dict)
    still_stopping: bool = False
    state: str = "idle"

    def as_dict(self) -> dict:
        return {
            "committed": self.committed,
            "path": None if self.path is None else str(self.path),
            "lowdim_rows": self.lowdim_rows,
            "camera_frames": self.camera_frames,
            "observations": self.observations,
            "action_records": self.action_records,
            "dropped_lowdim": self.dropped_lowdim,
            "dropped_observations": self.dropped_observations,
            "dropped_actions": self.dropped_actions,
            "error": self.error,
            "still_stopping": self.still_stopping,
            "state": self.state,
            "sync_report": dict(self.sync_report),
        }


class EpisodeRecorder:
    def __init__(
        self,
        writer: RawEpisodeWriter,
        config: RecorderConfig,
        *,
        clock: Optional[Clock] = None,
        on_warning: Optional[Callable[[str, dict], None]] = None,
    ) -> None:
        self._writer = writer
        self._config = config
        self._clock = clock or MonotonicClock()
        self._on_warning = on_warning
        self._lock = threading.Lock()
        self._lowdim: list[RobotStateSample] = []
        self._events: Queue = Queue(maxsize=config.queue_size)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Lifecycle: idle -> accepting -> closing -> committing -> finished.
        # closing is entered by stop(); committing is the non-cancellable
        # authorization point owned by the writer thread.
        self._state = "idle"
        self._accepting = False
        self._stop_requested = False
        self._commit_cancelled = False
        self._inflight = 0
        self._fault: Optional[str] = None
        self._abort_reason: Optional[str] = None
        self._end_reason = "duration"
        self._warned: set[str] = set()
        self._dropped_lowdim = 0
        self._dropped_observations = 0
        self._dropped_actions = 0
        self._sync_report: Mapping[str, Any] = {}
        self._result: Optional[RecorderResult] = None

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            raise RecordingError("recorder already started", code=ErrorCode.ILLEGAL_TRANSITION)
        try:
            self._writer.check_free_space(int(self._config.min_free_gb * 1024**3))
        except Exception as exc:
            self._writer.abort(f"startup failed: {type(exc).__name__}: {exc}")
            raise
        self._stop.clear()
        with self._lock:
            self._state = "accepting"
            self._accepting = True
            self._stop_requested = False
            self._commit_cancelled = False
            self._inflight = 0
            self._fault = None
            self._abort_reason = None
            self._end_reason = "duration"
        self._thread = threading.Thread(target=self._run, name="episode-recorder", daemon=True)
        self._thread.start()

    def stop(
        self,
        *,
        timeout_s: Optional[float] = None,
        end_reason: str = "duration",
        abort_reason: Optional[str] = None,
    ) -> RecorderResult:
        timeout = self._config.stop_timeout_s if timeout_s is None else float(timeout_s)
        with self._lock:
            if self._result is not None:
                return self._result
            if self._state == "accepting":
                self._state = "closing"
            self._accepting = False
            self._stop_requested = True
            self._end_reason = str(end_reason)
            if abort_reason is not None:
                self._abort_reason = str(abort_reason)
        self._stop.set()
        thread = self._thread
        if thread is None:
            # start() may have failed before spawning the writer thread (for
            # example disk-low); make sure the temporary episode carries a
            # failure marker instead of looking merely abandoned.
            with self._lock:
                reason = self._fault or self._abort_reason or "recorder never started"
                self._fault = self._fault or reason
                self._state = "finished"
            try:
                self._writer.abort(reason)
            except Exception:  # pragma: no cover - already committed/closed
                pass
            return self.result

        thread.join(timeout)
        if not thread.is_alive():
            return self.result

        # The writer might be in the non-cancellable COMMITTING section.  If it
        # has not crossed that point, cancel and let it abort; if it has, report
        # still_stopping rather than pretending the episode is final.
        with self._lock:
            if self._state != "committing":
                self._commit_cancelled = True
                self._state = "closing"
                self._fault = self._fault or "stop timeout before commit"
                if self._abort_reason is None:
                    self._abort_reason = "stop timeout before commit"
        thread.join(min(timeout, 1.0))
        if thread.is_alive():
            return self._snapshot_result(still_stopping=True)
        return self.result

    @property
    def result(self) -> RecorderResult:
        return self._snapshot_result(still_stopping=False)

    def wait_until_final(
        self,
        timeout_s: Optional[float] = None,
        *,
        poll_interval_s: float = 0.05,
    ) -> RecorderResult:
        """Wait for a final result after stop() returned still_stopping."""
        timeout = self._config.finalize_timeout_s if timeout_s is None else float(timeout_s)
        if timeout <= 0 or poll_interval_s <= 0:
            raise ValueError("finalize timeout/poll interval must be positive")
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                if self._result is not None:
                    return self._result
                thread = self._thread
                state = self._state
            if thread is None or not thread.is_alive():
                return self.result
            if state == "committing":
                # Once the writer has entered the non-cancellable rename
                # section, a deadline must not let the caller exit and leave the
                # daemon writer to finish (or be killed) independently.  Wait
                # for the authoritative final result.
                time.sleep(poll_interval_s)
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self._snapshot_result(still_stopping=True)
            time.sleep(min(poll_interval_s, remaining))

    def cancel_pending(self, reason: str) -> None:
        """Best-effort cancel if the writer has not entered commit."""
        with self._lock:
            if self._result is not None:
                return
            if self._state != "committing":
                self._commit_cancelled = True
                self._abort_reason = self._abort_reason or str(reason)
                self._fault = self._fault or str(reason)
                self._stop_requested = True
        self._stop.set()

    def _snapshot_result(self, *, still_stopping: bool) -> RecorderResult:
        with self._lock:
            if self._result is not None:
                return self._result
            state = self._state
            still = still_stopping or (
                state not in ("finished", "idle")
                and self._thread is not None
                and self._thread.is_alive()
            )
            error = self._fault or self._abort_reason
            if error is None:
                error = "recorder still stopping" if still else "recorder did not finish"
            return RecorderResult(
                committed=False,
                path=None,
                lowdim_rows=self._writer.lowdim_rows,
                camera_frames=self._writer.camera_frames,
                observations=self._writer.observation_records,
                action_records=self._writer.action_records,
                dropped_lowdim=self._dropped_lowdim,
                dropped_observations=self._dropped_observations,
                dropped_actions=self._dropped_actions,
                error=error,
                sync_report=dict(self._sync_report),
                still_stopping=still,
                state=state,
            )

    # -- producer side (never blocks on disk) -------------------------------
    def record_robot_sample(self, sample: RobotStateSample) -> bool:
        full = False
        with self._lock:
            if not self._accepting or self._state != "accepting" or self._fault is not None:
                self._record_drop_locked("lowdim")
                return False
            self._inflight += 1
            try:
                if len(self._lowdim) >= self._config.max_lowdim_rows:
                    self._record_drop_locked("lowdim")
                    full = True
                else:
                    self._lowdim.append(sample)
            finally:
                self._inflight -= 1
        if full:
            self._warn_once(
                "lowdim_buffer_full",
                {"max_rows": self._config.max_lowdim_rows},
            )
            return False
        return True

    def record_observation(self, observation: CanonicalObservation) -> bool:
        full = False
        with self._lock:
            if not self._accepting or self._state != "accepting" or self._fault is not None:
                self._record_drop_locked("observations")
                return False
            self._inflight += 1
            try:
                # Enqueue inside the lifecycle lock: stop() closes accepting
                # under the same lock, so an accepted event cannot be enqueued
                # after the writer has observed an idle queue.
                self._events.put_nowait(("observation", observation))
            except Full:
                self._record_drop_locked("observations")
                full = True
            finally:
                self._inflight -= 1
        if full:
            self._warn_once(
                "observation_queue_full",
                {"queue_size": self._config.queue_size},
            )
            return False
        return True

    def record_action(self, record: Mapping[str, Any]) -> bool:
        if not self._config.record_actions:
            return False
        full = False
        with self._lock:
            if not self._accepting or self._state != "accepting" or self._fault is not None:
                self._record_drop_locked("actions")
                return False
            self._inflight += 1
            try:
                self._events.put_nowait(("action", dict(record)))
            except Full:
                self._record_drop_locked("actions")
                full = True
            finally:
                self._inflight -= 1
        if full:
            self._warn_once("action_queue_full", {"queue_size": self._config.queue_size})
            return False
        return True

    def set_sync_report(self, report: Mapping[str, Any]) -> None:
        with self._lock:
            self._sync_report = dict(report)
        self._writer.set_sync_report(report)

    # -- writer thread ------------------------------------------------------
    def _run(self) -> None:
        last_disk_check = self._clock.now_ns()
        try:
            while True:
                self._drain_lowdim()
                event = self._next_event()
                if event is not None:
                    self._handle_event(event)
                now = self._clock.now_ns()
                if now - last_disk_check >= int(self._config.disk_check_interval_s * 1e9):
                    last_disk_check = now
                    self._writer.check_free_space(int(self._config.min_free_gb * 1024**3))
                with self._lock:
                    stop_requested = self._stop_requested
                    ready = (
                        stop_requested
                        and not self._accepting
                        and self._inflight == 0
                        and not self._lowdim
                        and self._events.empty()
                    )
                    if ready:
                        if (
                            self._fault is not None
                            or self._abort_reason is not None
                            or self._commit_cancelled
                        ):
                            self._state = "closing"
                        else:
                            # CAS-like transition under the lifecycle lock; after
                            # this point stop() must not claim a final failure.
                            self._state = "committing"
                        break
        except Exception as exc:  # noqa: BLE001 - writer boundary must not crash caller
            with self._lock:
                self._fault = f"{type(exc).__name__}: {exc}"
        finally:
            self._finish()

    def _drain_lowdim(self) -> None:
        with self._lock:
            if not self._lowdim:
                return
            batch = self._lowdim[: self._config.lowdim_batch_rows]
            del self._lowdim[: len(batch)]
        self._writer.append_lowdim(batch)

    def _next_event(self):
        try:
            return self._events.get(timeout=0.01)
        except Empty:
            return None

    def _handle_event(self, event) -> None:
        kind, payload = event
        if kind == "observation":
            self._write_observation(payload)
        elif kind == "action":
            self._writer.write_action(payload)
        else:  # pragma: no cover - internal invariant
            raise RecordingError(f"unknown recorder event {kind!r}")

    def _write_observation(self, observation: CanonicalObservation) -> None:
        image_keys = []
        for key, frame in observation.images.items():
            camera = camera_name_from_observation_key(key)
            kind = "depth" if frame.encoding == "depth16" else "color"
            relative_file = self._writer.write_image(camera, frame, kind=kind)
            image_keys.append(
                {
                    "camera": camera,
                    "kind": kind,
                    "key": key,
                    "frame_number": frame.frame_number,
                    "device_time_ns": frame.device_time_ns,
                    "host_time_ns": frame.host_time_ns,
                    "file": relative_file,
                }
            )
        state = observation.robot
        state_vector = CANONICAL_STATE_LAYOUT.encode(state).astype(np.float32)
        self._writer.write_observation(
            {
                "schema_version": observation.schema_version,
                "sequence_id": observation.sequence_id,
                "capture_time_ns": observation.capture_time_ns,
                "task": observation.task,
                "images": image_keys,
                # This is the exact state the online VLA observation used; it is
                # interpolated/ZOH by the synchronizer and must never be
                # reconstructed from raw lowdim during export.
                "canonical_state": {
                    "schema_version": "canonical-observation-state/v1",
                    "layout": CANONICAL_STATE_LAYOUT.to_dict(),
                    "values": state_vector.tolist(),
                    "capture_time_ns": observation.capture_time_ns,
                    "source": "ObservationSynchronizer",
                    "typed": {
                        "joint_position": state.joint_position.tolist(),
                        "joint_velocity": state.joint_velocity.tolist(),
                        "tcp_position": state.tcp_position.tolist(),
                        "tcp_quaternion_xyzw": state.tcp_quaternion_xyzw.tolist(),
                        "tcp_rotation_6d": state.tcp_rotation_6d.tolist(),
                        "tcp_velocity": state.tcp_velocity.tolist(),
                        "wrench_tcp": state.wrench_tcp.tolist(),
                        "gripper_width": state.gripper_width.tolist(),
                        "tcp_velocity_frame": state.tcp_velocity_frame,
                        "wrench_frame": state.wrench_frame,
                    },
                },
                "sync": {
                    "robot_sample_before_ns": observation.sync.robot_sample_before_ns,
                    "robot_sample_after_ns": observation.sync.robot_sample_after_ns,
                    "interpolation_alpha": observation.sync.interpolation_alpha,
                    "max_age_ns": observation.sync.max_age_ns,
                    "camera_skew_ns": observation.sync.camera_skew_ns,
                    "state_gap_ns": observation.sync.state_gap_ns,
                    "dropped_frames": dict(observation.sync.dropped_frames),
                },
            }
        )

    def _finish(self) -> None:
        # Drain anything still buffered.  At this point accepting is false, so
        # no accepted event can be added after the final emptiness check.
        try:
            while True:
                with self._lock:
                    pending_lowdim = bool(self._lowdim)
                if not pending_lowdim and self._events.empty():
                    break
                self._drain_lowdim()
                event = self._next_event()
                if event is not None:
                    self._handle_event(event)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._fault = self._fault or f"drain failed: {type(exc).__name__}: {exc}"

        with self._lock:
            authorized = (
                self._state == "committing"
                and self._fault is None
                and self._abort_reason is None
                and not self._commit_cancelled
            )
            end_reason = self._end_reason
            if authorized:
                self._state = "committing"
            else:
                self._state = "aborted"
            drops = (
                self._dropped_lowdim,
                self._dropped_observations,
                self._dropped_actions,
            )
            sync_report = dict(self._sync_report)
            fault = self._fault
            abort_reason = self._abort_reason

        if authorized:
            try:
                published = self._writer.commit(
                    end_reason=end_reason,
                    verdict="unreviewed",
                    extra_metadata={
                        "recorder": {
                            "dropped_lowdim": drops[0],
                            "dropped_observations": drops[1],
                            "dropped_actions": drops[2],
                        }
                    },
                )
                result = RecorderResult(
                    committed=True,
                    path=published.path,
                    lowdim_rows=published.lowdim_rows,
                    camera_frames=published.camera_frames,
                    observations=published.observation_records,
                    action_records=published.action_records,
                    dropped_lowdim=drops[0],
                    dropped_observations=drops[1],
                    dropped_actions=drops[2],
                    sync_report=sync_report,
                    state="finished",
                )
            except Exception as exc:  # noqa: BLE001
                failure = f"commit failed: {type(exc).__name__}: {exc}"
                with self._lock:
                    self._fault = failure
                self._writer.abort(failure)
                result = self._failed_result(failure)
        else:
            reason = fault or abort_reason or "recording aborted"
            self._writer.abort(reason)
            result = self._failed_result(reason)
        with self._lock:
            self._result = result
            self._accepting = False
            if self._state not in ("finished", "aborted"):
                self._state = "finished" if result.committed else "aborted"

    def _record_drop_locked(self, kind: str) -> None:
        """Increment a rejection counter; must be called while holding the lock."""
        if kind == "lowdim":
            self._dropped_lowdim += 1
        elif kind == "observations":
            self._dropped_observations += 1
        elif kind == "actions":
            self._dropped_actions += 1
        else:  # pragma: no cover
            raise ValueError(kind)
        if self._result is not None:
            # Post-stop rejections must remain visible in the final counters;
            # the committed result object is immutable but can be replaced.
            self._result = replace(
                self._result,
                dropped_lowdim=self._dropped_lowdim,
                dropped_observations=self._dropped_observations,
                dropped_actions=self._dropped_actions,
            )

    def _failed_result(self, reason: Optional[str] = None) -> RecorderResult:
        with self._lock:
            fault = self._fault
            drops = (
                self._dropped_lowdim,
                self._dropped_observations,
                self._dropped_actions,
            )
            sync_report = dict(self._sync_report)
        return RecorderResult(
            committed=False,
            path=None,
            lowdim_rows=self._writer.lowdim_rows,
            camera_frames=self._writer.camera_frames,
            observations=self._writer.observation_records,
            action_records=self._writer.action_records,
            dropped_lowdim=drops[0],
            dropped_observations=drops[1],
            dropped_actions=drops[2],
            error=reason or fault or "recording failed",
            sync_report=sync_report,
            still_stopping=False,
            state="aborted",
        )

    def _warn_once(self, reason: str, details: Mapping[str, Any]) -> None:
        if self._on_warning is None:
            return
        with self._lock:
            if reason in self._warned:
                return
            self._warned.add(reason)
        try:
            self._on_warning(reason, dict(details))
        except Exception:  # pragma: no cover - observability callback
            pass


__all__ = [
    "EpisodeRecorder",
    "RecorderConfig",
    "RecorderResult",
    "camera_name_from_observation_key",
]
