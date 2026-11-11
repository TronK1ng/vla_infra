"""M2 collection harness (fake backend first, read-only real adapters later).

It only reads devices, synchronises observations and writes an episode; it never
creates a policy client, an executor or a command sink.  Fake devices are used
for automated tests so the recording pipeline is reproducible without hardware.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional

from .. import __version__
from ..clock import Clock, MonotonicClock
from ..config import AppConfig
from ..contracts.constants import (
    ACTION_SCHEMA_VERSION,
    GRIPPER_SEMANTICS,
    IMAGE_KEY_BASE_DEPTH,
    IMAGE_KEY_BASE_RGB,
    IMAGE_KEY_WRIST_DEPTH,
    IMAGE_KEY_WRIST_RGB,
    OBSERVATION_SCHEMA_VERSION,
)
from ..contracts.policy_io import CANONICAL_STATE_LAYOUT
from ..devices.fake import FakeCamera, FakeGripper, FakeRobotArm
from ..devices.flexiv_rdk import FlexivRdkReadOnlyState
from ..devices.realsense import RealSenseCamera
from ..errors import ConfigError, ErrorCode, SyncError, UnsupportedBackendError
from ..recording.raw_writer import RawEpisodeWriter
from ..recording.recorder import EpisodeRecorder, RecorderConfig, RecorderResult
from ..sync.synchronizer import ObservationSynchronizer


@dataclass(frozen=True)
class CollectionResult:
    recorder: RecorderResult
    sync_report: Mapping[str, Any]
    observations: int
    robot_samples: int
    episode_path: Optional[str]
    error: str = ""

    def as_dict(self) -> dict:
        return {
            "recorder": self.recorder.as_dict(),
            "sync_report": dict(self.sync_report),
            "observations": self.observations,
            "robot_samples": self.robot_samples,
            "episode_path": self.episode_path,
            "error": self.error,
        }


def build_collection_metadata(
    app: AppConfig,
    *,
    control_source: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    station = app.station
    safety = app.safety
    cameras = {
        name: {
            "serial": camera.serial,
            "width": camera.width,
            "height": camera.height,
            "fps": camera.fps,
            "depth": camera.depth,
            "device_latency_ms": camera.device_latency_ms,
        }
        for name, camera in station.cameras.items()
    }
    metadata: Dict[str, Any] = {
        "station_id": station.station_id,
        "robot_type": "flexiv_rizon4s",
        "backend": station.backend,
        "control_source": control_source,
        "task_class": "collection",
        "software": {
            "vla_infra": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "devices": {
            "robot_serial": station.robot.serial,
            "robot_state_rate_hz": station.robot.state_rate_hz,
            "gripper": station.robot.gripper,
            "cameras": cameras,
            "rdk": {"version": "fake" if station.backend == "fake" else "unknown"},
            "tdk": {"version": "fake" if station.backend == "fake" else "unknown"},
            "firmware": "fake" if station.backend == "fake" else "unknown",
        },
        "calibration": {
            "version": station.calibration_version,
            "source": "not_collected_in_m1_m2_draft",
            "base_intrinsics_sha256": None,
            "wrist_intrinsics_sha256": None,
            "base_extrinsics_sha256": None,
            "wrist_extrinsics_sha256": None,
        },
        "tool": {
            "name": station.robot.gripper,
            "tcp": "station_configured_tcp",
            "cartesian_stiffness": None,
            "note": "TCP/stiffness tuning is station configuration, not inferred by M2",
        },
        "contracts": {
            "observation_schema": OBSERVATION_SCHEMA_VERSION,
            "action_schema": ACTION_SCHEMA_VERSION,
            "state_layout": CANONICAL_STATE_LAYOUT.to_dict(),
            "delta_semantics": "cumulative",
            "gripper_semantics": GRIPPER_SEMANTICS,
        },
        "safety_profile": {
            "workspace_min": list(safety.workspace_min),
            "workspace_max": list(safety.workspace_max),
            "tcp_height_min_m": safety.tcp_height_min_m,
            "tcp_height_max_m": safety.tcp_height_max_m,
            "max_linear_velocity_m_s": safety.max_linear_velocity_m_s,
            "max_angular_velocity_rad_s": safety.max_angular_velocity_rad_s,
            "max_force_n": safety.max_force_n,
            "max_torque_nm": safety.max_torque_nm,
            "max_step_translation_m": safety.max_step_translation_m,
            "max_step_rotation_rad": safety.max_step_rotation_rad,
        },
        "timing": {
            "timestamp_sources": {
                "host": "CLOCK_MONOTONIC via time.monotonic_ns",
                "robot_device": "RDK RobotStates.timestamp [sec,nsec]",
                "camera_device": "RealSense get_timestamp() milliseconds, preserved as ns",
            },
            "observation_rate_hz": min(camera.fps for camera in station.cameras.values()),
            "robot_state_rate_hz": station.robot.state_rate_hz,
            "sync": {
                "max_sync_error_ms": station.sync.max_sync_error_ns / 1e6,
                "max_state_gap_ms": station.sync.max_state_gap_ns / 1e6,
                "max_camera_skew_ms": station.sync.max_camera_skew_ns / 1e6,
                "wrench_policy": station.sync.wrench_policy,
                "gripper_policy": station.sync.gripper_policy,
            },
        },
        "end_reason": "duration",
        "verdict": "unreviewed",
    }
    if extra:
        metadata.update(extra)
    return metadata


class CollectionHarness:
    def __init__(
        self,
        app: AppConfig,
        *,
        task: str = "collection run",
        control_source: str = "operator",
        output_root: Optional[str] = None,
        recorder_config: Optional[RecorderConfig] = None,
        clock: Optional[Clock] = None,
        on_warning=None,
        gripper_width_provider=None,
    ) -> None:
        if app.station.backend not in ("fake", "real_readonly"):
            raise UnsupportedBackendError(
                "CollectionHarness supports backend fake or real_readonly",
                details={"backend": app.station.backend},
            )
        if app.station.backend == "real_readonly" and gripper_width_provider is None:
            raise ConfigError(
                "real_readonly collection requires a read-only gripper width provider; "
                "GN01 read access has not been implemented yet",
                code=ErrorCode.NOT_READY,
            )
        self.app = app
        self._gripper_width_provider = gripper_width_provider
        self.task = task
        self.control_source = control_source
        self.output_root = output_root
        self._clock = clock or MonotonicClock()
        self._recorder_config = recorder_config or RecorderConfig(
            min_free_gb=app.station.recording.min_free_gb,
            finalize_timeout_s=app.station.recording.finalize_timeout_s,
        )
        self._on_warning = on_warning
        self._cameras: Dict[str, Dict[str, Any]] = {}
        self._robot: Optional[Any] = None
        self._sync: Optional[ObservationSynchronizer] = None
        self._recorder: Optional[EpisodeRecorder] = None
        self._writer: Optional[RawEpisodeWriter] = None
        self._latest: Dict[str, Any] = {}
        self._built_frame_numbers: Dict[str, int] = {}
        self._observations = 0
        self._robot_samples = 0
        self._warnings: list[str] = []
        self._started = False
        self._finished = False
        self._stop_error = ""

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        if self._started:
            raise RuntimeError("collection harness already started")
        try:
            self._start_components()
        except BaseException as exc:  # noqa: BLE001 - never commit a failed startup
            self._stop_error = f"collection_startup_exception: {type(exc).__name__}: {exc}"
            self.stop(end_reason="startup_exception", abort_reason=self._stop_error)
            raise

    def _start_components(self) -> None:
        station = self.app.station
        if station.backend == "real_readonly":
            self._start_real_devices()
            root = self.output_root or station.recording.root
            metadata = build_collection_metadata(self.app, control_source=self.control_source)
            self._enrich_real_metadata(metadata)
        else:
            self._start_fake_devices()
            root = self.output_root or station.recording.root
            metadata = build_collection_metadata(
                self.app,
                control_source=self.control_source,
                extra={"warnings": []},
            )
        self._sync = ObservationSynchronizer(self._clock, station.sync)
        writer = RawEpisodeWriter(
            root,
            self.task,
            metadata=metadata,
            camera_names=tuple(station.cameras),
            clock=self._clock,
        )
        self._writer = writer
        self._recorder = EpisodeRecorder(
            writer,
            self._recorder_config,
            clock=self._clock,
            on_warning=self._on_recorder_warning,
        )
        self._recorder.start()
        self._started = True
        self._finished = False

    def _start_fake_devices(self) -> None:
        station = self.app.station
        for index, (name, camera) in enumerate(station.cameras.items()):
            color_key = IMAGE_KEY_BASE_RGB if name == "base" else IMAGE_KEY_WRIST_RGB
            depth_key = IMAGE_KEY_BASE_DEPTH if name == "base" else IMAGE_KEY_WRIST_DEPTH
            self._cameras[name] = {
                "color": FakeCamera(
                    color_key,
                    self._clock,
                    width=camera.width,
                    height=camera.height,
                    fps=camera.fps,
                    device_latency_ns=int(camera.device_latency_ms * 1_000_000),
                    seed=index,
                )
            }
            if camera.depth:
                self._cameras[name]["depth"] = FakeCamera(
                    depth_key,
                    self._clock,
                    width=camera.width,
                    height=camera.height,
                    fps=camera.fps,
                    encoding="depth16",
                    device_latency_ns=int(camera.device_latency_ms * 1_000_000),
                    seed=index + 10,
                )
            for device in self._cameras[name].values():
                device.start()
        self._robot = FakeRobotArm(
            self._clock,
            gripper=FakeGripper(
                width_min_m=self.app.safety.gripper_width_min_m,
                width_max_m=self.app.safety.gripper_width_max_m,
                initial_width_m=(
                    self.app.safety.gripper_width_min_m
                    + self.app.safety.gripper_width_max_m
                )
                / 2.0,
            ),
        )

    def _start_real_devices(self) -> None:
        station = self.app.station
        for name, camera in station.cameras.items():
            device = RealSenseCamera(
                name,
                camera.serial,
                width=camera.width,
                height=camera.height,
                fps=camera.fps,
                enable_depth=camera.depth,
                clock=self._clock,
            )
            device.start()
            self._cameras[name] = {"real": device}
        reader = FlexivRdkReadOnlyState(
            station.robot.serial,
            clock=self._clock,
            gripper_width_provider=self._gripper_width_provider,
        )
        reader.start()
        self._robot = reader

    def _enrich_real_metadata(self, metadata: dict) -> None:
        for name, cameras in self._cameras.items():
            device = cameras["real"]
            metadata["devices"]["cameras"][name].update(
                {
                    "device_info": device.device_info,
                    "intrinsics": device.device_info.get("intrinsics"),
                    "intrinsics_sha256": device.device_info.get("intrinsics_sha256"),
                    "depth_scale": device.depth_scale,
                }
            )
        if self._robot is not None and hasattr(self._robot, "device_info"):
            info = self._robot.device_info
            metadata["devices"]["rdk"] = info
            if info.get("firmware_version"):
                metadata["devices"]["firmware"] = info["firmware_version"]
        base = metadata["devices"]["cameras"]["base"]
        wrist = metadata["devices"]["cameras"]["wrist"]
        metadata["calibration"].update(
            {
                "base_intrinsics_sha256": base.get("intrinsics_sha256"),
                "wrist_intrinsics_sha256": wrist.get("intrinsics_sha256"),
                "source": "realsense_intrinsics_only; extrinsics pending calibration",
            }
        )

    def run(self, *, duration_s: float = 3.0) -> CollectionResult:
        self.start()
        period_ns = 10_000_000
        deadline = self._clock.now_ns() + int(duration_s * 1_000_000_000)
        next_tick = self._clock.now_ns()
        failure: Optional[BaseException] = None
        try:
            while self._clock.now_ns() < deadline:
                self.step()
                next_tick += period_ns
                now = self._clock.now_ns()
                if next_tick > now:
                    self._clock.sleep(next_tick - now)
                else:
                    next_tick = now
        except BaseException as exc:  # noqa: BLE001 - must abort before re-raising
            failure = exc
            self._stop_error = f"collection_exception: {type(exc).__name__}: {exc}"
        finally:
            result = self.stop(
                end_reason="exception" if failure is not None else "duration",
                abort_reason=self._stop_error if failure is not None else None,
            )
        if failure is not None:
            raise failure
        return result

    def stop(
        self,
        *,
        end_reason: str = "duration",
        abort_reason: Optional[str] = None,
        timeout_s: Optional[float] = None,
        finalize_timeout_s: Optional[float] = None,
    ) -> CollectionResult:
        if self._recorder is not None and not self._finished:
            if self._sync is not None:
                self._recorder.set_sync_report(self._sync.sync_report())
            recorder_result = self._recorder.stop(
                timeout_s=timeout_s,
                end_reason=end_reason,
                abort_reason=abort_reason,
            )
            if recorder_result.still_stopping:
                finalize = (
                    self._recorder_config.finalize_timeout_s
                    if finalize_timeout_s is None
                    else float(finalize_timeout_s)
                )
                recorder_result = self._recorder.wait_until_final(finalize)
                if recorder_result.still_stopping:
                    # Commit-before-timeout is not cancellable; try to cancel
                    # only outside the committing critical section.
                    self._recorder.cancel_pending("collection finalize timeout")
                    recorder_result = self._recorder.wait_until_final(
                        min(1.0, max(0.05, finalize))
                    )
                if recorder_result.still_stopping:
                    self._stop_error = (
                        "finalize timeout: recorder did not reach finished/aborted; "
                        "temporary episode must be treated as incomplete"
                    )
                    recorder_result = replace(
                        recorder_result,
                        error=self._stop_error,
                        still_stopping=True,
                    )
                else:
                    self._stop_error = ""
        elif self._writer is not None and not self._finished and not self._writer.committed:
            # Startup failed after creating the writer but before/inside the
            # recorder; leave a failure marker rather than a silent temp dir.
            self._writer.abort(abort_reason or "collection startup failed")
            recorder_result = RecorderResult(
                committed=False,
                path=None,
                lowdim_rows=self._writer.lowdim_rows,
                camera_frames=self._writer.camera_frames,
                observations=self._writer.observation_records,
                action_records=self._writer.action_records,
                dropped_lowdim=0,
                dropped_observations=0,
                dropped_actions=0,
                error=abort_reason or "collection startup failed",
            )
        elif self._recorder is not None:
            recorder_result = self._recorder.result
        else:
            recorder_result = RecorderResult(
                committed=False,
                path=None,
                lowdim_rows=0,
                camera_frames=0,
                observations=0,
                action_records=0,
                dropped_lowdim=0,
                dropped_observations=0,
                dropped_actions=0,
                error="collection was not started",
            )
        for cameras in self._cameras.values():
            for camera in cameras.values():
                camera.stop()
        if self._robot is not None and hasattr(self._robot, "close"):
            try:
                self._robot.close()
            except Exception:  # pragma: no cover - read-only close best effort
                pass
        self._finished = True
        self._started = False
        sync_report = dict(self._sync.sync_report()) if self._sync is not None else {}
        return CollectionResult(
            recorder=recorder_result,
            sync_report=sync_report,
            observations=self._observations,
            robot_samples=self._robot_samples,
            episode_path=None if recorder_result.path is None else str(recorder_result.path),
            error=self._stop_error,
        )

    # -- loop ---------------------------------------------------------------
    def step(self) -> None:
        assert self._robot is not None and self._sync is not None and self._recorder is not None
        samples_per_step = max(
            1, int(round(self.app.station.robot.state_rate_hz * 0.01))
        )
        before = max(1, samples_per_step // 2)
        for _ in range(before):
            sample = self._robot.read_state()
            self._sync.add_robot_sample(sample)
            self._recorder.record_robot_sample(sample)
            self._robot_samples += 1
        for name, cameras in self._cameras.items():
            color_key = IMAGE_KEY_BASE_RGB if name == "base" else IMAGE_KEY_WRIST_RGB
            depth_key = IMAGE_KEY_BASE_DEPTH if name == "base" else IMAGE_KEY_WRIST_DEPTH
            if "real" in cameras:
                pair = cameras["real"].read_pair()
                if pair is not None:
                    self._latest[color_key] = replace(pair.color, name=color_key)
                    if pair.depth is not None:
                        self._latest[depth_key] = replace(pair.depth, name=depth_key)
                continue
            # Fake path: read depth first so its host timestamp is not newer.
            depth_device = cameras.get("depth")
            if depth_device is not None:
                frame = depth_device.read()
                if frame is not None:
                    self._latest[frame.name] = frame
            color = cameras["color"].read()
            if color is not None:
                self._latest[color.name] = color
        for _ in range(max(1, samples_per_step - before)):
            sample = self._robot.read_state()
            self._sync.add_robot_sample(sample)
            self._recorder.record_robot_sample(sample)
            self._robot_samples += 1
        color_base = self._latest.get(IMAGE_KEY_BASE_RGB)
        color_wrist = self._latest.get(IMAGE_KEY_WRIST_RGB)
        if color_base is None or color_wrist is None:
            return
        both_new = (
            self._built_frame_numbers.get(IMAGE_KEY_BASE_RGB) != color_base.frame_number
            and self._built_frame_numbers.get(IMAGE_KEY_WRIST_RGB) != color_wrist.frame_number
        )
        if not both_new:
            return
        images = {
            IMAGE_KEY_BASE_RGB: color_base,
            IMAGE_KEY_WRIST_RGB: color_wrist,
        }
        target_host = max(color_base.host_time_ns, color_wrist.host_time_ns)
        for key in (IMAGE_KEY_BASE_DEPTH, IMAGE_KEY_WRIST_DEPTH):
            depth = self._latest.get(key)
            if (
                depth is not None
                and depth.host_time_ns <= target_host
                and target_host - depth.host_time_ns <= self.app.station.sync.max_camera_skew_ns
            ):
                images[key] = depth
        try:
            observation = self._sync.build(images, self.task)
        except SyncError:
            return
        self._recorder.record_observation(observation)
        self._built_frame_numbers[IMAGE_KEY_BASE_RGB] = color_base.frame_number
        self._built_frame_numbers[IMAGE_KEY_WRIST_RGB] = color_wrist.frame_number
        self._observations += 1

    # -- callbacks ----------------------------------------------------------
    def _on_recorder_warning(self, reason: str, details: Mapping[str, Any]) -> None:
        self._warnings.append(reason)
        if self._on_warning is not None:
            self._on_warning(reason, details)

    @property
    def recorder(self) -> Optional[EpisodeRecorder]:
        return self._recorder


__all__ = [
    "CollectionHarness",
    "CollectionResult",
    "build_collection_metadata",
]
