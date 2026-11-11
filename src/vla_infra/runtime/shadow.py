"""Hardware-free shadow harness (M1 acceptance vehicle).

This is **not** the production runtime.  It wires the M1 components together in
one process with fake devices and a loopback dummy policy server so that the
whole chain (fake cameras/robot -> sync -> policy observation -> WebSocket
policy -> inbox -> safety -> interpolation -> fake executor -> supervisor) can
be exercised and profiled without touching the real robot.

``run(..., operator_ack=True)`` enables motion on the *fake* backend only.
Without the acknowledgement the supervisor stays in HOLD and the policy runs in
shadow mode (chunks are produced but never executed).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional

from ..clock import Clock, MonotonicClock
from ..config import AppConfig
from ..contracts.constants import IMAGE_KEY_BASE_RGB, IMAGE_KEY_WRIST_RGB
from ..contracts.policy_io import build_policy_observation
from ..devices.fake import FakeCamera, FakeGripper, FakeRobotArm
from ..errors import ConfigError, SyncError, VlaInfraError
from ..execution.authority import ControlAuthority
from ..execution.executor import ExecutionReport, Executor, ExecutorEvent, ExecutorEventKind
from ..execution.inbox import ActionInbox
from ..execution.safety import SafetyGate
from ..execution.supervisor import Supervisor
from ..policy.adapters.base import (
    FixedDeltaAdapter,
    PolicyAdapter,
    SineSweepAdapter,
    ZeroActionAdapter,
)
from ..policy.client import WebsocketPolicyClient, sanitize_endpoint
from ..policy.dummy_server import DummyPolicyServer
from ..policy.worker import PolicyWorker
from ..sync.synchronizer import ObservationSynchronizer


@dataclass(frozen=True)
class HarnessResult:
    final_state: str
    armed: bool
    iterations: int
    observations: int
    policy_chunks: int
    writes: int
    sync_dropped: Mapping[str, int]
    executor: Mapping[str, Any]
    worker: Mapping[str, Any]
    client: Mapping[str, Any]
    server: Mapping[str, Any]
    supervisor: Mapping[str, Any]
    transport: Mapping[str, Any] = field(default_factory=dict)
    error: str = ""

    def as_dict(self) -> dict:
        return {
            "final_state": self.final_state,
            "armed": self.armed,
            "iterations": self.iterations,
            "observations": self.observations,
            "policy_chunks": self.policy_chunks,
            "writes": self.writes,
            "sync_dropped": dict(self.sync_dropped),
            "executor": dict(self.executor),
            "worker": dict(self.worker),
            "client": dict(self.client),
            "server": dict(self.server),
            "transport": dict(self.transport),
            "supervisor": {
                "state": self.supervisor.get("state"),
                "epoch": self.supervisor.get("epoch"),
                "latched_fault": self.supervisor.get("latched_fault"),
                "transitions": self.supervisor.get("transitions", []),
            },
            "error": self.error,
        }


def build_adapter(name: str) -> PolicyAdapter:
    if name == "zero":
        return ZeroActionAdapter(horizon=10, action_dt_s=0.05, gripper_normalized=0.5)
    if name == "fixed":
        return FixedDeltaAdapter(
            delta_m=(0.002, 0.0, 0.0),
            horizon=10,
            action_dt_s=0.05,
            gripper_normalized=0.5,
        )
    if name == "sine":
        return SineSweepAdapter(
            amplitude_m=0.002, period_steps=40, horizon=10, action_dt_s=0.05
        )
    raise ConfigError(f"unknown dummy adapter {name!r}")


class ShadowHarness:
    def __init__(
        self,
        app: AppConfig,
        *,
        clock: Optional[Clock] = None,
        embedded_server: bool = True,
        adapter: Optional[PolicyAdapter] = None,
        task: str = "shadow run: move slowly",
        auth_token: Optional[str] = None,
        on_chunk=None,
        transport_kind: str = "custom",
        transport_factory: Optional[Callable[[str], Any]] = None,
        server_factory: Optional[Callable[[PolicyAdapter], Any]] = None,
    ) -> None:
        if app.station.backend != "fake":
            raise ConfigError("M1 shadow harness only supports backend: fake")
        self.app = app
        self._clock = clock or MonotonicClock()
        self._embedded_server = embedded_server
        self._adapter = adapter or build_adapter(app.policy.server.adapter)
        self._task = task
        self._auth_token = auth_token
        self._on_chunk = on_chunk
        if transport_kind not in ("custom", "official"):
            raise ConfigError(f"unsupported transport_kind {transport_kind!r}")
        self._transport_kind = transport_kind
        self._transport_factory = transport_factory
        self._server_factory = server_factory
        self._cameras: Dict[str, FakeCamera] = {}
        self._robot: Optional[FakeRobotArm] = None
        self._sync: Optional[ObservationSynchronizer] = None
        self._authority: Optional[ControlAuthority] = None
        self._inbox: Optional[ActionInbox] = None
        self._supervisor: Optional[Supervisor] = None
        self._executor: Optional[Executor] = None
        self._server: Optional[Any] = None
        self._client: Optional[Any] = None
        self._worker: Optional[PolicyWorker] = None
        self._latest_frames: Dict[str, Any] = {}
        self._built_frame_numbers: Dict[str, int] = {}
        self._observations = 0
        self._chunks = 0
        self._writes = 0
        self._events: list[ExecutorEvent] = []
        self._sync_drops: Counter = Counter()
        self._iteration = 0
        self._started = False
        self._endpoint = app.policy.endpoint

    # -- accessors for tests / CLI -----------------------------------------
    @property
    def supervisor(self) -> Supervisor:
        assert self._supervisor is not None
        return self._supervisor

    @property
    def executor(self) -> Executor:
        assert self._executor is not None
        return self._executor

    @property
    def worker(self) -> PolicyWorker:
        assert self._worker is not None
        return self._worker

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def transport(self) -> Mapping[str, Any]:
        identity: Dict[str, Any] = {
            "transport_kind": self._transport_kind,
            "endpoint": sanitize_endpoint(self._endpoint),
        }
        client = self._client
        server = self._server
        if client is not None and hasattr(client, "identity"):
            identity.update(dict(client.identity))
        else:
            identity["client_class"] = type(client).__name__ if client is not None else None
        if server is not None and hasattr(server, "identity"):
            identity.update(dict(server.identity))
        else:
            identity["server_class"] = type(server).__name__ if server is not None else None
        identity["endpoint"] = sanitize_endpoint(self._endpoint)
        return identity

    # -- lifecycle ----------------------------------------------------------
    def start(self, *, operator_ack: bool = False) -> None:
        if self._started:
            raise RuntimeError("harness already started")
        try:
            self._start_components(operator_ack=operator_ack)
        except BaseException:
            self.stop()
            raise

    def _start_components(self, *, operator_ack: bool) -> None:
        station = self.app.station
        safety_profile = self.app.safety
        image_keys = {"base": IMAGE_KEY_BASE_RGB, "wrist": IMAGE_KEY_WRIST_RGB}
        self._cameras = {
            image_keys[name]: FakeCamera(
                image_keys[name],
                self._clock,
                width=camera.width,
                height=camera.height,
                fps=camera.fps,
                device_latency_ns=int(camera.device_latency_ms * 1_000_000),
                seed=index,
            )
            for index, (name, camera) in enumerate(station.cameras.items())
        }
        for camera in self._cameras.values():
            camera.start()
        gripper = FakeGripper(
            width_min_m=safety_profile.gripper_width_min_m,
            width_max_m=safety_profile.gripper_width_max_m,
            initial_width_m=(safety_profile.gripper_width_min_m + safety_profile.gripper_width_max_m) / 2.0,
        )
        self._robot = FakeRobotArm(self._clock, gripper=gripper)
        self._sync = ObservationSynchronizer(self._clock, station.sync)
        self._authority = ControlAuthority(owner="supervisor")
        self._inbox = ActionInbox(self._authority)
        self._supervisor = Supervisor(
            clock=self._clock, authority=self._authority, inbox=self._inbox
        )
        self._executor = Executor(
            clock=self._clock,
            robot=self._robot,
            state_provider=self._sync.buffer.latest,
            authority=self._authority,
            inbox=self._inbox,
            safety=SafetyGate(safety_profile),
            tick_period_ns=10_000_000,
            on_event=self._on_executor_event,
        )
        self._executor.start()
        if self._embedded_server:
            if self._server_factory is not None:
                self._server = self._server_factory(self._adapter)
            elif self._transport_kind == "official":
                from ..policy.openpi_official import OfficialOpenPIServer

                self._server = OfficialOpenPIServer(
                    self._adapter,
                    host=self.app.policy.server.host,
                    port=self.app.policy.server.port,
                    clock=self._clock,
                )
            else:
                self._server = DummyPolicyServer(
                    self._adapter,
                    host=self.app.policy.server.host,
                    port=self.app.policy.server.port,
                    auth_token=self._auth_token,
                    allow_non_loopback=self.app.policy.server.allow_non_loopback,
                    clock=self._clock,
                )
            self._endpoint = self._server.start()
        if self._transport_factory is not None:
            self._client = self._transport_factory(self._endpoint)
        elif self._transport_kind == "official":
            from ..policy.openpi_official import OfficialOpenPIClient

            self._client = OfficialOpenPIClient(
                self._endpoint,
                timeout_s=self.app.policy.timeout_s,
                connect_timeout_s=self.app.policy.connect_timeout_s,
                expected_metadata=self.app.policy.expected_metadata(),
                clock=self._clock,
            )
        else:
            self._client = WebsocketPolicyClient(
                self._endpoint,
                timeout_s=self.app.policy.timeout_s,
                connect_timeout_s=self.app.policy.connect_timeout_s,
                auth_token=self._auth_token,
                proxy=self.app.policy.proxy,
                clock=self._clock,
                expected_metadata=self.app.policy.expected_metadata(),
            )
        self._worker = PolicyWorker(
            transport=self._client,
            inbox=self._inbox,
            authority=self._authority,
            supervisor=self._supervisor,
            clock=self._clock,
            on_action=self._on_policy_action,
        )
        self._supervisor.connect_devices()
        self._supervisor.devices_ready()
        self._worker.start()
        self._started = True
        if operator_ack:
            self.arm_and_run(timeout_s=self.app.policy.timeout_s)

    def arm_and_run(self, *, timeout_s: float = 5.0) -> None:
        """Wait for the first policy result, then arm and start execution."""
        assert self._supervisor is not None
        deadline = self._clock.now_ns() + int(timeout_s * 1_000_000_000)
        while self._clock.now_ns() < deadline:
            self._step()
            if self._worker is not None and self._worker.stats.results > 0:
                break
            self._clock.sleep(20_000_000)
        if self._worker is None or self._worker.stats.results == 0:
            raise VlaInfraError(
                "no policy result arrived before the arming deadline",
                details={"endpoint": self._endpoint, "timeout_s": timeout_s},
            )
        now = self._clock.now_ns()
        for source in ("robot", "camera", "executor"):
            self._supervisor.heartbeat(source, now_ns=now)
        self._supervisor.arm(operator_ack=True, now_ns=now)
        self._supervisor.start_running(now_ns=now)

    def run(self, *, duration_s: float = 2.0, operator_ack: bool = False) -> HarnessResult:
        self.start(operator_ack=operator_ack)
        period_ns = 10_000_000
        iterations = max(1, int(duration_s / (period_ns / 1_000_000_000)))
        next_tick = self._clock.now_ns()
        try:
            for _ in range(iterations):
                self._step()
                next_tick += period_ns
                now = self._clock.now_ns()
                if next_tick > now:
                    self._clock.sleep(next_tick - now)
                else:
                    next_tick = now
        finally:
            result = self._snapshot()
            self.stop()
        return result

    def stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()
        if self._client is not None:
            self._client.close()
        if self._server is not None:
            self._server.stop()
        if self._robot is not None:
            try:
                self._robot.release_writer()
            except Exception:  # pragma: no cover - best effort
                pass
        for camera in self._cameras.values():
            camera.stop()
        self._started = False

    # -- one control-loop iteration ----------------------------------------
    def _step(self) -> ExecutionReport:
        assert self._robot is not None
        assert self._sync is not None
        assert self._supervisor is not None
        assert self._executor is not None
        now = self._clock.now_ns()
        self._iteration += 1
        # The harness emulates the configured state rate in batches around the
        # camera reads so that frame targets are always bracketed by samples.
        samples_per_step = max(1, int(round(self.app.station.robot.state_rate_hz * 0.01)))
        before_count = max(1, samples_per_step // 2)
        for _ in range(before_count):
            self._sync.add_robot_sample(self._robot.read_state())
        for name, camera in self._cameras.items():
            frame = camera.read()
            if frame is not None:
                self._latest_frames[name] = frame
        for _ in range(max(1, samples_per_step - before_count)):
            self._sync.add_robot_sample(self._robot.read_state())
        base = self._latest_frames.get(IMAGE_KEY_BASE_RGB)
        wrist = self._latest_frames.get(IMAGE_KEY_WRIST_RGB)
        both_new = (
            base is not None
            and wrist is not None
            and self._built_frame_numbers.get(IMAGE_KEY_BASE_RGB) != base.frame_number
            and self._built_frame_numbers.get(IMAGE_KEY_WRIST_RGB) != wrist.frame_number
        )
        if both_new:
            try:
                observation = self._sync.build(
                    {IMAGE_KEY_BASE_RGB: base, IMAGE_KEY_WRIST_RGB: wrist},
                    self._task,
                    extra_dropped={
                        "camera_dropouts": sum(camera.stats.failures for camera in self._cameras.values())
                    },
                )
            except SyncError as exc:
                reason = str(exc.details.get("reason", exc.code.value))
                self._sync_drops[reason] += 1
            else:
                payload = build_policy_observation(
                    observation,
                    self.app.policy.state_layout,
                    prompt=self._task,
                    image_size=self.app.policy.image_size,
                )
                assert self._worker is not None
                self._worker.publish(
                    payload,
                    capture_time_ns=observation.capture_time_ns,
                    sequence_id=observation.sequence_id,
                )
                self._built_frame_numbers[IMAGE_KEY_BASE_RGB] = base.frame_number
                self._built_frame_numbers[IMAGE_KEY_WRIST_RGB] = wrist.frame_number
                self._observations += 1
        self._supervisor.heartbeat("robot", now_ns=now)
        self._supervisor.heartbeat("camera", now_ns=now)
        report = self._executor.tick(now)
        self._supervisor.heartbeat("executor", now_ns=now)
        if report.wrote:
            self._writes += 1
        self._supervisor.tick(now_ns=now)
        return report

    # -- callbacks ----------------------------------------------------------
    def _on_policy_action(self, chunk) -> None:
        self._chunks += 1
        if self._on_chunk is not None:
            try:
                self._on_chunk(chunk)
            except Exception:  # pragma: no cover - observability
                pass

    def _on_executor_event(self, event: ExecutorEvent) -> None:
        self._events.append(event)
        if self._supervisor is None:
            return
        try:
            if event.kind in (
                ExecutorEventKind.WRITE_FAILED,
                ExecutorEventKind.WRITE_PARTIAL,
                ExecutorEventKind.INTERNAL_ERROR,
            ):
                self._supervisor.fault(f"executor_{event.kind.value}", now_ns=event.time_ns)
            elif event.kind in (
                ExecutorEventKind.PLAN_REJECTED,
                ExecutorEventKind.PLAN_EXPIRED,
                ExecutorEventKind.STALE_STATE,
            ):
                self._supervisor.hold(
                    f"executor_{event.kind.value}", now_ns=event.time_ns
                )
            elif event.kind is ExecutorEventKind.IDLE:
                self._supervisor.on_executor_idle(now_ns=event.time_ns)
            elif event.kind is ExecutorEventKind.PLAN_STARTED:
                self._supervisor.on_executor_active(now_ns=event.time_ns)
        except VlaInfraError:
            pass

    # -- reporting ----------------------------------------------------------
    def _snapshot(self) -> HarnessResult:
        supervisor_status = self.supervisor.status()
        armed = any(
            transition["to"] == "ARMED" for transition in supervisor_status["transitions"]
        )
        return HarnessResult(
            final_state=supervisor_status["state"],
            armed=armed,
            iterations=self._iteration,
            observations=self._observations,
            policy_chunks=self._chunks,
            writes=self._writes,
            sync_dropped=dict(self._sync_drops),
            executor=self._executor.stats.as_dict() if self._executor else {},
            worker=self._worker.stats.as_dict() if self._worker else {},
            client=self._client.stats.as_dict() if self._client else {},
            server=self._server.stats.as_dict() if self._server else {},
            transport=self.transport,
            supervisor=supervisor_status,
        )


__all__ = ["HarnessResult", "ShadowHarness", "build_adapter"]
