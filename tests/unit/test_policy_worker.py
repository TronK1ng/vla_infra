"""Policy worker: async inference, epoch capture, timeout/reconnect handling."""

from __future__ import annotations

import threading
import time
from dataclasses import replace

import numpy as np
import pytest

from vla_infra.clock import FakeClock
from vla_infra.contracts.action import ActionChunk
from vla_infra.contracts.policy_io import CANONICAL_STATE_LAYOUT, build_policy_observation
from vla_infra.execution.authority import ControlAuthority
from vla_infra.execution.inbox import ActionInbox
from vla_infra.errors import ErrorCode, PolicyContractError, ValidationError
from vla_infra.execution.supervisor import Supervisor, SystemState
from vla_infra.policy.adapters.base import ZeroActionAdapter
from vla_infra.policy.client import InProcessPolicyTransport
from vla_infra.policy.worker import PolicyWorker
from vla_infra.testing import make_observation

SOURCES = ("robot", "camera", "executor", "policy")


def wait_until(predicate, timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


def _payload(
    image_size: int = 224,
    *,
    sequence_id: int = 0,
    capture_time_ns: int = 1_000_000_000,
) -> dict:
    return build_policy_observation(
        make_observation(sequence_id=sequence_id, capture_time_ns=capture_time_ns),
        CANONICAL_STATE_LAYOUT,
        image_size=image_size,
    )


def _setup(transport):
    clock = FakeClock(start_ns=1_000_000_000)
    authority = ControlAuthority()
    inbox = ActionInbox(authority)
    supervisor = Supervisor(clock=clock, authority=authority, inbox=inbox)
    supervisor.connect_devices(now_ns=clock.now_ns())
    supervisor.devices_ready(now_ns=clock.now_ns())
    worker = PolicyWorker(
        transport=transport,
        inbox=inbox,
        authority=authority,
        supervisor=supervisor,
        clock=clock,
        poll_interval_s=0.001,
        reconnect_backoff_s=0.05,
        max_backoff_s=0.1,
    )
    return clock, authority, inbox, supervisor, worker


def _arm(supervisor: Supervisor, clock: FakeClock) -> None:
    now = clock.now_ns()
    supervisor.on_policy_connected(now_ns=now)
    for source in SOURCES:
        supervisor.heartbeat(source, now_ns=now)
    supervisor.arm(operator_ack=True, now_ns=now)
    supervisor.start_running(now_ns=now)


def test_worker_publishes_chunk_with_snapshot_epoch() -> None:
    adapter = ZeroActionAdapter(horizon=2, action_dt_s=0.05)
    transport = InProcessPolicyTransport(adapter)
    clock, authority, inbox, supervisor, worker = _setup(transport)
    _arm(supervisor, clock)
    worker.start()
    try:
        worker.publish(_payload(), capture_time_ns=clock.now_ns(), sequence_id=0)
        assert wait_until(lambda: inbox.depth() == 1)
        entry = inbox.take(now_ns=clock.now_ns(), epoch=authority.epoch, max_age_ns=1_000_000_000)
        assert entry is not None
        assert entry.epoch == authority.epoch
        assert entry.chunk.request_sequence_id == 0
    finally:
        worker.stop()


def test_hold_during_inference_rejects_chunk() -> None:
    class BlockingTransport:
        def __init__(self) -> None:
            self._adapter = ZeroActionAdapter(horizon=2, action_dt_s=0.05)
            self.connected = False
            self.started = threading.Event()
            self.release = threading.Event()
            self._metadata = self._adapter.metadata

        @property
        def metadata(self):
            return self._metadata

        def connect(self):
            self.connected = True
            return self._metadata

        def close(self):
            self.connected = False

        def infer(self, observation, request_sequence_id):
            self.started.set()
            self.release.wait(2.0)
            return ActionChunk(
                request_sequence_id=request_sequence_id,
                observation_capture_time_ns=observation["meta/capture_time_ns"],
                action_dt_s=self._metadata.action_dt_s,
                action_frame=self._metadata.action_frame,
                action_type=self._metadata.action_type,
                actions=np.zeros((2, 7), dtype=np.float32),
            )

    transport = BlockingTransport()
    clock, authority, inbox, supervisor, worker = _setup(transport)
    _arm(supervisor, clock)
    worker.start()
    try:
        worker.publish(_payload(), capture_time_ns=clock.now_ns(), sequence_id=0)
        assert transport.started.wait(2.0)
        old_epoch = authority.epoch
        supervisor.hold("during_inference", now_ns=clock.now_ns())
        transport.release.set()
        assert wait_until(lambda: worker.stats.results >= 1)
        time.sleep(0.05)
        assert inbox.depth() == 0
        assert worker.stats.inbox_rejected >= 1
        assert authority.epoch > old_epoch
        assert supervisor.state is SystemState.HOLD
    finally:
        transport.release.set()
        worker.stop()


def test_timeout_forces_hold_and_reconnect_does_not_resume() -> None:
    adapter = ZeroActionAdapter(horizon=2, action_dt_s=0.05)
    transport = InProcessPolicyTransport(adapter, fail_next=1, fail_with_timeout=True)
    clock, authority, inbox, supervisor, worker = _setup(transport)
    _arm(supervisor, clock)
    worker.start()
    try:
        worker.publish(_payload(), capture_time_ns=clock.now_ns(), sequence_id=0)
        assert wait_until(lambda: supervisor.state is SystemState.HOLD)
        assert worker.stats.timeouts >= 1
        assert not authority.valid
        # The worker reconnects on the next observation; the supervisor must not resume.
        worker.publish(_payload(sequence_id=2), capture_time_ns=clock.now_ns(), sequence_id=2)
        assert wait_until(lambda: transport.connected and worker.stats.results >= 1)
        assert supervisor.state is SystemState.HOLD
        assert not authority.valid
    finally:
        worker.stop()


def test_worker_stops_cleanly() -> None:
    transport = InProcessPolicyTransport(ZeroActionAdapter(horizon=2))
    _, _, _, _, worker = _setup(transport)
    worker.start()
    worker.publish(_payload(), capture_time_ns=1_000_000_000, sequence_id=0)
    assert wait_until(lambda: transport.connected)
    worker.stop(timeout_s=2.0)
    assert not transport.connected


def test_contract_error_latches_fault() -> None:
    class BadTransport(InProcessPolicyTransport):
        def infer(self, observation, request_sequence_id):
            raise PolicyContractError("bad server", code=ErrorCode.SCHEMA_MISMATCH)

    transport = BadTransport(ZeroActionAdapter(horizon=2))
    clock, authority, inbox, supervisor, worker = _setup(transport)
    _arm(supervisor, clock)
    worker.start()
    try:
        worker.publish(_payload(), capture_time_ns=clock.now_ns(), sequence_id=0)
        assert wait_until(lambda: supervisor.state is SystemState.FAULT)
        assert worker.stats.contract_errors >= 1
        assert not authority.valid
    finally:
        worker.stop()


class _ConnectFailingTransport:
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc
        self._adapter = ZeroActionAdapter(horizon=2)
        self.connected = False

    @property
    def metadata(self):
        return self._adapter.metadata

    def connect(self):
        raise self._exc

    def close(self):
        self.connected = False

    def infer(self, observation, request_sequence_id):  # pragma: no cover - not reached
        raise AssertionError("infer must not be called when connect fails")


def test_unknown_connect_exception_enters_hold() -> None:
    transport = _ConnectFailingTransport(RuntimeError("proxy handshake bug"))
    clock, authority, inbox, supervisor, worker = _setup(transport)
    _arm(supervisor, clock)
    worker.start()
    try:
        worker.publish(_payload(), capture_time_ns=clock.now_ns(), sequence_id=0)
        assert wait_until(lambda: supervisor.state is SystemState.HOLD)
        assert not authority.valid
        assert worker.stats.internal_errors >= 1
        assert worker.stats.last_error.startswith("connect_internal")
    finally:
        worker.stop()


def test_contract_error_during_connect_latches_fault() -> None:
    transport = _ConnectFailingTransport(
        PolicyContractError("bad metadata", code=ErrorCode.SCHEMA_MISMATCH)
    )
    clock, authority, inbox, supervisor, worker = _setup(transport)
    _arm(supervisor, clock)
    worker.start()
    try:
        worker.publish(_payload(), capture_time_ns=clock.now_ns(), sequence_id=0)
        assert wait_until(lambda: supervisor.state is SystemState.FAULT)
        assert not authority.valid
        assert worker.stats.contract_errors >= 1
    finally:
        worker.stop()


def test_capture_mismatch_chunk_never_reaches_inbox() -> None:
    class MismatchedCaptureTransport(InProcessPolicyTransport):
        def infer(self, observation, request_sequence_id):
            chunk = super().infer(observation, request_sequence_id)
            return replace(
                chunk,
                observation_capture_time_ns=chunk.observation_capture_time_ns + 1,
            )

    transport = MismatchedCaptureTransport(ZeroActionAdapter(horizon=2))
    clock, authority, inbox, supervisor, worker = _setup(transport)
    _arm(supervisor, clock)
    worker.start()
    try:
        worker.publish(_payload(), capture_time_ns=clock.now_ns(), sequence_id=0)
        assert wait_until(lambda: supervisor.state is SystemState.FAULT)
        assert inbox.depth() == 0
        assert worker.stats.contract_errors >= 1
        assert not authority.valid
    finally:
        worker.stop()


def test_invalid_metadata_validation_error_during_connect_latches_fault() -> None:
    transport = _ConnectFailingTransport(
        ValidationError("bad state layout", code=ErrorCode.SCHEMA_MISMATCH)
    )
    clock, authority, inbox, supervisor, worker = _setup(transport)
    _arm(supervisor, clock)
    worker.start()
    try:
        worker.publish(_payload(), capture_time_ns=clock.now_ns(), sequence_id=0)
        assert wait_until(lambda: supervisor.state is SystemState.FAULT)
        assert worker.stats.contract_errors >= 1
        assert not authority.valid
    finally:
        worker.stop()


def test_publish_rejects_identity_mismatch_before_slot() -> None:
    transport = InProcessPolicyTransport(ZeroActionAdapter(horizon=2))
    clock, authority, inbox, supervisor, worker = _setup(transport)
    payload = _payload(capture_time_ns=1_000_000_000, sequence_id=0)

    with pytest.raises(ValidationError) as exc:
        worker.publish(payload, capture_time_ns=1_000_000_001, sequence_id=0)
    assert exc.value.code is ErrorCode.CAPTURE_MISMATCH
    assert worker.slot_stats.published == 0

    with pytest.raises(ValidationError) as exc:
        worker.publish(payload, capture_time_ns=1_000_000_000, sequence_id=1)
    assert exc.value.code is ErrorCode.SEQUENCE_MISMATCH
    assert worker.slot_stats.published == 0

    missing = dict(payload)
    missing.pop("meta/sequence_id")
    with pytest.raises(ValidationError) as exc:
        worker.publish(missing, capture_time_ns=1_000_000_000, sequence_id=0)
    assert exc.value.code is ErrorCode.SCHEMA_MISMATCH
    assert worker.slot_stats.published == 0

    bad_type = dict(payload)
    bad_type["meta/sequence_id"] = True
    with pytest.raises(ValidationError) as exc:
        worker.publish(bad_type, capture_time_ns=1_000_000_000, sequence_id=0)
    assert exc.value.code is ErrorCode.INVALID_DTYPE
    assert worker.slot_stats.published == 0

    worker.publish(payload, capture_time_ns=1_000_000_000, sequence_id=0)
    assert worker.slot_stats.published == 1


def test_cleanup_failure_faults_and_never_retries() -> None:
    from vla_infra.errors import PolicyCleanupError

    class CleanupFailingTransport:
        def __init__(self):
            adapter = ZeroActionAdapter(horizon=2, action_dt_s=0.05)
            self._metadata = adapter.metadata
            self.connected = False
            self.connects = 0
            self.infers = 0
            self.closes = 0

        @property
        def metadata(self):
            return self._metadata

        def connect(self):
            self.connected = True
            self.connects += 1
            return self._metadata

        def close(self):
            self.connected = False
            self.closes += 1

        def infer(self, observation, request_sequence_id):
            self.infers += 1
            raise PolicyCleanupError(
                "infer cleanup failed",
                details={"threads": ["openpi-infer"], "cleanup_failed": True},
            )

    transport = CleanupFailingTransport()
    clock, authority, inbox, supervisor, worker = _setup(transport)
    _arm(supervisor, clock)
    worker.start()
    try:
        worker.publish(_payload(), capture_time_ns=clock.now_ns(), sequence_id=0)
        assert wait_until(lambda: supervisor.state is SystemState.FAULT)
        assert transport.infers == 1
        assert transport.connects == 1
        assert transport.closes == 1
        stats = worker.stats
        assert stats.cleanup_failures == 1
        assert "policy_cleanup_failed" in stats.last_error
        time.sleep(0.1)
        assert transport.infers == 1
        assert transport.connects == 1
    finally:
        worker.stop()


@pytest.mark.parametrize(
    "code",
    [ErrorCode.INVALID_DTYPE, ErrorCode.INVALID_SHAPE, ErrorCode.SCHEMA_MISMATCH, ErrorCode.NON_FINITE],
)
def test_action_contract_validation_is_not_internal_error(code) -> None:
    class InvalidActionTransport:
        def __init__(self):
            adapter = ZeroActionAdapter(horizon=2, action_dt_s=0.05)
            self._metadata = adapter.metadata
            self.connected = False
            self.infers = 0
            self.connects = 0
            self.closes = 0

        @property
        def metadata(self):
            return self._metadata

        def connect(self):
            self.connected = True
            self.connects += 1
            return self._metadata

        def close(self):
            self.connected = False
            self.closes += 1

        def infer(self, observation, request_sequence_id):
            self.infers += 1
            raise ValidationError("bad remote action contract", code=code)

    transport = InvalidActionTransport()
    clock, authority, inbox, supervisor, worker = _setup(transport)
    _arm(supervisor, clock)
    worker.start()
    try:
        worker.publish(_payload(), capture_time_ns=clock.now_ns(), sequence_id=0)
        assert wait_until(lambda: supervisor.state is SystemState.FAULT)
        stats = worker.stats
        assert stats.contract_errors == 1
        assert stats.internal_errors == 0
        assert stats.cleanup_failures == 0
        assert code.value in stats.last_error
        assert transport.infers == 1
        assert transport.connects == 1
        assert transport.closes == 1
    finally:
        worker.stop()
