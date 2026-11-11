"""Ring buffer: bounded, thread-safe, strictly monotonic, no future samples."""

from __future__ import annotations

import threading

import pytest

from vla_infra.clock import FakeClock
from vla_infra.errors import ErrorCode
from vla_infra.sync.ring_buffer import NonMonotonicTimestampError, TimestampRingBuffer
from vla_infra.testing import make_robot_sample


def _sample(clock: FakeClock, sequence: int):
    return make_robot_sample(clock, sequence_id=sequence)


def test_bracket_exact_and_interpolated() -> None:
    clock = FakeClock(start_ns=1_000)
    buffer = TimestampRingBuffer(capacity=8)
    for seq in range(5):
        buffer.append(_sample(clock, seq))
        clock.advance(10)
    exact = buffer.bracket(1_020)
    assert exact is not None and exact.exact and exact.gap_ns == 0
    bracket = buffer.bracket(1_025)
    assert bracket is not None
    assert bracket.before.host_time_ns == 1_020
    assert bracket.after.host_time_ns == 1_030
    assert bracket.gap_ns == 10
    assert bracket.exact is False


def test_bracket_never_extrapolates() -> None:
    clock = FakeClock(start_ns=1_000)
    buffer = TimestampRingBuffer(capacity=8)
    for seq in range(3):
        buffer.append(_sample(clock, seq))
        clock.advance(10)
    assert buffer.bracket(1_040) is None  # future target
    assert buffer.bracket(900) is None  # older than oldest
    assert buffer.bracket(1_010) is not None


def test_non_monotonic_timestamp_rejected() -> None:
    clock = FakeClock(start_ns=1_000)
    buffer = TimestampRingBuffer(capacity=8)
    buffer.append(make_robot_sample(clock, sequence_id=0, host_time_ns=1_000))
    with pytest.raises(NonMonotonicTimestampError) as exc:
        buffer.append(make_robot_sample(clock, sequence_id=1, host_time_ns=1_000))
    assert exc.value.code is ErrorCode.BUFFER_NOT_MONOTONIC
    assert buffer.stats.rejected == 1
    assert len(buffer) == 1


def test_capacity_eviction_and_stats() -> None:
    clock = FakeClock(start_ns=1_000)
    buffer = TimestampRingBuffer(capacity=4)
    for seq in range(10):
        buffer.append(make_robot_sample(clock, sequence_id=seq, host_time_ns=1_000 + seq))
    stats = buffer.stats
    assert stats.size == 4
    assert stats.appended == 10
    assert stats.evicted == 6
    assert stats.oldest_host_ns == 1_006
    assert stats.newest_host_ns == 1_009


def test_thread_safety_under_concurrent_append_and_bracket() -> None:
    buffer = TimestampRingBuffer(capacity=512)
    errors: list[BaseException] = []
    stop = threading.Event()

    def producer() -> None:
        try:
            for i in range(2_000):
                buffer.append(make_robot_sample(FakeClock(), sequence_id=i, host_time_ns=1_000 + i))
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)
        finally:
            stop.set()

    def consumer() -> None:
        try:
            while not stop.is_set():
                buffer.bracket(3_000)
                buffer.snapshot()
                stop.wait(0.0001)
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=producer), threading.Thread(target=consumer)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not errors
    assert buffer.stats.appended == 2_000
    assert buffer.stats.size == 512


def test_snapshot_is_immutable_tuple() -> None:
    clock = FakeClock(start_ns=1_000)
    buffer = TimestampRingBuffer(capacity=4)
    buffer.append(_sample(clock, 0))
    snapshot = buffer.snapshot()
    assert isinstance(snapshot, tuple)
    assert len(snapshot) == 1
    assert snapshot[0].state.gripper_width.flags.writeable is False
