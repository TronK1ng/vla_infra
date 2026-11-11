"""Clock behaviour: monotonic only, deterministic fake time."""

from __future__ import annotations

import pytest

from vla_infra.clock import FakeClock, MonotonicClock, ns_to_seconds, seconds_to_ns


def test_fake_clock_is_deterministic_and_monotonic() -> None:
    clock = FakeClock(start_ns=1_000)
    assert clock.now_ns() == 1_000
    assert clock.advance(500) == 1_500
    clock.sleep(250)
    assert clock.now_ns() == 1_750
    clock.set(2_000)
    assert clock.now_ns() == 2_000


def test_fake_clock_refuses_backwards_time() -> None:
    clock = FakeClock(start_ns=1_000)
    with pytest.raises(ValueError):
        clock.advance(-1)
    with pytest.raises(ValueError):
        clock.set(999)


def test_monotonic_clock_never_decreases() -> None:
    clock = MonotonicClock()
    samples = [clock.now_ns() for _ in range(1_000)]
    assert all(b >= a for a, b in zip(samples, samples[1:]))


def test_seconds_conversions() -> None:
    assert seconds_to_ns(0.05) == 50_000_000
    assert ns_to_seconds(50_000_000) == pytest.approx(0.05)
