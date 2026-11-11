"""Bounded latest-only action inbox semantics."""

from __future__ import annotations

import threading

import pytest

from vla_infra.errors import ConfigError
from vla_infra.execution.authority import ControlAuthority
from vla_infra.execution.inbox import ActionInbox
from vla_infra.testing import make_chunk

NOW = 1_000_000_000


def _armed() -> tuple[ControlAuthority, ActionInbox, int]:
    authority = ControlAuthority()
    epoch = authority.grant()
    authority.activate()
    return authority, ActionInbox(authority), epoch


def test_submit_and_take() -> None:
    _, inbox, epoch = _armed()
    chunk = make_chunk()
    assert inbox.submit(chunk, epoch=epoch, received_time_ns=NOW)
    assert inbox.depth() == 1
    entry = inbox.take(now_ns=NOW + 1_000, epoch=epoch, max_age_ns=10_000)
    assert entry is not None and entry.chunk is chunk and entry.epoch == epoch
    assert inbox.depth() == 0
    assert inbox.take(now_ns=NOW + 1_000, epoch=epoch, max_age_ns=10_000) is None


def test_epoch_mismatch_rejected() -> None:
    authority, inbox, epoch = _armed()
    authority.revoke("hold")
    assert not inbox.submit(make_chunk(), epoch=epoch, received_time_ns=NOW)
    assert inbox.depth() == 0
    assert inbox.stats.rejected_epoch == 1


def test_sequence_regression_rejected_even_after_clear() -> None:
    _, inbox, epoch = _armed()
    assert inbox.submit(make_chunk(request_sequence_id=5), epoch=epoch, received_time_ns=NOW)
    inbox.clear("hold")
    assert not inbox.submit(make_chunk(request_sequence_id=5), epoch=epoch, received_time_ns=NOW)
    assert not inbox.submit(make_chunk(request_sequence_id=4), epoch=epoch, received_time_ns=NOW)
    assert inbox.submit(make_chunk(request_sequence_id=6), epoch=epoch, received_time_ns=NOW)
    assert inbox.pending_sequence == 6
    assert inbox.stats.rejected_sequence == 2


def test_latest_only_replacement() -> None:
    _, inbox, epoch = _armed()
    first = make_chunk(request_sequence_id=1)
    second = make_chunk(request_sequence_id=2)
    assert inbox.submit(first, epoch=epoch, received_time_ns=NOW)
    assert inbox.submit(second, epoch=epoch, received_time_ns=NOW)
    entry = inbox.take(now_ns=NOW, epoch=epoch, max_age_ns=10_000)
    assert entry is not None and entry.chunk is second
    assert inbox.stats.replaced == 1


def test_stale_chunk_rejected_on_take() -> None:
    _, inbox, epoch = _armed()
    inbox.submit(make_chunk(), epoch=epoch, received_time_ns=NOW)
    entry = inbox.take(now_ns=NOW + 100_000, epoch=epoch, max_age_ns=10_000)
    assert entry is None
    assert inbox.depth() == 0
    assert inbox.stats.rejected_stale == 1


def test_clear_removes_pending_chunk() -> None:
    _, inbox, epoch = _armed()
    inbox.submit(make_chunk(), epoch=epoch, received_time_ns=NOW)
    inbox.clear("hold")
    assert inbox.depth() == 0
    assert inbox.stats.cleared == 1


def test_capacity_must_be_one() -> None:
    authority = ControlAuthority()
    with pytest.raises(ConfigError):
        ActionInbox(authority, capacity=4)


def test_submit_and_take_are_thread_safe() -> None:
    _, inbox, epoch = _armed()
    errors: list[BaseException] = []

    def producer() -> None:
        try:
            for sequence in range(200):
                inbox.submit(
                    make_chunk(request_sequence_id=sequence),
                    epoch=epoch,
                    received_time_ns=NOW + sequence,
                )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def consumer() -> None:
        try:
            for _ in range(200):
                inbox.take(now_ns=NOW + 1_000_000, epoch=epoch, max_age_ns=10_000_000)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=producer), threading.Thread(target=consumer)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert not errors
