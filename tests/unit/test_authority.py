"""Control authority: epoch validity and synchronous revocation."""

from __future__ import annotations

import threading
import time

import pytest

from vla_infra.errors import ErrorCode, NotArmedError
from vla_infra.execution.authority import ControlAuthority


def test_grant_activate_revoke() -> None:
    authority = ControlAuthority()
    assert not authority.valid and not authority.running
    with pytest.raises(NotArmedError):
        authority.activate()
    epoch = authority.grant("armed")
    assert epoch == 1 and authority.valid and not authority.running
    with pytest.raises(NotArmedError):
        with authority.write_permit(epoch):
            pass
    authority.activate()
    with authority.write_permit(epoch):
        pass
    revoked_epoch = authority.revoke("hold")
    assert revoked_epoch == 2
    assert not authority.valid and not authority.running
    with pytest.raises(NotArmedError) as exc:
        with authority.write_permit(epoch):
            pass
    assert exc.value.code is ErrorCode.EPOCH_MISMATCH
    assert not authority.is_current(epoch)


def test_stale_epoch_rejected() -> None:
    authority = ControlAuthority()
    first = authority.grant()
    authority.activate()
    second = authority.revoke("hold")
    assert second > first
    with pytest.raises(NotArmedError):
        with authority.write_permit(first):
            pass
    new = authority.grant()
    authority.activate()
    with authority.write_permit(new):
        pass


def test_revocation_waits_for_in_flight_write() -> None:
    authority = ControlAuthority()
    epoch = authority.grant()
    authority.activate()
    writes = []
    started = threading.Event()

    def write() -> None:
        with authority.write_permit(epoch):
            started.set()
            writes.append("begin")
            time.sleep(0.15)
            writes.append("end")

    thread = threading.Thread(target=write)
    thread.start()
    assert started.wait(2.0)
    began = time.monotonic()
    authority.revoke("hold")
    elapsed = time.monotonic() - began
    thread.join(timeout=2.0)
    assert elapsed >= 0.1
    assert writes == ["begin", "end"]
    with pytest.raises(NotArmedError):
        with authority.write_permit(epoch):
            writes.append("after-revoke")
    assert "after-revoke" not in writes
