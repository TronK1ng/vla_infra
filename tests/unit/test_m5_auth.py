"""M5 authorization chain negative tests."""

from __future__ import annotations

import pytest

from vla_infra.devices.m5_auth import M5AuthorizationIssuer, MotionAuthorization
from vla_infra.devices.preflight import (
    M5PreflightReport,
    M5PreflightRunner,
    PreflightVerdict,
    REQUIRED_ENABLE_PROBES,
    REQUIRED_SECTIONS,
)
from vla_infra.errors import RealHardwareDisabledError


def _report(station="S1", config="cfg", readonly_status="PASS", enable_status="PASS", blockers=()):
    readonly = {
        name: (lambda n=name: {"status": readonly_status, "details": {"name": n}})
        for name in REQUIRED_SECTIONS
    }
    enable = {
        name: (lambda n=name: {"status": enable_status, "details": {"name": n}})
        for name in REQUIRED_ENABLE_PROBES
    }
    return M5PreflightRunner(
        station_id=station,
        config_hash=config,
        readonly_probes=readonly,
        enable_probes=enable,
        clock_ns=lambda: 1000,
    ).run(allow_enable=True, operator_ack=True)


def _issuer(station="S1", config="cfg"):
    return M5AuthorizationIssuer(
        station_id=station, config_hash=config, secret=b"auth-test-secret-32-bytes-long!!"
    )


def test_issuer_accepts_only_validated_enable_report() -> None:
    issuer = _issuer()
    report = _report()
    assert report.verdict is PreflightVerdict.PASS_ENABLE_ALLOWED
    auth = issuer.issue(report, operator_ack=True, now_ns=1000)
    assert issuer.verify(auth, now_ns=1000) is True

    with pytest.raises(RealHardwareDisabledError):
        issuer.issue(object(), operator_ack=True, now_ns=1000)


def test_issuer_rejects_forged_verdict_sections_and_blockers() -> None:
    issuer = _issuer()
    valid = _report()
    sections = valid.as_dict()["sections"]
    forged = M5PreflightReport(
        schema_version="m5-preflight/v1",
        generated_at_ns=1000,
        expires_at_ns=2000,
        station_id="S1",
        config_hash="cfg",
        sections={},
        verdict=PreflightVerdict.PASS_ENABLE_ALLOWED,
    )
    with pytest.raises(RealHardwareDisabledError):
        issuer.issue(forged, operator_ack=True, now_ns=1000)

    blocked = M5PreflightReport(
        schema_version="m5-preflight/v1",
        generated_at_ns=1000,
        expires_at_ns=2000,
        station_id="S1",
        config_hash="cfg",
        sections=sections,
        verdict=PreflightVerdict.PASS_ENABLE_ALLOWED,
        blockers=("fatal",),
    )
    with pytest.raises(RealHardwareDisabledError):
        issuer.issue(blocked, operator_ack=True, now_ns=1000)

    failed = _report(enable_status="FAIL")
    with pytest.raises(RealHardwareDisabledError):
        issuer.issue(failed, operator_ack=True, now_ns=1000)


def test_issuer_binds_expected_station_and_config() -> None:
    report = _report(station="OTHER", config="other")
    with pytest.raises(RealHardwareDisabledError):
        _issuer().issue(report, operator_ack=True, now_ns=1000)
    valid = _report()
    auth = _issuer().issue(valid, operator_ack=True, now_ns=1000)
    assert _issuer(station="OTHER", config="cfg").verify(auth, now_ns=1000) is False
    assert _issuer(station="S1", config="other").verify(auth, now_ns=1000) is False


def test_issuer_rejects_expired_report_and_fabricated_auth() -> None:
    issuer = _issuer()
    report = _report()
    with pytest.raises(RealHardwareDisabledError):
        issuer.issue(report, operator_ack=True, now_ns=61_000_000_001)
    auth = issuer.issue(report, operator_ack=True, now_ns=1000)
    forged = MotionAuthorization(
        station_id=auth.station_id,
        config_hash=auth.config_hash,
        report_hash=auth.report_hash,
        issued_at_ns=auth.issued_at_ns,
        expires_at_ns=auth.expires_at_ns,
        operator_ack=True,
        token="0" * 64,
    )
    assert issuer.verify(forged, now_ns=1000) is False
    assert issuer.verify(auth, now_ns=61_000_000_001) is False


def test_report_sections_are_deeply_immutable() -> None:
    report = _report()
    with pytest.raises(TypeError):
        report.sections["software"] = {"status": "FAIL"}
    with pytest.raises(TypeError):
        report.sections["software"]["details"]["name"] = "mutated"
    assert report.validate_for_enable(
        expected_station_id="S1", expected_config_hash="cfg", now_ns=1000
    ) is None


def test_motion_capability_concurrent_grant_revoke_is_safe() -> None:
    import threading

    from vla_infra.devices.m5_auth import MotionCapability

    issuer = _issuer()
    report = _report()
    auth = issuer.issue(report, operator_ack=True, now_ns=1000)
    capability = MotionCapability()
    errors = []

    def worker(idx):
        try:
            for _ in range(200):
                capability.grant(auth, idx % 5)
                capability.check(auth, idx % 5)
                capability.revoke()
                capability.active
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2.0)
    assert errors == []
    assert capability.active is False


def test_tool_readiness_binds_approved_hashes() -> None:
    from vla_infra.devices.tool_state import ToolReadiness
    from vla_infra.errors import DeviceError

    readiness = ToolReadiness(
        expected_name="G1", expected_tcp_hash="approved-tcp", expected_payload_hash="approved-payload"
    )
    readiness.mark_switched("G1")
    with pytest.raises(DeviceError):
        readiness.verify(name="G1", tcp_hash="other", payload_hash="approved-payload")
    assert readiness.verified is False
    readiness.verify(name="G1", tcp_hash="approved-tcp", payload_hash="approved-payload")
    assert readiness.verified is True
    with pytest.raises(DeviceError):
        readiness.verify(name="G1", tcp_hash="other", payload_hash="approved-payload")
