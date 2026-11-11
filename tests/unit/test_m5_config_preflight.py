"""M5 Round 01 config/schema/preflight reporting tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vla_infra.config_m5 import load_m5_config
from vla_infra.devices.preflight import (
    M5PreflightReport,
    M5PreflightRunner,
    PreflightVerdict,
    ProbeResult,
    ProbeStatus,
    REQUIRED_ENABLE_PROBES,
    REQUIRED_SECTIONS,
)
from vla_infra.errors import ConfigError, RealHardwareDisabledError

ROOT = Path(__file__).resolve().parents[2]
STATION = ROOT / "configs" / "station.real.template.yaml"
SAFETY = ROOT / "configs" / "safety.real.template.yaml"


def test_m5_templates_load_strictly_and_are_disabled_by_default() -> None:
    config = load_m5_config(STATION, SAFETY)
    assert config.station.backend == "real"
    assert config.station.allow_real_motion is False
    assert config.station.nrt_limits.max_linear_vel_m_s == 0.02
    assert config.station.gn01.width_for_normalized(0.0) == 0.0
    assert config.station.gn01.width_for_normalized(1.0) == 0.1
    assert config.safety.stop_timeout_s == 2.0
    assert len(config.config_hash()) == 64
    assert config.config_hash() == load_m5_config(STATION, SAFETY).config_hash()
    with pytest.raises(RealHardwareDisabledError):
        config.assert_real_motion_gate()
    with pytest.raises(ConfigError):
        config.validate_tool_profile()


def test_m5_config_rejects_missing_or_unknown_keys(tmp_path: Path) -> None:
    raw = STATION.read_text()
    bad = tmp_path / "bad.yaml"
    bad.write_text(raw + "\ntypo_key: 1\n")
    with pytest.raises(ConfigError):
        load_m5_config(bad, SAFETY)
    missing = tmp_path / "missing.yaml"
    missing.write_text(raw.replace("nrt_limits:\n", "not_nrt_limits:\n"))
    with pytest.raises(ConfigError):
        load_m5_config(missing, SAFETY)


def _probes(value="PASS"):
    return {name: (lambda n=name: {"status": value, "details": {"name": n}}) for name in REQUIRED_SECTIONS}


def _enable_probes(value="PASS"):
    return {name: (lambda n=name: {"status": value, "details": {"name": n}}) for name in REQUIRED_ENABLE_PROBES}


def test_preflight_default_is_readonly_even_with_enable_probes() -> None:
    runner = M5PreflightRunner(
        station_id="S1",
        config_hash="abc",
        readonly_probes=_probes(),
        enable_probes=_enable_probes(),
        clock_ns=lambda: 123,
    )
    report = runner.run()
    assert report.verdict is PreflightVerdict.PASS_READONLY
    assert report.blockers == ()
    assert set(REQUIRED_SECTIONS).issubset(report.sections)
    assert "enable_enable_guard" not in report.sections


def test_preflight_enable_requires_explicit_request_and_probes() -> None:
    runner = M5PreflightRunner(
        station_id="S1",
        config_hash="abc",
        readonly_probes=_probes(),
        enable_probes=_enable_probes(),
        clock_ns=lambda: 123,
    )
    report = runner.run(allow_enable=True, operator_ack=True)
    assert report.verdict is PreflightVerdict.PASS_ENABLE_ALLOWED
    assert report.sections["enable_operator_ack"]["status"] == "PASS"


def test_preflight_missing_or_failing_probe_is_fail() -> None:
    probes = _probes()
    probes.pop("storage")
    report = M5PreflightRunner(station_id="S1", config_hash="abc", readonly_probes=probes).run()
    assert report.verdict is PreflightVerdict.FAIL
    assert "missing_probe:storage" in report.blockers

    probes = _probes()

    def boom():
        raise RuntimeError("disk")

    probes["storage"] = boom
    report = M5PreflightRunner(station_id="S1", config_hash="abc", readonly_probes=probes).run()
    assert report.verdict is PreflightVerdict.FAIL
    assert "probe_error:storage" in report.blockers


def test_preflight_report_json_roundtrip(tmp_path: Path) -> None:
    report = M5PreflightRunner(
        station_id="S1", config_hash="abc", readonly_probes=_probes(), clock_ns=lambda: 123
    ).run()
    path = tmp_path / "preflight.json"
    report.to_json(path)
    raw = json.loads(path.read_text())
    assert raw["verdict"] == "PASS_READONLY"
    assert raw["generated_at_ns"] == 123
    restored = M5PreflightReport.from_dict(raw)
    assert restored.as_dict() == report.as_dict()


def test_preflight_rejects_fail_unknown_missing_and_nonmapping() -> None:
    fails = [
        _probes("FAIL"),
        {name: (lambda: {"status": "UNKNOWN"}) for name in REQUIRED_SECTIONS},
        {name: (lambda: {"details": {}}) for name in REQUIRED_SECTIONS},
        {name: (lambda: "PASS") for name in REQUIRED_SECTIONS},
    ]
    for probes in fails:
        report = M5PreflightRunner(station_id="S1", config_hash="abc", readonly_probes=probes).run()
        assert report.verdict is PreflightVerdict.FAIL
        assert report.blockers


def test_preflight_enable_fails_closed_without_fixed_probes_or_ack() -> None:
    base = M5PreflightRunner(
        station_id="S1", config_hash="abc", readonly_probes=_probes(), enable_probes={}
    )
    assert base.run(allow_enable=True, operator_ack=True).verdict is PreflightVerdict.FAIL

    partial = _enable_probes()
    partial.pop("writer_lock")
    runner = M5PreflightRunner(
        station_id="S1", config_hash="abc", readonly_probes=_probes(), enable_probes=partial
    )
    assert runner.run(allow_enable=True, operator_ack=True).verdict is PreflightVerdict.FAIL

    runner = M5PreflightRunner(
        station_id="S1",
        config_hash="abc",
        readonly_probes=_probes(),
        enable_probes=_enable_probes(),
    )
    assert runner.run(allow_enable=True, operator_ack=False).verdict is PreflightVerdict.FAIL


def test_preflight_accepts_typed_probe_results() -> None:
    probes = {name: (lambda: ProbeResult(ProbeStatus.PASS, {"ok": True})) for name in REQUIRED_SECTIONS}
    report = M5PreflightRunner(station_id="S1", config_hash="abc", readonly_probes=probes).run()
    assert report.verdict is PreflightVerdict.PASS_READONLY
