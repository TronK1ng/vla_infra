"""Strict configuration loading and fail-closed startup."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from vla_infra.cli import doctor, record, rollout
from vla_infra.config import (
    load_app_config,
    load_policy,
    load_safety,
    load_station,
)
from vla_infra.errors import ConfigError, ErrorCode, UnsupportedBackendError

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
STATION = CONFIGS / "station.example.yaml"
SAFETY = CONFIGS / "safety.example.yaml"
POLICY = CONFIGS / "policy.example.yaml"


def _mapping(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write(tmp_path: Path, name: str, value: dict) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_example_configs_load() -> None:
    app = load_app_config(station_path=STATION, safety_path=SAFETY, policy_path=POLICY)
    assert app.station.backend == "fake"
    assert set(app.station.cameras) == {"base", "wrist"}
    assert app.policy.state_layout.dim == 22
    assert app.policy.expected_delta_semantics == "cumulative"
    assert app.safety.max_force_n == 30.0
    assert app.station.sync.max_sync_error_ns == 50_000_000


def test_missing_file_fails_closed() -> None:
    with pytest.raises(ConfigError):
        load_station(CONFIGS / "does-not-exist.yaml")


def test_missing_required_key_fails_closed(tmp_path: Path) -> None:
    raw = _mapping(STATION)
    del raw["robot"]["serial"]
    with pytest.raises(ConfigError) as exc:
        load_station(_write(tmp_path, "station.yaml", raw))
    assert "serial" in exc.value.details["missing"]


def test_unknown_key_fails_closed(tmp_path: Path) -> None:
    raw = _mapping(STATION)
    raw["robot"]["serial_number_typo"] = "x"
    with pytest.raises(ConfigError) as exc:
        load_station(_write(tmp_path, "station.yaml", raw))
    assert "serial_number_typo" in exc.value.details["unknown"]


def test_real_backend_refused_in_m1(tmp_path: Path) -> None:
    raw = _mapping(STATION)
    raw["backend"] = "real"
    with pytest.raises(UnsupportedBackendError) as exc:
        load_station(_write(tmp_path, "station.yaml", raw))
    assert exc.value.code is ErrorCode.UNSUPPORTED_BACKEND


def test_duplicate_camera_serials_refused(tmp_path: Path) -> None:
    raw = _mapping(STATION)
    raw["cameras"]["wrist"]["serial"] = raw["cameras"]["base"]["serial"]
    with pytest.raises(ConfigError):
        load_station(_write(tmp_path, "station.yaml", raw))


def test_camera_binding_is_exactly_base_and_wrist(tmp_path: Path) -> None:
    raw = _mapping(STATION)
    raw["cameras"].pop("wrist")
    with pytest.raises(ConfigError):
        load_station(_write(tmp_path, "station.yaml", raw))


def test_invalid_safety_workspace_refused(tmp_path: Path) -> None:
    raw = _mapping(SAFETY)
    raw["workspace"]["min"][0] = 1.0
    with pytest.raises(ConfigError):
        load_safety(_write(tmp_path, "safety.yaml", raw))


def test_missing_safety_threshold_refused(tmp_path: Path) -> None:
    raw = _mapping(SAFETY)
    del raw["motion"]["max_linear_velocity_m_s"]
    with pytest.raises(ConfigError) as exc:
        load_safety(_write(tmp_path, "safety.yaml", raw))
    assert "max_linear_velocity_m_s" in exc.value.details["missing"]


def test_unknown_safety_section_refused(tmp_path: Path) -> None:
    raw = _mapping(SAFETY)
    raw["extra_safety"] = {"force": 1.0}
    with pytest.raises(ConfigError):
        load_safety(_write(tmp_path, "safety.yaml", raw))


def test_policy_endpoint_scheme(tmp_path: Path) -> None:
    raw = _mapping(POLICY)
    raw["endpoint"] = "http://127.0.0.1:8765"
    with pytest.raises(ConfigError):
        load_policy(_write(tmp_path, "policy.yaml", raw))


def test_unknown_state_layout_refused(tmp_path: Path) -> None:
    raw = _mapping(POLICY)
    raw["state_layout"] = "mystery_layout"
    with pytest.raises(ConfigError):
        load_policy(_write(tmp_path, "policy.yaml", raw))


def test_unknown_adapter_refused(tmp_path: Path) -> None:
    raw = _mapping(POLICY)
    raw["server"]["adapter"] = "force_policy"
    with pytest.raises(ConfigError):
        load_policy(_write(tmp_path, "policy.yaml", raw))


def test_expected_metadata_matches_example() -> None:
    policy = load_policy(POLICY)
    metadata = policy.expected_metadata()
    assert metadata.horizon == policy.expected_horizon
    assert metadata.state_layout.dim == 22


def test_doctor_cli_reports_and_does_not_contact_hardware(capsys) -> None:
    code = doctor.main(["--station", str(STATION), "--safety", str(SAFETY), "--policy", str(POLICY), "--json"])
    assert code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True
    assert output["hardware_access"].startswith("none")
    assert output["station"]["backend"] == "fake"


def test_doctor_cli_fails_on_bad_config(tmp_path: Path, capsys) -> None:
    raw = _mapping(POLICY)
    raw["endpoint"] = "http://bad"
    bad = _write(tmp_path, "policy.yaml", raw)
    code = doctor.main(["--station", str(STATION), "--safety", str(SAFETY), "--policy", str(bad)])
    assert code == 1
    assert "CONFIG ERROR" in capsys.readouterr().out


def test_record_replay_and_rollout_cli(tmp_path: Path, capsys) -> None:
    lerobot = tmp_path / "lerobot"
    code = record.main([
        "--duration", "0.2", "--output", str(tmp_path), "--min-free-gb", "0",
        "--lerobot-out", str(lerobot), "--json",
    ])
    assert code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["recorder"]["committed"] is True
    episode = Path(output["episode_path"])
    assert (episode / "metadata.json").exists()
    assert (lerobot / "meta" / "info.json").exists()

    from vla_infra.cli import replay

    code = replay.main([str(episode), "--json"])
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["issues"] == []
    assert summary["lowdim_rows"] > 0

    assert rollout.main([]) == 3
    assert "NOT IMPLEMENTED" in capsys.readouterr().err


def test_policy_proxy_setting_loaded_and_validated(tmp_path: Path) -> None:
    raw = _mapping(POLICY)
    assert raw.get("proxy") == "none"
    raw["proxy"] = "env"
    loaded = load_policy(_write(tmp_path, "policy.yaml", raw))
    assert loaded.proxy == "env"
    raw["proxy"] = "http://proxy.local:3128"
    assert load_policy(_write(tmp_path, "policy.yaml", raw)).proxy == "http://proxy.local:3128"
    raw["proxy"] = "bogus-proxy"
    with pytest.raises(ConfigError):
        load_policy(_write(tmp_path, "policy.yaml", raw))


def test_absolute_action_requires_absolute_delta_semantics(tmp_path: Path) -> None:
    raw = _mapping(POLICY)
    raw["expected_action_type"] = "absolute_tcp"
    raw["expected_delta_semantics"] = "cumulative"
    with pytest.raises(ConfigError):
        load_policy(_write(tmp_path, "policy.yaml", raw))


def test_real_readonly_requires_gate_ack_and_real_serials(tmp_path: Path, monkeypatch) -> None:
    from vla_infra.errors import RealHardwareDisabledError

    raw = _mapping(STATION)
    raw["backend"] = "real_readonly"
    raw["on_site_readonly_ack"] = True
    raw["cameras"]["base"]["serial"] = "BASE-REAL-SN"
    raw["cameras"]["wrist"]["serial"] = "WRIST-REAL-SN"
    raw["robot"]["serial"] = "ROBOT-REAL-SN"
    path = _write(tmp_path, "station.yaml", raw)

    monkeypatch.delenv("VLA_INFRA_ALLOW_REAL_HARDWARE", raising=False)
    with pytest.raises(RealHardwareDisabledError):
        load_station(path)

    monkeypatch.setenv("VLA_INFRA_ALLOW_REAL_HARDWARE", "1")
    config = load_station(path)
    assert config.backend == "real_readonly"
    assert config.calibration_version == "uncalibrated"

    raw["on_site_readonly_ack"] = False
    with pytest.raises(RealHardwareDisabledError):
        load_station(_write(tmp_path, "station.yaml", raw))

    raw["on_site_readonly_ack"] = True
    raw["cameras"]["base"]["serial"] = "REPLACE_ME_BASE"
    with pytest.raises(ConfigError):
        load_station(_write(tmp_path, "station.yaml", raw))


def test_real_writer_backend_still_refused(tmp_path: Path) -> None:
    raw = _mapping(STATION)
    raw["backend"] = "real"
    with pytest.raises(Exception):
        load_station(_write(tmp_path, "station.yaml", raw))


def test_record_cli_reports_still_stopping_as_incomplete(monkeypatch, capsys) -> None:
    from vla_infra.recording.recorder import RecorderResult
    from vla_infra.runtime.collection import CollectionResult

    class SlowHarness:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def run(self, **kwargs):
            return CollectionResult(
                recorder=RecorderResult(
                    committed=False,
                    path=None,
                    lowdim_rows=10,
                    camera_frames=0,
                    observations=0,
                    action_records=0,
                    dropped_lowdim=0,
                    dropped_observations=0,
                    dropped_actions=0,
                    error="finalize timeout",
                    still_stopping=True,
                    state="committing",
                ),
                sync_report={},
                observations=0,
                robot_samples=0,
                episode_path=None,
                error="finalize timeout",
            )

    monkeypatch.setattr("vla_infra.cli.record.CollectionHarness", SlowHarness)
    code = record.main([
        "--duration", "0.1", "--min-free-gb", "0",
        "--finalize-timeout", "1", "--json",
    ])
    assert code == 3
    assert "FINALIZE TIMEOUT" in capsys.readouterr().out


def test_doctor_preflight_disk_gate(monkeypatch, capsys, tmp_path: Path) -> None:
    from types import SimpleNamespace

    low = SimpleNamespace(total=10, used=10, free=1)
    high = SimpleNamespace(total=10**12, used=1, free=10**12)
    monkeypatch.setattr("vla_infra.cli.doctor.shutil.disk_usage", lambda _path: low)
    code = doctor.main(["--station", str(STATION), "--safety", str(SAFETY), "--policy", str(POLICY), "--preflight", "--json"])
    assert code == 1
    assert "preflight_problems" in json.loads(capsys.readouterr().out)
    monkeypatch.setattr("vla_infra.cli.doctor.shutil.disk_usage", lambda _path: high)
    assert doctor.main(["--station", str(STATION), "--safety", str(SAFETY), "--policy", str(POLICY), "--preflight", "--json"]) == 0
