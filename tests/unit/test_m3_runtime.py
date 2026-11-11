"""M3 fake execution: 100 Hz thread, watchdogs, logs and fault injection."""

from __future__ import annotations

import json
import threading

import pytest
import time
from pathlib import Path

import numpy as np
from vla_infra.config import load_app_config
from vla_infra.runtime.m3 import M3Config, M3Runtime

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def _app():
    return load_app_config(
        station_path=CONFIGS / "station.example.yaml",
        safety_path=CONFIGS / "safety.example.yaml",
        policy_path=CONFIGS / "policy.example.yaml",
    )


def _runtime(tmp_path: Path, **config):
    values = dict(control_period_ns=10_000_000, policy_period_ns=50_000_000, state_period_ns=5_000_000)
    values.update(config)
    return M3Runtime(
        _app(),
        output=tmp_path / "execution.jsonl",
        config=M3Config(**values),
    )


def _wait_until(predicate, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not reached")


def test_m3_100hz_metrics_and_audit_log(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    result = runtime.run(duration_s=0.7)
    assert result.armed is True
    assert result.ticks >= 40
    assert result.writes > 0
    assert result.policy_chunks > 0
    assert result.log_records == result.ticks
    assert result.log_dropped == 0
    assert result.metrics["tick_period_ns"]["p50_ns"] is not None
    assert 8_000_000 <= result.metrics["tick_period_ns"]["p50_ns"] <= 15_000_000
    assert result.metrics["runtime_ns"]["p95_ns"] < 20_000_000
    assert result.metrics["overruns"] <= max(1, result.ticks // 10)
    assert any(t["to"] == "RUNNING" for t in result.supervisor["transitions"])
    lines = (tmp_path / "execution.jsonl").read_text().splitlines()
    assert len(lines) == result.ticks
    first = json.loads(lines[0])
    assert {"raw_state", "filtered_target", "commanded_target", "measured_state"}.issubset(first)
    assert first["schema_version"] == "m3-execution/v1"


def test_m3_policy_watchdog_hold(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.start()
    _wait_until(lambda: runtime._result().writes > 0)
    runtime.set_policy_enabled(False)
    time.sleep(1.2)
    result = runtime.stop()
    assert result.final_state in ("HOLD", "FAULT")
    reasons = [t["reason"] for t in result.supervisor["transitions"]]
    assert any("watchdog_expired:policy" in reason or "policy_disconnected" in reason for reason in reasons)


def test_m3_stale_state_hold(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.start()
    _wait_until(lambda: runtime._result().writes > 0)
    runtime.set_state_enabled(False)
    time.sleep(0.45)
    result = runtime.stop()
    assert result.final_state in ("HOLD", "FAULT")
    reasons = [t["reason"] for t in result.supervisor["transitions"]]
    assert any("executor_stale_state" in reason or "watchdog_expired:robot" in reason for reason in reasons)


def test_m3_invalid_workspace_chunk_zero_extra_writes(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.start()
    _wait_until(lambda: runtime._result().writes > 0)
    runtime.set_policy_enabled(False)
    bad = np.zeros((1, 7), dtype=np.float32)
    bad[0, 0] = 10.0  # far outside the workspace
    assert runtime.submit_actions(bad) is True
    _wait_until(lambda: runtime.supervisor.state.value in ("HOLD", "FAULT"), 2.0)
    before = runtime.executor.stats.writes
    time.sleep(0.2)
    after = runtime.executor.stats.writes
    result = runtime.stop()
    assert after == before
    assert result.executor["plans_rejected"] >= 1
    assert result.final_state in ("HOLD", "FAULT")


def test_m3_writer_error_faults_and_stops_writes(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.start()
    _wait_until(lambda: runtime._result().writes > 0)

    def boom(*args, **kwargs):
        raise RuntimeError("injected command sink failure")

    runtime._robot.command_pose = boom
    _wait_until(lambda: runtime.supervisor.state.value == "FAULT", 2.0)
    before = runtime.executor.stats.writes
    time.sleep(0.2)
    result = runtime.stop()
    assert runtime.executor.stats.writes == before
    assert result.executor["write_failures"] >= 1
    assert result.final_state == "FAULT"
    records = [json.loads(line) for line in (tmp_path / "execution.jsonl").read_text().splitlines()]
    failed = [record for record in records if record.get("kind") == "write_failed"]
    assert failed
    assert failed[0]["commanded_target"] is not None
    assert failed[0]["commanded_status"] == "attempted"
    assert failed[0]["pose_attempted"] is True
    assert failed[0]["pose_ok"] is False


def test_m3_old_chunk_rejected(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.start()
    _wait_until(lambda: runtime._result().writes > 0)
    runtime.set_policy_enabled(False)
    stale = np.zeros((1, 7), dtype=np.float32)
    stale[0, 0] = 0.001
    assert runtime.submit_actions(stale, capture_offset_ns=-2_000_000_000) is True
    _wait_until(lambda: runtime.supervisor.state.value in ("HOLD", "FAULT"), 2.0)
    result = runtime.stop()
    assert result.executor["plans_rejected"] >= 1
    assert result.final_state in ("HOLD", "FAULT")


def test_m3_latest_only_inbox_replaces_chunks(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.start()
    _wait_until(lambda: runtime._result().writes > 0)
    runtime.set_policy_enabled(False)
    for value in (0.001, 0.002, 0.003):
        actions = np.zeros((1, 7), dtype=np.float32)
        actions[0, 0] = value
        assert runtime.submit_actions(actions) is True
    stats = runtime._inbox.stats
    assert stats.replaced >= 2
    runtime.stop()


def test_m3_camera_watchdog_hold(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.start()
    _wait_until(lambda: runtime._result().writes > 0)
    runtime.set_camera_enabled(False)
    time.sleep(0.7)
    result = runtime.stop()
    assert result.final_state in ("HOLD", "FAULT")
    reasons = [t["reason"] for t in result.supervisor["transitions"]]
    assert any("watchdog_expired:camera" in reason for reason in reasons)


def test_m3_cli_runs_fake_profile(tmp_path: Path, capsys) -> None:
    from vla_infra.cli import m3

    code = m3.main([
        "--duration", "0.3",
        "--output", str(tmp_path / "cli.jsonl"),
        "--json",
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["armed"] is True
    assert payload["ticks"] > 0
    assert payload["log_records"] == payload["ticks"]


def test_m3_hold_rearm_resumes_writes(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.start()
    _wait_until(lambda: runtime._result().writes > 0)
    runtime.set_policy_enabled(False)
    _wait_until(lambda: runtime.supervisor.state.value in ("HOLD", "FAULT"), 2.0)
    assert runtime.supervisor.state.value == "HOLD"
    runtime.set_policy_enabled(True)
    time.sleep(0.1)
    runtime.rearm()
    _wait_until(lambda: runtime.executor.stats.writes > 0, 2.0)
    result = runtime.stop()
    assert any(t["to"] == "ARMED" for t in result.supervisor["transitions"])


def test_m3_four_stage_action_audit_correlates_and_differs(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.run(duration_s=0.5)
    lines = [json.loads(line) for line in (tmp_path / "execution.jsonl").read_text().splitlines()]
    successful = [
        record
        for record in lines
        if record.get("commanded_status") == "succeeded"
        and record.get("raw_action")
        and record.get("filtered_target")
        and record.get("commanded_target")
    ]
    assert successful
    record = successful[0]
    raw = record["raw_action"]
    assert raw["request_sequence_id"] >= 1
    assert raw["epoch"] == record["epoch"]
    assert raw["received_time_ns"] is not None
    assert raw["action_row"] is not None
    assert any(abs(value) > 0 for value in raw["action_row"])
    # The raw action is a delta; filtered is the segment endpoint; commanded is
    # the interpolated target actually attempted this tick.
    filtered_position = np.asarray(record["filtered_target"]["position"])
    commanded_position = np.asarray(record["commanded_target"]["position"])
    assert not np.allclose(filtered_position, commanded_position)
    assert record["commanded_status"] == "succeeded"
    assert record["measured_state"]["sequence_id"] > record["raw_state"]["sequence_id"]


def _camera_hold(runtime, camera_name: str) -> tuple:
    _wait_until(lambda: runtime._result().writes > 0)
    _wait_until(lambda: runtime.supervisor.state.value in ("HOLD", "FAULT"), 2.0)
    before = runtime.executor.stats.writes
    time.sleep(0.2)
    after = runtime.executor.stats.writes
    result = runtime.stop()
    reasons = [t["reason"] for t in result.supervisor["transitions"]]
    assert any(f"watchdog_expired:camera:{camera_name}" in reason for reason in reasons)
    assert after == before
    return result, reasons


def test_m3_camera_continuous_none_hold(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.start()
    _wait_until(lambda: runtime._result().writes > 0)
    runtime._cameras["base"].inject_failures(10_000)
    result, _ = _camera_hold(runtime, "base")
    assert result.final_state in ("HOLD", "FAULT")


def test_m3_camera_exception_hold(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.start()
    _wait_until(lambda: runtime._result().writes > 0)

    def boom():
        raise RuntimeError("injected camera exception")

    runtime._cameras["base"].read = boom
    result, _ = _camera_hold(runtime, "base")
    assert result.final_state in ("HOLD", "FAULT")


def test_m3_duplicate_old_camera_frame_is_not_heartbeat(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.start()
    _wait_until(lambda: runtime._result().writes > 0)
    from vla_infra.contracts.constants import IMAGE_KEY_BASE_RGB
    from vla_infra.testing import make_image

    old_frame = make_image(IMAGE_KEY_BASE_RGB, frame_number=0, host_time_ns=1)
    runtime._cameras["base"].read = lambda: old_frame
    result, _ = _camera_hold(runtime, "base")
    assert result.final_state in ("HOLD", "FAULT")


def test_m3_stop_timeout_on_blocked_callback_is_explicit(tmp_path: Path) -> None:
    import threading

    entered = threading.Event()
    release = threading.Event()

    def block_tick(report, info):
        entered.set()
        assert release.wait(2.0)

    runtime = M3Runtime(
        _app(),
        output=tmp_path / "blocked.jsonl",
        config=M3Config(
            control_period_ns=10_000_000,
            policy_period_ns=50_000_000,
            state_period_ns=5_000_000,
            control_thread_join_s=0.05,
        ),
        on_tick=block_tick,
    )
    runtime.start()
    assert entered.wait(2.0)
    result = runtime.stop()
    assert result.stop_error.startswith("m3_stop_timeout")
    assert "m3-callback" in result.stop_error
    # The blocked thread still owns resources; release it and clean up.
    release.set()
    result = runtime.stop()
    assert runtime.supervisor.state.value in ("HOLD", "FAULT")


def test_m3_blocked_command_sink_stop_does_not_claim_early(tmp_path: Path) -> None:
    import threading

    runtime = _runtime(
        tmp_path,
        control_thread_join_s=0.05,
    )
    runtime.start()
    _wait_until(lambda: runtime._result().writes > 0)
    entered = threading.Event()
    release = threading.Event()
    original = runtime._robot.command_pose

    def blocked(pose, *, epoch, now_ns):
        entered.set()
        assert release.wait(2.0)
        return original(pose, epoch=epoch, now_ns=now_ns)

    runtime._robot.command_pose = blocked
    assert entered.wait(2.0)
    started = time.monotonic()
    result = runtime.stop()
    assert time.monotonic() - started < 1.0
    assert result.stop_error.startswith("m3_stop_timeout")
    assert result.lifecycle == "STOP_FAILED"
    # A live blocked sink must not be reported as a clean stop or release.
    release.set()
    time.sleep(0.1)
    cleaned = runtime.stop()
    assert cleaned.stop_error == ""
    assert cleaned.lifecycle == "STOPPED"


def test_m3_permanent_blocked_sink_stop_subprocess_returns(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    code = """
import time
from vla_infra.config import load_app_config
from vla_infra.runtime.m3 import M3Config, M3Runtime
app = load_app_config(
    station_path="configs/station.example.yaml",
    safety_path="configs/safety.example.yaml",
    policy_path="configs/policy.example.yaml",
)
runtime = M3Runtime(app, output=r"%s", config=M3Config(control_thread_join_s=0.1))
runtime.start()
time.sleep(0.2)
def block(*args, **kwargs):
    time.sleep(3600)
runtime._robot.command_pose = block
time.sleep(0.1)
print("CALL_STOP", flush=True)
start = time.monotonic()
result = runtime.stop()
print("STOP_RETURNED", result.stop_error, result.lifecycle, round(time.monotonic() - start, 3), flush=True)
""" % (tmp_path / "blocked.jsonl")
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    started = time.monotonic()
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    elapsed = time.monotonic() - started
    assert completed.returncode == 0, completed.stderr
    assert "CALL_STOP" in completed.stdout
    assert "STOP_RETURNED" in completed.stdout
    assert "m3_stop_timeout" in completed.stdout
    assert elapsed < 3.0


def _assert_rearm_rejected_while_source_stale(tmp_path: Path, source: str) -> None:
    runtime = _runtime(tmp_path)
    runtime.start()
    _wait_until(lambda: runtime._result().writes > 0)
    if source == "camera":
        runtime.set_camera_enabled(False)
    elif source == "policy":
        runtime.set_policy_enabled(False)
    else:
        runtime.set_state_enabled(False)
    _wait_until(lambda: runtime.supervisor.state.value in ("HOLD", "FAULT"), 2.0)
    before = runtime.executor.stats.writes
    with pytest.raises(Exception) as exc:
        runtime.rearm()
    assert "fresh" in str(exc.value).lower() or "not allowed" in str(exc.value).lower()
    time.sleep(0.2)
    assert runtime.executor.stats.writes == before
    runtime.stop()


def test_m3_rearm_rejected_when_camera_still_stale(tmp_path: Path) -> None:
    _assert_rearm_rejected_while_source_stale(tmp_path, "camera")


def test_m3_rearm_rejected_when_policy_still_disabled(tmp_path: Path) -> None:
    _assert_rearm_rejected_while_source_stale(tmp_path, "policy")


def test_m3_rearm_rejected_when_state_still_disabled(tmp_path: Path) -> None:
    _assert_rearm_rejected_while_source_stale(tmp_path, "state")


def test_m3_audit_raw_state_is_exact_executor_sample_for_all_writes(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.run(duration_s=0.5)
    successful = []
    for line in (tmp_path / "execution.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record.get("commanded_status") in ("succeeded", "partial", "attempted"):
            successful.append(record)
    assert successful
    for record in successful:
        raw = record["raw_state"]
        measured = record["measured_state"]
        assert raw is not None and measured is not None
        assert raw["sequence_id"] < measured["sequence_id"]
        assert raw["host_time_ns"] <= record["start_time_ns"]
        assert measured["host_time_ns"] >= record["start_time_ns"]


def _startup_refused(tmp_path: Path, monkeypatch, patch, *, timeout=0.15):
    from vla_infra.runtime.m3 import M3StartupError

    patch(monkeypatch)
    runtime = _runtime(tmp_path, startup_timeout_s=timeout)
    with pytest.raises(M3StartupError) as exc:
        runtime.start()
    assert "missing" in exc.value.details
    assert runtime._robot.write_count == 0
    assert runtime._lifecycle == "START_FAILED"
    assert all(not thread.is_alive() for thread in runtime._threads)
    return runtime, exc.value


def test_m3_startup_refused_when_all_cameras_return_none(tmp_path: Path, monkeypatch) -> None:
    _, exc = _startup_refused(
        tmp_path,
        monkeypatch,
        lambda mp: mp.setattr("vla_infra.runtime.m3.FakeCamera.read", lambda self: None),
    )
    assert any(item.startswith("camera:") for item in exc.details["missing"])


def test_m3_startup_refused_when_camera_raises(tmp_path: Path, monkeypatch) -> None:
    def boom(self):
        raise RuntimeError("camera startup failure")

    _, exc = _startup_refused(
        tmp_path,
        monkeypatch,
        lambda mp: mp.setattr("vla_infra.runtime.m3.FakeCamera.read", boom),
    )
    assert any(item.startswith("camera:") for item in exc.details["missing"])


def test_m3_startup_refused_without_state(tmp_path: Path, monkeypatch) -> None:
    def boom(self):
        raise RuntimeError("no state")

    _, exc = _startup_refused(
        tmp_path,
        monkeypatch,
        lambda mp: mp.setattr("vla_infra.runtime.m3.FakeRobotArm.read_state", boom),
    )
    assert "robot" in exc.details["missing"]


def test_m3_startup_refused_without_policy_response(tmp_path: Path, monkeypatch) -> None:
    def boom(self, state):
        raise RuntimeError("policy startup failure")

    _, exc = _startup_refused(
        tmp_path,
        monkeypatch,
        lambda mp: mp.setattr("vla_infra.runtime.m3.M3Runtime._make_chunk", boom),
    )
    assert "policy" in exc.details["missing"]


def test_m3_startup_refused_without_executor_tick(tmp_path: Path, monkeypatch) -> None:
    def no_tick(self):
        return

    _, exc = _startup_refused(
        tmp_path,
        monkeypatch,
        lambda mp: mp.setattr("vla_infra.runtime.m3.M3Runtime._executor_loop", no_tick),
    )
    assert "executor" in exc.details["missing"]


class _BlockingHandle:
    def __init__(self, entered, release):
        self._entered = entered
        self._release = release
        self._closed = False

    def write(self, text):
        self._entered.set()
        assert self._release.wait(2.0)
        return len(text)

    def flush(self):
        return None

    def fileno(self):
        return -1

    def close(self):
        self._closed = True


def test_execution_log_close_reports_live_writer(tmp_path: Path) -> None:
    from vla_infra.runtime.execution_log import ExecutionLog

    entered = threading.Event()
    release = threading.Event()
    log = ExecutionLog(tmp_path / "blocked-log.jsonl")
    log.start()
    log._handle = _BlockingHandle(entered, release)
    assert log.append({"tick": 0})
    assert entered.wait(2.0)
    assert log.close(timeout_s=0.05) is False
    assert log.thread_alive() is True
    release.set()
    assert log.close(timeout_s=1.0) is True
    assert log.closed is True


def test_m3_stop_reports_logger_stall_not_stopped(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, control_thread_join_s=0.2)
    runtime.start()
    _wait_until(lambda: runtime._result().writes > 0)
    entered = threading.Event()
    release = threading.Event()
    runtime._log._handle = _BlockingHandle(entered, release)
    # Force at least one queued audit record to enter the blocked writer.
    runtime._append_log(0, runtime._clock.now_ns(), runtime._clock.now_ns(), None, None, None, False, "test")
    assert entered.wait(2.0)
    result = runtime.stop()
    assert result.lifecycle == "STOP_FAILED"
    assert "m3-execution-log" in result.stop_error
    # No false STOPPED while the logger thread is alive.
    assert runtime._log.thread_alive() is True
    release.set()
    cleaned = runtime.stop()
    assert cleaned.lifecycle == "STOPPED"
    assert cleaned.stop_error == ""


def test_m3_partial_gripper_failure_jsonl_has_commanded_target(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.start()
    _wait_until(lambda: runtime._result().writes > 0)
    runtime.set_policy_enabled(False)
    _wait_until(lambda: runtime.executor.active_plan is None, 2.0)

    def boom(*args, **kwargs):
        raise RuntimeError("injected gripper failure")

    runtime._robot.command_gripper = boom
    actions = np.zeros((1, 7), dtype=np.float32)
    actions[0, 6] = 0.9
    assert runtime.submit_actions(actions) is True
    _wait_until(lambda: runtime.supervisor.state.value in ("HOLD", "FAULT"), 2.0)
    runtime.stop()
    records = [json.loads(line) for line in (tmp_path / "execution.jsonl").read_text().splitlines()]
    partial = [record for record in records if record.get("kind") == "write_partial"]
    assert partial
    record = partial[0]
    assert record["commanded_target"] is not None
    assert record["commanded_status"] == "partial"
    assert record["pose_attempted"] is True and record["pose_ok"] is True
    assert record["gripper_attempted"] is True and record["gripper_ok"] is False


def test_m3_startup_cleanup_blocked_callback_subprocess(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    code = """
import time
from unittest.mock import patch
from vla_infra.config import load_app_config
from vla_infra.runtime.m3 import M3Config, M3Runtime, M3StartupError
app = load_app_config(
    station_path="configs/station.example.yaml",
    safety_path="configs/safety.example.yaml",
    policy_path="configs/policy.example.yaml",
)
def block(report, info):
    time.sleep(3600)
runtime = M3Runtime(
    app,
    output=r"%s",
    config=M3Config(startup_timeout_s=0.1, startup_cleanup_timeout_s=0.1),
    on_tick=block,
)
try:
    with patch("vla_infra.runtime.m3.FakeCamera.read", return_value=None):
        runtime.start()
except M3StartupError:
    print("START_RETURNED", runtime._lifecycle, runtime._robot.write_count, flush=True)
print("THREADS", [(t.name, t.is_alive()) for t in runtime._threads], flush=True)
""" % (tmp_path / "startup-callback.jsonl")
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    assert completed.returncode == 0, completed.stderr
    assert "START_RETURNED START_CLEANUP_FAILED 0" in completed.stdout
    assert "m3-callback" in completed.stdout


def test_m3_startup_cleanup_blocked_logger_subprocess(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    code = """
import time
from unittest.mock import patch
from vla_infra.config import load_app_config
from vla_infra.runtime.execution_log import ExecutionLog
from vla_infra.runtime.m3 import M3Config, M3Runtime, M3StartupError
class BlockingHandle:
    def write(self, text):
        time.sleep(3600)
    def flush(self):
        pass
    def fileno(self):
        return -1
    def close(self):
        pass
original_start = ExecutionLog.start
def start_and_block(self):
    original_start(self)
    self._handle = BlockingHandle()
ExecutionLog.start = start_and_block
app = load_app_config(
    station_path="configs/station.example.yaml",
    safety_path="configs/safety.example.yaml",
    policy_path="configs/policy.example.yaml",
)
runtime = M3Runtime(
    app,
    output=r"%s",
    config=M3Config(startup_timeout_s=0.1, startup_cleanup_timeout_s=0.1),
)
try:
    with patch("vla_infra.runtime.m3.FakeCamera.read", return_value=None):
        runtime.start()
except M3StartupError:
    print("START_RETURNED", runtime._lifecycle, runtime._robot.write_count, flush=True)
print("LOG_ALIVE", runtime._log.thread_alive(), flush=True)
""" % (tmp_path / "startup-log.jsonl")
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    assert completed.returncode == 0, completed.stderr
    assert "START_RETURNED START_CLEANUP_FAILED 0" in completed.stdout
    assert "LOG_ALIVE True" in completed.stdout


def test_m3_slow_callback_does_not_block_executor_and_counts_drops(tmp_path: Path) -> None:
    import time as _time

    def slow_callback(report, info):
        _time.sleep(0.15)

    runtime = M3Runtime(
        _app(),
        output=tmp_path / "slow-callback.jsonl",
        config=M3Config(control_period_ns=10_000_000, startup_timeout_s=1.0),
        on_tick=slow_callback,
    )
    result = runtime.run(duration_s=0.7)
    callback = result.metrics["callback"]
    assert callback["enqueued"] > 0
    assert callback["dropped"] > 0
    assert callback["processed"] >= 1
    assert callback["processed"] <= callback["enqueued"]
    assert callback["submitted"] == callback["enqueued"] + callback["dropped"]
    assert result.ticks >= 50
    assert result.metrics["tick_period_ns"]["p95_ns"] < 20_000_000


def test_m3_callback_errors_are_counted(tmp_path: Path) -> None:
    def failing(report, info):
        raise RuntimeError("telemetry callback failure")

    runtime = M3Runtime(
        _app(),
        output=tmp_path / "callback-error.jsonl",
        config=M3Config(startup_timeout_s=1.0),
        on_tick=failing,
    )
    result = runtime.run(duration_s=0.4)
    callback = result.metrics["callback"]
    assert callback["errors"] >= 1
    assert callback["last_error_type"] == "RuntimeError"
    assert callback["last_error_tick"] is not None
