"""M5 runtime/authority/stop skeleton tests (fake only)."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from vla_infra.contracts.pose import CartesianPose
from vla_infra.devices.flexiv_rdk import FlexivRdkNrtCommandSink, NrtLimits
from vla_infra.devices.gn01 import Gn01Config
from vla_infra.devices.m5_auth import M5AuthorizationIssuer
from vla_infra.devices.preflight import M5PreflightRunner, REQUIRED_ENABLE_PROBES, REQUIRED_SECTIONS
from vla_infra.devices.stop_control import run_stop_with_deadline
from vla_infra.runtime.m5 import M5Runtime, M5State


def _identity():
    return np.eye(4)


def _limits():
    return NrtLimits(0.02, 0.10, 0.05, 0.10)


def _pose():
    return CartesianPose(np.array([0.5, 0.0, 0.3]), np.array([0.0, 0.0, 0.0, 1.0]))


def _capture_error(errors, callable_):
    try:
        callable_()
    except Exception as exc:
        errors.append(exc)


class _PermitMode:
    NRT_CARTESIAN_MOTION_FORCE = "NRT_CARTESIAN_MOTION_FORCE"


class _PermitRobot:
    def __init__(self):
        self.calls = []
        self._operational = False
        self._mode = None
        self.fault = lambda: False
        self.recovery = lambda: False
        self.estop_released = lambda: True

    def Enable(self):
        self.calls.append("Enable")
        self._operational = True

    def operational(self):
        return self._operational

    def mode(self):
        return self._mode

    def SwitchMode(self, mode):
        self._mode = mode
        self.calls.append("SwitchMode")

    def SetForceControlAxis(self, axes):
        self.calls.append("SetForceControlAxis")

    def SendCartesianMotionForce(self, *args):
        self.calls.append("SendCartesianMotionForce")

    def Stop(self):
        self.calls.append("Stop")


class _PermitSdk:
    Mode = _PermitMode

    def __init__(self):
        self.robots = []

    def Robot(self, serial):
        robot = _PermitRobot()
        self.robots.append(robot)
        return robot


def _permit_auth(issuer):
    probes = {n: (lambda: {"status": "PASS"}) for n in REQUIRED_SECTIONS}
    enable = {n: (lambda: {"status": "PASS"}) for n in REQUIRED_ENABLE_PROBES}
    report = M5PreflightRunner(
        station_id="S1", config_hash="cfg", readonly_probes=probes,
        enable_probes=enable, clock_ns=lambda: 1000,
    ).run(allow_enable=True, operator_ack=True)
    return issuer.issue(report, operator_ack=True, now_ns=1000)


class FakeSession:
    def __init__(self):
        self.connect_calls = 0

    def connect(self):
        self.connect_calls += 1
        return None

    def close(self, timeout_s=2.0):
        return None


class FakeArm:
    def __init__(self):
        self.enabled = 0
        self.stops = 0
        self.stop_ok = True

    def enable_and_switch_nrt(self, auth, *, authority_epoch, now_ns):
        self.enabled += 1
        self.auth = auth
        self.epoch = authority_epoch

    def stop_motion(self, timeout_s=2.0):
        self.stops += 1
        return self.stop_ok


class FakeGn:
    def __init__(self):
        self.enabled = 0
        self.stops = 0
        self.stop_ok = True
        self.switches = 0
        self.verifications = 0

    def enable(self, auth, *, authority_epoch):
        self.enabled += 1
        self.auth = auth
        self.epoch = authority_epoch

    def stop(self, timeout_s=2.0):
        self.stops += 1
        return self.stop_ok

    def switch_tool(self):
        self.switches += 1

    def verify_tool(self, *, name, tcp_hash, payload_hash):
        self.verifications += 1


def _runner():
    probes = {n: (lambda: {"status": "PASS"}) for n in REQUIRED_SECTIONS}
    enable = {n: (lambda: {"status": "PASS"}) for n in REQUIRED_ENABLE_PROBES}
    return M5PreflightRunner(
        station_id="S1",
        config_hash="cfg",
        readonly_probes=probes,
        enable_probes=enable,
        clock_ns=lambda: 1000,
    )


class FakeConfig:
    class _Safety:
        stop_timeout_s = 2.0
    safety = _Safety()
    class _Station:
        tool_name = "G1"
        tool_tcp_hash = "a" * 64
        tool_payload_hash = "b" * 64
    station = _Station()

    def config_hash(self):
        return "cfg"

    def assert_real_motion_gate(self):
        return None


def _runtime():
    from vla_infra.execution.authority import ControlAuthority

    authority = ControlAuthority()
    issuer = M5AuthorizationIssuer(
        station_id="S1", config_hash="cfg", secret=b"runtime-secret-32-bytes-long!!!"
    )
    arm = FakeArm(); arm._authority = authority
    gn = FakeGn(); gn._authority = authority
    return M5Runtime(
        session=FakeSession(),
        arm=arm,
        gn01=gn,
        preflight_runner=_runner(),
        issuer=issuer,
        authority=authority,
        config=FakeConfig(),
        clock_ns=lambda: 1000,
    ), issuer


def test_m5_runtime_ack_enable_stop_lifecycle() -> None:
    runtime, issuer = _runtime()
    runtime.connect_readonly()
    assert runtime.state is M5State.CONNECTED_READONLY
    report = runtime.run_preflight(operator_ack=True)
    assert report.verdict.value == "PASS_ENABLE_ALLOWED"
    auth = runtime.issue_authorization(operator_ack=True, ack_text=report.report_hash())
    assert issuer.verify(auth, now_ns=1000) is True
    _heartbeat_all(runtime)
    epoch = runtime.enable()
    assert runtime.state is M5State.MODE_READY
    assert runtime.authority.running is False
    runtime.prepare_tool(name="G1", tcp_hash="a" * 64, payload_hash="b" * 64)
    assert runtime.state is M5State.ARMED
    assert runtime.authority.running is True
    assert runtime.arm.enabled == 1 and runtime.gn01.enabled == 1
    assert runtime.arm.epoch == epoch == runtime.gn01.epoch
    outcome = runtime.stop(timeout_s=0.5)
    assert outcome.confirmed is True
    assert runtime.state is M5State.STOPPED
    assert runtime.authority.valid is False


def test_m5_runtime_rejects_enable_without_ack() -> None:
    runtime, _ = _runtime()
    runtime.connect_readonly()
    with pytest.raises(Exception):
        runtime.run_preflight(operator_ack=False)
    assert runtime.state is M5State.PREFLIGHT_FAILED
    with pytest.raises(Exception):
        runtime.enable()


def test_bounded_stop_observes_blocked_sdk() -> None:
    release = threading.Event()

    def blocking_stop():
        release.wait(2.0)
        return True

    outcome = run_stop_with_deadline(blocking_stop, 0.05)
    assert outcome.timed_out is True
    assert outcome.confirmed is False
    assert outcome.thread_alive is True
    release.set()


def test_authority_permit_wraps_check_to_sdk_write() -> None:
    from vla_infra.execution.authority import ControlAuthority
    from vla_infra.errors import NotArmedError
    from vla_infra.devices.tool_state import ToolReadiness

    sdk = _PermitSdk()
    authority = ControlAuthority()
    epoch = authority.grant("test")
    authority.activate("test")
    issuer = M5AuthorizationIssuer(
        station_id="S1", config_hash="cfg", secret=b"permit-secret-32-bytes-long!!!"
    )
    auth = _permit_auth(issuer)
    tool = ToolReadiness()
    tool.mark_switched("G1")
    tool.verify(name="G1", tcp_hash="tcp", payload_hash="payload")
    sink = FlexivRdkNrtCommandSink(
        "R1",
        limits=_limits(),
        base_to_world=_identity(),
        sdk=sdk,
        authorization_verifier=lambda a: issuer.verify(a, now_ns=1000),
        tool_readiness=tool,
        authority=authority,
    )
    sink.connect(); sink.claim_writer("executor")
    sink.enable_and_switch_nrt(auth, authority_epoch=epoch, now_ns=5)
    sink.command_pose(_pose(), epoch=epoch, now_ns=10)
    writes = sdk.robots[0].calls
    authority.revoke("test_revoke")
    with pytest.raises(NotArmedError):
        sink.command_pose(_pose(), epoch=epoch, now_ns=20)
    assert sdk.robots[0].calls == writes


def test_runtime_production_rejects_mismatched_authority() -> None:

    runtime, _ = _runtime()
    with pytest.raises(Exception):
        M5Runtime(
            session=FakeSession(),
            arm=FakeArm(),
            gn01=FakeGn(),
            preflight_runner=_runner(),
            issuer=runtime.issuer,
            authority=runtime.authority,
            config=FakeConfig(),
        )


def test_enable_failure_is_transactional_and_stops_both_devices() -> None:
    runtime, _ = _runtime()
    runtime.connect_readonly()
    report = runtime.run_preflight(operator_ack=True)
    runtime.issue_authorization(operator_ack=True, ack_text=report.report_hash())
    runtime.gn01.enable = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("gn enable failed"))
    _heartbeat_all(runtime)
    with pytest.raises(Exception):
        runtime.enable()
    assert runtime.state in (M5State.FAULT, M5State.STOP_FAILED)
    assert runtime.authority.valid is False
    assert runtime.arm.stops == 1
    assert runtime.gn01.stops == 1


def test_parallel_stop_does_not_skip_gripper_when_arm_blocks() -> None:
    release = threading.Event()
    runtime, _ = _runtime()
    runtime.connect_readonly()
    runtime.arm.stop_motion = lambda timeout_s=2.0: release.wait(2.0)
    outcome = runtime.stop(timeout_s=0.05)
    assert outcome.confirmed is False
    assert outcome.timed_out is True
    assert runtime.gn01.stops == 1
    assert runtime.state is M5State.STOP_FAILED
    release.set()


def test_connect_failure_marks_fault() -> None:
    runtime, _ = _runtime()
    runtime.session.connect = lambda: (_ for _ in ()).throw(RuntimeError("connect failed"))
    with pytest.raises(RuntimeError):
        runtime.connect_readonly()
    assert runtime.state is M5State.FAULT


def test_stop_helper_missing_result_is_failure() -> None:
    outcome = run_stop_with_deadline(lambda: None, 0.1)
    assert outcome.confirmed is False


class _NoMotionConfig(FakeConfig):
    def assert_real_motion_gate(self):
        from vla_infra.errors import RealHardwareDisabledError

        raise RealHardwareDisabledError("allow_real_motion false")


def test_runtime_enable_requires_motion_gate_before_sdk() -> None:
    runtime, _ = _runtime()
    runtime._config = _NoMotionConfig()
    runtime.connect_readonly()
    report = runtime.run_preflight(operator_ack=True)
    runtime.issue_authorization(operator_ack=True, ack_text=report.report_hash())
    _heartbeat_all(runtime)
    with pytest.raises(Exception):
        runtime.enable()
    assert runtime.arm.enabled == 0
    assert runtime.state is M5State.AUTHORIZED


def test_fault_and_running_invalidate_stop_both_devices() -> None:
    runtime, _ = _runtime()
    runtime.connect_readonly()
    report = runtime.run_preflight(operator_ack=True)
    runtime.issue_authorization(operator_ack=True, ack_text=report.report_hash())
    _heartbeat_all(runtime)
    runtime.enable()
    runtime.fault("test")
    assert runtime.state in (M5State.FAULT, M5State.STOP_FAILED)
    assert runtime.arm.stops >= 1
    assert runtime.gn01.stops >= 1

    runtime2, _ = _runtime()
    runtime2.connect_readonly(); report2 = runtime2.run_preflight(operator_ack=True)
    runtime2.issue_authorization(operator_ack=True, ack_text=report2.report_hash()); _heartbeat_all(runtime2); runtime2.enable()
    runtime2.invalidate_authorization("reconnect")
    assert runtime2.arm.stops >= 1
    assert runtime2.gn01.stops >= 1
    assert runtime2.state is M5State.CONNECTED_READONLY


def _heartbeat_all(runtime, now_ns=1000):
    for source in ("policy", "executor", "state"):
        runtime.heartbeat(source, now_ns=now_ns)


def _authorized_runtime():
    runtime, _ = _runtime()
    runtime.connect_readonly()
    report = runtime.run_preflight(operator_ack=True)
    runtime.issue_authorization(operator_ack=True, ack_text=report.report_hash())
    _heartbeat_all(runtime)
    return runtime


def test_stop_failed_is_sticky_until_explicit_recovery() -> None:
    runtime = _authorized_runtime()
    runtime.arm.stop_ok = False
    runtime.gn01.stop_ok = False
    outcome = runtime.stop(timeout_s=0.2)
    assert outcome.confirmed is False
    assert runtime.state is M5State.STOP_FAILED

    with pytest.raises(Exception):
        runtime.invalidate_authorization("x")
    runtime.fault("x")
    assert runtime.state is M5State.STOP_FAILED
    with pytest.raises(Exception):
        runtime.run_preflight(operator_ack=True)
    with pytest.raises(Exception):
        runtime.connect_readonly()
    assert runtime.state is M5State.STOP_FAILED

    with pytest.raises(Exception):
        runtime.recover_stop_failed(operator_ack=True, e_stop_released=False, physical_stop_confirmed=True)
    runtime.recover_stop_failed(operator_ack=True, e_stop_released=True, physical_stop_confirmed=True)
    assert runtime.state is M5State.NEW


def test_enable_and_stop_are_serialized_by_lifecycle_mutex() -> None:
    runtime = _authorized_runtime()
    entered = threading.Event()
    release = threading.Event()

    def blocking_enable(auth, *, authority_epoch, now_ns):
        entered.set()
        release.wait(2.0)
        runtime.arm.enabled += 1

    runtime.arm.enable_and_switch_nrt = blocking_enable
    enable_thread = threading.Thread(target=runtime.enable)
    enable_thread.start()
    assert entered.wait(2.0)

    def do_stop():
        runtime.stop(timeout_s=0.2)

    stop_thread = threading.Thread(target=do_stop)
    stop_thread.start()
    time.sleep(0.05)
    release.set()
    enable_thread.join(2.0)
    stop_thread.join(2.0)
    assert runtime.state in (M5State.STOPPED, M5State.STOP_FAILED)


def test_watchdog_staleness_and_authorization_expiry_stop_devices() -> None:
    runtime = _authorized_runtime()
    runtime.enable()
    later = 1000 + 1_500_000_000
    runtime._clock_ns = lambda: later
    runtime.watchdog(now_ns=later)
    assert runtime.state in (M5State.FAULT, M5State.STOP_FAILED)
    assert runtime.arm.stops >= 1
    assert runtime.gn01.stops >= 1

    runtime2 = _authorized_runtime()
    runtime2.enable()
    runtime2._clock_ns = lambda: 61_000_000_000
    for source in ("policy", "executor", "state"):
        runtime2.heartbeat(source, now_ns=61_000_000_000)
    runtime2.watchdog(now_ns=61_000_000_000)
    assert runtime2.state in (M5State.FAULT, M5State.STOP_FAILED)


def test_compose_uses_expected_shared_tool_readiness() -> None:
    class ComposeArm(FakeArm):
        pass

    class ComposeGn(FakeGn):
        pass

    class Composite:
        def __init__(self, arm, gn):
            self._arm = arm
            self._gripper = gn
            self.connected = False

        def connect(self):
            self.connected = True

    class ComposeSession:
        def __init__(self):
            self.connected = False
            self.claimed = 0
            self.tool = None

        def claim_runtime(self):
            self.claimed += 1
            return object()

        def release_runtime(self, token=None):
            pass

        def connect(self):
            self.connected = True
            return None

        def create_composite(self, *, limits, base_to_world, gn01_name, gn01_config,
                             authorization_verifier, authority, tool_readiness, cleanup_timeout_s, owner_token):
            self.tool = tool_readiness
            arm = ComposeArm(); arm._authority = authority; arm._tool_readiness = tool_readiness
            gn = ComposeGn(); gn._authority = authority; gn._tool_readiness = tool_readiness
            return Composite(arm, gn)

        def close(self):
            pass

        def quarantine(self, reason):
            return None

    class Station:
        nrt_limits = _limits()
        base_to_world = _identity()
        gn01_name = "G1"
        gn01 = None
        tool_name = "G1"
        tool_tcp_hash = "a" * 64
        tool_payload_hash = "b" * 64

    class Config:
        station = Station()

        def config_hash(self):
            return "cfg"

        def validate_tool_profile(self):
            return None

        def assert_real_motion_gate(self):
            return None

    class Safety:
        stop_timeout_s = 2.0

    config = Config(); config.safety = Safety()
    session = ComposeSession()
    issuer = M5AuthorizationIssuer(station_id="S1", config_hash="cfg", secret=b"compose-secret-32-bytes-long!!!!")
    runtime = M5Runtime.compose(
        session=session,
        config=config,
        preflight_runner=_runner(),
        issuer=issuer,
    )
    assert runtime.state is M5State.CONNECTED_READONLY
    assert runtime.arm._authority is runtime.authority
    assert runtime.arm._tool_readiness is runtime.gn01._tool_readiness
    tool = runtime.arm._tool_readiness
    assert tool.expected_name == "G1"
    assert tool.expected_tcp_hash == "a" * 64
    assert tool.expected_payload_hash == "b" * 64


def test_watchdog_stops_without_external_poll_call() -> None:
    now = {"t": 1000}
    runtime, _ = _runtime()
    runtime._clock_ns = lambda: now["t"]
    runtime.connect_readonly()
    report = runtime.run_preflight(operator_ack=True)
    runtime.issue_authorization(operator_ack=True, ack_text=report.report_hash())
    _heartbeat_all(runtime)
    runtime.enable()
    now["t"] = 1000 + 1_500_000_000
    deadline = time.time() + 1.0
    while time.time() < deadline and runtime.state not in (M5State.FAULT, M5State.STOP_FAILED):
        time.sleep(0.01)
    assert runtime.state in (M5State.FAULT, M5State.STOP_FAILED)
    assert runtime.arm.stops >= 1
    runtime.close(timeout_s=0.2)


def test_enable_rejects_zero_or_missing_heartbeat() -> None:
    runtime, _ = _runtime()
    runtime.connect_readonly()
    report = runtime.run_preflight(operator_ack=True)
    runtime.issue_authorization(operator_ack=True, ack_text=report.report_hash())
    with pytest.raises(Exception):
        runtime.enable()
    assert runtime.arm.enabled == 0
    assert runtime.gn01.enabled == 0

    runtime2, _ = _runtime()
    runtime2.connect_readonly()
    report2 = runtime2.run_preflight(operator_ack=True)
    runtime2.issue_authorization(operator_ack=True, ack_text=report2.report_hash())
    runtime2.heartbeat("policy", now_ns=1000)
    runtime2.heartbeat("state", now_ns=1000)
    with pytest.raises(Exception):
        runtime2.enable()
    assert runtime2.arm.enabled == 0


def test_future_or_regressed_heartbeat_rejected() -> None:
    runtime, _ = _runtime()
    runtime.heartbeat("policy", now_ns=1000)
    with pytest.raises(Exception):
        runtime.heartbeat("policy", now_ns=1000)
    with pytest.raises(Exception):
        runtime.heartbeat("policy", now_ns=2000)
    with pytest.raises(Exception):
        runtime.heartbeat("unknown", now_ns=1001)


def test_watchdog_terminal_exception_revokes_and_stops() -> None:
    runtime = _authorized_runtime()
    runtime.enable()
    runtime.watchdog = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    deadline = time.time() + 2.0
    while time.time() < deadline and runtime.state not in (M5State.FAULT, M5State.STOP_FAILED):
        time.sleep(0.01)
    assert runtime.state in (M5State.FAULT, M5State.STOP_FAILED)
    assert runtime.arm.stops >= 1
    assert runtime.gn01.stops >= 1
    assert runtime._watchdog_healthy is False


def test_abort_stop_failure_quarantines_session() -> None:
    from vla_infra.devices.flexiv_rdk import NrtLimits
    from vla_infra.devices.real_session import FlexivRdkSession

    sdk = _PermitSdk()
    session = FlexivRdkSession("R1", sdk=sdk)
    session.connect()
    token = session.claim_runtime()
    composite = session.create_composite(
        limits=NrtLimits(0.02, 0.10, 0.05, 0.10),
        base_to_world=np.eye(4),
        gn01_name="G1",
        gn01_config=Gn01Config(0.0, 0.1, 0.01, 2.0),
        authorization_verifier=lambda a: False,
        owner_token=token,
    )
    sink = composite._arm
    sink._robot.Stop = lambda: (_ for _ in ()).throw(RuntimeError("stop blocked"))
    with pytest.raises(Exception):
        session.abort_owned_resources()
    assert session.quarantined is True
    with pytest.raises(Exception):
        session.claim_runtime()


def test_compose_failure_releases_claim() -> None:
    class Session:
        def __init__(self):
            self.owner = None

        def claim_runtime(self):
            if self.owner is not None:
                raise RuntimeError("already claimed")
            self.owner = object()
            return self.owner

        def release_runtime(self, token):
            assert token is self.owner
            self.owner = None

        def connect(self):
            return None

        def create_composite(self, **kwargs):
            raise RuntimeError("create failed")

        def abort_owned_resources(self):
            return None

        def quarantine(self, reason):
            return None

    class Station:
        nrt_limits = _limits()
        base_to_world = _identity()
        gn01_name = "G1"
        gn01 = None
        tool_name = "G1"
        tool_tcp_hash = "a" * 64
        tool_payload_hash = "b" * 64

    class Config:
        station = Station()

        def config_hash(self):
            return "cfg"

        def validate_tool_profile(self):
            return None

        def assert_real_motion_gate(self):
            return None

    class Safety:
        stop_timeout_s = 2.0

    config = Config(); config.safety = Safety()
    session = Session()
    issuer = M5AuthorizationIssuer(station_id="S1", config_hash="cfg", secret=b"rollback-secret-32-bytes-long!!!")
    with pytest.raises(RuntimeError):
        M5Runtime.compose(
            session=session, config=config, preflight_runner=_runner(), issuer=issuer
        )
    assert session.owner is None


def test_close_failure_is_not_silent_and_owner_remains() -> None:
    runtime = _authorized_runtime()
    runtime.enable()
    runtime.session.close_fail = True

    def fail_close(timeout_s=2.0):
        raise RuntimeError("session close failed")

    runtime.session.close = fail_close
    with pytest.raises(Exception):
        runtime.close(timeout_s=0.2)
    assert runtime.closed is False
    with pytest.raises(Exception):
        runtime.close(timeout_s=0.2)
    assert runtime.closed is False


def test_terminal_failure_does_not_overwrite_stopped() -> None:
    runtime = _authorized_runtime()
    runtime.enable()
    runtime.stop(timeout_s=0.2)
    assert runtime.state is M5State.STOPPED
    runtime._watchdog_terminal_fail("late_terminal")
    assert runtime.state in (M5State.STOPPED, M5State.STOP_FAILED)


def test_tool_identity_mismatch_never_arms_runtime() -> None:
    runtime = _authorized_runtime()
    runtime.enable()
    assert runtime.state is M5State.MODE_READY
    assert runtime.authority.running is False
    with pytest.raises(Exception):
        runtime.prepare_tool(
            name="G1", tcp_hash="c" * 64, payload_hash="b" * 64
        )
    assert runtime.state in (M5State.FAULT, M5State.STOP_FAILED)
    assert runtime.gn01.switches == 0
    assert runtime.authority.running is False


def test_concurrent_watchdog_start_keeps_single_thread() -> None:
    runtime, _ = _runtime()
    runtime.connect_readonly()
    first = runtime._watchdog_thread
    barrier = threading.Barrier(9)
    observed = []

    def start_again():
        barrier.wait()
        runtime.start_watchdog(interval_s=0.01)
        observed.append(runtime._watchdog_thread)

    threads = [threading.Thread(target=start_again) for _ in range(8)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(1.0)
    assert all(thread is first for thread in observed)
    runtime.close(timeout_s=0.2)


def test_terminal_revoke_during_enable_blocks_gn_enable() -> None:
    runtime = _authorized_runtime()
    entered = threading.Event()
    release = threading.Event()

    def blocking_arm(auth, *, authority_epoch, now_ns):
        entered.set()
        release.wait(1.0)
        runtime.arm.enabled += 1

    runtime.arm.enable_and_switch_nrt = blocking_arm
    enable_errors = []
    enable_thread = threading.Thread(
        target=lambda: _capture_error(enable_errors, runtime.enable)
    )
    enable_thread.start()
    assert entered.wait(1.0)
    terminal_thread = threading.Thread(
        target=runtime._watchdog_terminal_fail, args=("test_terminal",)
    )
    terminal_thread.start()
    deadline = time.time() + 1.0
    while runtime._watchdog_healthy and time.time() < deadline:
        time.sleep(0.001)
    assert runtime._watchdog_healthy is False
    release.set()
    enable_thread.join(2.0)
    terminal_thread.join(2.0)
    assert enable_errors
    assert runtime.gn01.enabled == 0
    assert runtime.authority.valid is False
    assert runtime.state in (M5State.FAULT, M5State.STOP_FAILED)


def test_quarantined_session_blocks_recovery_and_enable() -> None:
    runtime = _authorized_runtime()
    runtime.session.quarantined = True
    runtime.session.quarantine_reason = "stop_failed"
    with pytest.raises(Exception):
        runtime.recover_stop_failed(
            operator_ack=True, e_stop_released=True, physical_stop_confirmed=True
        )
    with pytest.raises(Exception):
        runtime.run_preflight(operator_ack=True)
    with pytest.raises(Exception):
        runtime.issue_authorization(operator_ack=True, ack_text="x")
    runtime._state = M5State.AUTHORIZED
    with pytest.raises(Exception):
        runtime.enable()
    assert runtime.arm.enabled == 0
    assert runtime.gn01.enabled == 0
    assert runtime.state is M5State.QUARANTINED
