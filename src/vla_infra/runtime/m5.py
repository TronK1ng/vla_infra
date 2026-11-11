"""M5 runtime skeleton (Round 02R2; injected devices only).

No real hardware is authorized by this module.  It is the single composition
root for authority/capability/tool readiness and enforces the config/env/
config-hash/ACK gate before any Enable.
"""

from __future__ import annotations

import threading
import time
from enum import Enum
from typing import Any, Dict, Optional

from ..devices.m5_auth import M5AuthorizationIssuer, MotionAuthorization
from ..devices.preflight import M5PreflightReport, M5PreflightRunner, PreflightVerdict
from ..devices.stop_control import StopOutcome, stop_pair_bounded
from ..devices.tool_state import ToolReadiness
from ..errors import DeviceError, ErrorCode, IllegalTransitionError, RealHardwareDisabledError
from ..execution.authority import ControlAuthority


class M5State(str, Enum):
    NEW = "NEW"
    CONNECTED_READONLY = "CONNECTED_READONLY"
    PREFLIGHT_OK = "PREFLIGHT_OK"
    PREFLIGHT_FAILED = "PREFLIGHT_FAILED"
    WAITING_OPERATOR_ACK = "WAITING_OPERATOR_ACK"
    AUTHORIZED = "AUTHORIZED"
    ENABLED = "ENABLED"
    MODE_READY = "MODE_READY"
    ARMED = "ARMED"
    RUNNING = "RUNNING"
    HOLD = "HOLD"
    FAULT = "FAULT"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    STOP_FAILED = "STOP_FAILED"
    QUARANTINED = "QUARANTINED"


_ACTIVE_STATES = {M5State.ENABLED, M5State.MODE_READY, M5State.ARMED, M5State.RUNNING, M5State.HOLD}
_REQUIRED_HEARTBEATS = frozenset({"policy", "executor", "state"})


def _serialized(method):
    def wrapper(self, *args, **kwargs):
        with self._op_lock:
            return method(self, *args, **kwargs)
    return wrapper


class M5Runtime:
    def __init__(
        self,
        *,
        session: Any,
        arm: Any,
        gn01: Any,
        preflight_runner: M5PreflightRunner,
        issuer: M5AuthorizationIssuer,
        authority: Optional[ControlAuthority] = None,
        config: Any = None,
        composite: Any = None,
        clock_ns=None,
    ) -> None:
        self.session = session
        self.arm = arm
        self.gn01 = gn01
        self.composite = composite
        self.preflight_runner = preflight_runner
        self.issuer = issuer
        self.authority = authority or ControlAuthority()
        for name, sink in (("arm", arm), ("gn01", gn01)):
            if getattr(sink, "_authority", None) is not self.authority:
                raise DeviceError(
                    f"{name} sink is not bound to the runtime ControlAuthority",
                    code=ErrorCode.CONFIG_INVALID,
                    details={"sink": type(sink).__name__},
                )
        self._config = config
        self._config_hash = config.config_hash() if config is not None else ""
        self._stop_timeout_s = float(config.safety.stop_timeout_s) if config is not None else 2.0
        self._clock_ns = clock_ns or time.monotonic_ns
        self._state = M5State.NEW
        self._report: Optional[M5PreflightReport] = None
        self._authorization: Optional[MotionAuthorization] = None
        self._authority_epoch: Optional[int] = None
        self._last_stop: Dict[str, StopOutcome] = {}
        self._lock = threading.RLock()
        self._op_lock = threading.Lock()
        self._heartbeats: Dict[str, int] = {}
        self._watchdog_max_age_ns = 1_000_000_000
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: Optional[threading.Thread] = None
        self._watchdog_healthy = False
        self._watchdog_lifecycle_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closing = False
        self._closed = False
        self._normal_watchdog_shutdown = False
        self._watchdog_terminal_reason = ""

    @classmethod
    def compose(
        cls,
        *,
        session: Any,
        config: Any,
        preflight_runner: M5PreflightRunner,
        issuer: M5AuthorizationIssuer,
        authority: Optional[ControlAuthority] = None,
        clock_ns=None,
    ) -> "M5Runtime":
        config.validate_tool_profile()
        claim_token = session.claim_runtime()
        authority = authority or ControlAuthority()
        tool = ToolReadiness(
            expected_name=config.station.tool_name,
            expected_tcp_hash=config.station.tool_tcp_hash,
            expected_payload_hash=config.station.tool_payload_hash,
        )
        try:
            session.connect()
            composite = session.create_composite(
                limits=config.station.nrt_limits,
                base_to_world=config.station.base_to_world,
                gn01_name=config.station.gn01_name,
                gn01_config=config.station.gn01,
                authorization_verifier=lambda a: issuer.verify(a),
                authority=authority,
                tool_readiness=tool,
                cleanup_timeout_s=float(config.safety.stop_timeout_s),
                owner_token=claim_token,
            )
            runtime = cls(
                session=session,
                arm=composite._arm,
                gn01=composite._gripper,
                composite=composite,
                preflight_runner=preflight_runner,
                issuer=issuer,
                authority=authority,
                config=config,
                clock_ns=clock_ns,
            )
            runtime._claim_token = claim_token
            runtime.connect_readonly()
            return runtime
        except Exception as original:
            runtime_obj = locals().get("runtime")
            try:
                if runtime_obj is not None:
                    runtime_obj._abort_composition()
                else:
                    session.abort_owned_resources()
            except Exception as cleanup:
                session.quarantine(f"compose_rollback_failed:{type(cleanup).__name__}")
                raise DeviceError(
                    "compose failed and rollback could not confirm safe cleanup; session quarantined",
                    code=ErrorCode.STOP_FAILED,
                    details={"original": type(original).__name__, "cleanup": type(cleanup).__name__},
                ) from original
            if getattr(session, "quarantined", False) or getattr(session, "has_pending_resources", False):
                raise DeviceError(
                    "compose rollback is quarantined; owner retained",
                    code=ErrorCode.STOP_FAILED,
                    details={"quarantine_reason": getattr(session, "quarantine_reason", "")},
                ) from original
            session.release_runtime(claim_token)
            raise

    @property
    def state(self) -> M5State:
        with self._lock:
            return self._state

    @property
    def report(self) -> Optional[M5PreflightReport]:
        with self._lock:
            return self._report

    @property
    def last_stop(self) -> Dict[str, StopOutcome]:
        with self._lock:
            return dict(self._last_stop)

    def _session_quarantined(self) -> bool:
        try:
            return bool(getattr(self.session, "quarantined", False))
        except Exception:
            return True

    def _sync_session_gate(self) -> None:
        if self._session_quarantined():
            self.authority.revoke("session_quarantined")
            with self._lock:
                self._authorization = None
                self._report = None
                self._state = M5State.QUARANTINED

    def _assert_open(self) -> None:
        self._sync_session_gate()
        if self._closed:
            raise DeviceError("runtime is closed", code=ErrorCode.ILLEGAL_TRANSITION)
        if self._session_quarantined():
            raise DeviceError(
                "session is quarantined; process restart/new session required",
                code=ErrorCode.STOP_FAILED,
            )

    def _abort_composition(self) -> None:
        self._watchdog_stop.set()
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(self._stop_timeout_s)
            if self._watchdog_thread.is_alive():
                raise DeviceError("watchdog did not exit during rollback", code=ErrorCode.STOP_FAILED)
            self._watchdog_thread = None
        self.authority.revoke("compose_rollback")
        self.session.abort_owned_resources(self._stop_timeout_s)
        self._set_state(M5State.STOPPED)

    def _set_state(self, state: M5State) -> None:
        with self._lock:
            if self._state is M5State.QUARANTINED and state is not M5State.QUARANTINED:
                raise DeviceError("QUARANTINED is terminal", code=ErrorCode.STOP_FAILED)
            if state is M5State.QUARANTINED:
                self._state = state
                return
            if self._state is M5State.STOP_FAILED and state is not M5State.STOP_FAILED:
                raise DeviceError(
                    "STOP_FAILED is sticky; use recover_stop_failed()",
                    code=ErrorCode.ILLEGAL_TRANSITION,
                )
            if self._state is M5State.STOPPING and state not in (
                M5State.STOPPING, M5State.STOPPED, M5State.STOP_FAILED, M5State.FAULT
            ):
                raise DeviceError(
                    "STOPPING is sticky until stop completes",
                    code=ErrorCode.ILLEGAL_TRANSITION,
                )
            self._state = state

    def _transition(self, expected: tuple[M5State, ...], new: M5State) -> None:
        with self._lock:
            if self._state not in expected:
                raise IllegalTransitionError(
                    f"illegal M5 transition {self._state.value} -> {new.value}",
                    code=ErrorCode.ILLEGAL_TRANSITION,
                )
            self._state = new

    @_serialized
    def connect_readonly(self) -> None:
        self._assert_open()
        with self._lock:
            if self._state is M5State.CONNECTED_READONLY:
                return
            if self._state is not M5State.NEW:
                raise IllegalTransitionError("connect is only valid from NEW", code=ErrorCode.ILLEGAL_TRANSITION)
        try:
            returned = self.session.connect()
            if returned is not None:
                raise DeviceError("session.connect must not return a raw Robot", code=ErrorCode.DEVICE_FAULT)
            if self.composite is not None:
                self.composite.connect()
        except Exception:
            self._set_state(M5State.FAULT)
            raise
        self.start_watchdog()
        self._set_state(M5State.CONNECTED_READONLY)

    def start_watchdog(self, *, interval_s: float = 0.05) -> None:
        self._assert_open()
        with self._watchdog_lifecycle_lock:
            if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
                return
            self._start_watchdog_locked(interval_s)

    def _start_watchdog_locked(self, interval_s: float) -> None:
        if isinstance(interval_s, bool) or not isinstance(interval_s, (int, float)) or interval_s <= 0:
            raise DeviceError("watchdog interval must be positive", code=ErrorCode.VALUE_OUT_OF_RANGE)
        self._watchdog_stop.clear()

        def loop() -> None:
            terminal_reason = "watchdog_unexpected_exit"
            try:
                while not self._watchdog_stop.wait(float(interval_s)):
                    self.watchdog(now_ns=self._clock_ns())
                terminal_reason = ""
            except BaseException as exc:  # noqa: BLE001
                terminal_reason = f"watchdog_exception:{type(exc).__name__}"
            finally:
                if terminal_reason:
                    try:
                        self._watchdog_terminal_fail(terminal_reason)
                    except BaseException:
                        self.authority.revoke("watchdog_terminal_failure")

        thread = threading.Thread(target=loop, name="m5-watchdog", daemon=True)
        self._watchdog_thread = thread
        thread.start()
        if not thread.is_alive():
            raise DeviceError("watchdog thread failed to start", code=ErrorCode.DEVICE_FAULT)
        self._watchdog_healthy = True

    def _watchdog_terminal_fail(self, reason: str) -> None:
        with self._lock:
            self._watchdog_healthy = False
            self._watchdog_terminal_reason = reason
            normal_shutdown = self._normal_watchdog_shutdown
        self.authority.revoke(reason)
        if normal_shutdown:
            return
        with self._op_lock:
            with self._lock:
                current = self._state
                closed = self._closed
            if closed or current in (
                M5State.STOPPED,
                M5State.STOP_FAILED,
                M5State.FAULT,
                M5State.NEW,
                M5State.CONNECTED_READONLY,
            ):
                return
            outcomes = self._stop_devices_parallel(self._stop_timeout_s)
            self._set_state(M5State.FAULT if all(o.confirmed for o in outcomes.values()) else M5State.STOP_FAILED)

    def close(self, *, timeout_s: Optional[float] = None) -> None:
        timeout = self._stop_timeout_s if timeout_s is None else float(timeout_s)
        with self._close_lock:
            if self._closed:
                return
            if self._closing:
                raise DeviceError("close already in progress", code=ErrorCode.ILLEGAL_TRANSITION)
            self._closing = True
        try:
            self.authority.revoke("m5_close")
            outcome = self.stop(timeout_s=timeout)
            if not outcome.confirmed:
                raise DeviceError(
                    "cannot close runtime: physical stop not confirmed",
                    code=ErrorCode.STOP_FAILED,
                )
            with self._watchdog_lifecycle_lock:
                self._normal_watchdog_shutdown = True
                self._watchdog_stop.set()
                if self._watchdog_thread is not None:
                    self._watchdog_thread.join(timeout)
                    if self._watchdog_thread.is_alive():
                        raise DeviceError("watchdog thread did not exit", code=ErrorCode.STOP_FAILED)
                    self._watchdog_thread = None
                self._watchdog_healthy = False
            self.session.close(timeout_s=timeout)
            token = getattr(self, "_claim_token", None)
            if token is not None:
                self.session.release_runtime(token)
                self._claim_token = None
            with self._close_lock:
                self._closed = True
        finally:
            with self._close_lock:
                self._closing = False

    @property
    def closed(self) -> bool:
        return self._closed

    @_serialized
    def invalidate_authorization(self, reason: str) -> None:
        self._assert_open()
        with self._lock:
            if self._state in (M5State.STOP_FAILED, M5State.STOPPING):
                raise DeviceError(
                    "cannot invalidate while STOPPING/STOP_FAILED",
                    code=ErrorCode.ILLEGAL_TRANSITION,
                )
            active = self._state in _ACTIVE_STATES
        if active:
            self._set_state(M5State.STOPPING)
            self.authority.revoke(reason)
            outcomes = self._stop_devices_parallel(self._stop_timeout_s)
            if not all(o.confirmed for o in outcomes.values()):
                self._set_state(M5State.STOP_FAILED)
                return
            self._set_state(M5State.STOPPED)
        else:
            self.authority.revoke(reason)
        with self._lock:
            self._authorization = None
            self._report = None
        self._set_state(M5State.CONNECTED_READONLY)

    @_serialized
    def run_preflight(self, *, operator_ack: bool = False) -> M5PreflightReport:
        self._assert_open()
        if self._state not in (M5State.NEW, M5State.CONNECTED_READONLY):
            raise IllegalTransitionError("preflight is only valid before Enable", code=ErrorCode.ILLEGAL_TRANSITION)
        report = self.preflight_runner.run(allow_enable=True, operator_ack=operator_ack)
        with self._lock:
            self._report = report
            self._state = M5State.PREFLIGHT_OK if report.verdict is PreflightVerdict.PASS_ENABLE_ALLOWED else M5State.PREFLIGHT_FAILED
        if report.verdict is not PreflightVerdict.PASS_ENABLE_ALLOWED:
            raise DeviceError("preflight did not authorize Enable", code=ErrorCode.NOT_READY)
        return report

    @_serialized
    def issue_authorization(self, *, operator_ack: bool, ack_text: str) -> MotionAuthorization:
        self._assert_open()
        with self._lock:
            if self._state is not M5State.PREFLIGHT_OK or self._report is None:
                raise IllegalTransitionError("authorization requires PREFLIGHT_OK", code=ErrorCode.ILLEGAL_TRANSITION)
            report = self._report
            self._state = M5State.WAITING_OPERATOR_ACK
        if not isinstance(ack_text, str) or ack_text.strip() not in (
            report.report_hash(),
            report.config_hash,
        ):
            self._set_state(M5State.PREFLIGHT_OK)
            raise DeviceError("operator ACK must echo the report or config hash", code=ErrorCode.AUTH_FAILED)
        try:
            auth = self.issuer.issue(report, operator_ack=operator_ack, now_ns=self._clock_ns())
        except Exception:
            self._set_state(M5State.PREFLIGHT_OK)
            raise
        with self._lock:
            self._authorization = auth
            self._state = M5State.AUTHORIZED
        return auth

    def _check_motion_gate(self) -> None:
        self._assert_open()
        if self._config is None:
            raise RealHardwareDisabledError("M5Runtime has no immutable M5Config")
        self._config.assert_real_motion_gate()
        if self._config_hash != self._config.config_hash():
            raise RealHardwareDisabledError("M5 config hash changed after composition")
        with self._lock:
            report = self._report
            auth = self._authorization
        if report is None or auth is None:
            raise RealHardwareDisabledError("M5 preflight/authorization is missing")
        if report.config_hash != self._config_hash or auth.config_hash != self._config_hash:
            raise RealHardwareDisabledError("M5 report/authorization config hash mismatch")

    def _heartbeat_fault_locked(self, now_ns: int) -> Optional[str]:
        for source in _REQUIRED_HEARTBEATS:
            last = self._heartbeats.get(source)
            if last is None:
                return f"heartbeat_missing:{source}"
            age = int(now_ns) - last
            if age < 0:
                return f"heartbeat_clock_regressed:{source}"
            if age > self._watchdog_max_age_ns:
                return f"heartbeat_stale:{source}"
        return None

    @_serialized
    def enable(self) -> int:
        self._assert_open()
        with self._lock:
            if self._state is not M5State.AUTHORIZED or self._authorization is None:
                raise IllegalTransitionError("Enable requires an ACK-bound authorization", code=ErrorCode.ILLEGAL_TRANSITION)
            auth = self._authorization
        self._check_motion_gate()
        if (
            self._watchdog_thread is None
            or not self._watchdog_thread.is_alive()
            or not self._watchdog_healthy
        ):
            raise DeviceError("watchdog is not running/healthy; Enable refused", code=ErrorCode.NOT_READY)
        with self._lock:
            heartbeat_fault = self._heartbeat_fault_locked(self._clock_ns())
        if heartbeat_fault is not None:
            raise DeviceError(heartbeat_fault, code=ErrorCode.NOT_READY)
        epoch = self.authority.grant("m5_enable")
        self._authority_epoch = epoch
        stage = "arm_enable"
        try:
            with self.authority.write_permit(epoch, require_running=False):
                self.arm.enable_and_switch_nrt(
                    auth, authority_epoch=epoch, now_ns=self._clock_ns()
                )
            if not self._watchdog_healthy or not self.authority.is_current(epoch):
                raise DeviceError(
                    "watchdog/authority was revoked during arm Enable",
                    code=ErrorCode.EPOCH_MISMATCH,
                )
            stage = "gn01_enable"
            with self.authority.write_permit(epoch, require_running=False):
                self.gn01.enable(auth, authority_epoch=epoch)
        except Exception as exc:
            self.authority.revoke(f"m5_enable_failed:{stage}")
            self._set_state(M5State.STOPPING)
            outcomes = self._stop_devices_parallel(self._stop_timeout_s)
            with self._lock:
                self._authorization = None
                self._report = None
                self._state = M5State.FAULT if all(o.confirmed for o in outcomes.values()) else M5State.STOP_FAILED
            details = {
                "stage": stage,
                "stop": {name: outcome.as_dict() for name, outcome in outcomes.items()},
            }
            if isinstance(exc, DeviceError):
                raise DeviceError(exc.message, code=exc.code, details={**exc.details, **details}) from exc
            raise DeviceError(
                "M5 enable failed; authority revoked and Stop attempted",
                code=ErrorCode.DEVICE_FAULT,
                details={"reason": type(exc).__name__, **details},
            ) from exc
        # Physical interfaces are enabled, but execution remains inactive until
        # Tool.Switch and the observed tool identity are explicitly verified.
        self._set_state(M5State.MODE_READY)
        return epoch

    @_serialized
    def prepare_tool(
        self,
        *,
        name: str,
        tcp_hash: str,
        payload_hash: str,
    ) -> None:
        """Switch and verify the tool before activating command execution."""
        self._assert_open()
        with self._lock:
            if self._state is not M5State.MODE_READY:
                raise IllegalTransitionError(
                    "tool preparation requires MODE_READY",
                    code=ErrorCode.ILLEGAL_TRANSITION,
                )
        expected = self._config.station
        if (
            name != expected.tool_name
            or tcp_hash != expected.tool_tcp_hash
            or payload_hash != expected.tool_payload_hash
        ):
            self.authority.revoke("tool_identity_mismatch")
            self._set_state(M5State.STOPPING)
            outcomes = self._stop_devices_parallel(self._stop_timeout_s)
            self._set_state(
                M5State.FAULT
                if all(o.confirmed for o in outcomes.values())
                else M5State.STOP_FAILED
            )
            raise DeviceError(
                "observed tool identity does not match the approved profile",
                code=ErrorCode.CONFIG_INVALID,
            )
        try:
            epoch = self._authority_epoch
            if epoch is None:
                raise DeviceError("missing control epoch", code=ErrorCode.NOT_ARMED)
            with self.authority.write_permit(epoch, require_running=False):
                self.gn01.switch_tool()
                self.gn01.verify_tool(
                    name=name,
                    tcp_hash=tcp_hash,
                    payload_hash=payload_hash,
                )
            self.authority.activate("m5_tool_verified")
        except Exception:
            self.authority.revoke("tool_prepare_failed")
            self._set_state(M5State.STOPPING)
            outcomes = self._stop_devices_parallel(self._stop_timeout_s)
            self._set_state(
                M5State.FAULT
                if all(o.confirmed for o in outcomes.values())
                else M5State.STOP_FAILED
            )
            raise
        self._set_state(M5State.ARMED)

    def _stop_devices_parallel(self, timeout_s: float) -> Dict[str, StopOutcome]:
        outcomes = stop_pair_bounded(
            lambda: self.arm.stop_motion(timeout_s),
            lambda: self.gn01.stop(timeout_s),
            timeout_s,
        )
        with self._lock:
            self._last_stop = dict(outcomes)
        return outcomes

    @_serialized
    def stop(self, *, timeout_s: Optional[float] = None) -> StopOutcome:
        self._assert_open()
        timeout = self._stop_timeout_s if timeout_s is None else float(timeout_s)
        with self._lock:
            if self._state in (M5State.STOPPED, M5State.STOP_FAILED):
                return StopOutcome(confirmed=self._state is M5State.STOPPED, timed_out=False, thread_alive=False)
        self.authority.revoke("m5_stop")
        self._set_state(M5State.STOPPING)
        outcomes = self._stop_devices_parallel(timeout)
        confirmed = len(outcomes) == 2 and all(o.confirmed for o in outcomes.values())
        self._set_state(M5State.STOPPED if confirmed else M5State.STOP_FAILED)
        return StopOutcome(
            confirmed=confirmed,
            timed_out=any(o.timed_out for o in outcomes.values()),
            thread_alive=any(o.thread_alive for o in outcomes.values()),
            error="" if confirmed else "stop_failed",
        )

    @_serialized
    def heartbeat(self, source: str, *, now_ns: Optional[int] = None) -> None:
        self._assert_open()
        if source not in ("policy", "executor", "state"):
            raise DeviceError("unknown heartbeat source", code=ErrorCode.VALUE_OUT_OF_RANGE)
        now = int(self._clock_ns() if now_ns is None else now_ns)
        with self._lock:
            last = self._heartbeats.get(source)
            if last is not None and now <= last:
                raise DeviceError("heartbeat timestamp did not advance", code=ErrorCode.BUFFER_NOT_MONOTONIC)
            if now > int(self._clock_ns()):
                raise DeviceError("heartbeat timestamp is in the future", code=ErrorCode.STALE_OBSERVATION)
            self._heartbeats[source] = now

    def policy_disconnected(self, reason: str) -> None:
        self.fault(f"policy_disconnected:{reason}")

    def watchdog(self, *, now_ns: int) -> None:
        self._assert_open()
        with self._lock:
            state = self._state
            auth = self._authorization
            heartbeat_fault = self._heartbeat_fault_locked(int(now_ns))
        if state not in _ACTIVE_STATES:
            return
        if auth is None or not self.issuer.verify(auth, now_ns=now_ns):
            self.fault("authorization_expired")
            return
        if heartbeat_fault is not None:
            self.fault(heartbeat_fault)

    @_serialized
    def recover_stop_failed(
        self, *, operator_ack: bool, e_stop_released: bool, physical_stop_confirmed: bool
    ) -> None:
        if self._session_quarantined():
            raise DeviceError(
                "quarantined session cannot recover; restart process/new session",
                code=ErrorCode.STOP_FAILED,
            )
        if self._state is not M5State.STOP_FAILED:
            raise IllegalTransitionError("not in STOP_FAILED", code=ErrorCode.ILLEGAL_TRANSITION)
        if not (operator_ack and e_stop_released and physical_stop_confirmed):
            raise DeviceError(
                "STOP_FAILED recovery requires operator ACK, released E-stop and confirmed physical stop",
                code=ErrorCode.SAFETY_STATE,
            )
        with self._lock:
            self._authorization = None
            self._report = None
            self._state = M5State.NEW

    @_serialized
    def fault(self, reason: str) -> None:
        self._assert_open()
        with self._lock:
            if self._state is M5State.STOP_FAILED:
                return
            active = self._state in _ACTIVE_STATES
        self.authority.revoke(reason)
        if active:
            self._set_state(M5State.STOPPING)
            outcomes = self._stop_devices_parallel(self._stop_timeout_s)
            self._set_state(M5State.FAULT if all(o.confirmed for o in outcomes.values()) else M5State.STOP_FAILED)
        else:
            self._set_state(M5State.FAULT)


__all__ = ["M5Runtime", "M5State"]
