"""Lifecycle state machine and control-authority arbitrator (master plan §9).

States (transitions are explicit and audited)::

    OFFLINE -> CONNECTED -> DEVICES_READY -> HOLD -> ARMED -> RUNNING
                                              ^       |        |
                                              +-------+--------+
                                                       |
                                                      FAULT -> OFFLINE (operator reset)

Only ``RUNNING`` allows the executor to write.  ``ARMED`` is therefore a grace
state: the authority is granted, but ``write_permit`` refuses writes until the
operator explicitly starts execution.  This is stricter than the plan's
"ARMED/RUNNING may command" and is deliberate fail-safe behaviour.

The supervisor is the only writer of the control authority; policy/network
callbacks can request transitions but cannot write to devices.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Mapping, Optional, Tuple

from ..clock import Clock
from ..errors import ErrorCode, IllegalTransitionError, NotArmedError, VlaInfraError
from .authority import ControlAuthority
from .inbox import ActionInbox

logger = logging.getLogger(__name__)

_VALID_SEVERITIES = ("hold", "fault")


class SystemState(str, Enum):
    OFFLINE = "OFFLINE"
    CONNECTED = "CONNECTED"
    DEVICES_READY = "DEVICES_READY"
    HOLD = "HOLD"
    ARMED = "ARMED"
    RUNNING = "RUNNING"
    FAULT = "FAULT"


_ALLOWED: Mapping[SystemState, Tuple[SystemState, ...]] = {
    SystemState.OFFLINE: (SystemState.CONNECTED, SystemState.FAULT),
    SystemState.CONNECTED: (SystemState.DEVICES_READY, SystemState.HOLD, SystemState.FAULT),
    SystemState.DEVICES_READY: (SystemState.HOLD, SystemState.FAULT),
    SystemState.HOLD: (SystemState.ARMED, SystemState.FAULT),
    SystemState.ARMED: (SystemState.RUNNING, SystemState.HOLD, SystemState.FAULT),
    SystemState.RUNNING: (SystemState.HOLD, SystemState.FAULT),
    SystemState.FAULT: (SystemState.OFFLINE,),
}


@dataclass(frozen=True)
class WatchdogConfig:
    timeout_ns: int
    severity: str = "hold"

    def __post_init__(self) -> None:
        if not isinstance(self.timeout_ns, int) or self.timeout_ns <= 0:
            raise ValueError("watchdog timeout_ns must be a positive integer")
        if self.severity not in _VALID_SEVERITIES:
            raise ValueError(f"watchdog severity must be one of {_VALID_SEVERITIES}")


@dataclass(frozen=True)
class SupervisorConfig:
    watchdogs: Mapping[str, WatchdogConfig] = field(
        default_factory=lambda: {
            "robot": WatchdogConfig(250_000_000, "fault"),
            "camera": WatchdogConfig(500_000_000, "hold"),
            "executor": WatchdogConfig(250_000_000, "fault"),
            "policy": WatchdogConfig(1_000_000_000, "hold"),
        }
    )
    chunk_starvation_ns: int = 1_000_000_000
    require_operator_ack: bool = True
    history_limit: int = 1024

    def __post_init__(self) -> None:
        if self.chunk_starvation_ns <= 0:
            raise ValueError("chunk_starvation_ns must be positive")
        if self.history_limit < 8:
            raise ValueError("history_limit must be >= 8")
        for name, config in self.watchdogs.items():
            if not isinstance(name, str) or not name:
                raise ValueError("watchdog names must be non-empty strings")
            if not isinstance(config, WatchdogConfig):
                raise TypeError("watchdogs must map to WatchdogConfig")


@dataclass(frozen=True)
class TransitionRecord:
    time_ns: int
    from_state: SystemState
    to_state: SystemState
    reason: str
    epoch: int

    def as_dict(self) -> dict:
        return {
            "time_ns": self.time_ns,
            "from": self.from_state.value,
            "to": self.to_state.value,
            "reason": self.reason,
            "epoch": self.epoch,
        }


@dataclass(frozen=True)
class WatchdogStatus:
    name: str
    timeout_ns: int
    severity: str
    last_heartbeat_ns: Optional[int]
    age_ns: Optional[int]
    fresh: bool

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "timeout_ns": self.timeout_ns,
            "severity": self.severity,
            "last_heartbeat_ns": self.last_heartbeat_ns,
            "age_ns": self.age_ns,
            "fresh": self.fresh,
        }


class Supervisor:
    def __init__(
        self,
        *,
        clock: Clock,
        config: Optional[SupervisorConfig] = None,
        authority: Optional[ControlAuthority] = None,
        inbox: Optional[ActionInbox] = None,
    ) -> None:
        self._clock = clock
        self.config = config or SupervisorConfig()
        self._authority = authority or ControlAuthority(owner="supervisor")
        self._inbox = inbox or ActionInbox(self._authority)
        self._lock = threading.RLock()
        self._state = SystemState.OFFLINE
        self._devices_ready = False
        self._policy_connected = False
        self._operator_ack = False
        self._latched_fault: Optional[str] = None
        self._idle_since_ns: Optional[int] = None
        self._last_policy_result_ns: Optional[int] = None
        self._heartbeats: dict[str, int] = {}
        self._history: Deque[TransitionRecord] = deque(maxlen=self.config.history_limit)
        self._history.append(
            TransitionRecord(
                time_ns=self._clock.now_ns(),
                from_state=SystemState.OFFLINE,
                to_state=SystemState.OFFLINE,
                reason="init",
                epoch=self._authority.epoch,
            )
        )

    # -- accessors ----------------------------------------------------------
    @property
    def state(self) -> SystemState:
        with self._lock:
            return self._state

    @property
    def authority(self) -> ControlAuthority:
        return self._authority

    @property
    def inbox(self) -> ActionInbox:
        return self._inbox

    @property
    def epoch(self) -> int:
        return self._authority.epoch

    @property
    def latched_fault(self) -> Optional[str]:
        with self._lock:
            return self._latched_fault

    @property
    def is_devices_ready(self) -> bool:
        with self._lock:
            return self._devices_ready

    @property
    def policy_connected(self) -> bool:
        with self._lock:
            return self._policy_connected

    def history(self) -> Tuple[TransitionRecord, ...]:
        with self._lock:
            return tuple(self._history)

    # -- lifecycle transitions ---------------------------------------------
    def connect_devices(self, *, now_ns: Optional[int] = None) -> SystemState:
        now = self._now(now_ns)
        with self._lock:
            self._require_state(SystemState.OFFLINE, "connect_devices")
            self._transition(SystemState.CONNECTED, "devices_connected", now)
        return SystemState.CONNECTED

    def devices_ready(self, *, now_ns: Optional[int] = None) -> SystemState:
        """Preflight passed: enter DEVICES_READY and immediately HOLD.

        Preflight is *not* the same as arming; an operator must still arm.  In
        M1 the "preflight" is structural (config validated and fake devices
        constructed); M2 adds the real read-only preflight evidence gate.
        """
        now = self._now(now_ns)
        with self._lock:
            self._require_state(SystemState.CONNECTED, "devices_ready")
            self._devices_ready = True
            self._transition(SystemState.DEVICES_READY, "devices_ready", now)
            self._revoke_and_clear("devices_ready")
            self._transition(SystemState.HOLD, "ready_hold", now)
        return SystemState.HOLD

    def arm(self, *, operator_ack: bool = False, now_ns: Optional[int] = None) -> SystemState:
        now = self._now(now_ns)
        with self._lock:
            self._require_state(SystemState.HOLD, "arm")
            if not self._devices_ready:
                raise VlaInfraError(
                    "cannot arm before device preflight passed",
                    code=ErrorCode.NOT_READY,
                    details={"state": self._state.value},
                )
            if not self._policy_connected:
                raise VlaInfraError(
                    "cannot arm while the policy endpoint is disconnected",
                    code=ErrorCode.NOT_READY,
                )
            if self.config.require_operator_ack and not operator_ack:
                raise VlaInfraError(
                    "arming requires explicit operator acknowledgement",
                    code=ErrorCode.NOT_READY,
                )
            stale = self._stale_watchdogs(now)
            if stale:
                raise VlaInfraError(
                    f"cannot arm with stale watchdogs: {', '.join(sorted(stale))}",
                    code=ErrorCode.NOT_READY,
                    details={"stale": sorted(stale)},
                )
            self._operator_ack = True
            self._authority.grant("armed")
            self._transition(SystemState.ARMED, "arm", now)
        return SystemState.ARMED

    def start_running(self, *, now_ns: Optional[int] = None) -> SystemState:
        now = self._now(now_ns)
        with self._lock:
            self._require_state(SystemState.ARMED, "start_running")
            stale = self._stale_watchdogs(now)
            if stale:
                raise VlaInfraError(
                    f"cannot start with stale watchdogs: {', '.join(sorted(stale))}",
                    code=ErrorCode.NOT_READY,
                    details={"stale": sorted(stale)},
                )
            try:
                self._authority.activate("running")
            except NotArmedError as exc:
                raise VlaInfraError(
                    "control authority is not armed",
                    code=ErrorCode.NOT_ARMED,
                    details=exc.details,
                ) from exc
            self._idle_since_ns = None
            self._transition(SystemState.RUNNING, "start_running", now)
        return SystemState.RUNNING

    def hold(self, reason: str, *, now_ns: Optional[int] = None) -> SystemState:
        now = self._now(now_ns)
        with self._lock:
            if self._state is SystemState.HOLD:
                return self._state
            if self._state in (SystemState.FAULT, SystemState.OFFLINE):
                raise IllegalTransitionError(
                    f"cannot HOLD from {self._state.value}; FAULT requires an explicit reset",
                    details={"state": self._state.value, "reason": reason},
                )
            self._revoke_and_clear(reason)
            self._idle_since_ns = None
            self._transition(SystemState.HOLD, reason, now)
            logger.warning("control_hold reason=%s epoch=%d", reason, self._authority.epoch)
        return SystemState.HOLD

    def fault(self, reason: str, *, now_ns: Optional[int] = None) -> SystemState:
        now = self._now(now_ns)
        with self._lock:
            if self._state is SystemState.FAULT:
                return self._state
            self._latched_fault = reason
            self._revoke_and_clear(reason)
            self._idle_since_ns = None
            self._transition(SystemState.FAULT, reason, now)
            logger.error("control_fault reason=%s epoch=%d", reason, self._authority.epoch)
        return SystemState.FAULT

    def reset(self, *, operator_ack: bool = False, now_ns: Optional[int] = None) -> SystemState:
        now = self._now(now_ns)
        with self._lock:
            self._require_state(SystemState.FAULT, "reset")
            if not operator_ack:
                raise VlaInfraError(
                    "FAULT reset requires an explicit on-site operator acknowledgement",
                    code=ErrorCode.NOT_READY,
                )
            self._revoke_and_clear("fault_reset")
            self._latched_fault = None
            self._devices_ready = False
            self._policy_connected = False
            self._idle_since_ns = None
            self._transition(SystemState.OFFLINE, "fault_reset", now)
        return SystemState.OFFLINE

    # -- event inputs -------------------------------------------------------
    def on_policy_connected(self, *, now_ns: Optional[int] = None) -> None:
        now = self._now(now_ns)
        with self._lock:
            self._policy_connected = True
            self._heartbeat_locked("policy", now)

    def on_policy_disconnected(self, reason: str, *, now_ns: Optional[int] = None) -> None:
        now = self._now(now_ns)
        with self._lock:
            self._policy_connected = False
            if self._state in (SystemState.ARMED, SystemState.RUNNING):
                self._revoke_and_clear(f"policy_disconnected:{reason}")
                self._idle_since_ns = None
                self._transition(SystemState.HOLD, f"policy_disconnected:{reason}", now)
            logger.warning("policy_disconnected reason=%s state=%s", reason, self._state.value)

    def on_policy_result(self, *, now_ns: Optional[int] = None) -> None:
        now = self._now(now_ns)
        with self._lock:
            self._last_policy_result_ns = now
            self._heartbeat_locked("policy", now)

    def on_chunk_accepted(self, *, now_ns: Optional[int] = None) -> None:
        with self._lock:
            self._idle_since_ns = None

    def on_executor_idle(self, *, now_ns: Optional[int] = None) -> None:
        now = self._now(now_ns)
        with self._lock:
            if self._idle_since_ns is None:
                self._idle_since_ns = now
            self._heartbeat_locked("executor", now)

    def on_executor_active(self, *, now_ns: Optional[int] = None) -> None:
        now = self._now(now_ns)
        with self._lock:
            self._idle_since_ns = None
            self._heartbeat_locked("executor", now)

    def heartbeat(self, source: str, *, now_ns: Optional[int] = None) -> None:
        now = self._now(now_ns)
        with self._lock:
            if source not in self.config.watchdogs:
                raise VlaInfraError(
                    f"unknown watchdog source {source!r}",
                    code=ErrorCode.CONFIG_INVALID,
                    details={"known": sorted(self.config.watchdogs)},
                )
            self._heartbeat_locked(source, now)

    # -- watchdog evaluation ------------------------------------------------
    def tick(self, *, now_ns: Optional[int] = None) -> SystemState:
        now = self._now(now_ns)
        with self._lock:
            if self._state in (SystemState.ARMED, SystemState.RUNNING):
                for name, config in self.config.watchdogs.items():
                    last = self._heartbeats.get(name)
                    if last is None or now - last > config.timeout_ns:
                        reason = f"watchdog_expired:{name}"
                        logger.warning("watchdog_expired source=%s severity=%s", name, config.severity)
                        if config.severity == "fault":
                            self._latched_fault = reason
                            self._revoke_and_clear(reason)
                            self._idle_since_ns = None
                            self._transition(SystemState.FAULT, reason, now)
                            return self._state
                        self._revoke_and_clear(reason)
                        self._idle_since_ns = None
                        self._transition(SystemState.HOLD, reason, now)
                        return self._state
                if (
                    self._state is SystemState.RUNNING
                    and self._idle_since_ns is not None
                    and now - self._idle_since_ns > self.config.chunk_starvation_ns
                ):
                    self._revoke_and_clear("chunk_starvation")
                    self._idle_since_ns = None
                    self._transition(SystemState.HOLD, "chunk_starvation", now)
                    return self._state
            return self._state

    # -- status -------------------------------------------------------------
    def watchdog_status(self, *, now_ns: Optional[int] = None) -> Tuple[WatchdogStatus, ...]:
        now = self._now(now_ns)
        with self._lock:
            result = []
            for name, config in self.config.watchdogs.items():
                last = self._heartbeats.get(name)
                age = None if last is None else now - last
                fresh = last is not None and age is not None and age <= config.timeout_ns
                result.append(
                    WatchdogStatus(
                        name=name,
                        timeout_ns=config.timeout_ns,
                        severity=config.severity,
                        last_heartbeat_ns=last,
                        age_ns=age,
                        fresh=fresh,
                    )
                )
            return tuple(result)

    def status(self, *, now_ns: Optional[int] = None) -> dict:
        with self._lock:
            return {
                "state": self._state.value,
                "epoch": self._authority.epoch,
                "authority": {
                    "valid": self._authority.valid,
                    "running": self._authority.running,
                    "reason": self._authority.reason,
                },
                "devices_ready": self._devices_ready,
                "policy_connected": self._policy_connected,
                "latched_fault": self._latched_fault,
                "watchdogs": [w.as_dict() for w in self.watchdog_status(now_ns=now_ns)],
                "inbox": self._inbox.stats.as_dict(),
                "transitions": [t.as_dict() for t in self._history],
            }

    # -- internals ----------------------------------------------------------
    def _now(self, now_ns: Optional[int]) -> int:
        return self._clock.now_ns() if now_ns is None else int(now_ns)

    def _require_state(self, expected: SystemState, action: str) -> None:
        if self._state is not expected:
            raise IllegalTransitionError(
                f"illegal transition: cannot {action} from {self._state.value}",
                details={
                    "action": action,
                    "state": self._state.value,
                    "expected": expected.value,
                },
            )

    def _transition(self, to_state: SystemState, reason: str, now_ns: int) -> None:
        from_state = self._state
        if to_state not in _ALLOWED[from_state]:
            raise IllegalTransitionError(
                f"illegal transition {from_state.value} -> {to_state.value}",
                details={"from": from_state.value, "to": to_state.value, "reason": reason},
            )
        self._state = to_state
        record = TransitionRecord(
            time_ns=now_ns,
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            epoch=self._authority.epoch,
        )
        self._history.append(record)
        logger.info(
            "state_transition %s->%s reason=%s epoch=%d",
            from_state.value,
            to_state.value,
            reason,
            self._authority.epoch,
        )

    def _heartbeat_locked(self, source: str, now_ns: int) -> None:
        self._heartbeats[source] = now_ns

    def _stale_watchdogs(self, now_ns: int) -> set[str]:
        stale: set[str] = set()
        for name, config in self.config.watchdogs.items():
            last = self._heartbeats.get(name)
            if last is None or now_ns - last > config.timeout_ns:
                stale.add(name)
        return stale

    def _revoke_and_clear(self, reason: str) -> None:
        if self._authority.valid:
            self._authority.revoke(reason)
        self._inbox.clear(reason)


__all__ = [
    "Supervisor",
    "SupervisorConfig",
    "SystemState",
    "TransitionRecord",
    "WatchdogConfig",
    "WatchdogStatus",
]
