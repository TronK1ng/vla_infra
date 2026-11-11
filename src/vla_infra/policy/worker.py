"""Asynchronous policy worker.

The control loop must never block on GPU inference or the network.  The worker
owns a *latest-only* observation slot, performs one inference at a time on a
background thread, and publishes the resulting chunk into the bounded action
inbox.

Critical epoch rule: the worker snapshots the control epoch **before** sending
the request and submits the result with that epoch.  If a HOLD/FAULT happens
while the model is thinking, the chunk is rejected by the inbox instead of
being executed after a later ARM.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Optional

import numpy as np

from ..clock import Clock
from ..contracts.action import ActionChunk
from ..contracts.constants import POLICY_KEY_CAPTURE_TIME, POLICY_KEY_SEQUENCE
from ..errors import (
    ErrorCode,
    PolicyContractError,
    PolicyDisconnectedError,
    PolicyCleanupError,
    PolicyTimeoutError,
    ValidationError,
    VlaInfraError,
)
from ..execution.authority import ControlAuthority
from ..execution.inbox import ActionInbox
from ..execution.supervisor import Supervisor
from .client import PolicyTransport
from .adapters.base import PolicyMetadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObservationBundle:
    payload: Mapping[str, Any]
    capture_time_ns: int
    sequence_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.capture_time_ns, int) or self.capture_time_ns <= 0:
            raise ValueError("capture_time_ns must be a positive integer")
        if not isinstance(self.sequence_id, int) or self.sequence_id < 0:
            raise ValueError("sequence_id must be a non-negative integer")


@dataclass(frozen=True)
class SlotStats:
    published: int = 0
    replaced: int = 0
    taken: int = 0
    empty_takes: int = 0

    def as_dict(self) -> dict:
        return {
            "published": self.published,
            "replaced": self.replaced,
            "taken": self.taken,
            "empty_takes": self.empty_takes,
        }


class LatestSlot:
    """A one-element thread-safe slot; a newer value overwrites an older one."""

    def __init__(self, name: str = "observations") -> None:
        self.name = name
        self._lock = threading.Lock()
        self._value: Optional[ObservationBundle] = None
        self._stats = SlotStats()

    def put(self, bundle: ObservationBundle) -> bool:
        if not isinstance(bundle, ObservationBundle):
            raise TypeError("LatestSlot accepts ObservationBundle objects")
        with self._lock:
            replaced = self._value is not None
            self._value = bundle
            self._stats = _bump(
                self._stats,
                published=1,
                replaced=1 if replaced else 0,
            )
            return replaced

    def take(self) -> Optional[ObservationBundle]:
        with self._lock:
            value = self._value
            self._value = None
            if value is None:
                self._stats = _bump(self._stats, empty_takes=1)
            else:
                self._stats = _bump(self._stats, taken=1)
            return value

    @property
    def stats(self) -> SlotStats:
        with self._lock:
            return self._stats


@dataclass(frozen=True)
class WorkerStats:
    observations_published: int = 0
    observations_replaced: int = 0
    requests: int = 0
    results: int = 0
    accepted: int = 0
    inbox_rejected: int = 0
    timeouts: int = 0
    disconnects: int = 0
    contract_errors: int = 0
    internal_errors: int = 0
    cleanup_failures: int = 0
    connects: int = 0
    last_error: str = ""

    def as_dict(self) -> dict:
        return {
            "observations_published": self.observations_published,
            "observations_replaced": self.observations_replaced,
            "requests": self.requests,
            "results": self.results,
            "accepted": self.accepted,
            "inbox_rejected": self.inbox_rejected,
            "timeouts": self.timeouts,
            "disconnects": self.disconnects,
            "contract_errors": self.contract_errors,
            "internal_errors": self.internal_errors,
            "cleanup_failures": self.cleanup_failures,
            "connects": self.connects,
            "last_error": self.last_error,
        }


class PolicyWorker:
    def __init__(
        self,
        *,
        transport: PolicyTransport,
        inbox: ActionInbox,
        authority: ControlAuthority,
        supervisor: Supervisor,
        clock: Clock,
        name: str = "policy",
        poll_interval_s: float = 0.005,
        reconnect_backoff_s: float = 0.5,
        max_backoff_s: float = 5.0,
        on_action: Optional[Callable[[ActionChunk], None]] = None,
    ) -> None:
        if poll_interval_s <= 0 or reconnect_backoff_s <= 0 or max_backoff_s < reconnect_backoff_s:
            raise ValueError("invalid worker timing configuration")
        self._transport = transport
        self._inbox = inbox
        self._authority = authority
        self._supervisor = supervisor
        self._clock = clock
        self.name = name
        self._poll_interval_s = float(poll_interval_s)
        self._backoff_s = float(reconnect_backoff_s)
        self._max_backoff_s = float(max_backoff_s)
        self._on_action = on_action
        self._slot = LatestSlot(f"{name}-observations")
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stats = WorkerStats()

    # -- public API ---------------------------------------------------------
    @property
    def slot_stats(self) -> SlotStats:
        return self._slot.stats

    @property
    def metadata(self) -> PolicyMetadata:
        return self._transport.metadata

    @property
    def connected(self) -> bool:
        return self._transport.connected

    def publish(
        self,
        payload: Mapping[str, Any],
        *,
        capture_time_ns: int,
        sequence_id: int,
    ) -> bool:
        if not isinstance(payload, Mapping):
            raise ValidationError(
                "policy payload must be a mapping",
                code=ErrorCode.INVALID_SHAPE,
            )
        payload_capture = _payload_int(
            payload,
            POLICY_KEY_CAPTURE_TIME,
            minimum=1,
        )
        payload_sequence = _payload_int(
            payload,
            POLICY_KEY_SEQUENCE,
            minimum=0,
        )
        if isinstance(capture_time_ns, bool) or not isinstance(
            capture_time_ns, (int, np.integer)
        ):
            raise ValidationError(
                "capture_time_ns must be an integer",
                code=ErrorCode.INVALID_DTYPE,
            )
        if isinstance(sequence_id, bool) or not isinstance(sequence_id, (int, np.integer)):
            raise ValidationError(
                "sequence_id must be an integer",
                code=ErrorCode.INVALID_DTYPE,
            )
        if int(capture_time_ns) != payload_capture:
            raise ValidationError(
                "publish capture_time_ns does not match payload meta/capture_time_ns",
                code=ErrorCode.CAPTURE_MISMATCH,
                details={
                    "argument": int(capture_time_ns),
                    "payload": payload_capture,
                },
            )
        if int(sequence_id) != payload_sequence:
            raise ValidationError(
                "publish sequence_id does not match payload meta/sequence_id",
                code=ErrorCode.SEQUENCE_MISMATCH,
                details={
                    "argument": int(sequence_id),
                    "payload": payload_sequence,
                },
            )
        bundle = ObservationBundle(
            payload=payload,
            capture_time_ns=int(capture_time_ns),
            sequence_id=int(sequence_id),
        )
        replaced = self._slot.put(bundle)
        with self._lock:
            self._stats = _bump(
                self._stats,
                observations_published=1,
                observations_replaced=1 if replaced else 0,
            )
        return replaced

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("policy worker already started")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"policy-worker-{self.name}", daemon=True)
        self._thread.start()

    def stop(self, *, timeout_s: float = 5.0) -> None:
        self._stop.set()
        try:
            self._transport.close()
        except Exception:  # pragma: no cover - best effort
            pass
        thread = self._thread
        if thread is not None:
            thread.join(timeout_s)
        self._thread = None

    @property
    def stats(self) -> WorkerStats:
        with self._lock:
            return self._stats

    # -- loop ---------------------------------------------------------------
    def _run(self) -> None:
        backoff = self._backoff_s
        while not self._stop.is_set():
            bundle = self._slot.take()
            if bundle is None:
                self._stop.wait(self._poll_interval_s)
                continue
            if not self._transport.connected:
                try:
                    self._transport.connect()
                except (PolicyContractError, ValidationError) as exc:
                    # A handshake/metadata contract violation is not retryable
                    # with the same configuration: fail closed and stop.
                    with self._lock:
                        self._stats = replace(
                            _bump(self._stats, contract_errors=1),
                            last_error=f"{exc.code.value}: {exc.message}",
                        )
                    logger.error("policy_connect_contract_error code=%s", exc.code.value)
                    if not self._stop.is_set():
                        self._supervisor.fault(
                            f"policy_connect_contract:{exc.code.value}",
                            now_ns=self._clock.now_ns(),
                        )
                    self._transport.close()
                    break
                except VlaInfraError as exc:
                    with self._lock:
                        self._stats = replace(
                            _bump(self._stats, disconnects=1),
                            last_error=f"{exc.code.value}: {exc.message}",
                        )
                    logger.warning(
                        "policy_connect_failed code=%s reason=%s",
                        exc.code.value,
                        type(exc).__name__,
                    )
                    if not self._stop.is_set():
                        self._supervisor.on_policy_disconnected(
                            f"connect_failed:{exc.code.value}", now_ns=self._clock.now_ns()
                        )
                        if not self._stop.wait(backoff):
                            backoff = min(self._max_backoff_s, backoff * 2.0)
                    continue
                except Exception as exc:
                    # Unknown transport/connect error: at minimum HOLD.
                    with self._lock:
                        self._stats = replace(
                            _bump(self._stats, internal_errors=1),
                            last_error=f"connect_internal: {type(exc).__name__}",
                        )
                    logger.warning(
                        "policy_connect_internal_error reason=%s", type(exc).__name__
                    )
                    if not self._stop.is_set():
                        self._supervisor.on_policy_disconnected(
                            "policy_connect_internal", now_ns=self._clock.now_ns()
                        )
                        self._transport.close()
                        if not self._stop.wait(backoff):
                            backoff = min(self._max_backoff_s, backoff * 2.0)
                    continue
                backoff = self._backoff_s
                with self._lock:
                    self._stats = _bump(self._stats, connects=1)
                self._supervisor.on_policy_connected(now_ns=self._clock.now_ns())

            epoch = self._authority.epoch
            sequence = bundle.sequence_id
            with self._lock:
                self._stats = _bump(self._stats, requests=1)
            try:
                chunk = self._transport.infer(bundle.payload, sequence)
            except PolicyCleanupError as exc:
                with self._lock:
                    self._stats = replace(
                        _bump(self._stats, cleanup_failures=1),
                        last_error=f"{exc.code.value}: {exc.message}",
                    )
                if self._stop.is_set():
                    break
                self._supervisor.fault(
                    f"policy_cleanup_failed:{exc.code.value}", now_ns=self._clock.now_ns()
                )
                self._transport.close()
                break
            except (PolicyTimeoutError, PolicyDisconnectedError) as exc:
                with self._lock:
                    self._stats = replace(
                        _bump(
                            self._stats,
                            timeouts=1 if isinstance(exc, PolicyTimeoutError) else 0,
                            disconnects=1,
                        ),
                        last_error=f"{exc.code.value}: {exc.message}",
                    )
                if self._stop.is_set():
                    break
                self._supervisor.on_policy_disconnected(
                    exc.code.value, now_ns=self._clock.now_ns()
                )
                self._transport.close()
                if self._stop.wait(backoff) is False:
                    backoff = min(self._max_backoff_s, backoff * 2.0)
                continue
            except (PolicyContractError, ValidationError) as exc:
                with self._lock:
                    self._stats = replace(
                        _bump(self._stats, contract_errors=1),
                        last_error=f"{exc.code.value}: {exc.message}",
                    )
                if self._stop.is_set():
                    break
                self._supervisor.fault(
                    f"policy_contract:{exc.code.value}", now_ns=self._clock.now_ns()
                )
                self._transport.close()
                break
            except Exception as exc:  # pragma: no cover - defensive boundary
                with self._lock:
                    self._stats = replace(
                        _bump(self._stats, internal_errors=1), last_error=repr(exc)
                    )
                if self._stop.is_set():
                    break
                self._supervisor.fault("policy_internal_error", now_ns=self._clock.now_ns())
                self._transport.close()
                break

            if chunk.observation_capture_time_ns != bundle.capture_time_ns:
                # Defense in depth: even a buggy transport/server cannot move a
                # chunk for a different observation into the execution inbox.
                with self._lock:
                    self._stats = replace(
                        _bump(self._stats, contract_errors=1),
                        last_error=(
                            "capture_mismatch: "
                            f"{chunk.observation_capture_time_ns} != {bundle.capture_time_ns}"
                        ),
                    )
                if not self._stop.is_set():
                    self._supervisor.fault(
                        "policy_capture_mismatch", now_ns=self._clock.now_ns()
                    )
                self._transport.close()
                break
            now = self._clock.now_ns()
            accepted = self._inbox.submit(chunk, epoch=epoch, received_time_ns=now)
            with self._lock:
                self._stats = _bump(
                    self._stats,
                    results=1,
                    accepted=1 if accepted else 0,
                    inbox_rejected=0 if accepted else 1,
                )
            self._supervisor.on_policy_result(now_ns=now)
            if accepted:
                self._supervisor.on_chunk_accepted(now_ns=now)
            if self._on_action is not None:
                try:
                    self._on_action(chunk)
                except Exception:  # pragma: no cover - observability callback
                    pass
            backoff = self._backoff_s


def _payload_int(payload: Mapping[str, Any], key: str, *, minimum: int) -> int:
    if key not in payload:
        raise ValidationError(
            f"policy payload is missing {key!r}",
            code=ErrorCode.SCHEMA_MISMATCH,
            details={"missing": key},
        )
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValidationError(
            f"policy payload {key!r} must be an integer",
            code=ErrorCode.INVALID_DTYPE,
            details={"key": key},
        )
    value = int(value)
    if value < minimum:
        raise ValidationError(
            f"policy payload {key!r} must be >= {minimum}",
            code=ErrorCode.VALUE_OUT_OF_RANGE,
            details={"key": key, "value": value},
        )
    return value


def _bump(stats: Any, **increments: int) -> Any:
    return replace(
        stats,
        **{name: getattr(stats, name) + int(delta) for name, delta in increments.items()},
    )


__all__ = [
    "LatestSlot",
    "ObservationBundle",
    "PolicyWorker",
    "SlotStats",
    "WorkerStats",
]
