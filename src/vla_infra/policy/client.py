"""Policy client wrapper.

The wrapper owns endpoint handling, timeouts, connect/handshake, one-in-flight
request discipline, sequence echo validation, RTT metrics and error-code
translation.  Business code never calls ``websockets`` directly.

Important clock-domain rule: ``server_*_time_ns`` fields are the server's
monotonic clock and are **never** compared with local timestamps.  Only the
local RTT measurement and the local chunk receive time are used for safety.
"""

from __future__ import annotations

import ipaddress
import logging
import threading
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Deque, Mapping, Optional, Protocol, Tuple, Union, runtime_checkable
from urllib.parse import urlsplit

import numpy as np
from websockets.exceptions import ConnectionClosed, WebSocketException
from websockets.sync.client import connect as connect_sync

from ..clock import Clock, MonotonicClock
from ..contracts.action import ActionChunk
from ..contracts.constants import ACTION_DIM, POLICY_KEY_CAPTURE_TIME
from ..contracts.validation import validate_policy_observation
from ..errors import (
    ConfigError,
    ErrorCode,
    PolicyContractError,
    PolicyDisconnectedError,
    PolicyTimeoutError,
    ValidationError,
    VlaInfraError,
)
from .adapters.base import PolicyAdapter, PolicyMetadata
from .wire import (
    build_request,
    get_codec,
    parse_response,
    validate_metadata_message,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class PolicyTransport(Protocol):
    @property
    def metadata(self) -> PolicyMetadata: ...

    @property
    def connected(self) -> bool: ...

    def connect(self) -> PolicyMetadata: ...

    def close(self) -> None: ...

    def infer(
        self, observation: Mapping[str, Any], request_sequence_id: int
    ) -> ActionChunk: ...


def sanitize_endpoint(endpoint: str) -> str:
    """Strip credentials and token query parameters from an endpoint for logs."""
    from urllib.parse import urlunsplit

    try:
        parts = urlsplit(endpoint)
    except ValueError:
        return "<invalid-endpoint>"
    netloc = parts.netloc
    if "@" in netloc:
        netloc = netloc.rsplit("@", 1)[1]
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def sanitize_proxy(proxy: Union[bool, str, None]) -> str:
    """Describe a proxy setting for logs without credentials."""
    if proxy is None or proxy is False:
        return "direct"
    if proxy is True:
        return "env"
    try:
        parts = urlsplit(str(proxy))
    except ValueError:
        return "<invalid-proxy>"
    netloc = parts.netloc
    if "@" in netloc:
        netloc = netloc.rsplit("@", 1)[1]
    return f"{parts.scheme}://{netloc}"


def is_loopback_endpoint(endpoint: str) -> bool:
    """True for localhost / 127.0.0.0-8 / ::1 style WebSocket endpoints."""
    try:
        host = urlsplit(endpoint).hostname
    except ValueError:
        return False
    if not host:
        return False
    if host.lower() in ("localhost", "localhost.localdomain"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def resolve_proxy(endpoint: str, proxy_setting: str = "none") -> Union[bool, str, None]:
    """Resolve the explicit proxy policy for a policy endpoint.

    ``proxy_setting`` is one of ``none``/``off`` (direct), ``env``/``auto``
    (inherit the environment, recorded explicitly), or an explicit proxy URL.
    Loopback endpoints always bypass proxies: a local dummy/robot-side policy
    server must never depend on an operator's system proxy configuration.
    """
    if is_loopback_endpoint(endpoint):
        return None
    setting = (proxy_setting or "none").strip()
    lowered = setting.lower()
    if lowered in ("none", "off", "false", "direct"):
        return None
    if lowered in ("env", "auto", "true"):
        return True
    if "://" in setting:
        return setting
    raise ConfigError(
        "policy proxy must be 'none', 'env' or a proxy URL",
        details={"proxy": sanitize_proxy(setting)},
    )


def _rtt_percentile(samples: Tuple[int, ...], fraction: float) -> Optional[int]:
    if not samples:
        return None
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


@dataclass(frozen=True)
class ClientStats:
    connects: int = 0
    connection_failures: int = 0
    requests: int = 0
    successes: int = 0
    timeouts: int = 0
    disconnects: int = 0
    protocol_errors: int = 0
    last_rtt_ns: Optional[int] = None
    rtt_p50_ns: Optional[int] = None
    rtt_p95_ns: Optional[int] = None

    def as_dict(self) -> dict:
        return {
            "connects": self.connects,
            "connection_failures": self.connection_failures,
            "requests": self.requests,
            "successes": self.successes,
            "timeouts": self.timeouts,
            "disconnects": self.disconnects,
            "protocol_errors": self.protocol_errors,
            "last_rtt_ns": self.last_rtt_ns,
            "rtt_p50_ns": self.rtt_p50_ns,
            "rtt_p95_ns": self.rtt_p95_ns,
        }


class WebsocketPolicyClient:
    def __init__(
        self,
        endpoint: str,
        *,
        timeout_s: float = 5.0,
        connect_timeout_s: float = 3.0,
        codec: str = "auto",
        auth_token: Optional[str] = None,
        proxy: str = "none",
        clock: Optional[Clock] = None,
        expected_metadata: Optional[PolicyMetadata] = None,
        max_message_bytes: int = 32 * 1024 * 1024,
        name: str = "policy-client",
    ) -> None:
        if not endpoint.startswith(("ws://", "wss://")):
            raise ConfigError(
                "policy endpoint must use ws:// or wss://",
                details={"endpoint": sanitize_endpoint(endpoint)},
            )
        if timeout_s <= 0 or connect_timeout_s <= 0:
            raise ConfigError("client timeouts must be positive")
        self.endpoint = endpoint
        self._timeout_s = float(timeout_s)
        self._connect_timeout_s = float(connect_timeout_s)
        self._codec = get_codec(codec)
        self._auth_token = auth_token
        self._proxy_setting = str(proxy)
        self._resolved_proxy = resolve_proxy(endpoint, self._proxy_setting)
        self._clock = clock or MonotonicClock()
        self._expected = expected_metadata
        self._max_message_bytes = int(max_message_bytes)
        self.name = name
        self._ws: Any = None
        self._metadata: Optional[PolicyMetadata] = None
        self._connected = False
        self._inflight = False
        self._lock = threading.RLock()
        self._rtt: Deque[int] = deque(maxlen=256)
        self._stats = ClientStats()

    # -- properties ---------------------------------------------------------
    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    @property
    def metadata(self) -> PolicyMetadata:
        with self._lock:
            if self._metadata is None:
                raise PolicyDisconnectedError(
                    "policy metadata is unavailable before a successful connect"
                )
            return self._metadata

    @property
    def stats(self) -> ClientStats:
        with self._lock:
            return self._stats

    def _connect_headers(self) -> Optional[dict[str, str]]:
        if not self._auth_token:
            return None
        return {"Authorization": f"Bearer {self._auth_token}"}

    # -- lifecycle ----------------------------------------------------------
    def connect(self) -> PolicyMetadata:
        with self._lock:
            if self._connected:
                assert self._metadata is not None
                return self._metadata
            try:
                ws = connect_sync(
                    self.endpoint,
                    open_timeout=self._connect_timeout_s,
                    close_timeout=2.0,
                    max_size=self._max_message_bytes,
                    compression=None,
                    additional_headers=self._connect_headers(),
                    proxy=self._resolved_proxy,
                )
            except Exception as exc:  # proxy/DNS/TLS/handshake failures are all connection failures
                self._stats = _bump(self._stats, connection_failures=1)
                logger.warning(
                    "policy_connect_failed endpoint=%s proxy=%s reason=%s",
                    sanitize_endpoint(self.endpoint),
                    sanitize_proxy(self._resolved_proxy),
                    type(exc).__name__,
                )
                raise PolicyDisconnectedError(
                    f"could not connect to policy endpoint {sanitize_endpoint(self.endpoint)}",
                    details={"reason": type(exc).__name__},
                ) from exc
            try:
                raw = ws.recv(timeout=self._timeout_s)
                message = self._codec.decode(raw)
                metadata = validate_metadata_message(message)
                if self._expected is not None:
                    issues = metadata.compatibility_issues(self._expected)
                    if issues:
                        raise PolicyContractError(
                            "policy server contract is incompatible with local configuration",
                            code=ErrorCode.SCHEMA_MISMATCH,
                            details={"fields": list(issues)},
                        )
            except PolicyContractError:
                ws.close()
                raise
            except ValidationError as exc:
                # Malformed metadata is a contract violation, not a transient
                # network failure; the worker must latch FAULT for it.
                ws.close()
                raise PolicyContractError(
                    "policy server returned invalid metadata",
                    code=exc.code,
                    details=exc.details,
                ) from exc
            except VlaInfraError:
                ws.close()
                raise
            except Exception as exc:  # proxy handshake / close / decode boundary
                ws.close()
                self._stats = _bump(self._stats, connection_failures=1)
                logger.warning(
                    "policy_handshake_failed endpoint=%s proxy=%s reason=%s",
                    sanitize_endpoint(self.endpoint),
                    sanitize_proxy(self._resolved_proxy),
                    type(exc).__name__,
                )
                raise PolicyDisconnectedError(
                    "policy handshake failed",
                    details={"reason": type(exc).__name__},
                ) from exc
            self._ws = ws
            self._metadata = metadata
            self._connected = True
            self._stats = _bump(self._stats, connects=1)
            logger.info(
                "policy_connected endpoint=%s proxy=%s policy=%s version=%s",
                sanitize_endpoint(self.endpoint),
                sanitize_proxy(self._resolved_proxy),
                metadata.name,
                metadata.version,
            )
            return metadata

    def close(self) -> None:
        with self._lock:
            ws = self._ws
            self._ws = None
            self._connected = False
            self._inflight = False
        if ws is not None:
            try:
                ws.close()
            except Exception:  # pragma: no cover - best effort close
                pass

    # -- inference ----------------------------------------------------------
    def infer(self, observation: Mapping[str, Any], request_sequence_id: int) -> ActionChunk:
        expected_capture_ns = observation.get(POLICY_KEY_CAPTURE_TIME)
        if (
            isinstance(expected_capture_ns, bool)
            or not isinstance(expected_capture_ns, (int, np.integer))
            or int(expected_capture_ns) <= 0
        ):
            raise ValidationError(
                "observation meta/capture_time_ns must be a positive integer",
                code=ErrorCode.INVALID_DTYPE,
                details={"field": POLICY_KEY_CAPTURE_TIME},
            )
        with self._lock:
            if not self._connected or self._ws is None:
                raise PolicyDisconnectedError(
                    "policy client is not connected", details={"endpoint": sanitize_endpoint(self.endpoint)}
                )
            if self._inflight:
                raise VlaInfraError(
                    "a policy request is already in flight on this connection",
                    code=ErrorCode.POLICY_CONTRACT,
                )
            self._inflight = True
            ws = self._ws
            metadata = self._metadata
            assert metadata is not None
        expected_capture_ns = int(expected_capture_ns)
        request = build_request(
            observation,
            request_sequence_id=request_sequence_id,
            client_send_time_ns=self._clock.now_ns(),
        )
        started = self._clock.now_ns()
        try:
            with self._lock:
                self._stats = _bump(self._stats, requests=1)
            ws.send(self._codec.encode(request))
            raw = ws.recv(timeout=self._timeout_s)
            message = self._codec.decode(raw)
            chunk = parse_response(message, expected_sequence_id=request_sequence_id)
            self._validate_chunk_against_metadata(chunk, metadata)
            if chunk.observation_capture_time_ns != expected_capture_ns:
                raise PolicyContractError(
                    "policy response does not belong to the requested observation",
                    code=ErrorCode.CAPTURE_MISMATCH,
                    details={
                        "expected_capture_time_ns": int(expected_capture_ns),
                        "received_capture_time_ns": int(chunk.observation_capture_time_ns),
                        "request_sequence_id": int(request_sequence_id),
                    },
                )
        except TimeoutError as exc:
            self._fail_connection(timeouts=1)
            raise PolicyTimeoutError(
                f"policy request timed out after {self._timeout_s:.3f} s",
                details={"endpoint": sanitize_endpoint(self.endpoint)},
            ) from exc
        except (ConnectionClosed, OSError, WebSocketException) as exc:
            self._fail_connection(disconnects=1)
            raise PolicyDisconnectedError(
                "policy connection lost during inference",
                details={"reason": type(exc).__name__},
            ) from exc
        except (ValidationError, ValueError, KeyError) as exc:
            with self._lock:
                self._stats = _bump(self._stats, protocol_errors=1)
            self.close()
            raise PolicyContractError(
                "malformed message from the policy server",
                code=ErrorCode.POLICY_CONTRACT,
                details={"reason": str(exc)},
            ) from exc
        except PolicyContractError:
            with self._lock:
                self._stats = _bump(self._stats, protocol_errors=1)
            self.close()
            raise
        finally:
            with self._lock:
                self._inflight = False
        rtt = self._clock.now_ns() - started
        with self._lock:
            self._rtt.append(int(rtt))
            samples = tuple(self._rtt)
            self._stats = replace(
                _bump(self._stats, successes=1),
                last_rtt_ns=int(rtt),
                rtt_p50_ns=_rtt_percentile(samples, 0.5),
                rtt_p95_ns=_rtt_percentile(samples, 0.95),
            )
        return chunk

    def _validate_chunk_against_metadata(self, chunk: ActionChunk, metadata: PolicyMetadata) -> None:
        if chunk.horizon != metadata.horizon:
            raise PolicyContractError(
                "server action horizon does not match its metadata",
                code=ErrorCode.INVALID_SHAPE,
                details={"horizon": chunk.horizon, "declared": metadata.horizon},
            )
        if chunk.action_frame != metadata.action_frame:
            raise PolicyContractError(
                "server action frame does not match its metadata",
                code=ErrorCode.INVALID_FRAME,
                details={"frame": chunk.action_frame, "declared": metadata.action_frame},
            )
        if chunk.action_type != metadata.action_type:
            raise PolicyContractError(
                "server action type does not match its metadata",
                code=ErrorCode.INVALID_ACTION,
                details={"type": chunk.action_type, "declared": metadata.action_type},
            )
        if abs(chunk.action_dt_s - metadata.action_dt_s) > 1e-6:
            raise PolicyContractError(
                "server action_dt_s does not match its metadata",
                code=ErrorCode.POLICY_CONTRACT,
                details={"dt": chunk.action_dt_s, "declared": metadata.action_dt_s},
            )

    def _fail_connection(self, *, timeouts: int = 0, disconnects: int = 0) -> None:
        with self._lock:
            self._stats = _bump(self._stats, timeouts=timeouts, disconnects=disconnects)
        self.close()
        logger.warning(
            "policy_connection_failed endpoint=%s timeouts=%d disconnects=%d",
            sanitize_endpoint(self.endpoint),
            timeouts,
            disconnects,
        )


class InProcessPolicyTransport:
    """Adapter-backed transport used by deterministic tests and offline demos.

    It applies the same request validation and metadata checks as the WebSocket
    server, but without sockets, so tests can use a fake clock.  It is not used
    by the production shadow/rollout entry points.
    """

    def __init__(
        self,
        adapter: PolicyAdapter,
        *,
        clock: Optional[Clock] = None,
        latency_ns: int = 0,
        fail_next: int = 0,
        fail_with_timeout: bool = False,
        expected_metadata: Optional[PolicyMetadata] = None,
    ) -> None:
        self._adapter = adapter
        self._clock = clock or MonotonicClock()
        self._latency_ns = int(latency_ns)
        self._fail_next = int(fail_next)
        self._fail_with_timeout = fail_with_timeout
        self._expected = expected_metadata
        self._connected = False
        self._metadata = adapter.metadata
        self._requests = 0

    @property
    def metadata(self) -> PolicyMetadata:
        return self._metadata

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def requests(self) -> int:
        return self._requests

    def connect(self) -> PolicyMetadata:
        if self._expected is not None:
            issues = self._metadata.compatibility_issues(self._expected)
            if issues:
                raise PolicyContractError(
                    "in-process policy is incompatible with local configuration",
                    code=ErrorCode.SCHEMA_MISMATCH,
                    details={"fields": list(issues)},
                )
        self._connected = True
        return self._metadata

    def close(self) -> None:
        self._connected = False

    def infer(self, observation: Mapping[str, Any], request_sequence_id: int) -> ActionChunk:
        if not self._connected:
            raise PolicyDisconnectedError("in-process policy transport is not connected")
        if self._fail_next > 0:
            self._fail_next -= 1
            self._connected = False
            if self._fail_with_timeout:
                raise PolicyTimeoutError("simulated in-process timeout")
            raise PolicyDisconnectedError("simulated in-process disconnect")
        self._requests += 1
        validate_policy_observation(
            observation,
            expected_layout=self._metadata.state_layout,
            expected_image_size=self._metadata.image_size,
        )
        if self._latency_ns:
            self._clock.sleep(self._latency_ns)
        result = self._adapter.infer(observation)
        actions = result.get("actions")
        if not isinstance(actions, np.ndarray):
            raise PolicyContractError("adapter actions must be a numpy array")
        if actions.dtype != np.float32:
            raise PolicyContractError(
                "adapter actions must be float32",
                code=ErrorCode.INVALID_DTYPE,
                details={"dtype": str(actions.dtype)},
            )
        expected_shape = (self._metadata.horizon, ACTION_DIM)
        if actions.shape != expected_shape:
            raise PolicyContractError(
                "adapter actions shape does not match metadata",
                code=ErrorCode.INVALID_SHAPE,
                details={"shape": list(actions.shape), "expected": list(expected_shape)},
            )
        from ..contracts.constants import POLICY_KEY_CAPTURE_TIME

        return ActionChunk(
            request_sequence_id=request_sequence_id,
            observation_capture_time_ns=int(observation[POLICY_KEY_CAPTURE_TIME]),
            action_dt_s=self._metadata.action_dt_s,
            action_frame=self._metadata.action_frame,
            action_type=self._metadata.action_type,
            actions=actions,
            extras=dict(result.get("extras", {})),
            server_receive_time_ns=self._clock.now_ns(),
            server_send_time_ns=self._clock.now_ns(),
        )


# Master-plan naming compatibility.
PolicyClient = WebsocketPolicyClient

__all__ = [
    "ClientStats",
    "InProcessPolicyTransport",
    "PolicyClient",
    "PolicyTransport",
    "WebsocketPolicyClient",
    "sanitize_endpoint",
]


def _bump(stats: ClientStats, **increments: int) -> ClientStats:
    return replace(
        stats,
        **{name: getattr(stats, name) + int(delta) for name, delta in increments.items()},
    )
