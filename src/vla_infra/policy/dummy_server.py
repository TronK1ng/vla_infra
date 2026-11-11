"""Dummy OpenPI-shaped WebSocket policy server.

Security defaults: bind to loopback only; non-loopback binding requires both an
explicit opt-in and a bearer token.  M1 does not provide TLS — production
deployments must use a VPN or TLS reverse proxy (master plan §8.2).

The server never imports RDK, never touches a camera and has no robot network
route.  It validates every request against its adapter metadata and turns
adapter failures into error responses without leaking tracebacks or secrets.
"""

from __future__ import annotations

import hmac
import ipaddress
import logging
import threading
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Tuple

import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve as serve_sync

from ..clock import Clock, MonotonicClock
from ..contracts.action import ActionChunk
from ..contracts.constants import ACTION_DIM
from ..contracts.validation import validate_policy_observation
from ..errors import ConfigError, ErrorCode, PolicyContractError, ValidationError, VlaInfraError
from .adapters.base import PolicyAdapter, PolicyMetadata
from .wire import (
    build_error_message,
    build_metadata_message,
    build_result_message,
    get_codec,
    observation_capture_time,
    validate_request,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServerStats:
    connections: int = 0
    requests: int = 0
    successes: int = 0
    errors: int = 0
    auth_failures: int = 0
    last_sequence_id: int = -1

    def as_dict(self) -> dict:
        return {
            "connections": self.connections,
            "requests": self.requests,
            "successes": self.successes,
            "errors": self.errors,
            "auth_failures": self.auth_failures,
            "last_sequence_id": self.last_sequence_id,
        }


def _is_loopback(host: str) -> bool:
    if host in ("localhost", "127.0.0.1", "::1"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _display_host(host: str) -> str:
    if host in ("0.0.0.0", "::"):
        return "127.0.0.1"
    if ":" in host:
        return f"[{host}]"
    return host


class DummyPolicyServer:
    def __init__(
        self,
        adapter: PolicyAdapter,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        codec: str = "auto",
        auth_token: Optional[str] = None,
        allow_non_loopback: bool = False,
        clock: Optional[Clock] = None,
        max_message_bytes: int = 32 * 1024 * 1024,
    ) -> None:
        if not isinstance(adapter, PolicyAdapter):
            raise TypeError("adapter must be a PolicyAdapter")
        if not _is_loopback(host) and not allow_non_loopback:
            raise ConfigError(
                "refusing to bind a policy server to a non-loopback interface without explicit opt-in",
                details={"host": host},
            )
        if allow_non_loopback and not auth_token:
            raise ConfigError(
                "non-loopback policy server requires an auth token; M1 still needs a VPN/TLS boundary",
                details={"host": host},
            )
        if auth_token is not None and len(auth_token) < 8:
            raise ConfigError("auth token must be at least 8 characters")
        self._adapter = adapter
        self._adapter_lock = threading.Lock()
        self._host = host
        self._requested_port = int(port)
        self._bound_port = -1
        self._codec = get_codec(codec)
        self._auth_token = auth_token
        self._clock = clock or MonotonicClock()
        self._max_message_bytes = int(max_message_bytes)
        self._stats = ServerStats()
        self._stats_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._server: Any = None
        self._connections: set = set()
        self._connections_lock = threading.Lock()
        self._ready = threading.Event()
        self._start_error: Optional[BaseException] = None
        self._started = False

    # -- lifecycle ----------------------------------------------------------
    def start(self, *, timeout_s: float = 5.0) -> str:
        if self._started:
            raise RuntimeError("server already started")
        self._started = True
        self._thread = threading.Thread(target=self._run, name=f"policy-server-{self.metadata.name}", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=timeout_s):
            raise TimeoutError("policy server did not become ready in time")
        if self._start_error is not None:
            raise self._start_error
        return self.endpoint

    def _run(self) -> None:
        try:
            with serve_sync(
                self._handler,
                self._host,
                self._requested_port,
                max_size=self._max_message_bytes,
                compression=None,
            ) as server:
                self._server = server
                self._bound_port = int(server.socket.getsockname()[1])
                self._ready.set()
                server.serve_forever()
        except BaseException as exc:  # noqa: BLE001 - surfaced to the caller
            self._start_error = exc
            self._ready.set()
            logger.exception("policy_server_crashed")

    def stop(self, *, timeout_s: float = 5.0) -> None:
        # websockets' sync Server does not track connections: close them first,
        # otherwise an idle client would keep a dead endpoint "connected".
        with self._connections_lock:
            connections = list(self._connections)
        for connection in connections:
            try:
                connection.close(1001, "server shutdown")
            except Exception:  # pragma: no cover - best effort
                pass
        server = self._server
        if server is not None:
            try:
                server.shutdown()
            except Exception:  # pragma: no cover - best effort shutdown
                logger.exception("policy_server_shutdown_failed")
        thread = self._thread
        if thread is not None:
            thread.join(timeout_s)
        self._server = None
        self._thread = None
        self._started = False

    @property
    def endpoint(self) -> str:
        if self._bound_port < 0:
            raise RuntimeError("server is not started")
        return f"ws://{_display_host(self._host)}:{self._bound_port}"

    @property
    def metadata(self) -> PolicyMetadata:
        return self._adapter.metadata

    @property
    def stats(self) -> ServerStats:
        with self._stats_lock:
            return self._stats

    # -- connection handling ------------------------------------------------
    def _handler(self, connection: Any) -> None:
        with self._stats_lock:
            self._stats = _bump(self._stats, connections=1)
        with self._connections_lock:
            self._connections.add(connection)
        try:
            if self._auth_token is not None:
                token = self._header_token(connection)
                if token is None or not hmac.compare_digest(token, self._auth_token):
                    with self._stats_lock:
                        self._stats = _bump(self._stats, auth_failures=1)
                    connection.close(1008, "unauthorized")
                    logger.warning("policy_server_auth_failed")
                    return
            connection.send(
                self._codec.encode(
                    build_metadata_message(self.metadata, server_time_ns=self._clock.now_ns())
                )
            )
            last_sequence: Optional[int] = None
            for raw in connection:
                response: Mapping[str, Any]
                accepted_sequence = last_sequence
                try:
                    request = self._codec.decode(raw)
                    response, accepted_sequence = self._handle_request(request, last_sequence)
                except ValidationError as exc:
                    response = build_error_message(
                        code=exc.code, message=exc.message, request_sequence_id=None
                    )
                except VlaInfraError as exc:
                    response = build_error_message(
                        code=exc.code, message=exc.message, request_sequence_id=None
                    )
                except Exception:  # pragma: no cover - defensive boundary
                    logger.exception("policy_server_decode_failed")
                    response = build_error_message(
                        code=ErrorCode.INTERNAL_ERROR, message="internal server error"
                    )
                last_sequence = accepted_sequence
                connection.send(self._codec.encode(response))
        except ConnectionClosed:
            pass
        except Exception:  # pragma: no cover - defensive connection boundary
            logger.exception("policy_server_connection_failed")
        finally:
            with self._connections_lock:
                self._connections.discard(connection)
            logger.info("policy_server_connection_closed")

    def _handle_request(
        self, request: Any, last_sequence: Optional[int]
    ) -> Tuple[Mapping[str, Any], Optional[int]]:
        sequence: Optional[int] = None
        try:
            sequence, observation = validate_request(request)
            if last_sequence is not None and sequence <= last_sequence:
                raise VlaInfraError(
                    f"request sequence {sequence} does not advance past {last_sequence}",
                    code=ErrorCode.SEQUENCE_REGRESSION,
                    details={"sequence": sequence, "last": last_sequence},
                )
            with self._stats_lock:
                self._stats = _bump(self._stats, requests=1)
            metadata = self.metadata
            validate_policy_observation(
                observation,
                expected_layout=metadata.state_layout,
                expected_image_size=metadata.image_size,
            )
            receive_ns = self._clock.now_ns()
            with self._adapter_lock:
                raw_result = self._adapter.infer(observation)
            if not isinstance(raw_result, Mapping):
                raise PolicyContractError("adapter result must be a mapping")
            actions = raw_result.get("actions")
            if not isinstance(actions, np.ndarray):
                raise PolicyContractError("adapter actions must be a numpy array")
            if actions.dtype != np.float32:
                raise PolicyContractError(
                    "adapter actions must be float32",
                    code=ErrorCode.INVALID_DTYPE,
                    details={"dtype": str(actions.dtype)},
                )
            if actions.shape != (metadata.horizon, ACTION_DIM):
                raise PolicyContractError(
                    "adapter actions shape does not match the declared horizon",
                    code=ErrorCode.INVALID_SHAPE,
                    details={
                        "shape": list(actions.shape),
                        "expected": [metadata.horizon, ACTION_DIM],
                    },
                )
            extras = raw_result.get("extras", {})
            if not isinstance(extras, Mapping):
                raise PolicyContractError("adapter extras must be a mapping")
            send_ns = self._clock.now_ns()
            chunk = ActionChunk(
                request_sequence_id=sequence,
                observation_capture_time_ns=observation_capture_time(observation),
                action_dt_s=metadata.action_dt_s,
                action_frame=metadata.action_frame,
                action_type=metadata.action_type,
                actions=actions,
                extras=dict(extras),
                server_receive_time_ns=receive_ns,
                server_send_time_ns=send_ns,
            )
            response = build_result_message(
                chunk, server_receive_time_ns=receive_ns, server_send_time_ns=send_ns
            )
            with self._stats_lock:
                self._stats = replace(
                    _bump(self._stats, successes=1),
                    last_sequence_id=int(sequence),
                )
            return response, int(sequence)
        except VlaInfraError as exc:
            with self._stats_lock:
                self._stats = _bump(self._stats, errors=1)
            logger.warning(
                "policy_server_rejected seq=%s code=%s message=%s",
                sequence,
                exc.code.value,
                exc.message,
            )
            return (
                build_error_message(
                    code=exc.code,
                    message=exc.message,
                    request_sequence_id=sequence,
                    details=exc.details,
                ),
                last_sequence,
            )
        except Exception:
            with self._stats_lock:
                self._stats = _bump(self._stats, errors=1)
            logger.exception("policy_server_internal_error seq=%s", sequence)
            return (
                build_error_message(
                    code=ErrorCode.INTERNAL_ERROR,
                    message="internal server error",
                    request_sequence_id=sequence,
                ),
                last_sequence,
            )

    @staticmethod
    def _header_token(connection: Any) -> Optional[str]:
        try:
            value = connection.request.headers.get("Authorization", "")
        except Exception:  # pragma: no cover - fake connection
            return None
        prefix = "Bearer "
        return value[len(prefix) :] if value.startswith(prefix) else None


def _bump(stats: ServerStats, **increments: int) -> ServerStats:
    return replace(
        stats,
        **{name: getattr(stats, name) + int(delta) for name, delta in increments.items()},
    )


__all__ = ["DummyPolicyServer", "ServerStats"]
