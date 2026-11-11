"""Pinned official OpenPI client adapter (M4).

The official ``openpi-client`` package is an optional dependency pinned in the
``m4`` extra.  This module is the only place that imports it.  The existing
``policy/openpi.py`` classes are retained as **custom protocol** wrappers; they
must not be mistaken for official OpenPI compatibility.

Official client API (0.1.2)::

    WebsocketClientPolicy(host=endpoint) -> metadata via get_server_metadata()
    client.infer(dict) -> dict with ``actions``

The official client has no timeout/reconnect state machine.  This adapter adds
a bounded connect/infer wrapper and explicit sequence checks, without changing
the package's wire format.
"""

from __future__ import annotations

import importlib.metadata
import logging
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlsplit

import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve as serve_sync

from ..clock import Clock, MonotonicClock
from ..contracts.action import ActionChunk
from ..contracts.policy_io import CANONICAL_STATE_LAYOUT
from ..errors import (
    ErrorCode,
    PolicyCleanupError,
    PolicyContractError,
    PolicyDisconnectedError,
    PolicyTimeoutError,
    ValidationError,
)
from .adapters.base import PolicyAdapter, PolicyMetadata, ZeroActionAdapter
from .client import is_loopback_endpoint, sanitize_endpoint

OPENPI_PACKAGE = "openpi-client"
OPENPI_PINNED_VERSION = "0.1.2"
OPENPI_CLIENT_MODULE = "openpi_client.websocket_client_policy.WebsocketClientPolicy"


class OpenPIUnavailableError(ImportError):
    """Official OpenPI client is not installed; install the pinned m4 extra."""


def _require_openpi():
    try:
        from openpi_client import msgpack_numpy  # type: ignore
        from openpi_client.websocket_client_policy import WebsocketClientPolicy  # type: ignore
        try:
            version = importlib.metadata.version(OPENPI_PACKAGE)
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise OpenPIUnavailableError(
            f"official {OPENPI_PACKAGE}=={OPENPI_PINNED_VERSION} is required for the official M4 adapter"
        ) from exc
    if version != "unknown" and version != OPENPI_PINNED_VERSION:
        raise OpenPIUnavailableError(
            f"official OpenPI package version {version!r} is not the pinned {OPENPI_PINNED_VERSION}"
        )
    return WebsocketClientPolicy, msgpack_numpy, version


@dataclass(frozen=True)
class OfficialClientStats:
    connects: int = 0
    connection_failures: int = 0
    requests: int = 0
    successes: int = 0
    timeouts: int = 0
    cleanup_failures: int = 0
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
            "cleanup_failures": self.cleanup_failures,
            "disconnects": self.disconnects,
            "protocol_errors": self.protocol_errors,
            "last_rtt_ns": self.last_rtt_ns,
            "rtt_p50_ns": self.rtt_p50_ns,
            "rtt_p95_ns": self.rtt_p95_ns,
        }


class OfficialOpenPIClient:
    def __init__(
        self,
        endpoint: str,
        *,
        timeout_s: float = 5.0,
        connect_timeout_s: float = 3.0,
        expected_metadata: Optional[PolicyMetadata] = None,
        clock: Optional[Clock] = None,
        name: str = "m4-official-openpi-client",
    ) -> None:
        if not endpoint.startswith(("ws://", "wss://")):
            raise ValueError("official OpenPI endpoint must be ws:// or wss://")
        self.endpoint = endpoint
        self.name = name
        self._timeout_s = float(timeout_s)
        self._connect_timeout_s = float(connect_timeout_s)
        self._expected = expected_metadata
        self._clock = clock or MonotonicClock()
        self._client: Any = None
        self._metadata: Optional[PolicyMetadata] = None
        self._connected = False
        self._lock = threading.Lock()
        self._last_sequence: Optional[int] = None
        self._rtt: list[int] = []
        self._stats = OfficialClientStats()
        self._package_version = "unknown"
        self._client_class = OPENPI_CLIENT_MODULE
        self._server_metadata: Dict[str, Any] = {}

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    @property
    def metadata(self) -> PolicyMetadata:
        with self._lock:
            if self._metadata is None:
                raise PolicyDisconnectedError("official OpenPI metadata unavailable before connect")
            return self._metadata

    @property
    def stats(self) -> OfficialClientStats:
        with self._lock:
            return self._stats

    @property
    def identity(self) -> dict:
        return {
            "transport_kind": "official_openpi",
            "client_class": self._client_class,
            "package": OPENPI_PACKAGE,
            "package_version": self._package_version,
            "pinned_version": OPENPI_PINNED_VERSION,
            "endpoint": sanitize_endpoint(self.endpoint),
        }

    def connect(self) -> PolicyMetadata:
        WebsocketClientPolicy, _msgpack, version = _require_openpi()
        self._package_version = version
        with self._lock:
            if self._connected:
                assert self._metadata is not None
                return self._metadata
        try:
            client = _construct_official_client(
                WebsocketClientPolicy,
                _msgpack,
                self.endpoint,
                self._connect_timeout_s,
            )
        except TimeoutError as exc:
            with self._lock:
                self._stats = replace(self._stats, timeouts=self._stats.timeouts + 1)
            raise PolicyTimeoutError(
                "official OpenPI client construction timed out",
                details={"endpoint": self.endpoint, "timeout_s": self._connect_timeout_s},
            ) from exc
        except (OSError, ConnectionError) as exc:
            with self._lock:
                self._stats = replace(self._stats, connection_failures=self._stats.connection_failures + 1)
            raise PolicyDisconnectedError(
                "official OpenPI client construction failed",
                details={"endpoint": self.endpoint, "reason": type(exc).__name__},
            ) from exc
        try:
            metadata_raw = client.get_server_metadata()
        except Exception as exc:  # noqa: BLE001
            try:
                client._ws.close()
            except Exception:
                pass
            raise PolicyDisconnectedError(
                "official OpenPI metadata handshake failed",
                details={"reason": type(exc).__name__},
            ) from exc
        try:
            metadata = _metadata_from_official(metadata_raw, self._expected)
        except Exception:
            # The temporary client is not yet owned by self._client; close it on
            # every metadata parse/compatibility failure.
            try:
                client._ws.close()
            except Exception:
                pass
            raise
        with self._lock:
            # Ownership transfer happens only after all validation succeeds.
            self._client = client
            self._metadata = metadata
            self._server_metadata = dict(metadata_raw) if isinstance(metadata_raw, Mapping) else {}
            self._connected = True
            self._stats = replace(self._stats, connects=self._stats.connects + 1)
        return metadata

    def _protocol_error(self) -> None:
        with self._lock:
            self._stats = replace(
                self._stats, protocol_errors=self._stats.protocol_errors + 1
            )

    def close(self) -> None:
        with self._lock:
            client = self._client
            self._client = None
            self._connected = False
        if client is not None:
            try:
                client._ws.close()
            except Exception:
                pass

    def reconnect(self) -> PolicyMetadata:
        self.close()
        return self.connect()

    def infer(self, observation: Mapping[str, Any], request_sequence_id: int) -> ActionChunk:
        if (
            isinstance(request_sequence_id, bool)
            or not isinstance(request_sequence_id, (int, np.integer))
            or int(request_sequence_id) < 0
        ):
            raise ValidationError(
                "request sequence must be a non-negative integer",
                code=ErrorCode.SEQUENCE_MISMATCH,
                details={"value": repr(request_sequence_id)},
            )
        request_sequence_id = int(request_sequence_id)
        with self._lock:
            if not self._connected or self._client is None or self._metadata is None:
                raise PolicyDisconnectedError("official OpenPI client is not connected")
            if self._last_sequence is not None and request_sequence_id <= self._last_sequence:
                duplicate = request_sequence_id == self._last_sequence
                self._stats = replace(self._stats, protocol_errors=self._stats.protocol_errors + 1)
                raise PolicyContractError(
                    "official OpenPI request sequence is duplicate or regressed",
                    code=ErrorCode.SEQUENCE_REGRESSION,
                    details={
                        "sequence": request_sequence_id,
                        "last": self._last_sequence,
                        "duplicate": duplicate,
                    },
                )
            client = self._client
            metadata = self._metadata
        official_obs = _official_observation(observation, metadata)
        observation_sequence = int(official_obs["meta/sequence_id"])
        if request_sequence_id != observation_sequence:
            raise PolicyContractError(
                "request sequence does not match observation sequence",
                code=ErrorCode.SEQUENCE_MISMATCH,
                details={
                    "request_sequence_id": request_sequence_id,
                    "observation_sequence_id": observation_sequence,
                },
            )
        with self._lock:
            self._stats = replace(self._stats, requests=self._stats.requests + 1)
        started = self._clock.now_ns()
        ws = client._ws
        result: dict = {}

        def invoke() -> None:
            try:
                result["value"] = client.infer(official_obs)
            except Exception as exc:  # noqa: BLE001
                result["error"] = exc

        thread = threading.Thread(target=invoke, name="openpi-infer", daemon=True)
        thread.start()
        thread.join(self._timeout_s)
        if thread.is_alive():
            try:
                ws.close()
            except Exception:
                pass
            thread.join(1.0)
            cleanup_failed = thread.is_alive()
            with self._lock:
                self._connected = False
                self._stats = replace(
                    self._stats,
                    timeouts=self._stats.timeouts + 1,
                    cleanup_failures=self._stats.cleanup_failures + (1 if cleanup_failed else 0),
                )
            details = {
                "endpoint": self.endpoint,
                "timeout_s": self._timeout_s,
                "cleanup_failed": cleanup_failed,
                "thread_alive": thread.is_alive(),
                "threads": [thread.name] if cleanup_failed else [],
            }
            if cleanup_failed:
                raise PolicyCleanupError(
                    "official OpenPI inference cleanup failed; retry is disabled",
                    details=details,
                )
            raise PolicyTimeoutError("official OpenPI inference timed out", details=details)
        if "error" in result:
            exc = result["error"]
            with self._lock:
                if isinstance(exc, (OSError, ConnectionError, ConnectionClosed)):
                    self._connected = False
                    self._stats = replace(self._stats, disconnects=self._stats.disconnects + 1)
                else:
                    self._stats = replace(self._stats, protocol_errors=self._stats.protocol_errors + 1)
            raise PolicyDisconnectedError(
                "official OpenPI inference failed",
                details={"reason": type(exc).__name__},
            ) from exc
        value = result["value"]
        if not isinstance(value, Mapping) or "actions" not in value:
            with self._lock:
                self._stats = replace(self._stats, protocol_errors=self._stats.protocol_errors + 1)
            raise PolicyContractError(
                "official OpenPI response has no actions",
                code=ErrorCode.POLICY_CONTRACT,
            )
        actions = np.asarray(value["actions"])
        if actions.dtype != np.float32:
            self._protocol_error()
            raise ValidationError(
                "official OpenPI actions must be float32",
                code=ErrorCode.INVALID_DTYPE,
                details={"dtype": str(actions.dtype)},
            )
        if actions.ndim != 2 or actions.shape[1] != metadata.action_dim or actions.shape[0] < 1:
            self._protocol_error()
            raise ValidationError(
                "official OpenPI actions must be (H, action_dim)",
                code=ErrorCode.INVALID_SHAPE,
                details={"shape": list(actions.shape), "action_dim": metadata.action_dim},
            )
        expected_shape = (metadata.horizon, metadata.action_dim)
        if actions.shape != expected_shape:
            self._protocol_error()
            raise ValidationError(
                "official OpenPI actions do not match metadata horizon/action_dim",
                code=ErrorCode.SCHEMA_MISMATCH,
                details={"shape": list(actions.shape), "expected": list(expected_shape)},
            )
        if not np.all(np.isfinite(actions)):
            self._protocol_error()
            raise ValidationError("official OpenPI actions contain NaN/Inf", code=ErrorCode.NON_FINITE)
        extras = {key: value[key] for key in value if key != "actions"}
        if not all(isinstance(key, str) for key in extras):
            self._protocol_error()
            raise PolicyContractError(
                "official OpenPI response extras keys must be strings",
                code=ErrorCode.POLICY_CONTRACT,
            )
        chunk = ActionChunk(
            request_sequence_id=int(request_sequence_id),
            observation_capture_time_ns=int(official_obs["meta/capture_time_ns"]),
            action_dt_s=float(metadata.action_dt_s),
            action_frame=metadata.action_frame,
            action_type=metadata.action_type,
            actions=actions,
            extras=extras,
        )
        rtt = self._clock.now_ns() - started
        with self._lock:
            self._last_sequence = int(request_sequence_id)
            self._rtt.append(rtt)
            self._rtt = self._rtt[-256:]
            ordered = sorted(self._rtt)
            self._stats = replace(
                self._stats,
                successes=self._stats.successes + 1,
                last_rtt_ns=rtt,
                rtt_p50_ns=ordered[min(len(ordered) - 1, int(round(0.50 * (len(ordered) - 1))))],
                rtt_p95_ns=ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))],
            )
        return chunk


@dataclass(frozen=True)
class OfficialServerStats:
    connections: int = 0
    requests: int = 0
    successes: int = 0
    errors: int = 0

    def as_dict(self) -> dict:
        return {
            "connections": self.connections,
            "requests": self.requests,
            "successes": self.successes,
            "errors": self.errors,
        }


class OfficialOpenPIServer:
    """Minimal OpenPI wire-compatible dummy server using the pinned package."""

    def __init__(
        self,
        adapter: Optional[PolicyAdapter] = None,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        clock: Optional[Clock] = None,
        metadata_override: Optional[Mapping[str, Any]] = None,
    ) -> None:
        WebsocketClientPolicy, msgpack_numpy, version = _require_openpi()
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("official dummy server is loopback-only")
        self._adapter = adapter or ZeroActionAdapter()
        self._host = host
        self._requested_port = int(port)
        self._clock = clock or MonotonicClock()
        self._msgpack = msgpack_numpy
        self._server = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._bound_port = -1
        self._package_version = version
        self._stats = OfficialServerStats()
        self._metadata_override = dict(metadata_override) if metadata_override is not None else None
        self._connections: set = set()
        self._handler_threads: set = set()
        self._connections_lock = threading.Lock()
        logging.getLogger("websockets.server").setLevel(logging.CRITICAL)

    @property
    def endpoint(self) -> str:
        if self._bound_port < 0:
            raise RuntimeError("server is not started")
        return f"ws://127.0.0.1:{self._bound_port}"

    @property
    def metadata(self) -> dict:
        if self._metadata_override is not None:
            return dict(self._metadata_override)
        m = self._adapter.metadata
        return {
            "name": m.name,
            "version": m.version,
            "horizon": m.horizon,
            "action_dt_s": m.action_dt_s,
            "action_frame": m.action_frame,
            "action_type": m.action_type,
            "delta_semantics": m.delta_semantics,
            "action_dim": m.action_dim,
            "state_layout": m.state_layout.to_dict(),
            "image_size": m.image_size,
        }

    @property
    def stats(self) -> OfficialServerStats:
        return self._stats

    @property
    def identity(self) -> dict:
        return {
            "transport_kind": "official_openpi",
            "server_class": "vla_infra.policy.openpi_official.OfficialOpenPIServer",
            "package": OPENPI_PACKAGE,
            "package_version": self._package_version,
            "endpoint": sanitize_endpoint(self.endpoint) if self._bound_port >= 0 else None,
        }

    def start(self, *, timeout_s: float = 5.0) -> str:
        if self._thread is not None:
            raise RuntimeError("server already started")
        self._thread = threading.Thread(target=self._run, name="openpi-official-dummy", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout_s):
            raise TimeoutError("official OpenPI dummy server did not start")
        return self.endpoint

    def _run(self) -> None:
        packer = self._msgpack.Packer()
        metadata = self.metadata

        def handler(connection):
            handler_thread = threading.current_thread()
            with self._connections_lock:
                self._connections.add(connection)
                self._handler_threads.add(handler_thread)
            self._stats = replace(self._stats, connections=self._stats.connections + 1)
            try:
                connection.send(packer.pack(metadata))
                for raw in connection:
                    try:
                        obs = self._msgpack.unpackb(raw)
                        result = self._adapter.infer(obs)
                        actions = np.asarray(result["actions"])
                        response = {"actions": actions}
                        response.update({k: v for k, v in result.items() if k != "actions"})
                        connection.send(packer.pack(response))
                        self._stats = replace(
                            self._stats,
                            requests=self._stats.requests + 1,
                            successes=self._stats.successes + 1,
                        )
                    except Exception as exc:  # noqa: BLE001
                        self._stats = replace(
                            self._stats,
                            requests=self._stats.requests + 1,
                            errors=self._stats.errors + 1,
                        )
                        connection.send(f"openpi dummy error: {type(exc).__name__}: {exc}")
            except Exception:
                pass
            finally:
                with self._connections_lock:
                    self._connections.discard(connection)
                    self._handler_threads.discard(handler_thread)

        try:
            with serve_sync(handler, self._host, self._requested_port, compression=None, max_size=None) as server:
                self._server = server
                self._bound_port = int(server.socket.getsockname()[1])
                self._ready.set()
                server.serve_forever()
        finally:
            self._ready.set()

    def stop(self, *, timeout_s: float = 5.0) -> None:
        with self._connections_lock:
            connections = list(self._connections)
            self._connections.clear()
        for connection in connections:
            try:
                connection.close()
            except Exception:
                pass
        server = self._server
        if server is not None:
            try:
                server.shutdown()
            except Exception:
                pass
        deadline = time.monotonic() + timeout_s
        thread = self._thread
        if thread is not None:
            thread.join(max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                raise PolicyTimeoutError(
                    "official OpenPI dummy server stop timed out",
                    details={"thread": thread.name, "timeout_s": timeout_s},
                )
        with self._connections_lock:
            handlers = list(self._handler_threads)
        for handler_thread in handlers:
            handler_thread.join(max(0.0, deadline - time.monotonic()))
        alive = [handler.name for handler in handlers if handler.is_alive()]
        if alive:
            raise PolicyTimeoutError(
                "official OpenPI dummy server handler stop timed out",
                details={"threads": alive, "timeout_s": timeout_s},
            )
        self._server = None
        self._thread = None
        self._bound_port = -1


def _construct_official_client(client_class, msgpack_numpy, endpoint: str, timeout_s: float):
    """Create an official WebsocketClientPolicy around an explicitly managed connection.

    No process-global function is patched.  The websockets connection is created
    with explicit proxy/open_timeout, metadata is read with a timeout, and the
    official class is instantiated with a preconnected ``_wait_for_server``
    override.  ``infer`` remains the official implementation.
    """
    import websockets.sync.client as ws_client

    kwargs = {
        "open_timeout": timeout_s,
        "close_timeout": 2.0,
        "compression": None,
        "max_size": None,
    }
    if is_loopback_endpoint(endpoint):
        kwargs["proxy"] = None
    connection = ws_client.connect(endpoint, **kwargs)
    try:
        metadata = msgpack_numpy.unpackb(connection.recv(timeout=timeout_s))
        class Preconnected(client_class):
            def _wait_for_server(self):
                return connection, metadata
        return Preconnected(host=endpoint)
    except Exception:
        try:
            connection.close()
        except Exception:
            pass
        raise


def _host_port(endpoint: str) -> tuple[str, int]:
    parts = urlsplit(endpoint)
    if parts.hostname is None or parts.port is None:
        raise ValueError("endpoint must include host and port")
    return parts.hostname, int(parts.port)


def _official_observation(observation: Mapping[str, Any], metadata: PolicyMetadata) -> dict:
    if not isinstance(observation, Mapping):
        raise ValidationError("official OpenPI observation must be a mapping", code=ErrorCode.SCHEMA_MISMATCH)

    def required(key: str):
        if key not in observation or observation[key] is None:
            raise ValidationError(
                f"official OpenPI observation is missing {key!r}",
                code=ErrorCode.SCHEMA_MISMATCH,
                details={"key": key},
            )
        return observation[key]

    state = np.asarray(required("observation/state"))
    if state.dtype != np.float32:
        raise ValidationError(
            "observation/state must be float32",
            code=ErrorCode.INVALID_DTYPE,
            details={"dtype": str(state.dtype)},
        )
    if state.ndim != 1 or state.shape[0] != metadata.state_layout.dim:
        raise ValidationError(
            "observation/state shape does not match layout",
            code=ErrorCode.INVALID_SHAPE,
            details={"shape": list(state.shape), "expected": [metadata.state_layout.dim]},
        )
    if not np.all(np.isfinite(state)):
        raise ValidationError("observation/state contains NaN/Inf", code=ErrorCode.NON_FINITE)

    images = {}
    for key in ("observation/image", "observation/wrist_image"):
        image = np.asarray(required(key))
        if image.dtype != np.uint8:
            raise ValidationError(
                f"{key} must be uint8",
                code=ErrorCode.INVALID_DTYPE,
                details={"key": key, "dtype": str(image.dtype)},
            )
        if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise ValidationError(
                f"{key} must be non-empty HWC RGB",
                code=ErrorCode.INVALID_SHAPE,
                details={"key": key, "shape": list(image.shape)},
            )
        if image.shape[0] != metadata.image_size or image.shape[1] != metadata.image_size:
            raise ValidationError(
                f"{key} resolution does not match metadata",
                code=ErrorCode.INVALID_SHAPE,
                details={"key": key, "shape": list(image.shape), "expected": metadata.image_size},
            )
        images[key] = image

    prompt = observation.get("prompt", observation.get("task"))
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValidationError("official OpenPI prompt/task must be non-empty", code=ErrorCode.SCHEMA_MISMATCH)

    capture = required("meta/capture_time_ns")
    if isinstance(capture, bool) or not isinstance(capture, (int, np.integer)) or int(capture) <= 0:
        raise ValidationError(
            "meta/capture_time_ns must be a positive integer",
            code=ErrorCode.VALUE_OUT_OF_RANGE,
            details={"value": repr(capture)},
        )
    sequence = required("meta/sequence_id")
    if isinstance(sequence, bool) or not isinstance(sequence, (int, np.integer)) or int(sequence) < 0:
        raise ValidationError(
            "meta/sequence_id must be a non-negative integer",
            code=ErrorCode.SEQUENCE_MISMATCH,
            details={"value": repr(sequence)},
        )
    return {
        "state": state,
        "image": images["observation/image"],
        "wrist_image": images["observation/wrist_image"],
        "prompt": prompt.strip(),
        "meta/capture_time_ns": int(capture),
        "meta/sequence_id": int(sequence),
    }


def _metadata_from_official(raw: Any, expected: Optional[PolicyMetadata]) -> PolicyMetadata:
    if not isinstance(raw, Mapping):
        raise PolicyContractError("official OpenPI metadata is not a mapping", code=ErrorCode.POLICY_CONTRACT)
    required = (
        "horizon",
        "action_dt_s",
        "action_frame",
        "action_type",
        "delta_semantics",
        "state_layout",
        "image_size",
        "action_dim",
    )
    missing = [key for key in required if key not in raw]
    if missing and expected is None:
        raise PolicyContractError(
            "official OpenPI metadata is missing safety-critical fields",
            code=ErrorCode.SCHEMA_MISMATCH,
            details={"missing": missing},
        )

    def value(key: str, cast):
        if key in raw:
            try:
                return cast(raw[key])
            except (TypeError, ValueError) as exc:
                raise PolicyContractError(
                    f"official OpenPI metadata field {key!r} is invalid",
                    code=ErrorCode.SCHEMA_MISMATCH,
                    details={"field": key},
                ) from exc
        assert expected is not None
        return getattr(expected, key)

    layout_value = raw.get("state_layout")
    if layout_value is not None:
        from ..contracts.policy_io import StateLayout
        state_layout = StateLayout.from_dict(layout_value)
    elif expected is not None:
        state_layout = expected.state_layout
    else:  # pragma: no cover - covered by missing check above
        state_layout = CANONICAL_STATE_LAYOUT
    metadata = PolicyMetadata(
        name=str(raw.get("name", "official-openpi-dummy")),
        version=str(raw.get("version", "unknown")),
        horizon=value("horizon", int),
        action_dt_s=value("action_dt_s", float),
        action_frame=value("action_frame", str),
        action_type=value("action_type", str),
        delta_semantics=value("delta_semantics", str),
        state_layout=state_layout,
        image_size=value("image_size", int),
        action_dim=value("action_dim", int),
    )
    if expected is not None:
        issues = expected.compatibility_issues(metadata)
        if issues:
            raise PolicyContractError(
                "official OpenPI server contract is incompatible",
                code=ErrorCode.SCHEMA_MISMATCH,
                details={"fields": list(issues)},
            )
    return metadata


__all__ = [
    "OPENPI_CLIENT_MODULE",
    "OPENPI_PACKAGE",
    "OPENPI_PINNED_VERSION",
    "OfficialClientStats",
    "OfficialOpenPIClient",
    "OfficialOpenPIServer",
    "OpenPIUnavailableError",
]
