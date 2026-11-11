"""Dummy policy server/client over a real loopback WebSocket."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from websockets.sync.server import serve

from vla_infra.contracts.action import ActionChunk
from vla_infra.contracts.policy_io import CANONICAL_STATE_LAYOUT, build_policy_observation
from vla_infra.errors import (
    ConfigError,
    ErrorCode,
    PolicyContractError,
    PolicyDisconnectedError,
    PolicyTimeoutError,
    ValidationError,
)
from vla_infra.policy.adapters.base import PolicyAdapter, PolicyMetadata, ZeroActionAdapter
from vla_infra.policy.client import (
    WebsocketPolicyClient,
    is_loopback_endpoint,
    resolve_proxy,
    sanitize_endpoint,
    sanitize_proxy,
)
from vla_infra.policy.dummy_server import DummyPolicyServer
from vla_infra.policy.wire import (
    build_metadata_message,
    build_result_message,
    get_codec,
    observation_capture_time,
    validate_request,
)
from vla_infra.testing import make_observation


def _payload(image_size: int = 64) -> dict:
    return build_policy_observation(make_observation(), CANONICAL_STATE_LAYOUT, image_size=image_size)


def _client(endpoint: str, **kwargs) -> WebsocketPolicyClient:
    options = {"timeout_s": 2.0, "connect_timeout_s": 2.0}
    options.update(kwargs)
    return WebsocketPolicyClient(endpoint, **options)


@pytest.fixture()
def zero_server():
    adapter = ZeroActionAdapter(horizon=4, action_dt_s=0.05, gripper_normalized=0.5, image_size=64)
    server = DummyPolicyServer(adapter, port=0)
    endpoint = server.start()
    try:
        yield server, endpoint
    finally:
        server.stop()


def test_metadata_handshake_and_result(zero_server) -> None:
    server, endpoint = zero_server
    client = _client(endpoint)
    try:
        metadata = client.connect()
        assert metadata.horizon == 4
        assert metadata.state_layout.name == "canonical_v1"
        chunk = client.infer(_payload(), 0)
        assert chunk.request_sequence_id == 0
        assert chunk.actions.shape == (4, 7)
        assert chunk.actions.dtype == np.float32
        assert chunk.server_receive_time_ns is not None
        assert chunk.server_send_time_ns >= chunk.server_receive_time_ns
        assert chunk.extras["policy_mode"] == "dummy_zero"
        assert client.stats.successes == 1
        assert client.stats.last_rtt_ns is not None
        assert server.stats.successes == 1
    finally:
        client.close()


def test_incompatible_metadata_is_refused_at_handshake(zero_server) -> None:
    _, endpoint = zero_server
    expected = PolicyMetadata(
        name="expected",
        version="0",
        horizon=5,  # server declares 4
        action_dt_s=0.05,
        action_frame="base",
        action_type="delta_tcp",
    )
    client = _client(endpoint, expected_metadata=expected)
    with pytest.raises(PolicyContractError) as exc:
        client.connect()
    assert exc.value.code is ErrorCode.SCHEMA_MISMATCH
    assert not client.connected


def test_duplicate_sequence_rejected(zero_server) -> None:
    server, endpoint = zero_server
    client = _client(endpoint)
    try:
        client.connect()
        client.infer(_payload(), 7)
        with pytest.raises(PolicyContractError) as exc:
            client.infer(_payload(), 7)
        assert exc.value.details["remote_code"] == ErrorCode.SEQUENCE_REGRESSION.value
        assert server.stats.errors == 1
    finally:
        client.close()


def test_malformed_observation_returns_error(zero_server) -> None:
    _, endpoint = zero_server
    client = _client(endpoint)
    try:
        client.connect()
        payload = _payload()
        payload["observation/image"] = np.zeros((32, 32, 3), dtype=np.uint8)  # wrong size
        with pytest.raises(PolicyContractError) as exc:
            client.infer(payload, 1)
        assert exc.value.details["remote_code"] == ErrorCode.INVALID_SHAPE.value
    finally:
        client.close()


def test_adapter_output_dtype_violation_is_reported() -> None:
    class BadAdapter(PolicyAdapter):
        @property
        def metadata(self) -> PolicyMetadata:
            return PolicyMetadata(
                name="bad",
                version="0",
                horizon=2,
                action_dt_s=0.05,
                action_frame="base",
                action_type="delta_tcp",
                image_size=64,
            )

        def infer(self, observation):
            return {"actions": np.zeros((2, 7), dtype=np.float64)}

    server = DummyPolicyServer(BadAdapter(), port=0)
    endpoint = server.start()
    client = _client(endpoint)
    try:
        client.connect()
        with pytest.raises(PolicyContractError) as exc:
            client.infer(_payload(), 0)
        assert exc.value.details["remote_code"] == ErrorCode.INVALID_DTYPE.value
        assert server.stats.errors == 1
    finally:
        client.close()
        server.stop()


def test_timeout_disconnects_client() -> None:
    class SlowAdapter(ZeroActionAdapter):
        def infer(self, observation):
            time.sleep(0.4)
            return super().infer(observation)

    server = DummyPolicyServer(SlowAdapter(horizon=2, image_size=64), port=0)
    endpoint = server.start()
    client = WebsocketPolicyClient(endpoint, timeout_s=0.05, connect_timeout_s=1.0)
    try:
        client.connect()
        with pytest.raises(PolicyTimeoutError):
            client.infer(_payload(), 0)
        assert client.stats.timeouts == 1
        assert not client.connected
    finally:
        client.close()
        server.stop()


def test_auth_token_required_and_accepted() -> None:
    adapter = ZeroActionAdapter(horizon=2, image_size=64)
    server = DummyPolicyServer(adapter, port=0, auth_token="correct-horse-battery")
    endpoint = server.start()
    try:
        unauthorized = _client(endpoint)
        with pytest.raises(PolicyDisconnectedError):
            unauthorized.connect()
        assert not unauthorized.connected
        authorized = _client(endpoint, auth_token="correct-horse-battery")
        try:
            authorized.connect()
            chunk = authorized.infer(_payload(), 0)
            assert chunk.horizon == 2
        finally:
            authorized.close()
    finally:
        server.stop()


def test_non_loopback_bind_requires_explicit_optin_and_token() -> None:
    adapter = ZeroActionAdapter(horizon=2, image_size=64)
    with pytest.raises(ConfigError):
        DummyPolicyServer(adapter, host="0.0.0.0", port=0)
    with pytest.raises(ConfigError):
        DummyPolicyServer(adapter, host="0.0.0.0", port=0, allow_non_loopback=True)
    with pytest.raises(ConfigError):
        DummyPolicyServer(adapter, port=0, auth_token="short")


def test_server_stop_causes_disconnect() -> None:
    adapter = ZeroActionAdapter(horizon=2, image_size=64)
    server = DummyPolicyServer(adapter, port=0)
    endpoint = server.start()
    client = _client(endpoint)
    client.connect()
    server.stop()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            client.infer(_payload(), 1)
        except (PolicyDisconnectedError, PolicyContractError):
            break
        time.sleep(0.02)
    assert not client.connected
    client.close()


def test_sanitize_endpoint_removes_credentials_and_token() -> None:
    assert sanitize_endpoint("ws://user:pass@example.com:8765/path?token=abc") == "ws://example.com:8765/path"
    assert sanitize_endpoint("ws://127.0.0.1:8765") == "ws://127.0.0.1:8765"


def test_client_refuses_non_ws_endpoint() -> None:
    with pytest.raises(ConfigError):
        WebsocketPolicyClient("http://127.0.0.1:8765")


def test_client_metadata_unavailable_before_connect() -> None:
    with pytest.raises(PolicyDisconnectedError):
        _client("ws://127.0.0.1:1").metadata


def test_connection_failure_is_clear() -> None:
    client = _client("ws://127.0.0.1:1", connect_timeout_s=0.2)
    with pytest.raises(PolicyDisconnectedError) as exc:
        client.connect()
    assert exc.value.code is ErrorCode.POLICY_DISCONNECTED
    assert client.stats.connection_failures == 1


def test_validation_error_on_sequence_metadata() -> None:
    payload = _payload()
    payload["meta/sequence_id"] = -1
    with pytest.raises(ValidationError):
        # Local validation is also available before sending.
        from vla_infra.contracts.validation import validate_policy_observation

        validate_policy_observation(payload, expected_layout=CANONICAL_STATE_LAYOUT, expected_image_size=64)


class _TamperingCaptureServer:
    """Rogue server that echoes the wrong observation capture timestamp."""

    def __init__(self) -> None:
        self._adapter = ZeroActionAdapter(horizon=2, image_size=64)
        self._codec = get_codec("auto")
        self._server = None
        self._thread = None
        self._ready = threading.Event()
        self._port = -1

    def start(self) -> str:
        self._server = serve(self._handler, "127.0.0.1", 0, compression=None)
        self._port = int(self._server.socket.getsockname()[1])
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self._ready.set()
        return f"ws://127.0.0.1:{self._port}"

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None

    def _handler(self, connection) -> None:
        metadata = self._adapter.metadata
        connection.send(
            self._codec.encode(build_metadata_message(metadata, server_time_ns=1))
        )
        try:
            for raw in connection:
                request = self._codec.decode(raw)
                sequence, observation = validate_request(request)
                actions = np.zeros((metadata.horizon, 7), dtype=np.float32)
                actions[:, 6] = 0.5
                chunk = ActionChunk(
                    request_sequence_id=sequence,
                    # Wrong on purpose: server should echo the request capture time.
                    observation_capture_time_ns=observation_capture_time(observation) + 1,
                    action_dt_s=metadata.action_dt_s,
                    action_frame=metadata.action_frame,
                    action_type=metadata.action_type,
                    actions=actions,
                    server_receive_time_ns=1,
                    server_send_time_ns=2,
                )
                connection.send(
                    self._codec.encode(
                        build_result_message(
                            chunk, server_receive_time_ns=1, server_send_time_ns=2
                        )
                    )
                )
        except Exception:
            pass


def test_proxy_resolution_is_explicit_and_loopback_is_direct() -> None:
    assert is_loopback_endpoint("ws://127.0.0.1:8765")
    assert is_loopback_endpoint("ws://localhost:8765")
    assert is_loopback_endpoint("wss://[::1]:8765")
    assert not is_loopback_endpoint("ws://192.168.1.20:8765")
    assert resolve_proxy("ws://127.0.0.1:8765", "env") is None
    assert resolve_proxy("ws://example.com:8765", "none") is None
    assert resolve_proxy("ws://example.com:8765", "env") is True
    assert resolve_proxy("ws://example.com:8765", "http://proxy.local:3128") == "http://proxy.local:3128"
    with pytest.raises(ConfigError):
        resolve_proxy("ws://example.com:8765", "bogus")
    assert sanitize_proxy("http://user:secret@proxy.local:3128") == "http://proxy.local:3128"


def test_loopback_ignores_forced_system_proxy(zero_server, monkeypatch) -> None:
    _, endpoint = zero_server
    for name in (
        "all_proxy",
        "ALL_PROXY",
        "http_proxy",
        "HTTP_PROXY",
        "https_proxy",
        "HTTPS_PROXY",
    ):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
    client = _client(endpoint)
    try:
        client.connect()
        chunk = client.infer(_payload(), 0)
        assert chunk.request_sequence_id == 0
    finally:
        client.close()


def test_capture_timestamp_tampering_is_rejected() -> None:
    server = _TamperingCaptureServer()
    endpoint = server.start()
    client = _client(endpoint)
    try:
        client.connect()
        with pytest.raises(PolicyContractError) as exc:
            client.infer(_payload(), 0)
        assert exc.value.code is ErrorCode.CAPTURE_MISMATCH
        assert not client.connected
    finally:
        client.close()
        server.stop()


def test_explicit_bad_proxy_is_a_normalized_connection_error() -> None:
    client = WebsocketPolicyClient(
        "ws://example.com:80",
        proxy="http://127.0.0.1:9",
        timeout_s=0.5,
        connect_timeout_s=0.5,
    )
    with pytest.raises(PolicyDisconnectedError) as exc:
        client.connect()
    assert exc.value.code is ErrorCode.POLICY_DISCONNECTED
    assert client.stats.connection_failures == 1
