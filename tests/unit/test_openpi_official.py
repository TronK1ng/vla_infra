"""Official pinned OpenPI client adapter tests (optional dependency)."""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

VENDOR = Path(__file__).resolve().parents[2] / ".openpi_vendor"
if VENDOR.exists() and str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

pytest.importorskip("openpi_client")

from vla_infra.config import load_app_config
from vla_infra.contracts.policy_io import CANONICAL_STATE_LAYOUT, build_policy_observation
from vla_infra.errors import (
    ErrorCode,
    PolicyContractError,
    PolicyDisconnectedError,
    PolicyTimeoutError,
    ValidationError,
)
from vla_infra.policy.adapters.base import ZeroActionAdapter
from vla_infra.policy.openpi_official import (
    OPENPI_PINNED_VERSION,
    OfficialOpenPIClient,
    OfficialOpenPIServer,
)
from vla_infra.testing import make_observation

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def _app():
    return load_app_config(
        station_path=CONFIGS / "station.example.yaml",
        safety_path=CONFIGS / "safety.example.yaml",
        policy_path=CONFIGS / "policy.example.yaml",
    )


def _adapter(app, *, slow: bool = False):
    base = ZeroActionAdapter(
        horizon=app.policy.expected_horizon,
        action_dt_s=app.policy.expected_action_dt_s,
        image_size=app.policy.image_size,
    )
    if not slow:
        return base

    class Slow:
        metadata = base.metadata

        def infer(self, observation):
            time.sleep(0.4)
            return base.infer(observation)

    return Slow()


def _payload(app, sequence_id: int = 0):
    return build_policy_observation(
        make_observation(sequence_id=sequence_id),
        CANONICAL_STATE_LAYOUT,
        image_size=app.policy.image_size,
    )


def _client(endpoint: str, app, **kwargs) -> OfficialOpenPIClient:
    values = {
        "timeout_s": 2.0,
        "connect_timeout_s": 2.0,
        "expected_metadata": app.policy.expected_metadata(),
    }
    values.update(kwargs)
    return OfficialOpenPIClient(endpoint, **values)


def _server(app, *, slow: bool = False, port: int = 0) -> OfficialOpenPIServer:
    return OfficialOpenPIServer(_adapter(app, slow=slow), port=port)


def test_official_package_identity_and_zero_round_trip() -> None:
    import importlib.metadata

    app = _app()
    assert importlib.metadata.version("openpi-client") == OPENPI_PINNED_VERSION
    server = _server(app)
    endpoint = server.start()
    client = _client(endpoint, app)
    try:
        metadata = client.connect()
        assert metadata.horizon == app.policy.expected_horizon
        chunk = client.infer(_payload(app, 0), 0)
        assert chunk.actions.shape == (app.policy.expected_horizon, 7)
        assert np.allclose(chunk.actions[:, :6], 0.0)
        assert np.allclose(chunk.actions[:, 6], 0.5)
        assert client.stats.successes == 1
        assert client.stats.last_rtt_ns is not None
        assert client.identity["package"] == "openpi-client"
        assert client.identity["package_version"] == OPENPI_PINNED_VERSION
    finally:
        client.close()
        server.stop()


def test_official_duplicate_and_regression_sequence_errors() -> None:
    app = _app()
    server = _server(app)
    endpoint = server.start()
    client = _client(endpoint, app)
    try:
        client.connect()
        client.infer(_payload(app, 5), 5)
        with pytest.raises(PolicyContractError) as duplicate:
            client.infer(_payload(app, 5), 5)
        assert duplicate.value.code is ErrorCode.SEQUENCE_REGRESSION
        assert duplicate.value.details["duplicate"] is True
        with pytest.raises(PolicyContractError) as regression:
            client.infer(_payload(app, 4), 4)
        assert regression.value.code is ErrorCode.SEQUENCE_REGRESSION
        assert regression.value.details["duplicate"] is False
    finally:
        client.close()
        server.stop()


def test_official_infer_timeout_is_bounded() -> None:
    app = _app()
    server = _server(app, slow=True)
    endpoint = server.start()
    client = _client(endpoint, app, timeout_s=0.05, connect_timeout_s=2.0)
    try:
        client.connect()
        with pytest.raises(PolicyTimeoutError):
            client.infer(_payload(app, 0), 0)
        assert client.stats.timeouts >= 1
        assert client.connected is False
    finally:
        client.close()
        server.stop()


def test_official_connect_timeout_is_bounded() -> None:
    app = _app()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    accepted = []

    def accept_and_hold():
        try:
            connection, _ = listener.accept()
            accepted.append(connection)
            time.sleep(2.0)
        except OSError:
            pass

    thread = threading.Thread(target=accept_and_hold, daemon=True)
    thread.start()
    client = _client(f"ws://127.0.0.1:{port}", app, connect_timeout_s=0.1)
    try:
        with pytest.raises(PolicyTimeoutError):
            client.connect()
    finally:
        client.close()
        listener.close()
        for connection in accepted:
            connection.close()


def test_official_disconnect_is_explicit_and_reconnect_works() -> None:
    app = _app()
    server = _server(app)
    endpoint = server.start()
    port = int(endpoint.rsplit(":", 1)[1])
    client = _client(endpoint, app)
    try:
        client.connect()
        server.stop()
        with pytest.raises((PolicyDisconnectedError, PolicyTimeoutError)):
            client.infer(_payload(app, 0), 0)
        assert client.connected is False
        replacement = _server(app, port=port)
        replacement.start()
        try:
            metadata = client.reconnect()
            assert metadata.horizon == app.policy.expected_horizon
            chunk = client.infer(_payload(app, 1), 1)
            assert chunk.actions.shape[1] == 7
        finally:
            replacement.stop()
    finally:
        client.close()
        server.stop()


def test_official_connect_timeout_leaves_no_threads_and_is_repeatable() -> None:
    app = _app()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(5)
    port = listener.getsockname()[1]
    accepted = []

    def accept_and_hold():
        while True:
            try:
                connection, _ = listener.accept()
                accepted.append(connection)
            except OSError:
                return

    thread = threading.Thread(target=accept_and_hold, daemon=True)
    thread.start()
    try:
        for _ in range(3):
            client = _client(f"ws://127.0.0.1:{port}", app, connect_timeout_s=0.1)
            with pytest.raises(PolicyTimeoutError):
                client.connect()
            assert not [t for t in threading.enumerate() if t.name == "openpi-connect"]
            assert client.connected is False
            client.close()
    finally:
        listener.close()
        for connection in accepted:
            connection.close()


def test_official_infer_timeout_leaves_no_threads() -> None:
    app = _app()
    server = _server(app, slow=True)
    endpoint = server.start()
    client = _client(endpoint, app, timeout_s=0.05, connect_timeout_s=2.0)
    try:
        client.connect()
        with pytest.raises(PolicyTimeoutError) as exc:
            client.infer(_payload(app, 0), 0)
        assert exc.value.details["thread_alive"] is False
        assert not [t for t in threading.enumerate() if t.name == "openpi-infer"]
    finally:
        client.close()
        server.stop()


def test_official_observation_missing_or_invalid_never_reaches_server() -> None:
    app = _app()
    server = _server(app)
    endpoint = server.start()
    client = _client(endpoint, app)
    try:
        client.connect()
        baseline = server.stats.requests
        cases = []
        cases.append(("missing-state", {k: v for k, v in _payload(app).items() if k != "observation/state"}))
        bad_state = _payload(app)
        bad_state["observation/state"] = np.zeros((21,), dtype=np.float32)
        cases.append(("state-shape", bad_state))
        bad_dtype = _payload(app)
        bad_dtype["observation/state"] = bad_dtype["observation/state"].astype(np.float64)
        cases.append(("state-dtype", bad_dtype))
        bad_finite = _payload(app)
        state = bad_finite["observation/state"].copy()
        state[0] = np.nan
        bad_finite["observation/state"] = state
        cases.append(("state-nan", bad_finite))
        bad_image = _payload(app)
        bad_image["observation/image"] = np.zeros((1, 1, 3), dtype=np.uint8)
        cases.append(("image-shape", bad_image))
        bad_image_dtype = _payload(app)
        bad_image_dtype["observation/image"] = bad_image_dtype["observation/image"].astype(np.float32)
        cases.append(("image-dtype", bad_image_dtype))
        bad_prompt = _payload(app)
        bad_prompt["prompt"] = "   "
        cases.append(("prompt-empty", bad_prompt))
        bad_capture = _payload(app)
        bad_capture["meta/capture_time_ns"] = 0
        cases.append(("capture", bad_capture))
        bad_sequence = _payload(app)
        bad_sequence["meta/sequence_id"] = -1
        cases.append(("sequence", bad_sequence))
        for name, payload in cases:
            with pytest.raises((ValidationError, PolicyContractError)):
                client.infer(payload, 0)
            assert server.stats.requests == baseline, name
        assert client.stats.requests == 0
    finally:
        client.close()
        server.stop()


def test_official_action_dtype_shape_and_finite_are_enforced() -> None:
    app = _app()
    zero = ZeroActionAdapter(
        horizon=app.policy.expected_horizon,
        action_dt_s=app.policy.expected_action_dt_s,
        image_size=app.policy.image_size,
    )

    class BadAdapter:
        metadata = zero.metadata

        def __init__(self, actions):
            self.actions = actions

        def infer(self, observation):
            return {"actions": self.actions}

    cases = [
        ("dtype-int", np.zeros((10, 7), dtype=np.int64), ErrorCode.INVALID_DTYPE),
        ("dtype-f64", np.zeros((10, 7), dtype=np.float64), ErrorCode.INVALID_DTYPE),
        ("shape", np.zeros((10, 6), dtype=np.float32), ErrorCode.INVALID_SHAPE),
        ("short-horizon", np.zeros((1, 7), dtype=np.float32), ErrorCode.SCHEMA_MISMATCH),
        ("long-horizon", np.zeros((20, 7), dtype=np.float32), ErrorCode.SCHEMA_MISMATCH),
        ("nan", np.full((10, 7), np.nan, dtype=np.float32), ErrorCode.NON_FINITE),
    ]
    for name, actions, code in cases:
        server = OfficialOpenPIServer(BadAdapter(actions), port=0)
        endpoint = server.start()
        client = _client(endpoint, app)
        try:
            client.connect()
            with pytest.raises(ValidationError) as exc:
                client.infer(_payload(app), 0)
            assert exc.value.code is code, name
            assert client.stats.protocol_errors == 1, name
        finally:
            client.close()
            server.stop()


def test_official_server_stop_does_not_false_report_stopped() -> None:
    app = _app()
    entered = threading.Event()
    release = threading.Event()
    zero = ZeroActionAdapter(
        horizon=app.policy.expected_horizon,
        action_dt_s=app.policy.expected_action_dt_s,
        image_size=app.policy.image_size,
    )

    class BlockingAdapter:
        metadata = zero.metadata

        def infer(self, observation):
            entered.set()
            release.wait(5.0)
            return zero.infer(observation)

    server = OfficialOpenPIServer(BlockingAdapter(), port=0)
    endpoint = server.start()
    client = _client(endpoint, app)
    worker = None
    try:
        client.connect()

        def call_infer():
            try:
                client.infer(_payload(app), 0)
            except Exception:
                pass

        worker = threading.Thread(target=call_infer, daemon=True)
        worker.start()
        assert entered.wait(2.0)
        with pytest.raises(PolicyTimeoutError):
            server.stop(timeout_s=0.05)
        assert server._thread is not None
        assert any(thread.is_alive() for thread in server._handler_threads)
        assert server.endpoint
    finally:
        release.set()
        if worker is not None:
            worker.join(2.0)
        client.close()
        server.stop(timeout_s=2.0)


def test_official_operation_does_not_mutate_proxy_environment() -> None:
    app = _app()
    keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "NO_PROXY", "no_proxy")
    before = {key: os.environ.get(key) for key in keys}
    server = _server(app)
    endpoint = server.start()
    clients = [_client(endpoint, app) for _ in range(2)]
    try:
        for client in clients:
            client.connect()
        threads = [threading.Thread(target=lambda c=c: c.infer(_payload(app), 0)) for c in clients]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(2.0)
        assert all(client.stats.successes == 1 for client in clients)
        assert {key: os.environ.get(key) for key in keys} == before
    finally:
        for client in clients:
            client.close()
        server.stop()


def test_official_infer_cleanup_failure_is_non_retryable() -> None:
    from vla_infra.policy.openpi_official import OfficialClientStats

    app = _app()
    entered = threading.Event()
    release = threading.Event()

    class NoCloseSocket:
        def close(self):
            pass

    class BlockingClient:
        def __init__(self):
            self._ws = NoCloseSocket()

        def infer(self, observation):
            entered.set()
            release.wait(2.0)
            return {"actions": np.zeros((app.policy.expected_horizon, 7), dtype=np.float32)}

    client = OfficialOpenPIClient(
        "ws://127.0.0.1:1",
        timeout_s=0.05,
        connect_timeout_s=0.1,
        expected_metadata=app.policy.expected_metadata(),
    )
    client._client = BlockingClient()
    client._metadata = app.policy.expected_metadata()
    client._connected = True
    client._stats = OfficialClientStats()
    try:
        with pytest.raises(Exception) as exc:
            client.infer(_payload(app), 0)
        from vla_infra.errors import PolicyCleanupError

        assert isinstance(exc.value, PolicyCleanupError)
        assert exc.value.details["cleanup_failed"] is True
        assert exc.value.details["thread_alive"] is True
        assert "openpi-infer" in exc.value.details["threads"]
        assert client.stats.cleanup_failures == 1
    finally:
        release.set()
        for thread in threading.enumerate():
            if thread.name == "openpi-infer":
                thread.join(2.0)
        client.close()


def test_official_metadata_missing_safety_fields_fail_closed() -> None:
    from vla_infra.policy.openpi_official import _metadata_from_official

    app = _app()
    with pytest.raises(PolicyContractError) as exc:
        _metadata_from_official({}, None)
    assert exc.value.code is ErrorCode.SCHEMA_MISMATCH
    assert "horizon" in exc.value.details["missing"]
    assert "action_frame" in exc.value.details["missing"]
    explicit = _metadata_from_official({}, app.policy.expected_metadata())
    assert explicit.horizon == app.policy.expected_horizon
    assert explicit.action_frame == app.policy.expected_action_frame


def test_official_request_and_observation_sequence_are_bound() -> None:
    app = _app()
    server = _server(app)
    endpoint = server.start()
    client = _client(endpoint, app)
    try:
        client.connect()
        baseline = server.stats.requests
        with pytest.raises(PolicyContractError) as exc:
            client.infer(_payload(app, 0), 5)
        assert exc.value.code is ErrorCode.SEQUENCE_MISMATCH
        assert exc.value.details["request_sequence_id"] == 5
        assert exc.value.details["observation_sequence_id"] == 0
        assert server.stats.requests == baseline
        assert client.stats.requests == 0
        for bad in (True, 1.5, -1):
            with pytest.raises(ValidationError) as bad_exc:
                client.infer(_payload(app, 0), bad)
            assert bad_exc.value.code is ErrorCode.SEQUENCE_MISMATCH
        assert server.stats.requests == baseline
    finally:
        client.close()
        server.stop()


def test_official_metadata_failure_closes_temporary_connection_repeatedly() -> None:
    app = _app()
    expected = app.policy.expected_metadata()
    base_meta = {
        "name": expected.name,
        "version": expected.version,
        "horizon": 4,  # expected 10 -> compatibility failure
        "action_dt_s": expected.action_dt_s,
        "action_frame": expected.action_frame,
        "action_type": expected.action_type,
        "delta_semantics": expected.delta_semantics,
        "state_layout": expected.state_layout.to_dict(),
        "image_size": expected.image_size,
        "action_dim": expected.action_dim,
    }
    cases = [
        ("mismatch", base_meta),
        ("missing", {"name": "broken", "version": "0"}),
        ("invalid", {**base_meta, "horizon": "not-an-int"}),
    ]
    for name, metadata in cases:
        server = OfficialOpenPIServer(
            ZeroActionAdapter(horizon=10, action_dt_s=0.05, image_size=app.policy.image_size),
            port=0,
            metadata_override=metadata,
        )
        endpoint = server.start()
        for _ in range(3):
            client = _client(endpoint, app, expected_metadata=None if name == "missing" else app.policy.expected_metadata())
            with pytest.raises((PolicyContractError, ValidationError)):
                client.connect()
            assert client.connected is False
            assert client._client is None
            assert not [t for t in threading.enumerate() if t.name in ("openpi-connect", "openpi-infer")]
            for _ in range(100):
                if not server._connections:
                    break
                time.sleep(0.005)
            assert not server._connections, name
            client.close()
        server.stop()


def test_official_missing_actions_increments_protocol_errors() -> None:
    from vla_infra.policy.openpi_official import OfficialClientStats

    app = _app()

    class MissingActionsClient:
        class _ws:
            @staticmethod
            def close():
                pass

        def infer(self, observation):
            return {"server_timing": {"infer_ms": 1.0}}

    client = OfficialOpenPIClient("ws://127.0.0.1:1", expected_metadata=app.policy.expected_metadata())
    client._client = MissingActionsClient()
    client._metadata = app.policy.expected_metadata()
    client._connected = True
    client._stats = OfficialClientStats()
    try:
        with pytest.raises(PolicyContractError) as exc:
            client.infer(_payload(app), 0)
        assert exc.value.code is ErrorCode.POLICY_CONTRACT
        assert client.stats.protocol_errors == 1
    finally:
        client.close()
