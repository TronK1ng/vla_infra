"""Custom-protocol compatibility shims (NOT the official OpenPI adapter).

The wire protocol and security boundaries live in ``client.py``/
``dummy_server.py``.  These classes are the **project's custom protocol**, not
the official ``openpi-client`` package.  They are retained as compatibility
shims and must not be used as official-OpenPI acceptance evidence.

The official pinned adapter lives in ``policy/openpi_official.py``.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..clock import Clock
from ..contracts.action import ActionChunk
from .adapters.base import PolicyAdapter, PolicyMetadata, ZeroActionAdapter
from .client import ClientStats, WebsocketPolicyClient
from .dummy_server import DummyPolicyServer, ServerStats


class CustomPolicyClient:
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
        name: str = "m4-openpi-client",
    ) -> None:
        self._client = WebsocketPolicyClient(
            endpoint,
            timeout_s=timeout_s,
            connect_timeout_s=connect_timeout_s,
            codec=codec,
            auth_token=auth_token,
            proxy=proxy,
            clock=clock,
            expected_metadata=expected_metadata,
            name=name,
        )
        self.endpoint = endpoint
        self.name = name

    @property
    def connected(self) -> bool:
        return self._client.connected

    @property
    def metadata(self) -> PolicyMetadata:
        return self._client.metadata

    @property
    def stats(self) -> ClientStats:
        return self._client.stats

    def connect(self) -> PolicyMetadata:
        return self._client.connect()

    def infer(self, observation: Mapping[str, Any], request_sequence_id: int) -> ActionChunk:
        return self._client.infer(observation, request_sequence_id)

    def close(self) -> None:
        self._client.close()


class CustomDummyPolicyServer:
    def __init__(
        self,
        adapter: Optional[PolicyAdapter] = None,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        codec: str = "auto",
        auth_token: Optional[str] = None,
        allow_non_loopback: bool = False,
        clock: Optional[Clock] = None,
    ) -> None:
        self._server = DummyPolicyServer(
            adapter or ZeroActionAdapter(),
            host=host,
            port=port,
            codec=codec,
            auth_token=auth_token,
            allow_non_loopback=allow_non_loopback,
            clock=clock,
        )

    @property
    def endpoint(self) -> str:
        return self._server.endpoint

    @property
    def metadata(self) -> PolicyMetadata:
        return self._server.metadata

    @property
    def stats(self) -> ServerStats:
        return self._server.stats

    def start(self, *, timeout_s: float = 5.0) -> str:
        return self._server.start(timeout_s=timeout_s)

    def stop(self, *, timeout_s: float = 5.0) -> None:
        self._server.stop(timeout_s=timeout_s)


# Backward-compatible aliases; docs/tests must prefer the explicit Custom* names.
OpenPIClient = CustomPolicyClient
OpenPIServer = CustomDummyPolicyServer

__all__ = [
    "CustomDummyPolicyServer",
    "CustomPolicyClient",
    "OpenPIClient",
    "OpenPIServer",
]
