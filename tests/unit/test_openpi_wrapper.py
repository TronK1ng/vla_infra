"""M4 OpenPI-shaped wrapper over the loopback dummy policy boundary."""

from __future__ import annotations

import numpy as np
import pytest

from vla_infra.contracts.policy_io import CANONICAL_STATE_LAYOUT, build_policy_observation
from vla_infra.errors import ErrorCode, PolicyContractError, PolicyDisconnectedError
from vla_infra.policy.adapters.base import ZeroActionAdapter
from vla_infra.policy.openpi import OpenPIClient, OpenPIServer
from vla_infra.testing import make_observation


def _payload(sequence_id: int = 0) -> dict:
    observation = make_observation(sequence_id=sequence_id)
    return build_policy_observation(observation, CANONICAL_STATE_LAYOUT, image_size=64)


def test_openpi_wrapper_zero_action_round_trip() -> None:
    adapter = ZeroActionAdapter(horizon=4, action_dt_s=0.05, gripper_normalized=0.5, image_size=64)
    server = OpenPIServer(adapter, port=0)
    endpoint = server.start()
    client = OpenPIClient(endpoint, timeout_s=2.0, connect_timeout_s=2.0)
    try:
        metadata = client.connect()
        assert metadata.horizon == 4
        chunk = client.infer(_payload(0), 0)
        assert chunk.actions.shape == (4, 7)
        assert np.all(chunk.actions[:, :6] == 0.0)
        assert np.allclose(chunk.actions[:, 6], 0.5)
        assert client.stats.successes == 1
        assert client.stats.last_rtt_ns is not None
        assert server.stats.successes == 1
    finally:
        client.close()
        server.stop()


def test_openpi_wrapper_close_makes_inference_explicitly_invalid() -> None:
    server = OpenPIServer(port=0)
    endpoint = server.start()
    client = OpenPIClient(endpoint, timeout_s=1.0, connect_timeout_s=1.0)
    try:
        client.connect()
        client.close()
        with pytest.raises(PolicyDisconnectedError):
            client.infer(_payload(), 1)
    finally:
        server.stop()


def test_openpi_wrapper_sequence_regression_is_rejected() -> None:
    server = OpenPIServer(ZeroActionAdapter(image_size=64), port=0)
    endpoint = server.start()
    client = OpenPIClient(endpoint, timeout_s=2.0, connect_timeout_s=2.0)
    try:
        client.connect()
        client.infer(_payload(1), 1)
        with pytest.raises(PolicyContractError) as duplicate:
            client.infer(_payload(1), 1)
        assert duplicate.value.details["remote_code"] == ErrorCode.SEQUENCE_REGRESSION.value
    finally:
        client.close()
    regression_client = OpenPIClient(endpoint, timeout_s=2.0, connect_timeout_s=2.0)
    try:
        regression_client.connect()
        regression_client.infer(_payload(5), 5)
        with pytest.raises(PolicyContractError) as regression:
            regression_client.infer(_payload(4), 4)
        assert regression.value.details["remote_code"] == ErrorCode.SEQUENCE_REGRESSION.value
        assert server.stats.errors >= 2
    finally:
        regression_client.close()
        server.stop()
