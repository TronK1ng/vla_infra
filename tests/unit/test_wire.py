"""Wire codec and message validation."""

from __future__ import annotations

import numpy as np
import pytest

from vla_infra.contracts.constants import POLICY_KEY_CAPTURE_TIME, POLICY_KEY_SEQUENCE
from vla_infra.errors import ErrorCode, PolicyContractError, ValidationError
from vla_infra.policy.adapters.base import ZeroActionAdapter
from vla_infra.policy.wire import (
    JsonCodec,
    build_error_message,
    build_metadata_message,
    build_request,
    build_result_message,
    get_codec,
    parse_response,
    validate_metadata_message,
    validate_request,
)
from vla_infra.testing import make_chunk


def test_json_codec_roundtrip_arrays_and_nesting() -> None:
    codec = JsonCodec()
    value = {
        "image": np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3),
        "state": np.linspace(0, 1, 5, dtype=np.float32),
        "nested": {"list": [1, 2.5, "x", None], "flag": True},
    }
    decoded = codec.decode(codec.encode(value))
    assert decoded["image"].dtype == np.uint8
    assert np.array_equal(decoded["image"], value["image"])
    assert decoded["state"].dtype == np.float32
    assert np.array_equal(decoded["state"], value["state"])
    assert decoded["nested"] == value["nested"]


def test_msgpack_codec_roundtrip_when_available() -> None:
    pytest.importorskip("msgpack")
    from vla_infra.policy.wire import MsgpackCodec

    codec = MsgpackCodec()
    value = {"actions": np.zeros((3, 7), dtype=np.float32)}
    decoded = codec.decode(codec.encode(value))
    assert decoded["actions"].dtype == np.float32
    assert decoded["actions"].shape == (3, 7)


def test_get_codec_names_and_unknown() -> None:
    assert get_codec("json").name == "json"
    assert get_codec("auto").name in ("json", "msgpack")
    with pytest.raises(ValidationError):
        get_codec("protobuf")


def test_float64_actions_do_not_silently_change_dtype_on_the_wire() -> None:
    codec = JsonCodec()
    payload = {"actions": np.zeros((2, 7), dtype=np.float64)}
    decoded = codec.decode(codec.encode(payload))
    assert decoded["actions"].dtype == np.float64


def test_build_and_validate_request() -> None:
    observation = {
        "observation/state": np.zeros(22, dtype=np.float32),
        POLICY_KEY_SEQUENCE: 3,
        POLICY_KEY_CAPTURE_TIME: 1_000,
    }
    request = build_request(observation, request_sequence_id=4, client_send_time_ns=999)
    sequence, decoded = validate_request(request)
    assert sequence == 4
    assert decoded is observation
    bad = dict(request)
    bad["schema_version"] = "wire/v999"
    with pytest.raises(ValidationError) as exc:
        validate_request(bad)
    assert exc.value.code is ErrorCode.SCHEMA_MISMATCH
    with pytest.raises(ValidationError):
        validate_request({**request, "type": "unknown"})
    with pytest.raises(ValidationError):
        validate_request({**request, "request_sequence_id": -1})


def test_metadata_roundtrip() -> None:
    metadata = ZeroActionAdapter(horizon=3).metadata
    message = JsonCodec().decode(JsonCodec().encode(build_metadata_message(metadata, server_time_ns=42)))
    decoded = validate_metadata_message(message)
    assert decoded == metadata
    assert message["schema_version"] == "wire/v1"
    with pytest.raises(ValidationError):
        validate_metadata_message({"type": "metadata", "schema_version": "wire/v9", "metadata": {}})


def test_result_roundtrip_and_sequence_check() -> None:
    chunk = make_chunk(request_sequence_id=11, observation_capture_time_ns=1_000)
    message = build_result_message(chunk, server_receive_time_ns=100, server_send_time_ns=200)
    decoded = parse_response(JsonCodec().decode(JsonCodec().encode(message)), expected_sequence_id=11)
    assert decoded.request_sequence_id == 11
    assert decoded.actions.dtype == np.float32
    assert np.array_equal(decoded.actions, chunk.actions)
    assert decoded.server_receive_time_ns == 100
    assert decoded.server_send_time_ns == 200
    with pytest.raises(PolicyContractError) as exc:
        parse_response(message, expected_sequence_id=12)
    assert exc.value.code is ErrorCode.SEQUENCE_MISMATCH


def test_error_response_becomes_exception_with_remote_code() -> None:
    message = build_error_message(
        code=ErrorCode.STALE_OBSERVATION,
        message="too old",
        request_sequence_id=3,
        details={"age_ns": 10},
    )
    with pytest.raises(PolicyContractError) as exc:
        parse_response(message, expected_sequence_id=3)
    assert exc.value.details["remote_code"] == "stale_observation"
    assert exc.value.details["remote_details"] == {"age_ns": 10}


def test_result_with_wrong_dtype_is_rejected() -> None:
    chunk = make_chunk()
    message = build_result_message(chunk, server_receive_time_ns=1, server_send_time_ns=2)
    message["result"]["actions"] = message["result"]["actions"].astype(np.float64)
    with pytest.raises(PolicyContractError) as exc:
        parse_response(message, expected_sequence_id=chunk.request_sequence_id)
    assert exc.value.code is ErrorCode.INVALID_DTYPE


def test_malformed_ndarray_payload_rejected() -> None:
    codec = JsonCodec()
    with pytest.raises(ValidationError):
        codec.decode(b'{"actions": {"__ndarray__": true, "dtype": "<f4", "shape": [2], "data": "AAAA"}}')


def test_delta_semantics_is_declared_and_restricted() -> None:
    from vla_infra.errors import ValidationError
    from vla_infra.policy.adapters.base import PolicyMetadata

    base = dict(
        name="x",
        version="0",
        horizon=2,
        action_dt_s=0.05,
        action_frame="base",
        action_type="delta_tcp",
    )
    assert PolicyMetadata(**base).delta_semantics == "cumulative"
    with pytest.raises(ValidationError):
        PolicyMetadata(**base, delta_semantics="from_observation")
    with pytest.raises(ValidationError):
        PolicyMetadata(**{**base, "action_type": "absolute_tcp"})
    assert PolicyMetadata(**{**base, "action_type": "absolute_tcp"}, delta_semantics="absolute").delta_semantics == "absolute"
