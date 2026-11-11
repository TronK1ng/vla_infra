"""Wire protocol for the dummy OpenPI-shaped policy endpoint.

The transport is msgpack when :mod:`msgpack` is installed, otherwise a JSON +
base64 fallback with the same logical structure.  Both preserve numpy dtype and
shape, so ``float32`` actions and ``uint8`` images survive intact.

Message framing (one WebSocket message per frame)::

    server -> client, once after connect: {"type": "metadata", ...}
    client -> server, per inference:       {"type": "infer", "schema_version": ...}
    server -> client:                      {"type": "result", ...} | {"type": "error", ...}

This is structurally compatible with OpenPI's dictionary-style observation /
action messages; M4 will add a thin adapter around the official
``openpi_client`` while keeping this interface.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Mapping, Optional, Protocol, Tuple

import numpy as np

from ..contracts.action import ActionChunk
from ..contracts.constants import (
    ACTION_DIM,
    ACTION_SCHEMA_VERSION,
    POLICY_KEY_CAPTURE_TIME,
    POLICY_KEY_SEQUENCE,
    SUPPORTED_WIRE_SCHEMAS,
    WIRE_SCHEMA_VERSION,
)
from ..errors import ErrorCode, PolicyContractError, ValidationError

_NDARRAY_KEY = "__ndarray__"
_BYTES_KEY = "__bytes_b64__"


class WireCodec(Protocol):
    name: str

    def encode(self, value: Any) -> bytes: ...

    def decode(self, data: bytes) -> Any: ...


def _encode_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        return {
            _NDARRAY_KEY: True,
            "dtype": arr.dtype.str,
            "shape": list(arr.shape),
            "data": arr.tobytes(order="C"),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        encoded = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"wire mapping keys must be strings, got {type(key).__name__}")
            encoded[key] = _encode_value(item)
        return encoded
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_encode_value(item) for item in value]
    if isinstance(value, (str, bytes, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported wire value type {type(value).__name__}")


def _decode_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if value.get(_NDARRAY_KEY) is True:
            try:
                dtype = np.dtype(str(value["dtype"]))
                shape = tuple(int(d) for d in value["shape"])
                raw = value["data"]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValidationError(
                    "malformed ndarray payload on the wire",
                    code=ErrorCode.POLICY_CONTRACT,
                    details={"reason": str(exc)},
                ) from exc
            if not isinstance(raw, (bytes, bytearray, memoryview)):
                raise ValidationError(
                    "ndarray payload data must be bytes",
                    code=ErrorCode.POLICY_CONTRACT,
                )
            count = int(np.prod(shape)) if shape else 1
            if dtype.itemsize * count != len(raw):
                raise ValidationError(
                    "ndarray payload length does not match dtype/shape",
                    code=ErrorCode.POLICY_CONTRACT,
                    details={"dtype": str(dtype), "shape": list(shape), "bytes": len(raw)},
                )
            arr = np.frombuffer(bytes(raw), dtype=dtype).reshape(shape).copy()
            return arr
        return {str(key): _decode_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    return value


def _bytes_to_b64(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return {_BYTES_KEY: base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, Mapping):
        return {key: _bytes_to_b64(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_bytes_to_b64(item) for item in value]
    return value


def _b64_to_bytes(value: Any) -> Any:
    if isinstance(value, Mapping):
        if set(value.keys()) == {_BYTES_KEY}:
            try:
                return base64.b64decode(value[_BYTES_KEY].encode("ascii"), validate=True)
            except Exception as exc:
                raise ValidationError(
                    "malformed base64 wire payload",
                    code=ErrorCode.POLICY_CONTRACT,
                ) from exc
        return {key: _b64_to_bytes(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_b64_to_bytes(item) for item in value]
    return value


class JsonCodec:
    name = "json"

    def encode(self, value: Any) -> bytes:
        return json.dumps(_bytes_to_b64(_encode_value(value)), separators=(",", ":")).encode("utf-8")

    def decode(self, data: bytes) -> Any:
        return _decode_value(_b64_to_bytes(json.loads(data.decode("utf-8"))))


class MsgpackCodec:
    name = "msgpack"

    def __init__(self) -> None:
        try:
            import msgpack  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ValidationError(
                "msgpack codec requested but the msgpack package is not installed",
                code=ErrorCode.CONFIG_INVALID,
            ) from exc
        self._msgpack = msgpack

    def encode(self, value: Any) -> bytes:
        return self._msgpack.packb(_encode_value(value), use_bin_type=True)

    def decode(self, data: bytes) -> Any:
        return _decode_value(self._msgpack.unpackb(data, raw=False))


def get_codec(name: str = "auto") -> WireCodec:
    if name == "json":
        return JsonCodec()
    if name == "msgpack":
        return MsgpackCodec()
    if name == "auto":
        try:
            return MsgpackCodec()
        except ValidationError:
            return JsonCodec()
    raise ValidationError(
        f"unknown wire codec {name!r}",
        code=ErrorCode.CONFIG_INVALID,
        details={"supported": ["auto", "json", "msgpack"]},
    )


# ---------------------------------------------------------------------------
# Message validation
# ---------------------------------------------------------------------------
def validate_metadata_message(message: Mapping[str, Any]):
    from .adapters.base import PolicyMetadata

    if not isinstance(message, Mapping) or message.get("type") != "metadata":
        raise ValidationError(
            "expected a metadata message from the policy server",
            code=ErrorCode.POLICY_CONTRACT,
            details={"received": message.get("type") if isinstance(message, Mapping) else None},
        )
    schema = message.get("schema_version")
    if schema not in SUPPORTED_WIRE_SCHEMAS:
        raise ValidationError(
            f"unsupported wire schema {schema!r}",
            code=ErrorCode.SCHEMA_MISMATCH,
            details={"supported": sorted(SUPPORTED_WIRE_SCHEMAS)},
        )
    metadata = message.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValidationError(
            "metadata message has no metadata mapping",
            code=ErrorCode.POLICY_CONTRACT,
        )
    return PolicyMetadata.from_dict(metadata)


def build_metadata_message(metadata: Any, *, server_time_ns: int) -> dict:
    return {
        "type": "metadata",
        "schema_version": WIRE_SCHEMA_VERSION,
        "server_time_ns": int(server_time_ns),
        "metadata": metadata.to_dict(),
    }


def build_request(
    observation: Mapping[str, Any],
    *,
    request_sequence_id: int,
    client_send_time_ns: int,
) -> dict:
    if not isinstance(request_sequence_id, int) or request_sequence_id < 0:
        raise ValueError("request_sequence_id must be a non-negative integer")
    return {
        "type": "infer",
        "schema_version": WIRE_SCHEMA_VERSION,
        "request_sequence_id": int(request_sequence_id),
        "client_send_time_ns": int(client_send_time_ns),
        "observation": observation,
    }


def validate_request(message: Mapping[str, Any]) -> Tuple[int, Mapping[str, Any]]:
    if not isinstance(message, Mapping):
        raise ValidationError("wire request must be a mapping", code=ErrorCode.POLICY_CONTRACT)
    if message.get("type") != "infer":
        raise ValidationError(
            f"unsupported request type {message.get('type')!r}",
            code=ErrorCode.POLICY_CONTRACT,
        )
    if message.get("schema_version") not in SUPPORTED_WIRE_SCHEMAS:
        raise ValidationError(
            f"unsupported wire schema {message.get('schema_version')!r}",
            code=ErrorCode.SCHEMA_MISMATCH,
        )
    seq = message.get("request_sequence_id")
    if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
        raise ValidationError(
            "request_sequence_id must be a non-negative integer",
            code=ErrorCode.INVALID_DTYPE,
        )
    observation = message.get("observation")
    if not isinstance(observation, Mapping):
        raise ValidationError(
            "request has no observation mapping",
            code=ErrorCode.POLICY_CONTRACT,
        )
    return seq, observation


def action_chunk_payload(chunk: ActionChunk) -> dict:
    return {
        "schema_version": chunk.schema_version,
        "request_sequence_id": chunk.request_sequence_id,
        "observation_capture_time_ns": chunk.observation_capture_time_ns,
        "action_dt_s": float(chunk.action_dt_s),
        "action_frame": chunk.action_frame,
        "action_type": chunk.action_type,
        "actions": chunk.actions,
        "extras": dict(chunk.extras),
        "server_receive_time_ns": chunk.server_receive_time_ns,
        "server_send_time_ns": chunk.server_send_time_ns,
    }


def build_result_message(
    chunk: ActionChunk,
    *,
    server_receive_time_ns: int,
    server_send_time_ns: int,
) -> dict:
    payload = action_chunk_payload(chunk)
    payload["server_receive_time_ns"] = int(server_receive_time_ns)
    payload["server_send_time_ns"] = int(server_send_time_ns)
    return {
        "type": "result",
        "schema_version": WIRE_SCHEMA_VERSION,
        "request_sequence_id": chunk.request_sequence_id,
        "server_receive_time_ns": int(server_receive_time_ns),
        "server_send_time_ns": int(server_send_time_ns),
        "result": payload,
    }


def build_error_message(
    *,
    code: ErrorCode,
    message: str,
    request_sequence_id: Optional[int] = None,
    details: Optional[Mapping[str, Any]] = None,
) -> dict:
    return {
        "type": "error",
        "schema_version": WIRE_SCHEMA_VERSION,
        "request_sequence_id": request_sequence_id,
        "error": {
            "code": code.value,
            "message": str(message),
            "details": dict(details or {}),
        },
    }


def parse_response(message: Mapping[str, Any], *, expected_sequence_id: int) -> ActionChunk:
    if not isinstance(message, Mapping):
        raise PolicyContractError(
            "policy response must be a mapping",
            code=ErrorCode.POLICY_CONTRACT,
        )
    if message.get("schema_version") not in SUPPORTED_WIRE_SCHEMAS:
        raise PolicyContractError(
            f"unsupported wire schema {message.get('schema_version')!r}",
            code=ErrorCode.SCHEMA_MISMATCH,
        )
    seq = message.get("request_sequence_id")
    if seq != expected_sequence_id:
        raise PolicyContractError(
            "policy response sequence does not match the request",
            code=ErrorCode.SEQUENCE_MISMATCH,
            details={"expected": expected_sequence_id, "received": seq},
        )
    if message.get("type") == "error":
        error = message.get("error") or {}
        remote_code = str(error.get("code", ErrorCode.POLICY_REMOTE_ERROR.value))
        raise PolicyContractError(
            f"policy server rejected the request: {error.get('message', 'unknown error')}",
            code=ErrorCode.POLICY_REMOTE_ERROR,
            details={
                "remote_code": remote_code,
                "remote_details": error.get("details", {}),
            },
        )
    if message.get("type") != "result":
        raise PolicyContractError(
            f"unexpected policy response type {message.get('type')!r}",
            code=ErrorCode.POLICY_CONTRACT,
        )
    payload = message.get("result")
    if not isinstance(payload, Mapping):
        raise PolicyContractError("result payload is missing", code=ErrorCode.POLICY_CONTRACT)
    if payload.get("request_sequence_id") != expected_sequence_id:
        raise PolicyContractError(
            "result payload sequence mismatch",
            code=ErrorCode.SEQUENCE_MISMATCH,
        )
    actions = payload.get("actions")
    if not isinstance(actions, np.ndarray):
        raise PolicyContractError(
            "actions must decode to a numpy array",
            code=ErrorCode.POLICY_CONTRACT,
        )
    if actions.dtype != np.float32:
        raise PolicyContractError(
            "actions dtype must be float32",
            code=ErrorCode.INVALID_DTYPE,
            details={"dtype": str(actions.dtype)},
        )
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise PolicyContractError(
            f"actions must have shape (H, {ACTION_DIM})",
            code=ErrorCode.INVALID_SHAPE,
            details={"shape": list(actions.shape)},
        )
    if payload.get("schema_version") not in (ACTION_SCHEMA_VERSION,):
        raise PolicyContractError(
            f"unsupported action schema {payload.get('schema_version')!r}",
            code=ErrorCode.SCHEMA_MISMATCH,
        )
    extras = payload.get("extras", {})
    if not isinstance(extras, Mapping):
        raise PolicyContractError("extras must be a mapping", code=ErrorCode.POLICY_CONTRACT)
    try:
        return ActionChunk(
            request_sequence_id=int(payload["request_sequence_id"]),
            observation_capture_time_ns=int(payload["observation_capture_time_ns"]),
            action_dt_s=float(payload["action_dt_s"]),
            action_frame=str(payload["action_frame"]),
            action_type=str(payload["action_type"]),
            actions=actions,
            extras=dict(extras),
            server_receive_time_ns=payload.get("server_receive_time_ns"),
            server_send_time_ns=payload.get("server_send_time_ns"),
            schema_version=str(payload["schema_version"]),
        )
    except KeyError as exc:
        raise PolicyContractError(
            f"result payload is missing {exc.args[0]!r}",
            code=ErrorCode.POLICY_CONTRACT,
        ) from exc


def observation_sequence(observation: Mapping[str, Any]) -> int:
    value = observation.get(POLICY_KEY_SEQUENCE)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(
            "observation meta/sequence_id must be a non-negative integer",
            code=ErrorCode.INVALID_DTYPE,
        )
    return int(value)


def observation_capture_time(observation: Mapping[str, Any]) -> int:
    value = observation.get(POLICY_KEY_CAPTURE_TIME)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValidationError(
            "observation meta/capture_time_ns must be a positive integer",
            code=ErrorCode.INVALID_DTYPE,
        )
    return int(value)


__all__ = [
    "JsonCodec",
    "MsgpackCodec",
    "WireCodec",
    "action_chunk_payload",
    "build_error_message",
    "build_metadata_message",
    "build_request",
    "build_result_message",
    "get_codec",
    "observation_capture_time",
    "observation_sequence",
    "parse_response",
    "validate_metadata_message",
    "validate_request",
]
