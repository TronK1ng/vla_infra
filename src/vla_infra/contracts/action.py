"""Unified action / policy-result contract (master plan §5.4).

A policy result is a *chunk*: ``H`` rows of seven-dimensional actions.

Universal action layout (fixed order, fixed units)::

    [dx, dy, dz, drx, dry, drz, gripper]

* translation: metres;
* rotation: axis-angle rotation vector, radians;
* gripper: normalised to ``[0, 1]`` (0 = closed, 1 = open); the local GN01
  adapter maps it to a physical width;
* ``action_frame`` and ``action_type`` are mandatory — no defaults, because a
  wrong frame is a dangerous silent failure.

``delta_tcp`` semantics: the first row is a delta relative to the *measured*
TCP pose at plan creation; each following row is a delta relative to the
previous commanded target.  ``absolute_tcp`` rows carry absolute positions and
absolute orientation expressed as a rotation vector.

Model-specific outputs (Force Policy wrench references, selection masks, ...)
go into ``extras``.  They are optional and no lower layer may require them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional

import numpy as np

from ..errors import ErrorCode, ValidationError
from .constants import (
    ACTION_DIM,
    ACTION_SCHEMA_VERSION,
    MAX_ACTION_HORIZON,
    PROPRIETARY_EXTRAS_KEYS,
)


def _freeze_actions(value: Any) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype != np.float32:
        raise ValidationError(
            "actions dtype must be float32",
            code=ErrorCode.INVALID_DTYPE,
            details={"dtype": str(raw.dtype)},
        )
    arr = np.array(raw, dtype=np.float32, copy=True, order="C")
    if arr.ndim != 2 or arr.shape[1] != ACTION_DIM:
        raise ValidationError(
            f"actions must have shape (H, {ACTION_DIM}), got {arr.shape}",
            code=ErrorCode.INVALID_SHAPE,
            details={"shape": list(arr.shape), "expected_dim": ACTION_DIM},
        )
    if arr.shape[0] < 1 or arr.shape[0] > MAX_ACTION_HORIZON:
        raise ValidationError(
            f"action horizon must be in [1, {MAX_ACTION_HORIZON}], got {arr.shape[0]}",
            code=ErrorCode.VALUE_OUT_OF_RANGE,
            details={"horizon": int(arr.shape[0])},
        )
    if not np.all(np.isfinite(arr)):
        raise ValidationError(
            "actions contain NaN/Inf",
            code=ErrorCode.NON_FINITE,
            details={"reason": "non_finite_action"},
        )
    gripper = arr[:, 6]
    if np.any(gripper < 0.0) or np.any(gripper > 1.0):
        raise ValidationError(
            "gripper channel must lie in [0, 1]",
            code=ErrorCode.SAFETY_GRIPPER,
            details={"min": float(gripper.min()), "max": float(gripper.max())},
        )
    arr.setflags(write=False)
    return arr


def _freeze_extra_value(value: Any, *, depth: int = 0) -> Any:
    """Recursively freeze extras so policy diagnostics cannot be mutated later.

    Mutable containers become tuples/mapping proxies and numpy arrays are copied
    and made read-only.  A depth limit turns accidental cyclic structures into a
    contract error instead of a recursion crash.
    """
    if depth > 8:
        raise ValidationError(
            "extras nesting is too deep or cyclic",
            code=ErrorCode.INVALID_SHAPE,
        )
    if isinstance(value, np.ndarray):
        frozen = np.array(value, copy=True, order="C")
        frozen.setflags(write=False)
        return frozen
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        frozen_mapping: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValidationError(
                    "extras keys must be strings",
                    code=ErrorCode.INVALID_SHAPE,
                    details={"key": repr(key)},
                )
            frozen_mapping[key] = _freeze_extra_value(item, depth=depth + 1)
        return MappingProxyType(frozen_mapping)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_extra_value(item, depth=depth + 1) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_extra_value(item, depth=depth + 1) for item in value)
    return value


def _freeze_extras(extras: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(extras, Mapping):
        raise ValidationError(
            "extras must be a mapping",
            code=ErrorCode.INVALID_SHAPE,
            details={"type": type(extras).__name__},
        )
    return _freeze_extra_value(extras)


@dataclass(frozen=True)
class ActionChunk:
    """Validated policy output chunk.

    The constructor canonicalises ``actions`` to a read-only float32 array; wire
    decoding performs an additional dtype check so that a float64 server cannot
    silently change the contract.
    """

    request_sequence_id: int
    observation_capture_time_ns: int
    action_dt_s: float
    action_frame: str
    action_type: str
    actions: np.ndarray
    extras: Mapping[str, Any] = field(default_factory=dict)
    server_receive_time_ns: Optional[int] = None
    server_send_time_ns: Optional[int] = None
    schema_version: str = ACTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", _freeze_actions(self.actions))
        object.__setattr__(self, "extras", _freeze_extras(self.extras))
        from .validation import validate_action_chunk

        validate_action_chunk(self)

    @property
    def horizon(self) -> int:
        return int(self.actions.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    @property
    def gripper(self) -> np.ndarray:
        return self.actions[:, 6]

    @property
    def has_proprietary_extras(self) -> bool:
        return any(key in self.extras for key in PROPRIETARY_EXTRAS_KEYS)

    def with_extras(self, **extra: Any) -> "ActionChunk":
        merged = dict(self.extras)
        merged.update(extra)
        return ActionChunk(
            request_sequence_id=self.request_sequence_id,
            observation_capture_time_ns=self.observation_capture_time_ns,
            action_dt_s=self.action_dt_s,
            action_frame=self.action_frame,
            action_type=self.action_type,
            actions=self.actions,
            extras=merged,
            server_receive_time_ns=self.server_receive_time_ns,
            server_send_time_ns=self.server_send_time_ns,
            schema_version=self.schema_version,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ActionChunk(seq={self.request_sequence_id}, H={self.horizon}, "
            f"dt={self.action_dt_s}, frame={self.action_frame!r}, "
            f"type={self.action_type!r}, extras={sorted(self.extras)})"
        )


# The master plan uses both names for the same structure.
PolicyResult = ActionChunk
