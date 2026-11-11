"""Policy adapter interface and dummy adapters.

An adapter maps a validated policy observation to actions.  It owns the
model-specific semantics (horizon, ``action_dt_s``, action frame/type, state
layout, extras) and declares them in :class:`PolicyMetadata`.  The server
rejects requests whose declared state layout does not match the adapter, and
rejects adapter output whose shape/dtype/horizon violates the metadata.

No adapter may import RDK, open a camera or know a robot exists.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from ...contracts.constants import (
    ACTION_DIM,
    ACTION_FRAMES,
    ACTION_SCHEMA_VERSION,
    ACTION_TYPES,
    DEFAULT_POLICY_IMAGE_SIZE,
    MAX_ACTION_DT_S,
    MAX_ACTION_HORIZON,
    MIN_ACTION_DT_S,
    GRIPPER_SEMANTICS,
    SUPPORTED_WIRE_SCHEMAS,
    WIRE_SCHEMA_VERSION,
)
from ...contracts.policy_io import CANONICAL_STATE_LAYOUT, StateLayout
from ...errors import ErrorCode, ValidationError


@dataclass(frozen=True)
class PolicyMetadata:
    name: str
    version: str
    horizon: int
    action_dt_s: float
    action_frame: str
    action_type: str
    delta_semantics: str = "cumulative"
    gripper_semantics: str = GRIPPER_SEMANTICS
    state_layout: StateLayout = CANONICAL_STATE_LAYOUT
    image_size: int = DEFAULT_POLICY_IMAGE_SIZE
    action_dim: int = ACTION_DIM
    extras_keys: Tuple[str, ...] = ()
    notes: str = ""
    wire_schema_version: str = WIRE_SCHEMA_VERSION
    action_schema_version: str = ACTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValidationError("policy name must be a non-empty string")
        if not isinstance(self.version, str) or not self.version:
            raise ValidationError("policy version must be a non-empty string")
        if self.wire_schema_version not in SUPPORTED_WIRE_SCHEMAS:
            raise ValidationError(
                f"unsupported wire schema {self.wire_schema_version!r}",
                code=ErrorCode.SCHEMA_MISMATCH,
            )
        if self.action_schema_version != ACTION_SCHEMA_VERSION:
            raise ValidationError(
                f"unsupported action schema {self.action_schema_version!r}",
                code=ErrorCode.SCHEMA_MISMATCH,
            )
        if self.action_dim != ACTION_DIM:
            raise ValidationError(
                f"M1 only supports {ACTION_DIM}-dimensional actions",
                code=ErrorCode.SCHEMA_MISMATCH,
                details={"action_dim": self.action_dim},
            )
        if not isinstance(self.horizon, int) or not (1 <= self.horizon <= MAX_ACTION_HORIZON):
            raise ValidationError(
                f"horizon must be in [1, {MAX_ACTION_HORIZON}]",
                code=ErrorCode.VALUE_OUT_OF_RANGE,
                details={"horizon": self.horizon},
            )
        dt = float(self.action_dt_s)
        if not np.isfinite(dt) or dt < MIN_ACTION_DT_S or dt > MAX_ACTION_DT_S:
            raise ValidationError(
                "action_dt_s is outside the contract range",
                code=ErrorCode.VALUE_OUT_OF_RANGE,
                details={"action_dt_s": dt},
            )
        if self.action_frame not in ACTION_FRAMES:
            raise ValidationError(
                f"unknown action frame {self.action_frame!r}",
                code=ErrorCode.INVALID_FRAME,
            )
        if self.action_type not in ACTION_TYPES:
            raise ValidationError(
                f"unknown action type {self.action_type!r}",
                code=ErrorCode.INVALID_ACTION,
            )
        if self.action_type == "delta_tcp" and self.delta_semantics != "cumulative":
            raise ValidationError(
                "M1 only implements cumulative delta_tcp semantics; "
                "adapters must convert from_observation deltas explicitly",
                code=ErrorCode.INVALID_ACTION,
                details={"delta_semantics": self.delta_semantics},
            )
        if self.action_type == "absolute_tcp" and self.delta_semantics != "absolute":
            raise ValidationError(
                "absolute_tcp metadata must declare delta_semantics='absolute'",
                code=ErrorCode.INVALID_ACTION,
                details={"delta_semantics": self.delta_semantics},
            )
        if self.gripper_semantics != GRIPPER_SEMANTICS:
            raise ValidationError(
                f"gripper_semantics must be {GRIPPER_SEMANTICS!r} in v1",
                code=ErrorCode.INVALID_ACTION,
                details={"gripper_semantics": self.gripper_semantics},
            )
        if not isinstance(self.image_size, int) or self.image_size < 32:
            raise ValidationError(
                "image_size must be an integer >= 32",
                code=ErrorCode.VALUE_OUT_OF_RANGE,
            )
        if not isinstance(self.state_layout, StateLayout):
            raise TypeError("state_layout must be a StateLayout")
        object.__setattr__(self, "extras_keys", tuple(str(k) for k in self.extras_keys))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "wire_schema_version": self.wire_schema_version,
            "action_schema_version": self.action_schema_version,
            "action_dim": self.action_dim,
            "horizon": self.horizon,
            "action_dt_s": float(self.action_dt_s),
            "action_frame": self.action_frame,
            "action_type": self.action_type,
            "delta_semantics": self.delta_semantics,
            "gripper_semantics": self.gripper_semantics,
            "state_layout": self.state_layout.to_dict(),
            "image_size": self.image_size,
            "extras_keys": list(self.extras_keys),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PolicyMetadata":
        if not isinstance(value, Mapping):
            raise ValidationError("policy metadata must be a mapping", code=ErrorCode.POLICY_CONTRACT)
        try:
            layout = StateLayout.from_dict(value["state_layout"])
            return cls(
                name=str(value["name"]),
                version=str(value["version"]),
                horizon=int(value["horizon"]),
                action_dt_s=float(value["action_dt_s"]),
                action_frame=str(value["action_frame"]),
                action_type=str(value["action_type"]),
                delta_semantics=str(value.get("delta_semantics", "cumulative")),
                gripper_semantics=str(value.get("gripper_semantics", GRIPPER_SEMANTICS)),
                state_layout=layout,
                image_size=int(value["image_size"]),
                action_dim=int(value.get("action_dim", ACTION_DIM)),
                extras_keys=tuple(value.get("extras_keys", ())),
                notes=str(value.get("notes", "")),
                wire_schema_version=str(value.get("wire_schema_version", WIRE_SCHEMA_VERSION)),
                action_schema_version=str(value.get("action_schema_version", ACTION_SCHEMA_VERSION)),
            )
        except KeyError as exc:
            raise ValidationError(
                f"policy metadata is missing {exc.args[0]!r}",
                code=ErrorCode.POLICY_CONTRACT,
            ) from exc

    def compatibility_issues(self, other: "PolicyMetadata") -> Tuple[str, ...]:
        """Contract fields that must match between client and server."""
        issues = []
        if self.action_schema_version != other.action_schema_version:
            issues.append("action_schema_version")
        if self.wire_schema_version != other.wire_schema_version:
            issues.append("wire_schema_version")
        if self.action_dim != other.action_dim:
            issues.append("action_dim")
        if self.image_size != other.image_size:
            issues.append("image_size")
        if self.state_layout.to_dict() != other.state_layout.to_dict():
            issues.append("state_layout")
        if self.horizon != other.horizon:
            issues.append("horizon")
        if abs(self.action_dt_s - other.action_dt_s) > 1e-6:
            issues.append("action_dt_s")
        if self.action_frame != other.action_frame:
            issues.append("action_frame")
        if self.action_type != other.action_type:
            issues.append("action_type")
        if self.delta_semantics != other.delta_semantics:
            issues.append("delta_semantics")
        if self.gripper_semantics != other.gripper_semantics:
            issues.append("gripper_semantics")
        return tuple(issues)


class PolicyAdapter(ABC):
    @property
    @abstractmethod
    def metadata(self) -> PolicyMetadata: ...

    @abstractmethod
    def infer(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return ``{"actions": float32[H,7], "extras": Mapping}``."""


class ZeroActionAdapter(PolicyAdapter):
    """Zero Cartesian delta; gripper held at a neutral normalised value."""

    def __init__(
        self,
        *,
        horizon: int = 10,
        action_dt_s: float = 0.05,
        action_frame: str = "base",
        action_type: str = "delta_tcp",
        gripper_normalized: float = 0.5,
        image_size: int = DEFAULT_POLICY_IMAGE_SIZE,
        name: str = "dummy_zero",
        version: str = "0.1",
    ) -> None:
        self._gripper = float(gripper_normalized)
        if not 0.0 <= self._gripper <= 1.0:
            raise ValueError("gripper_normalized must lie in [0, 1]")
        self._metadata = PolicyMetadata(
            name=name,
            version=version,
            horizon=horizon,
            action_dt_s=action_dt_s,
            action_frame=action_frame,
            action_type=action_type,
            delta_semantics="absolute" if action_type == "absolute_tcp" else "cumulative",
            image_size=image_size,
            notes="deterministic zero-motion dummy",
        )

    @property
    def metadata(self) -> PolicyMetadata:
        return self._metadata

    def infer(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        actions = np.zeros((self._metadata.horizon, ACTION_DIM), dtype=np.float32)
        actions[:, 6] = np.float32(self._gripper)
        return {"actions": actions, "extras": {"policy_mode": "dummy_zero"}}


class FixedDeltaAdapter(PolicyAdapter):
    """Constant small delta per step, useful for fake-executor acceptance runs."""

    def __init__(
        self,
        *,
        delta_m: Sequence[float] = (0.001, 0.0, 0.0),
        delta_rotvec_rad: Sequence[float] = (0.0, 0.0, 0.0),
        gripper_normalized: float = 0.5,
        horizon: int = 10,
        action_dt_s: float = 0.05,
        action_frame: str = "base",
        action_type: str = "delta_tcp",
        image_size: int = DEFAULT_POLICY_IMAGE_SIZE,
        name: str = "dummy_fixed",
    ) -> None:
        self._delta = np.asarray(delta_m, dtype=np.float32)
        self._delta_rot = np.asarray(delta_rotvec_rad, dtype=np.float32)
        if self._delta.shape != (3,) or self._delta_rot.shape != (3,):
            raise ValueError("delta_m and delta_rotvec_rad must have three components")
        self._gripper = float(gripper_normalized)
        if not 0.0 <= self._gripper <= 1.0:
            raise ValueError("gripper_normalized must lie in [0, 1]")
        self._metadata = PolicyMetadata(
            name=name,
            version="0.1",
            horizon=horizon,
            action_dt_s=action_dt_s,
            action_frame=action_frame,
            action_type=action_type,
            delta_semantics="absolute" if action_type == "absolute_tcp" else "cumulative",
            image_size=image_size,
            notes="constant delta dummy for fake-backend tests",
        )

    @property
    def metadata(self) -> PolicyMetadata:
        return self._metadata

    def infer(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        actions = np.zeros((self._metadata.horizon, ACTION_DIM), dtype=np.float32)
        actions[:, 0:3] = self._delta
        actions[:, 3:6] = self._delta_rot
        actions[:, 6] = np.float32(self._gripper)
        return {"actions": actions, "extras": {"policy_mode": "dummy_fixed"}}


class SineSweepAdapter(PolicyAdapter):
    """Deterministic small sinusoidal sweep driven by the observation sequence."""

    def __init__(
        self,
        *,
        amplitude_m: float = 0.002,
        period_steps: int = 20,
        horizon: int = 10,
        action_dt_s: float = 0.05,
        gripper_normalized: float = 0.5,
        image_size: int = DEFAULT_POLICY_IMAGE_SIZE,
        name: str = "dummy_sine",
    ) -> None:
        if amplitude_m <= 0 or period_steps <= 0:
            raise ValueError("amplitude/period must be positive")
        from ...contracts.constants import POLICY_KEY_SEQUENCE

        self._sequence_key = POLICY_KEY_SEQUENCE
        self._amplitude = float(amplitude_m)
        self._period = int(period_steps)
        self._gripper = float(gripper_normalized)
        self._metadata = PolicyMetadata(
            name=name,
            version="0.1",
            horizon=horizon,
            action_dt_s=action_dt_s,
            action_frame="base",
            action_type="delta_tcp",
            delta_semantics="cumulative",
            image_size=image_size,
            notes="sinusoidal dummy driven by meta/sequence_id",
        )

    @property
    def metadata(self) -> PolicyMetadata:
        return self._metadata

    def infer(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        sequence = int(observation.get(self._sequence_key, 0))
        phase = 2.0 * np.pi * (sequence % self._period) / self._period
        actions = np.zeros((self._metadata.horizon, ACTION_DIM), dtype=np.float32)
        actions[:, 0] = np.float32(self._amplitude * np.sin(phase))
        actions[:, 1] = np.float32(self._amplitude * np.cos(phase))
        actions[:, 6] = np.float32(self._gripper)
        return {"actions": actions, "extras": {"policy_mode": "dummy_sine", "phase_rad": float(phase)}}
