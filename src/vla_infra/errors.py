"""Error taxonomy shared by every layer.

Every failure that can cross a layer boundary (device -> sync -> policy ->
safety -> supervisor) carries a stable :class:`ErrorCode`.  Codes are the only
part of an error that is transmitted across the WebSocket boundary; free-text
messages are diagnostic and must not be parsed by control logic.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Optional


class ErrorCode(str, Enum):
    # --- contract / validation -------------------------------------------------
    SCHEMA_MISMATCH = "schema_mismatch"
    INVALID_SHAPE = "invalid_shape"
    INVALID_DTYPE = "invalid_dtype"
    INVALID_ENCODING = "invalid_encoding"
    INVALID_FRAME = "invalid_frame"
    INVALID_ACTION = "invalid_action"
    INVALID_QUATERNION = "invalid_quaternion"
    INVALID_SYNC = "invalid_sync"
    NON_FINITE = "non_finite"
    VALUE_OUT_OF_RANGE = "value_out_of_range"

    # --- sequencing / time -----------------------------------------------------
    SEQUENCE_REGRESSION = "sequence_regression"
    SEQUENCE_MISMATCH = "sequence_mismatch"
    CAPTURE_MISMATCH = "capture_mismatch"
    STALE_OBSERVATION = "stale_observation"
    STALE_PLAN = "stale_plan"
    CHUNK_EXPIRED = "chunk_expired"
    EPOCH_MISMATCH = "epoch_mismatch"
    BUFFER_NOT_MONOTONIC = "buffer_not_monotonic"
    BUFFER_EMPTY = "buffer_empty"

    # --- safety ----------------------------------------------------------------
    SAFETY_WORKSPACE = "safety_workspace"
    SAFETY_HEIGHT = "safety_height"
    SAFETY_STEP = "safety_step"
    SAFETY_VELOCITY = "safety_velocity"
    SAFETY_ACCELERATION = "safety_acceleration"
    SAFETY_FIRST_TARGET = "safety_first_target"
    SAFETY_FORCE = "safety_force"
    SAFETY_TORQUE = "safety_torque"
    SAFETY_GRIPPER = "safety_gripper"
    SAFETY_STATE = "safety_state"
    NOT_ARMED = "not_armed"
    WATCHDOG_EXPIRED = "watchdog_expired"

    # --- lifecycle / policy ----------------------------------------------------
    ILLEGAL_TRANSITION = "illegal_transition"
    NOT_READY = "not_ready"
    POLICY_TIMEOUT = "policy_timeout"
    POLICY_DISCONNECTED = "policy_disconnected"
    POLICY_REMOTE_ERROR = "policy_remote_error"
    POLICY_CONTRACT = "policy_contract"
    POLICY_CLEANUP_FAILED = "policy_cleanup_failed"
    STOP_FAILED = "stop_failed"
    POLICY_REJECTED = "policy_rejected"
    AUTH_FAILED = "auth_failed"
    RATE_LIMITED = "rate_limited"

    # --- devices / config ------------------------------------------------------
    DEVICE_FAULT = "device_fault"
    DEVICE_NOT_CONNECTED = "device_not_connected"
    SINGLE_WRITER_VIOLATION = "single_writer_violation"
    CONFIG_INVALID = "config_invalid"
    UNSUPPORTED_BACKEND = "unsupported_backend"
    REAL_HARDWARE_DISABLED = "real_hardware_disabled"
    INTERNAL_ERROR = "internal_error"


class VlaInfraError(Exception):
    """Base class for all infrastructure errors."""

    code: ErrorCode = ErrorCode.INTERNAL_ERROR

    def __init__(
        self,
        message: str,
        *,
        code: Optional[ErrorCode] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = str(message)
        if code is not None:
            self.code = code
        self.details: dict[str, Any] = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "details": dict(self.details),
        }

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        if self.details:
            return f"[{self.code.value}] {self.message} ({self.details})"
        return f"[{self.code.value}] {self.message}"


class ValidationError(VlaInfraError):
    code = ErrorCode.INVALID_ACTION


class SafetyViolation(VlaInfraError):
    code = ErrorCode.SAFETY_STATE


class ConfigError(VlaInfraError):
    code = ErrorCode.CONFIG_INVALID


class UnsupportedBackendError(ConfigError):
    code = ErrorCode.UNSUPPORTED_BACKEND


class RealHardwareDisabledError(UnsupportedBackendError):
    code = ErrorCode.REAL_HARDWARE_DISABLED


class IllegalTransitionError(VlaInfraError):
    code = ErrorCode.ILLEGAL_TRANSITION


class NotArmedError(VlaInfraError):
    code = ErrorCode.NOT_ARMED


class SequenceError(ValidationError):
    code = ErrorCode.SEQUENCE_REGRESSION


class SyncError(VlaInfraError):
    code = ErrorCode.STALE_OBSERVATION


class DeviceError(VlaInfraError):
    code = ErrorCode.DEVICE_FAULT


class SingleWriterViolationError(DeviceError):
    code = ErrorCode.SINGLE_WRITER_VIOLATION


class PolicyError(VlaInfraError):
    code = ErrorCode.POLICY_REMOTE_ERROR


class PolicyTimeoutError(PolicyError):
    code = ErrorCode.POLICY_TIMEOUT


class PolicyDisconnectedError(PolicyError):
    code = ErrorCode.POLICY_DISCONNECTED


class PolicyContractError(PolicyError):
    code = ErrorCode.POLICY_CONTRACT


class PolicyCleanupError(PolicyError):
    """A transport cleanup timed out; retrying would leak resources."""

    code = ErrorCode.POLICY_CLEANUP_FAILED
