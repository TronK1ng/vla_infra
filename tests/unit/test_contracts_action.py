"""Action/policy-result contract tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from vla_infra.contracts.action import ActionChunk, PolicyResult
from vla_infra.contracts.constants import ACTION_SCHEMA_VERSION, PROPRIETARY_EXTRAS_KEYS
from vla_infra.errors import ErrorCode, ValidationError
from vla_infra.testing import make_chunk


def test_valid_chunk_and_alias() -> None:
    chunk = make_chunk()
    assert PolicyResult is ActionChunk
    assert chunk.actions.dtype == np.float32
    assert chunk.actions.shape == (4, 7)
    assert chunk.horizon == 4
    assert chunk.actions.flags.writeable is False
    assert chunk.gripper.tolist() == [0.5] * 4


def test_chunk_is_frozen() -> None:
    chunk = make_chunk()
    with pytest.raises(FrozenInstanceError):
        chunk.action_dt_s = 0.1  # type: ignore[misc]
    with pytest.raises(ValueError):
        chunk.actions[0, 0] = 1.0


def test_actions_shape_checks() -> None:
    with pytest.raises(ValidationError) as exc:
        make_chunk(actions=np.zeros((4, 6), dtype=np.float32))
    assert exc.value.code is ErrorCode.INVALID_SHAPE
    with pytest.raises(ValidationError):
        make_chunk(actions=np.zeros((4,), dtype=np.float32))
    with pytest.raises(ValidationError):
        make_chunk(actions=np.zeros((0, 7), dtype=np.float32))
    with pytest.raises(ValidationError):
        make_chunk(actions=np.zeros((513, 7), dtype=np.float32))


def test_actions_dtype_must_be_float32() -> None:
    with pytest.raises(ValidationError) as exc:
        make_chunk(actions=np.zeros((2, 7), dtype=np.float64))
    assert exc.value.code is ErrorCode.INVALID_DTYPE
    with pytest.raises(ValidationError):
        make_chunk(actions=[[0.0] * 7])


def test_action_values_must_be_finite() -> None:
    actions = np.zeros((2, 7), dtype=np.float32)
    actions[1, 3] = np.nan
    with pytest.raises(ValidationError) as exc:
        make_chunk(actions=actions)
    assert exc.value.code is ErrorCode.NON_FINITE


def test_gripper_channel_bounds() -> None:
    actions = np.zeros((2, 7), dtype=np.float32)
    actions[:, 6] = 1.5
    with pytest.raises(ValidationError) as exc:
        make_chunk(actions=actions)
    assert exc.value.code is ErrorCode.SAFETY_GRIPPER
    actions[:, 6] = -0.1
    with pytest.raises(ValidationError):
        make_chunk(actions=actions)


def test_action_dt_bounds() -> None:
    for dt in (0.0, -0.01, 2.0, float("nan")):
        with pytest.raises(ValidationError):
            make_chunk(action_dt_s=dt)


def test_action_frame_and_type_must_be_explicit_and_valid() -> None:
    with pytest.raises(ValidationError) as exc:
        make_chunk(action_frame="world")
    assert exc.value.code is ErrorCode.INVALID_FRAME
    with pytest.raises(ValidationError) as exc:
        make_chunk(action_type="delta_joint")
    assert exc.value.code is ErrorCode.INVALID_ACTION
    with pytest.raises(TypeError):
        ActionChunk(  # missing action_frame / action_type
            request_sequence_id=1,
            observation_capture_time_ns=1,
            action_dt_s=0.05,
            actions=np.zeros((1, 7), dtype=np.float32),
        )  # type: ignore[call-arg]


def test_schema_version_must_be_supported() -> None:
    with pytest.raises(ValidationError) as exc:
        make_chunk(schema_version="action/v2")
    assert exc.value.code is ErrorCode.SCHEMA_MISMATCH


def test_sequence_and_capture_are_validated() -> None:
    with pytest.raises(ValidationError):
        make_chunk(request_sequence_id=-1)
    with pytest.raises(ValidationError):
        make_chunk(observation_capture_time_ns=0)


def test_extras_are_optional_and_preserve_force_policy_fields() -> None:
    extras = {
        "reference_wrench": np.zeros((2, 6), dtype=np.float32),
        "interaction_frame": "tcp",
        "selection_mask": [1, 0, 1],
        "policy_mode": "force",
    }
    chunk = make_chunk(extras=extras)
    assert chunk.has_proprietary_extras
    assert set(chunk.extras) == set(PROPRIETARY_EXTRAS_KEYS)
    assert chunk.extras["interaction_frame"] == "tcp"
    # Extras are not required by the contract.
    assert not make_chunk().has_proprietary_extras
    with pytest.raises(TypeError):
        chunk.extras["new_key"] = 1  # type: ignore[index]


def test_server_timestamps_optional_but_validated() -> None:
    chunk = make_chunk()
    assert chunk.server_receive_time_ns is None
    with pytest.raises(ValidationError):
        replace(chunk, server_receive_time_ns=-1)


def test_with_extras_returns_new_chunk() -> None:
    chunk = make_chunk()
    updated = chunk.with_extras(policy_mode="test")
    assert updated is not chunk
    assert updated.extras["policy_mode"] == "test"
    assert "policy_mode" not in chunk.extras
    assert np.array_equal(updated.actions, chunk.actions)


def test_schema_version_defaults_to_current() -> None:
    assert make_chunk().schema_version == ACTION_SCHEMA_VERSION


def test_extras_are_deeply_frozen() -> None:
    array = np.array([1.0, 2.0], dtype=np.float32)
    nested_list = [1, 2]
    nested_dict = {"inner": array}
    chunk = make_chunk(
        extras={
            "reference_wrench": array,
            "selection": nested_list,
            "meta": nested_dict,
            "nested": {"deep": [{"values": array}]},
        }
    )
    # Mutating the caller's objects must not change the chunk.
    array[0] = 999.0
    nested_list.append(3)
    nested_dict["inner"] = None
    assert chunk.extras["reference_wrench"][0] == pytest.approx(1.0)
    assert chunk.extras["selection"] == (1, 2)
    assert np.array_equal(chunk.extras["meta"]["inner"], chunk.extras["reference_wrench"])
    assert chunk.extras["nested"]["deep"][0]["values"].flags.writeable is False
    with pytest.raises(ValueError):
        chunk.extras["reference_wrench"][0] = 5.0
    with pytest.raises(TypeError):
        chunk.extras["meta"]["new"] = "x"  # type: ignore[index]
