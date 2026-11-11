"""Policy observation encoding: explicit state layout, letterboxing, RGB safety."""

from __future__ import annotations

import numpy as np
import pytest

from vla_infra.contracts.constants import (
    DEFAULT_POLICY_IMAGE_SIZE,
    POLICY_KEY_CAPTURE_TIME,
    POLICY_KEY_IMAGE,
    POLICY_KEY_IMAGE_ENCODING,
    POLICY_KEY_PROMPT,
    POLICY_KEY_SCHEMA,
    POLICY_KEY_SEQUENCE,
    POLICY_KEY_STATE,
    POLICY_KEY_STATE_DIM,
    POLICY_KEY_STATE_LAYOUT,
    POLICY_KEY_WRIST_IMAGE,
)
from vla_infra.contracts.policy_io import (
    CANONICAL_STATE_LAYOUT,
    JOINT_STATE_LAYOUT,
    StateLayout,
    build_policy_observation,
    letterbox_geometry,
    letterbox_rgb,
    state_scalar_values,
)
from vla_infra.contracts.validation import validate_policy_observation
from vla_infra.errors import ErrorCode, ValidationError
from vla_infra.testing import make_image, make_observation, make_robot_state


def test_canonical_layout_dim_and_axes() -> None:
    assert CANONICAL_STATE_LAYOUT.dim == 22
    assert CANONICAL_STATE_LAYOUT.axes[0:3] == ("tcp_x", "tcp_y", "tcp_z")
    assert CANONICAL_STATE_LAYOUT.axes[21] == "gripper_width"
    assert JOINT_STATE_LAYOUT.dim == 8


def test_layout_encode_respects_declared_order() -> None:
    state = make_robot_state(position=(0.1, 0.2, 0.3), gripper_width=0.07)
    layout = StateLayout(name="custom", axes=("tcp_z", "tcp_x", "gripper_width"), units=("m", "m", "m"))
    encoded = layout.encode(state)
    assert encoded.dtype == np.float32
    assert encoded.tolist() == pytest.approx([0.3, 0.1, 0.07])


def test_layout_rejects_unknown_axis() -> None:
    layout = StateLayout(name="bad", axes=("tcp_x", "mystery"), units=("m", "m"))
    with pytest.raises(ValidationError) as exc:
        layout.encode(make_robot_state())
    assert exc.value.code is ErrorCode.SCHEMA_MISMATCH


def test_layout_rejects_duplicate_axes_and_unit_mismatch() -> None:
    with pytest.raises(ValidationError):
        StateLayout(name="dup", axes=("tcp_x", "tcp_x"), units=("m", "m"))
    with pytest.raises(ValidationError):
        StateLayout(name="units", axes=("tcp_x",), units=("m", "m"))


def test_normalization_must_be_none() -> None:
    with pytest.raises(ValidationError):
        StateLayout(name="norm", axes=("tcp_x",), units=("m",), normalization="server_side")


def test_state_scalar_values_has_expected_keys() -> None:
    values = state_scalar_values(make_robot_state())
    for key in ("tcp_x", "rot6d_0", "tcp_wx", "fx", "tz", "q6", "dq6", "gripper_width"):
        assert key in values


def test_letterbox_shape_and_centering() -> None:
    image = np.zeros((4, 8, 3), dtype=np.uint8)
    image[:, :, 0] = 10
    image[:, :, 1] = 20
    image[:, :, 2] = 30
    out = letterbox_rgb(image, 8)
    assert out.shape == (8, 8, 3)
    x0, y0, new_w, new_h = letterbox_geometry(8, 4, 8)
    assert (x0, y0, new_w, new_h) == (0, 2, 8, 4)
    assert np.all(out[0:2] == 0)  # top padding
    assert np.all(out[2:6, :, 0] == 10)  # R preserved
    assert np.all(out[2:6, :, 1] == 20)  # G preserved
    assert np.all(out[2:6, :, 2] == 30)  # B preserved


def test_letterbox_rejects_non_rgb_uint8() -> None:
    with pytest.raises(ValidationError):
        letterbox_rgb(np.zeros((4, 4), dtype=np.uint8), 8)
    with pytest.raises(ValidationError):
        letterbox_rgb(np.zeros((4, 4, 3), dtype=np.uint16), 8)


def test_build_policy_observation_contract() -> None:
    obs = make_observation(sequence_id=9, capture_time_ns=1_000_000_000)
    payload = build_policy_observation(obs, CANONICAL_STATE_LAYOUT, image_size=64)
    assert payload[POLICY_KEY_IMAGE].shape == (64, 64, 3)
    assert payload[POLICY_KEY_WRIST_IMAGE].shape == (64, 64, 3)
    assert payload[POLICY_KEY_STATE].dtype == np.float32
    assert payload[POLICY_KEY_STATE].shape == (22,)
    assert payload[POLICY_KEY_PROMPT] == obs.task
    assert payload[POLICY_KEY_SEQUENCE] == 9
    assert payload[POLICY_KEY_CAPTURE_TIME] == 1_000_000_000
    assert payload[POLICY_KEY_IMAGE_ENCODING] == "rgb8"
    assert payload[POLICY_KEY_STATE_LAYOUT]["name"] == "canonical_v1"
    assert payload[POLICY_KEY_STATE_DIM] == 22
    validate_policy_observation(
        payload,
        expected_layout=CANONICAL_STATE_LAYOUT,
        expected_image_size=64,
    )


def test_default_image_size_is_224() -> None:
    payload = build_policy_observation(make_observation(), CANONICAL_STATE_LAYOUT)
    assert payload[POLICY_KEY_IMAGE].shape == (DEFAULT_POLICY_IMAGE_SIZE,) * 2 + (3,)


def test_validate_policy_observation_rejects_mutations() -> None:
    payload = build_policy_observation(make_observation(), CANONICAL_STATE_LAYOUT, image_size=32)
    bad_encoding = dict(payload)
    bad_encoding[POLICY_KEY_IMAGE_ENCODING] = "bgr8"
    with pytest.raises(ValidationError) as exc:
        validate_policy_observation(bad_encoding, expected_layout=CANONICAL_STATE_LAYOUT, expected_image_size=32)
    assert exc.value.code is ErrorCode.INVALID_ENCODING

    bad_state = dict(payload)
    bad_state[POLICY_KEY_STATE] = payload[POLICY_KEY_STATE].astype(np.float64)
    with pytest.raises(ValidationError):
        validate_policy_observation(bad_state, expected_layout=CANONICAL_STATE_LAYOUT, expected_image_size=32)

    bad_dim = dict(payload)
    bad_dim[POLICY_KEY_STATE_DIM] = 21
    with pytest.raises(ValidationError):
        validate_policy_observation(bad_dim, expected_layout=CANONICAL_STATE_LAYOUT, expected_image_size=32)

    bad_prompt = dict(payload)
    bad_prompt[POLICY_KEY_PROMPT] = "   "
    with pytest.raises(ValidationError):
        validate_policy_observation(bad_prompt, expected_layout=CANONICAL_STATE_LAYOUT, expected_image_size=32)

    other_layout = StateLayout(name="other", axes=("tcp_x",), units=("m",))
    with pytest.raises(ValidationError) as exc:
        validate_policy_observation(payload, expected_layout=other_layout, expected_image_size=32)
    assert exc.value.code is ErrorCode.SCHEMA_MISMATCH

    bad_schema = dict(payload)
    bad_schema[POLICY_KEY_SCHEMA] = "policy-observation/v999"
    with pytest.raises(ValidationError) as exc:
        validate_policy_observation(bad_schema, expected_layout=CANONICAL_STATE_LAYOUT, expected_image_size=32)
    assert exc.value.code is ErrorCode.SCHEMA_MISMATCH


def test_rgb_channels_are_not_swapped_in_letterbox() -> None:
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[:, :, 0] = 200
    payload = build_policy_observation(
        make_observation(
            images={
                "base_rgb": make_image("base_rgb", data=rgb, host_time_ns=1_000_000_000),
                "wrist_rgb": make_image("wrist_rgb", data=rgb, host_time_ns=1_000_000_000),
            }
        ),
        CANONICAL_STATE_LAYOUT,
        image_size=8,
    )
    assert payload[POLICY_KEY_IMAGE][4, 4].tolist() == [200, 0, 0]
