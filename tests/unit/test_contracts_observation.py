"""Canonical observation contract: shape, dtype, frame, timestamp and sync checks."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from vla_infra.contracts.constants import (
    IMAGE_ENCODING_DEPTH16,
    IMAGE_KEY_BASE_DEPTH,
    IMAGE_KEY_BASE_RGB,
    IMAGE_KEY_WRIST_RGB,
    OBSERVATION_SCHEMA_VERSION,
)
from vla_infra.contracts.observation import ImageFrame, RobotState, SyncInfo
from vla_infra.errors import ErrorCode, ValidationError
from vla_infra.testing import (
    make_depth_image,
    make_image,
    make_observation,
    make_robot_state,
    make_sync,
)


def test_valid_observation_roundtrip() -> None:
    obs = make_observation(sequence_id=3)
    assert obs.schema_version == OBSERVATION_SCHEMA_VERSION
    assert obs.sequence_id == 3
    assert set(obs.images) == {IMAGE_KEY_BASE_RGB, IMAGE_KEY_WRIST_RGB}
    assert obs.robot.gripper_width_m == pytest.approx(0.04)
    assert obs.base_rgb.data.flags.writeable is False


def test_observation_is_frozen() -> None:
    obs = make_observation()
    with pytest.raises(FrozenInstanceError):
        obs.task = "other"  # type: ignore[misc]
    with pytest.raises(ValueError):
        obs.base_rgb.data[0, 0, 0] = 1


def test_schema_mismatch_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        replace(make_observation(), schema_version="observation/v999")
    assert exc.value.code is ErrorCode.SCHEMA_MISMATCH


def test_missing_required_image_rejected() -> None:
    obs = make_observation()
    images = {IMAGE_KEY_BASE_RGB: obs.base_rgb}
    with pytest.raises(ValidationError) as exc:
        replace(obs, images=images)
    assert exc.value.code is ErrorCode.INVALID_SHAPE


def test_image_key_must_match_name() -> None:
    with pytest.raises(ValidationError) as exc:
        make_image(name=IMAGE_KEY_WRIST_RGB)
        ImageFrame(
            name=IMAGE_KEY_WRIST_RGB,
            data=np.zeros((4, 4, 3), dtype=np.uint8),
            device_time_ns=1,
            host_time_ns=1,
            frame_number=0,
            encoding="bgr8",  # type: ignore[arg-type]
        )
    assert exc.value.code is ErrorCode.INVALID_ENCODING


def test_rgb_image_dtype_and_shape() -> None:
    with pytest.raises(ValidationError) as exc:
        make_image(data=np.zeros((4, 4, 3), dtype=np.uint16))
    assert exc.value.code is ErrorCode.INVALID_DTYPE
    with pytest.raises(ValidationError) as exc:
        make_image(data=np.zeros((4, 4), dtype=np.uint8))
    assert exc.value.code is ErrorCode.INVALID_SHAPE
    with pytest.raises(ValidationError) as exc:
        make_image(data=np.zeros((4, 4, 4), dtype=np.uint8))
    assert exc.value.code is ErrorCode.INVALID_SHAPE


def test_depth_image_contract() -> None:
    depth = make_image(encoding=IMAGE_ENCODING_DEPTH16, value=1000)
    assert depth.data.dtype == np.uint16
    with pytest.raises(ValidationError):
        make_image(encoding=IMAGE_ENCODING_DEPTH16, data=np.zeros((4, 4), dtype=np.int16))


def test_optional_depth_images_accepted() -> None:
    obs = make_observation()
    images = {
        IMAGE_KEY_BASE_RGB: obs.base_rgb,
        IMAGE_KEY_WRIST_RGB: obs.wrist_rgb,
        IMAGE_KEY_BASE_DEPTH: make_depth_image(
            IMAGE_KEY_BASE_DEPTH, host_time_ns=obs.capture_time_ns, device_time_ns=1
        ),
    }
    updated = replace(obs, images=images)
    assert updated.base_depth is not None
    assert updated.wrist_depth is None


def test_rgb_and_depth_encoding_are_not_interchangeable() -> None:
    obs = make_observation()
    bad_depth_key = make_image(
        IMAGE_KEY_BASE_DEPTH, encoding=IMAGE_ENCODING_DEPTH16, host_time_ns=obs.capture_time_ns
    )
    images = {
        IMAGE_KEY_BASE_RGB: obs.base_rgb,
        IMAGE_KEY_WRIST_RGB: obs.wrist_rgb,
        IMAGE_KEY_BASE_DEPTH: ImageFrame(
            name=IMAGE_KEY_BASE_DEPTH,
            data=bad_depth_key.data,
            device_time_ns=1,
            host_time_ns=obs.capture_time_ns,
            frame_number=0,
            encoding=IMAGE_ENCODING_DEPTH16,
        ),
    }
    assert replace(obs, images=images).base_depth is not None
    with pytest.raises(ValidationError):
        replace(bad_depth_key, name=IMAGE_KEY_BASE_DEPTH, encoding="bgr8")  # type: ignore[arg-type]


def test_robot_state_non_finite_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        make_robot_state(joints=[0.0, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert exc.value.code is ErrorCode.NON_FINITE
    with pytest.raises(ValidationError):
        make_robot_state(position=[0.5, np.inf, 0.3])


def test_robot_state_rotation_consistency() -> None:
    state = make_robot_state()
    with pytest.raises(ValidationError) as exc:
        RobotState(
            joint_position=state.joint_position,
            joint_velocity=state.joint_velocity,
            tcp_position=state.tcp_position,
            tcp_quaternion_xyzw=state.tcp_quaternion_xyzw,
            tcp_rotation_6d=np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0]),
            tcp_velocity=state.tcp_velocity,
            wrench_tcp=state.wrench_tcp,
            gripper_width=state.gripper_width,
        )
    assert exc.value.code is ErrorCode.INVALID_QUATERNION


def test_robot_state_quaternion_must_be_unit() -> None:
    state = make_robot_state()
    with pytest.raises(ValidationError):
        RobotState(
            joint_position=state.joint_position,
            joint_velocity=state.joint_velocity,
            tcp_position=state.tcp_position,
            tcp_quaternion_xyzw=np.array([0.0, 0.0, 0.0, 0.9]),
            tcp_rotation_6d=state.tcp_rotation_6d,
            tcp_velocity=state.tcp_velocity,
            wrench_tcp=state.wrench_tcp,
            gripper_width=state.gripper_width,
        )


def test_gripper_width_sanity_bounds() -> None:
    with pytest.raises(ValidationError) as exc:
        make_robot_state(gripper_width=-0.01)
    assert exc.value.code is ErrorCode.SAFETY_GRIPPER
    with pytest.raises(ValidationError):
        make_robot_state(gripper_width=0.75)


def test_sync_alpha_must_match_sample_times() -> None:
    obs = make_observation()
    bad = replace(obs.sync, interpolation_alpha=0.9)
    with pytest.raises(ValidationError) as exc:
        replace(obs, sync=bad)
    assert exc.value.code is ErrorCode.INVALID_SYNC


def test_sync_bracket_must_contain_capture_time() -> None:
    obs = make_observation(capture_time_ns=1_000_000_000)
    sync = make_sync(before_ns=1_020_000_000, after_ns=1_030_000_000, capture_ns=1_025_000_000)
    with pytest.raises(ValidationError) as exc:
        replace(obs, sync=sync)
    assert exc.value.code is ErrorCode.INVALID_SYNC


def test_sync_max_age_must_be_honest() -> None:
    obs = make_observation(capture_time_ns=1_005_000_000)
    sync = make_sync(before_ns=1_000_000_000, after_ns=1_010_000_000, capture_ns=1_005_000_000)
    assert sync.max_age_ns == 5_000_000
    with pytest.raises(ValidationError):
        replace(obs, sync=replace(sync, max_age_ns=4_000_000))


def test_camera_skew_metadata_must_match_images() -> None:
    obs = make_observation()
    with pytest.raises(ValidationError) as exc:
        replace(obs, sync=replace(obs.sync, camera_skew_ns=5))
    assert exc.value.code is ErrorCode.INVALID_SYNC


def test_future_image_rejected() -> None:
    obs = make_observation(capture_time_ns=1_000_000_000)
    future = make_image(IMAGE_KEY_BASE_RGB, host_time_ns=1_000_000_100)
    images = {IMAGE_KEY_BASE_RGB: future, IMAGE_KEY_WRIST_RGB: obs.wrist_rgb}
    with pytest.raises(ValidationError) as exc:
        replace(obs, images=images)
    assert exc.value.code is ErrorCode.INVALID_SYNC


def test_extra_image_in_the_future_is_rejected() -> None:
    obs = make_observation(capture_time_ns=1_000_000_000)
    extra = make_image("extra_mono", host_time_ns=1_000_000_100)
    with pytest.raises(ValidationError) as exc:
        replace(obs, images={**obs.images, "extra_mono": extra})
    assert exc.value.code is ErrorCode.INVALID_SYNC


def test_task_must_be_non_empty() -> None:
    with pytest.raises(ValidationError):
        replace(make_observation(), task="   ")


def test_sequence_id_must_be_non_negative() -> None:
    with pytest.raises(ValidationError):
        replace(make_observation(), sequence_id=-1)
    with pytest.raises(ValidationError):
        replace(make_observation(), sequence_id="0")  # type: ignore[arg-type]


def test_dropped_frame_counters_must_be_non_negative() -> None:
    with pytest.raises(ValidationError):
        SyncInfo(
            robot_sample_before_ns=1,
            robot_sample_after_ns=1,
            interpolation_alpha=0.0,
            max_age_ns=0,
            dropped_frames={"sync": -1},
        )


def test_quaternion_sign_is_not_normalized_away() -> None:
    # The contract stores xyzw as given (normalized); q and -q are both valid.
    state = make_robot_state()
    flipped = replace(state, tcp_quaternion_xyzw=-state.tcp_quaternion_xyzw)
    assert np.allclose(flipped.tcp_rotation_6d, state.tcp_rotation_6d)
