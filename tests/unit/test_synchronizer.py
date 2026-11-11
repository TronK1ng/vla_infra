"""Observation synchroniser: interpolation, provenance, staleness rejection."""

from __future__ import annotations

import numpy as np
import pytest

from vla_infra.clock import FakeClock
from vla_infra.contracts.constants import IMAGE_ENCODING_DEPTH16, IMAGE_KEY_BASE_DEPTH, IMAGE_KEY_BASE_RGB, IMAGE_KEY_WRIST_RGB
from vla_infra.contracts.observation import ImageFrame
from vla_infra.errors import SyncError
from vla_infra.geometry import axis_angle_to_matrix, matrix_to_quaternion
from vla_infra.sync.synchronizer import ObservationSynchronizer, SyncConfig
from vla_infra.testing import make_image, make_robot_sample, make_robot_state


def _images(t: int, wrist_offset: int = 0):
    return {
        IMAGE_KEY_BASE_RGB: make_image(IMAGE_KEY_BASE_RGB, host_time_ns=t, device_time_ns=t - 1),
        IMAGE_KEY_WRIST_RGB: make_image(
            IMAGE_KEY_WRIST_RGB, host_time_ns=t + wrist_offset, device_time_ns=t + wrist_offset - 1
        ),
    }


def _synchronizer(**overrides) -> tuple[FakeClock, ObservationSynchronizer]:
    clock = FakeClock(start_ns=1_000_000_000)
    values = {
        "max_sync_error_ns": 5_000_000,
        "max_state_gap_ns": 20_000_000,
        "max_camera_skew_ns": 5_000_000,
    }
    values.update(overrides)
    return clock, ObservationSynchronizer(clock, SyncConfig(**values))


def test_linear_and_slerp_interpolation() -> None:
    clock, sync = _synchronizer()
    quat = matrix_to_quaternion(axis_angle_to_matrix([0.0, 0.0, np.pi / 2]))
    sync.add_robot_sample(make_robot_sample(clock, sequence_id=0, host_time_ns=1_000_000_000))
    sync.add_robot_sample(
        make_robot_sample(
            clock,
            sequence_id=1,
            host_time_ns=1_010_000_000,
            state=make_robot_state(position=(0.5, 0.1, 0.3), quaternion=quat, wrench=(1, 0, 0, 0, 0, 0), gripper_width=0.08),
        )
    )
    observation = sync.build(_images(1_005_000_000), task="pick")
    assert observation.sync.interpolation_alpha == pytest.approx(0.5)
    assert observation.sync.max_age_ns == 5_000_000
    assert observation.sync.state_gap_ns == 10_000_000
    assert np.allclose(observation.robot.tcp_position, [0.5, 0.05, 0.3], atol=1e-12)
    # 45 degrees about Z
    assert np.allclose(
        np.abs(observation.robot.tcp_quaternion_xyzw),
        np.abs([0.0, 0.0, np.sin(np.pi / 8), np.cos(np.pi / 8)]),
        atol=1e-9,
    )
    # wrench and gripper use zero-order hold (before sample)
    assert np.allclose(observation.robot.wrench_tcp, [0, 0, 0, 0, 0, 0])
    assert observation.robot.gripper_width_m == pytest.approx(0.04)
    assert observation.capture_time_ns == 1_005_000_000
    assert observation.sync.camera_skew_ns == 0


def test_exact_sample_uses_alpha_zero() -> None:
    clock, sync = _synchronizer()
    sync.add_robot_sample(make_robot_sample(clock, sequence_id=0, host_time_ns=1_000_000_000))
    sync.add_robot_sample(make_robot_sample(clock, sequence_id=1, host_time_ns=1_010_000_000))
    observation = sync.build(_images(1_000_000_000), task="pick")
    assert observation.sync.interpolation_alpha == 0.0
    assert observation.sync.robot_sample_before_ns == observation.sync.robot_sample_after_ns == 1_000_000_000
    assert observation.sync.max_age_ns == 0


def test_no_future_sample_is_used() -> None:
    clock, sync = _synchronizer()
    sync.add_robot_sample(make_robot_sample(clock, sequence_id=0, host_time_ns=1_000_000_000))
    sync.add_robot_sample(make_robot_sample(clock, sequence_id=1, host_time_ns=1_010_000_000))
    with pytest.raises(SyncError) as exc:
        sync.build(_images(1_020_000_000), task="pick")
    assert exc.value.details["reason"] == "robot_stale"
    assert sync.stats.dropped["robot_stale"] == 1


def test_robot_stale_when_buffer_empty() -> None:
    clock, sync = _synchronizer()
    with pytest.raises(SyncError):
        sync.build(_images(1_000_000_000), task="pick")


def test_camera_skew_rejected() -> None:
    clock, sync = _synchronizer()
    sync.add_robot_sample(make_robot_sample(clock, sequence_id=0, host_time_ns=1_000_000_000))
    sync.add_robot_sample(make_robot_sample(clock, sequence_id=1, host_time_ns=1_020_000_000))
    with pytest.raises(SyncError) as exc:
        sync.build(_images(1_000_000_000, wrist_offset=10_000_000), task="pick")
    assert exc.value.details["reason"] == "camera_skew"
    assert sync.stats.dropped["camera_skew"] == 1


def test_state_gap_rejected() -> None:
    clock, sync = _synchronizer(max_state_gap_ns=5_000_000, max_sync_error_ns=3_000_000)
    sync.add_robot_sample(make_robot_sample(clock, sequence_id=0, host_time_ns=1_000_000_000))
    sync.add_robot_sample(make_robot_sample(clock, sequence_id=1, host_time_ns=1_010_000_000))
    with pytest.raises(SyncError) as exc:
        sync.build(_images(1_005_000_000), task="pick")
    assert exc.value.details["reason"] == "state_gap"


def test_sync_error_rejected() -> None:
    clock, sync = _synchronizer(max_sync_error_ns=2_000_000, max_state_gap_ns=20_000_000)
    sync.add_robot_sample(make_robot_sample(clock, sequence_id=0, host_time_ns=1_000_000_000))
    sync.add_robot_sample(make_robot_sample(clock, sequence_id=1, host_time_ns=1_010_000_000))
    with pytest.raises(SyncError) as exc:
        sync.build(_images(1_005_000_000), task="pick")
    assert exc.value.details["reason"] == "sync_error"
    assert sync.stats.dropped["sync_error"] == 1


def test_unexpected_target_override() -> None:
    clock, sync = _synchronizer()
    sync.add_robot_sample(make_robot_sample(clock, sequence_id=0, host_time_ns=1_000_000_000))
    sync.add_robot_sample(make_robot_sample(clock, sequence_id=1, host_time_ns=1_010_000_000))
    with pytest.raises(SyncError) as exc:
        sync.build(_images(1_005_000_000), task="pick", target_time_ns=1_000_000_000)
    assert exc.value.details["reason"] == "future_image"


def test_sequence_only_advances_on_success() -> None:
    clock, sync = _synchronizer()
    sync.add_robot_sample(make_robot_sample(clock, sequence_id=0, host_time_ns=1_000_000_000))
    first = sync.build(_images(1_000_000_000), task="pick")
    assert first.sequence_id == 0
    with pytest.raises(SyncError):
        sync.build(_images(2_000_000_000), task="pick")
    sync.add_robot_sample(make_robot_sample(clock, sequence_id=1, host_time_ns=1_010_000_000))
    second = sync.build(_images(1_005_000_000), task="pick")
    assert second.sequence_id == 1
    assert sync.stats.built == 2


def test_depth_image_is_preserved() -> None:
    clock, sync = _synchronizer()
    sync.add_robot_sample(make_robot_sample(clock, sequence_id=0, host_time_ns=1_000_000_000))
    images = _images(1_000_000_000)
    images[IMAGE_KEY_BASE_DEPTH] = ImageFrame(
        name=IMAGE_KEY_BASE_DEPTH,
        data=np.full((4, 4), 1234, dtype=np.uint16),
        device_time_ns=999,
        host_time_ns=1_000_000_000,
        frame_number=0,
        encoding=IMAGE_ENCODING_DEPTH16,
    )
    observation = sync.build(images, task="pick")
    assert observation.base_depth is not None
    assert int(observation.base_depth.data[0, 0]) == 1234


def test_linear_policies_when_configured() -> None:
    clock, sync = _synchronizer(wrench_policy="linear", gripper_policy="linear")
    sync.add_robot_sample(make_robot_sample(clock, sequence_id=0, host_time_ns=1_000_000_000))
    sync.add_robot_sample(
        make_robot_sample(
            clock,
            sequence_id=1,
            host_time_ns=1_010_000_000,
            state=make_robot_state(wrench=(2, 0, 0, 0, 0, 0), gripper_width=0.08),
        )
    )
    observation = sync.build(_images(1_005_000_000), task="pick")
    assert np.allclose(observation.robot.wrench_tcp, [1, 0, 0, 0, 0, 0])
    assert observation.robot.gripper_width_m == pytest.approx(0.06)
