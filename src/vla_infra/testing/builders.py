"""Deterministic builders shared by unit and integration tests."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from vla_infra.clock import FakeClock
from vla_infra.contracts.action import ActionChunk
from vla_infra.contracts.constants import (
    ACTION_FRAME_BASE,
    ACTION_SCHEMA_VERSION,
    ACTION_TYPE_DELTA_TCP,
    IMAGE_ENCODING_DEPTH16,
    IMAGE_ENCODING_RGB8,
    IMAGE_KEY_BASE_RGB,
    IMAGE_KEY_WRIST_RGB,
    OBSERVATION_SCHEMA_VERSION,
)
from vla_infra.contracts.observation import (
    CanonicalObservation,
    ImageFrame,
    RobotState,
    RobotStateSample,
    SyncInfo,
)
from vla_infra.contracts.pose import CartesianPose
from vla_infra.geometry import matrix_to_rotation_6d, quaternion_to_matrix

IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def make_pose(
    position: Sequence[float] = (0.5, 0.0, 0.3),
    quaternion: Sequence[float] = IDENTITY_QUAT,
) -> CartesianPose:
    return CartesianPose(np.asarray(position, dtype=np.float64), np.asarray(quaternion, dtype=np.float64))


def make_robot_state(
    *,
    position: Sequence[float] = (0.5, 0.0, 0.3),
    quaternion: Sequence[float] = IDENTITY_QUAT,
    joints: Optional[Sequence[float]] = None,
    joint_velocity: Optional[Sequence[float]] = None,
    tcp_velocity: Optional[Sequence[float]] = None,
    wrench: Optional[Sequence[float]] = None,
    gripper_width: float = 0.04,
) -> RobotState:
    quat = np.asarray(quaternion, dtype=np.float64)
    rot6d = matrix_to_rotation_6d(quaternion_to_matrix(quat))
    return RobotState(
        joint_position=np.asarray(joints if joints is not None else np.zeros(7), dtype=np.float64),
        joint_velocity=np.asarray(joint_velocity if joint_velocity is not None else np.zeros(7), dtype=np.float64),
        tcp_position=np.asarray(position, dtype=np.float64),
        tcp_quaternion_xyzw=quat,
        tcp_rotation_6d=rot6d,
        tcp_velocity=np.asarray(tcp_velocity if tcp_velocity is not None else np.zeros(6), dtype=np.float64),
        wrench_tcp=np.asarray(wrench if wrench is not None else np.zeros(6), dtype=np.float64),
        gripper_width=np.array([gripper_width], dtype=np.float64),
    )


def make_image(
    name: str = IMAGE_KEY_BASE_RGB,
    *,
    value: int = 16,
    width: int = 8,
    height: int = 8,
    device_time_ns: int = 1_000_000_000,
    host_time_ns: int = 1_000_000_000,
    frame_number: int = 0,
    encoding: str = IMAGE_ENCODING_RGB8,
    data: Optional[np.ndarray] = None,
) -> ImageFrame:
    if data is None:
        if encoding == IMAGE_ENCODING_RGB8:
            data = np.full((height, width, 3), value, dtype=np.uint8)
        else:
            data = np.full((height, width), value, dtype=np.uint16)
    return ImageFrame(
        name=name,
        data=data,
        device_time_ns=device_time_ns,
        host_time_ns=host_time_ns,
        frame_number=frame_number,
        encoding=encoding,
    )


def make_sync(
    *,
    before_ns: int,
    after_ns: int,
    capture_ns: int,
    dropped: Optional[Mapping[str, int]] = None,
    camera_skew_ns: int = 0,
    interpolation_applied: bool = True,
) -> SyncInfo:
    gap = after_ns - before_ns
    alpha = 0.0 if gap == 0 else (capture_ns - before_ns) / float(gap)
    max_age = max(capture_ns - before_ns, after_ns - capture_ns)
    return SyncInfo(
        robot_sample_before_ns=before_ns,
        robot_sample_after_ns=after_ns,
        interpolation_alpha=alpha,
        max_age_ns=max_age,
        dropped_frames=dict(dropped or {}),
        interpolation_applied=interpolation_applied,
        camera_skew_ns=camera_skew_ns,
        state_gap_ns=gap,
    )


def make_observation(
    *,
    sequence_id: int = 0,
    capture_time_ns: int = 1_000_000_000,
    task: str = "pick up the red cube",
    robot: Optional[RobotState] = None,
    images: Optional[Mapping[str, ImageFrame]] = None,
    sync: Optional[SyncInfo] = None,
    base_offset_ns: int = 0,
    wrist_offset_ns: int = 0,
) -> CanonicalObservation:
    if images is None:
        images = {
            IMAGE_KEY_BASE_RGB: make_image(
                IMAGE_KEY_BASE_RGB,
                host_time_ns=capture_time_ns + base_offset_ns,
                device_time_ns=capture_time_ns + base_offset_ns - 1_000_000,
            ),
            IMAGE_KEY_WRIST_RGB: make_image(
                IMAGE_KEY_WRIST_RGB,
                host_time_ns=capture_time_ns + wrist_offset_ns,
                device_time_ns=capture_time_ns + wrist_offset_ns - 1_000_000,
            ),
        }
    skew = max(
        abs(images[IMAGE_KEY_BASE_RGB].host_time_ns - capture_time_ns),
        abs(images[IMAGE_KEY_WRIST_RGB].host_time_ns - capture_time_ns),
    )
    if sync is None:
        sync = make_sync(before_ns=capture_time_ns, after_ns=capture_time_ns, capture_ns=capture_time_ns, camera_skew_ns=skew)
    return CanonicalObservation(
        schema_version=OBSERVATION_SCHEMA_VERSION,
        sequence_id=sequence_id,
        capture_time_ns=capture_time_ns,
        task=task,
        images=images,
        robot=robot if robot is not None else make_robot_state(),
        sync=sync,
    )


def make_chunk(
    *,
    actions: Optional[np.ndarray] = None,
    request_sequence_id: int = 1,
    observation_capture_time_ns: int = 1_000_000_000,
    action_dt_s: float = 0.05,
    action_frame: str = ACTION_FRAME_BASE,
    action_type: str = ACTION_TYPE_DELTA_TCP,
    extras: Optional[Mapping[str, Any]] = None,
    horizon: int = 4,
    delta_m: Sequence[float] = (0.001, 0.0, 0.0),
    gripper: float = 0.5,
    schema_version: str = ACTION_SCHEMA_VERSION,
) -> ActionChunk:
    if actions is None:
        rows = np.zeros((horizon, 7), dtype=np.float32)
        rows[:, 0:3] = np.asarray(delta_m, dtype=np.float32)
        rows[:, 6] = np.float32(gripper)
        actions = rows
    return ActionChunk(
        request_sequence_id=request_sequence_id,
        observation_capture_time_ns=observation_capture_time_ns,
        action_dt_s=action_dt_s,
        action_frame=action_frame,
        action_type=action_type,
        actions=actions,
        extras=dict(extras or {}),
        schema_version=schema_version,
    )


def make_robot_sample(
    clock: FakeClock,
    *,
    sequence_id: int = 0,
    state: Optional[RobotState] = None,
    device_time_ns: Optional[int] = None,
    host_time_ns: Optional[int] = None,
) -> RobotStateSample:
    host = clock.now_ns() if host_time_ns is None else host_time_ns
    return RobotStateSample(
        state=state if state is not None else make_robot_state(),
        device_time_ns=(
            device_time_ns if device_time_ns is not None else max(1, host - 500_000)
        ),
        host_time_ns=host,
        sequence_id=sequence_id,
    )


def make_safety_profile(**overrides: Any):
    from vla_infra.execution.safety import SafetyProfile

    defaults: Dict[str, Any] = dict(
        workspace_min=(-0.5, -0.5, 0.0),
        workspace_max=(0.9, 0.5, 0.8),
        tcp_height_min_m=0.02,
        tcp_height_max_m=0.75,
        max_step_translation_m=0.05,
        max_step_rotation_rad=0.25,
        max_linear_velocity_m_s=1.0,
        max_angular_velocity_rad_s=3.0,
        max_first_target_translation_m=0.1,
        max_first_target_rotation_rad=0.3,
        max_observation_age_ns=1_000_000_000,
        max_chunk_age_ns=2_000_000_000,
        max_force_n=100.0,
        max_torque_nm=20.0,
        gripper_width_min_m=0.0,
        gripper_width_max_m=0.1,
        gripper_max_speed_m_s=0.2,
        allowed_action_frames=("base", "tcp"),
        allowed_action_types=("delta_tcp", "absolute_tcp"),
        min_segment_dt_ns=1_000_000,
    )
    defaults.update(overrides)
    return SafetyProfile(**defaults)


def make_depth_image(name: str = "base_depth", **kwargs: Any) -> ImageFrame:
    return make_image(name=name, encoding=IMAGE_ENCODING_DEPTH16, value=1000, **kwargs)
