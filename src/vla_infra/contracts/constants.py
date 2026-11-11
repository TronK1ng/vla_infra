"""Frozen constants that define the v1 contracts.

Changing anything here is a schema change and requires a new
``*_schema_version`` plus a migration note in ``exchange/worker``.
"""

from __future__ import annotations

from typing import Tuple

# --- schema versions ---------------------------------------------------------
OBSERVATION_SCHEMA_VERSION = "observation/v1"
ACTION_SCHEMA_VERSION = "action/v1"
POLICY_OBSERVATION_SCHEMA_VERSION = "policy-observation/v1"
WIRE_SCHEMA_VERSION = "wire/v1"
SUPPORTED_OBSERVATION_SCHEMAS = frozenset({OBSERVATION_SCHEMA_VERSION})
SUPPORTED_ACTION_SCHEMAS = frozenset({ACTION_SCHEMA_VERSION})
SUPPORTED_WIRE_SCHEMAS = frozenset({WIRE_SCHEMA_VERSION})

# --- images ------------------------------------------------------------------
IMAGE_ENCODING_RGB8 = "rgb8"
IMAGE_ENCODING_DEPTH16 = "depth16"
SUPPORTED_IMAGE_ENCODINGS = frozenset({IMAGE_ENCODING_RGB8, IMAGE_ENCODING_DEPTH16})
IMAGE_KEY_BASE_RGB = "base_rgb"
IMAGE_KEY_WRIST_RGB = "wrist_rgb"
IMAGE_KEY_BASE_DEPTH = "base_depth"
IMAGE_KEY_WRIST_DEPTH = "wrist_depth"
REQUIRED_RGB_IMAGES: Tuple[str, ...] = (IMAGE_KEY_BASE_RGB, IMAGE_KEY_WRIST_RGB)
OPTIONAL_DEPTH_IMAGES: Tuple[str, ...] = (IMAGE_KEY_BASE_DEPTH, IMAGE_KEY_WRIST_DEPTH)

# --- robot state -------------------------------------------------------------
JOINT_COUNT = 7
TCP_POSITION_DIM = 3
QUATERNION_DIM = 4
ROTATION_6D_DIM = 6
TCP_VELOCITY_DIM = 6
WRENCH_DIM = 6
GRIPPER_WIDTH_DIM = 1

ROBOT_STATE_SHAPES = {
    "joint_position": (JOINT_COUNT,),
    "joint_velocity": (JOINT_COUNT,),
    "tcp_position": (TCP_POSITION_DIM,),
    "tcp_quaternion_xyzw": (QUATERNION_DIM,),
    "tcp_rotation_6d": (ROTATION_6D_DIM,),
    "tcp_velocity": (TCP_VELOCITY_DIM,),
    "wrench_tcp": (WRENCH_DIM,),
    "gripper_width": (GRIPPER_WIDTH_DIM,),
}

ROBOT_STATE_UNITS = {
    "joint_position": "rad",
    "joint_velocity": "rad/s",
    "tcp_position": "m",
    "tcp_quaternion_xyzw": "unit",
    "tcp_rotation_6d": "unit",
    "tcp_velocity": "m/s,rad/s",
    "wrench_tcp": "N,Nm",
    "gripper_width": "m",
}

#: ``tcp_velocity`` is documented as a 6-vector in this frame.  M1 only
#: supports base-frame velocity; device adapters must rotate before publishing.
TCP_VELOCITY_FRAME = "base"
#: Wrench is measured at the TCP; the contract fixes its frame and ordering.
WRENCH_FRAME = "tcp"
WRENCH_AXES = ("fx", "fy", "fz", "tx", "ty", "tz")
TCP_POSITION_AXES = ("tcp_x", "tcp_y", "tcp_z")
ROTATION_6D_AXES = tuple(f"rot6d_{i}" for i in range(ROTATION_6D_DIM))
TCP_VELOCITY_AXES = ("tcp_vx", "tcp_vy", "tcp_vz", "tcp_wx", "tcp_wy", "tcp_wz")
JOINT_POSITION_AXES = tuple(f"q{i}" for i in range(JOINT_COUNT))
JOINT_VELOCITY_AXES = tuple(f"dq{i}" for i in range(JOINT_COUNT))
GRIPPER_AXES = ("gripper_width",)
#: Normalised gripper command convention; recorded in policy/data metadata.
GRIPPER_SEMANTICS = "closed_0_open_1"

# --- actions -----------------------------------------------------------------
ACTION_DIM = 7
ACTION_AXES: Tuple[str, ...] = ("dx", "dy", "dz", "drx", "dry", "drz", "gripper")
ACTION_UNITS: Tuple[str, ...] = ("m", "m", "m", "rad", "rad", "rad", "unit")
ACTION_FRAME_BASE = "base"
ACTION_FRAME_TCP = "tcp"
ACTION_FRAMES = frozenset({ACTION_FRAME_BASE, ACTION_FRAME_TCP})
ACTION_TYPE_DELTA_TCP = "delta_tcp"
ACTION_TYPE_ABSOLUTE_TCP = "absolute_tcp"
ACTION_TYPES = frozenset({ACTION_TYPE_DELTA_TCP, ACTION_TYPE_ABSOLUTE_TCP})
MAX_ACTION_HORIZON = 512
MIN_ACTION_DT_S = 1.0 / 1000.0
MAX_ACTION_DT_S = 1.0
#: Force-policy-specific keys.  They are *optional* extras; no lower layer may
#: require them.
PROPRIETARY_EXTRAS_KEYS: Tuple[str, ...] = (
    "reference_wrench",
    "interaction_frame",
    "selection_mask",
    "policy_mode",
)

# --- policy observation ------------------------------------------------------
POLICY_KEY_IMAGE = "observation/image"
POLICY_KEY_WRIST_IMAGE = "observation/wrist_image"
POLICY_KEY_STATE = "observation/state"
POLICY_KEY_PROMPT = "prompt"
POLICY_KEY_SEQUENCE = "meta/sequence_id"
POLICY_KEY_CAPTURE_TIME = "meta/capture_time_ns"
POLICY_KEY_SCHEMA = "meta/schema_version"
POLICY_KEY_IMAGE_ENCODING = "meta/image_encoding"
POLICY_KEY_STATE_LAYOUT = "meta/state_layout"
POLICY_KEY_STATE_DIM = "meta/state_dim"
POLICY_KEY_SOURCE_IMAGE_SIZE = "meta/source_image_size"

POLICY_OBSERVATION_KEYS: Tuple[str, ...] = (
    POLICY_KEY_IMAGE,
    POLICY_KEY_WRIST_IMAGE,
    POLICY_KEY_STATE,
    POLICY_KEY_PROMPT,
    POLICY_KEY_SEQUENCE,
    POLICY_KEY_CAPTURE_TIME,
    POLICY_KEY_SCHEMA,
    POLICY_KEY_IMAGE_ENCODING,
    POLICY_KEY_STATE_LAYOUT,
    POLICY_KEY_STATE_DIM,
)

DEFAULT_POLICY_IMAGE_SIZE = 224
