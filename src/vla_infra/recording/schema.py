"""On-disk low-dimensional schema for raw episodes (M2 draft).

The schema is explicit and self-describing: fields, dtype, shape, units and
frames are written next to the binary so a converter never has to guess.  The
binary format is a flat, append-only numpy structured array; it is deliberately
not Force Policy's HDF5 layout.  A field-mapping manifest is included in the
episode metadata so a future converter can produce the Force Policy layout
without copying its code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from ..contracts.constants import (
    GRIPPER_SEMANTICS,
    JOINT_COUNT,
    QUATERNION_DIM,
    ROTATION_6D_DIM,
    TCP_POSITION_DIM,
    TCP_VELOCITY_DIM,
    WRENCH_AXES,
    WRENCH_DIM,
    WRENCH_FRAME,
)
from ..contracts.observation import RobotStateSample

LOWDIM_SCHEMA_VERSION = "lowdim/v1"


@dataclass(frozen=True)
class LowdimField:
    name: str
    dtype: str
    shape: Tuple[int, ...]
    unit: str
    frame: str

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "unit": self.unit,
            "frame": self.frame,
        }


LOWDIM_FIELDS: Tuple[LowdimField, ...] = (
    LowdimField("host_time_ns", "<i8", (), "ns", "host_monotonic"),
    LowdimField("device_time_ns", "<i8", (), "ns", "device"),
    LowdimField("sequence_id", "<i8", (), "1", "counter"),
    LowdimField("joint_position", "<f8", (JOINT_COUNT,), "rad", "joint"),
    LowdimField("joint_velocity", "<f8", (JOINT_COUNT,), "rad/s", "joint"),
    LowdimField("tcp_position", "<f8", (TCP_POSITION_DIM,), "m", "base"),
    LowdimField("tcp_quaternion_xyzw", "<f8", (QUATERNION_DIM,), "unit", "base"),
    LowdimField("tcp_rotation_6d", "<f8", (ROTATION_6D_DIM,), "unit", "base"),
    LowdimField("tcp_velocity", "<f8", (TCP_VELOCITY_DIM,), "m/s,rad/s", "base"),
    LowdimField("wrench_tcp", "<f8", (WRENCH_DIM,), "N,Nm", WRENCH_FRAME),
    LowdimField("gripper_width", "<f8", (1,), "m", "tcp"),
)

LOWDIM_DTYPE = np.dtype([(field.name, field.dtype, field.shape) for field in LOWDIM_FIELDS])


def schema_dict() -> Dict[str, Any]:
    return {
        "schema_version": LOWDIM_SCHEMA_VERSION,
        "dtype": LOWDIM_DTYPE.str,
        "itemsize": LOWDIM_DTYPE.itemsize,
        "fields": [field.as_dict() for field in LOWDIM_FIELDS],
        "ordering": "little-endian structured record, append-only",
        "contracts": {
            "gripper_semantics": GRIPPER_SEMANTICS,
            "wrench_axes": list(WRENCH_AXES),
            "wrench_frame": WRENCH_FRAME,
        },
        # Mapping is documented, not applied: Force Policy conversion is an M6
        # adapter concern and may require its own HDF5 field names.
        "force_policy_mapping_note": (
            "Use host_time_ns/device_time_ns as raw timing provenance; convert "
            "tcp_quaternion_xyzw to the converter's required order explicitly."
        ),
    }


def samples_to_rows(samples: Iterable[RobotStateSample]) -> np.ndarray:
    samples = list(samples)
    rows = np.zeros(len(samples), dtype=LOWDIM_DTYPE)
    for index, sample in enumerate(samples):
        state = sample.state
        rows[index]["host_time_ns"] = sample.host_time_ns
        rows[index]["device_time_ns"] = sample.device_time_ns
        rows[index]["sequence_id"] = sample.sequence_id
        rows[index]["joint_position"] = state.joint_position
        rows[index]["joint_velocity"] = state.joint_velocity
        rows[index]["tcp_position"] = state.tcp_position
        rows[index]["tcp_quaternion_xyzw"] = state.tcp_quaternion_xyzw
        rows[index]["tcp_rotation_6d"] = state.tcp_rotation_6d
        rows[index]["tcp_velocity"] = state.tcp_velocity
        rows[index]["wrench_tcp"] = state.wrench_tcp
        rows[index]["gripper_width"] = state.gripper_width
    return rows


def row_to_robot_state(row: np.void):
    from ..contracts.observation import RobotState

    return RobotState(
        joint_position=row["joint_position"],
        joint_velocity=row["joint_velocity"],
        tcp_position=row["tcp_position"],
        tcp_quaternion_xyzw=row["tcp_quaternion_xyzw"],
        tcp_rotation_6d=row["tcp_rotation_6d"],
        tcp_velocity=row["tcp_velocity"],
        wrench_tcp=row["wrench_tcp"],
        gripper_width=row["gripper_width"],
    )


def row_to_dict(row: np.void) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for field in LOWDIM_FIELDS:
        value = row[field.name]
        result[field.name] = value.tolist() if hasattr(value, "tolist") else int(value)
    return result


__all__ = [
    "LOWDIM_DTYPE",
    "LOWDIM_FIELDS",
    "LOWDIM_SCHEMA_VERSION",
    "LowdimField",
    "row_to_dict",
    "row_to_robot_state",
    "samples_to_rows",
    "schema_dict",
]
