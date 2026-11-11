"""Test-support helpers shipped with the package.

These builders create deterministic contract objects, fake-device state and
safety profiles.  They are imported by the test suite and by the shadow harness
demo; they contain no hardware or network access.
"""

from __future__ import annotations

from .builders import (
    IDENTITY_QUAT,
    make_chunk,
    make_depth_image,
    make_image,
    make_observation,
    make_pose,
    make_robot_sample,
    make_robot_state,
    make_safety_profile,
    make_sync,
)

__all__ = [
    "IDENTITY_QUAT",
    "make_chunk",
    "make_depth_image",
    "make_image",
    "make_observation",
    "make_pose",
    "make_robot_sample",
    "make_robot_state",
    "make_safety_profile",
    "make_sync",
]
