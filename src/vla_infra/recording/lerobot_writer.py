"""LeRobot export draft (M2).

A full LeRobot dataset uses Parquet + encoded video.  This environment does not
ship PyArrow/FFmpeg, so M2 produces an explicitly labelled **JSONL draft** with
the standard feature names and metadata fields.  Two export modes are allowed:

* observation-only (`jsonl_observation_only_draft`) when the episode has no
  actions; no zero action labels are fabricated;
* with actions only when every observation has an explicitly correlated action
  record; otherwise export is refused.

The `observation.state` values are copied from the persisted canonical
observation state (the exact vector the online VLA input used), never rebuilt
from raw lowdim timing.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ..contracts.constants import (
    ACTION_AXES,
    ACTION_SCHEMA_VERSION,
    GRIPPER_SEMANTICS,
    OBSERVATION_SCHEMA_VERSION,
)
from ..contracts.policy_io import CANONICAL_STATE_LAYOUT, StateLayout
from ..errors import ErrorCode, VlaInfraError
from .image_codec import read_png
from .reader import RawEpisode, load_episode, validate_episode

LEROBOT_DRAFT_FORMAT = "jsonl_draft"
LEROBOT_OBSERVATION_ONLY_FORMAT = "jsonl_observation_only_draft"


class LeRobotExportError(VlaInfraError):
    code = ErrorCode.CONFIG_INVALID


@dataclass(frozen=True)
class LeRobotExportResult:
    path: Path
    frames: int
    images: int
    action_available: bool
    format: str = LEROBOT_DRAFT_FORMAT


def _json_line(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, default=_json_default) + "\n"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _observation_state(record: Mapping[str, Any]) -> Tuple[StateLayout, np.ndarray]:
    state = record.get("canonical_state")
    if not isinstance(state, Mapping):
        raise LeRobotExportError(
            "observation has no persisted canonical_state; refusing to reconstruct from raw lowdim",
            code=ErrorCode.SCHEMA_MISMATCH,
        )
    try:
        layout = StateLayout.from_dict(state["layout"])
        values = np.asarray(state["values"], dtype=np.float32).reshape(-1)
    except (KeyError, TypeError, ValueError) as exc:
        raise LeRobotExportError(
            "observation canonical_state is malformed",
            code=ErrorCode.INVALID_SHAPE,
        ) from exc
    if layout.axes != CANONICAL_STATE_LAYOUT.axes:
        raise LeRobotExportError(
            "exported observation state layout is not the canonical layout",
            code=ErrorCode.SCHEMA_MISMATCH,
            details={"layout": layout.to_dict()},
        )
    if values.shape != (layout.dim,) or not np.all(np.isfinite(values)):
        raise LeRobotExportError(
            "observation canonical_state values have the wrong shape or are non-finite",
            code=ErrorCode.INVALID_SHAPE,
            details={"shape": list(values.shape), "dim": layout.dim},
        )
    capture = state.get("capture_time_ns")
    if capture != record.get("capture_time_ns"):
        raise LeRobotExportError(
            "canonical_state capture time does not match observation capture time",
            code=ErrorCode.INVALID_SYNC,
        )
    return layout, values


def _index_images(record: Mapping[str, Any]) -> Dict[Tuple[str, str, int], Mapping[str, Any]]:
    result: Dict[Tuple[str, str, int], Mapping[str, Any]] = {}
    for entry in record.get("images", []):
        try:
            key = (str(entry["camera"]), str(entry["kind"]), int(entry["frame_number"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise LeRobotExportError(
                "observation image reference is malformed",
                code=ErrorCode.INVALID_SHAPE,
            ) from exc
        result[key] = entry
    return result


def _index_actions(actions: List[Mapping[str, Any]]) -> Dict[int, Mapping[str, Any]]:
    grouped: Dict[int, List[Mapping[str, Any]]] = {}
    for record in actions:
        sequence = record.get("observation_sequence_id")
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            raise LeRobotExportError(
                "action record is missing an integer observation_sequence_id; "
                "time proximity is not an acceptable action identity",
                code=ErrorCode.SCHEMA_MISMATCH,
            )
        grouped.setdefault(int(sequence), []).append(record)
    chosen: Dict[int, Mapping[str, Any]] = {}
    for sequence, records in grouped.items():
        commanded = [record for record in records if record.get("phase") == "commanded"]
        if len(commanded) == 1:
            chosen[sequence] = commanded[0]
        elif len(records) == 1:
            chosen[sequence] = records[0]
        else:
            raise LeRobotExportError(
                "multiple action records share one observation identity and none is "
                "unambiguously the commanded action",
                code=ErrorCode.INVALID_ACTION,
                details={"observation_sequence_id": sequence, "count": len(records)},
            )
    return chosen


def _action_vector(record: Mapping[str, Any]) -> List[float]:
    values = record.get("actions")
    if values is None:
        raise LeRobotExportError(
            "action record has no actions vector",
            code=ErrorCode.INVALID_ACTION,
        )
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.shape != (7,) or not np.all(np.isfinite(array)):
        raise LeRobotExportError(
            "action must be a finite 7-vector",
            code=ErrorCode.INVALID_SHAPE,
            details={"shape": list(array.shape)},
        )
    return array.tolist()


def export_lerobot_draft(
    episode_path: str | Path,
    output_dir: str | Path,
    *,
    copy_images: bool = True,
    require_calibrated: Optional[bool] = None,
) -> LeRobotExportResult:
    raw: RawEpisode = load_episode(episode_path)
    calibration_required = bool(raw.actions) if require_calibrated is None else bool(require_calibrated)
    issues = validate_episode(
        episode_path,
        max_dropped_lowdim=0,
        max_dropped_observations=0,
        max_dropped_actions=0,
        require_calibrated=calibration_required,
    )
    if issues:
        raise LeRobotExportError(
            "episode failed strict validation; refusing to export potentially corrupt data",
            code=ErrorCode.INVALID_SYNC,
            details={"issues": issues[:50]},
        )
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise LeRobotExportError(
            "LeRobot draft output directory exists and is not empty",
            code=ErrorCode.CONFIG_INVALID,
            details={"path": str(out)},
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(parents=True, exist_ok=True)

    # Validation above already enforces strict sequence/capture order; do not
    # sort here, because sorting would hide original order corruption.
    observations = list(raw.observations)
    if not observations:
        raise LeRobotExportError("raw episode has no observation records", code=ErrorCode.INVALID_SHAPE)

    action_by_observation = _index_actions(raw.actions)
    has_actions = bool(action_by_observation)
    if has_actions:
        missing = [
            int(record.get("sequence_id", -1))
            for record in observations
            if int(record.get("sequence_id", -1)) not in action_by_observation
        ]
        if missing:
            raise LeRobotExportError(
                "episode has partial action coverage; refusing to create training labels",
                code=ErrorCode.INVALID_ACTION,
                details={"missing_observation_sequences": missing[:20]},
            )

    episode_index = int(raw.metadata.get("episode_index", 0))
    task = str(raw.metadata.get("task", ""))
    contracts = raw.metadata.get("contracts", {})
    rows: List[dict] = []
    images_written = 0
    first_capture: Optional[int] = None
    state_layout: Optional[StateLayout] = None
    first_image_shape: Optional[List[int]] = None

    for observation in observations:
        layout, state_values = _observation_state(observation)
        if state_layout is None:
            state_layout = layout
        elif state_layout.axes != layout.axes:
            raise LeRobotExportError(
                "observation state layout changes within one episode",
                code=ErrorCode.SCHEMA_MISMATCH,
            )
        capture = int(observation["capture_time_ns"])
        if first_capture is None:
            first_capture = capture
        sequence = int(observation["sequence_id"])
        image_map = _index_images(observation)
        base_candidates = [entry for key, entry in image_map.items() if key[0] == "base" and key[1] == "color"]
        wrist_candidates = [entry for key, entry in image_map.items() if key[0] == "wrist" and key[1] == "color"]
        if len(base_candidates) != 1 or len(wrist_candidates) != 1:
            raise LeRobotExportError(
                "each observation must reference exactly one base and one wrist color frame",
                code=ErrorCode.INVALID_SHAPE,
                details={"sequence_id": sequence},
            )
        base_entry = base_candidates[0]
        wrist_entry = wrist_candidates[0]
        base_rel = f"images/base/{int(base_entry['frame_number']):012d}.png"
        wrist_rel = f"images/wrist/{int(wrist_entry['frame_number']):012d}.png"
        if copy_images:
            (out / "images" / "base").mkdir(parents=True, exist_ok=True)
            (out / "images" / "wrist").mkdir(parents=True, exist_ok=True)
            shutil.copy2(raw.path / str(base_entry["file"]), out / base_rel)
            shutil.copy2(raw.path / str(wrist_entry["file"]), out / wrist_rel)
            images_written += 2
        if first_image_shape is None:
            image = read_png(raw.path / str(base_entry["file"]), encoding="rgb8")
            first_image_shape = list(image.shape)
        row = {
            "observation.images.base": base_rel,
            "observation.images.wrist": wrist_rel,
            "observation.state": state_values.tolist(),
            "task": task,
            "timestamp": (capture - first_capture) / 1_000_000_000.0,
            "episode_index": episode_index,
            "frame_index": len(rows),
            "index": len(rows),
        }
        if has_actions:
            action_record = action_by_observation[sequence]
            declared_capture = action_record.get("observation_capture_time_ns")
            if declared_capture is not None and int(declared_capture) != capture:
                raise LeRobotExportError(
                    "action capture timestamp does not match its observation",
                    code=ErrorCode.INVALID_SYNC,
                    details={"sequence_id": sequence},
                )
            row["action"] = _action_vector(action_record)
        rows.append(row)

    if not rows:
        raise LeRobotExportError("no observation could be exported", code=ErrorCode.INVALID_SHAPE)
    assert state_layout is not None

    data_path = out / "data" / "chunk-000" / f"episode_{episode_index:06d}.jsonl"
    with data_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_json_line(row))

    features: Dict[str, Any] = {
        "observation.images.base": {
            "dtype": "image",
            "shape": first_image_shape,
            "names": ["height", "width", "channel"],
        },
        "observation.images.wrist": {
            "dtype": "image",
            "shape": first_image_shape,
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [state_layout.dim],
            "names": list(state_layout.axes),
            "units": list(state_layout.units),
            "frame": state_layout.frame,
        },
        "task": {"dtype": "string", "shape": []},
        "timestamp": {"dtype": "float32", "shape": []},
        "episode_index": {"dtype": "int64", "shape": []},
        "frame_index": {"dtype": "int64", "shape": []},
        "index": {"dtype": "int64", "shape": []},
    }
    calibration = raw.metadata.get("calibration", {})
    training_ready = bool(
        has_actions
        and calibration.get("version") not in (None, "", "uncalibrated")
        and calibration.get("base_intrinsics_sha256")
    )
    if has_actions:
        features["action"] = {
            "dtype": "float32",
            "shape": [len(ACTION_AXES)],
            "names": list(ACTION_AXES),
            "units": ["m", "m", "m", "rad", "rad", "rad", "unit"],
        }
    timestamps = [row["timestamp"] for row in rows]
    fps = 0.0
    if len(timestamps) > 1 and timestamps[-1] > 0:
        fps = (len(timestamps) - 1) / timestamps[-1]
    info = {
        "codebase_version": "v2.0",
        "robot_type": raw.metadata.get("robot_type", "flexiv_rizon4s"),
        "total_episodes": 1,
        "total_frames": len(rows),
        "total_tasks": 1,
        "fps": fps,
        "chunks_size": 1000,
        "data_path": f"data/chunk-000/episode_{episode_index:06d}.jsonl",
        "video_path": None,
        "image_path": "images",
        "format": LEROBOT_DRAFT_FORMAT if has_actions else LEROBOT_OBSERVATION_ONLY_FORMAT,
        "features": features,
        "state_layout": state_layout.to_dict(),
        "contracts": {
            "observation_schema": OBSERVATION_SCHEMA_VERSION,
            "action_schema": ACTION_SCHEMA_VERSION,
            "delta_semantics": contracts.get("delta_semantics", "cumulative"),
            "gripper_semantics": contracts.get("gripper_semantics", GRIPPER_SEMANTICS),
        },
        "action_available": has_actions,
        "action_identity": "observation_sequence_id" if has_actions else None,
        "action_identity_scope": "(episode_index, observation_sequence_id)" if has_actions else None,
        "training_ready": training_ready,
        "calibration_version": calibration.get("version"),
        "source_raw_episode": str(raw.path),
        "source_calibration_version": raw.metadata.get("calibration", {}).get("version"),
        "camera_intrinsics_sha256": {
            name: camera.get("intrinsics_sha256")
            for name, camera in raw.metadata.get("devices", {}).get("cameras", {}).items()
        },
        "camera_extrinsics_sha256": {
            name: raw.metadata.get("calibration", {}).get(f"{name}_extrinsics_sha256")
            for name in raw.metadata.get("devices", {}).get("cameras", {})
        },
        "notes": (
            "M2 JSONL draft; observation.state is the exact persisted canonical "
            "state used online. Action is omitted when the raw episode has no "
            "actions; zero actions are never fabricated. Parquet/video conversion "
            "requires PyArrow/FFmpeg and is not claimed."
        ),
    }
    (out / "meta" / "info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    with (out / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(_json_line({"episode_index": episode_index, "tasks": [task], "length": len(rows)}))
    with (out / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(_json_line({"task": task, "task_index": 0}))
    return LeRobotExportResult(
        path=out,
        frames=len(rows),
        images=images_written,
        action_available=has_actions,
        format=info["format"],
    )


class LerobotWriter:
    """Compatibility wrapper; the draft exporter is the real implementation."""

    def export(self, episode_path: str | Path, output_dir: str | Path) -> LeRobotExportResult:
        return export_lerobot_draft(episode_path, output_dir)


__all__ = [
    "LEROBOT_DRAFT_FORMAT",
    "LEROBOT_OBSERVATION_ONLY_FORMAT",
    "LeRobotExportError",
    "LeRobotExportResult",
    "LerobotWriter",
    "export_lerobot_draft",
]
