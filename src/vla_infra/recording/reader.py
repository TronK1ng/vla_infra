"""Raw episode reader and consistency checks (M2).

The reader is used by replay/LeRobot export and by the doctor/validation CLI.
It never infers fields from array length: lowdim fields come from the schema
written with the episode, and frame provenance comes from frames.jsonl.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

from ..contracts.observation import RobotState
from ..contracts.policy_io import CANONICAL_STATE_LAYOUT, StateLayout
from ..errors import ErrorCode
from .raw_writer import RAW_EPISODE_SCHEMA_VERSION, RecordingError


@dataclass(frozen=True)
class RawEpisode:
    path: Path
    metadata: Mapping[str, Any]
    lowdim: np.ndarray
    camera_frames: Mapping[str, List[Mapping[str, Any]]]
    observations: List[Mapping[str, Any]]
    actions: List[Mapping[str, Any]]
    schema: Mapping[str, Any]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RecordingError(
            f"required episode file is missing: {path.name}",
            code=ErrorCode.CONFIG_INVALID,
            details={"path": str(path)},
        ) from exc
    except json.JSONDecodeError as exc:
        raise RecordingError(
            f"episode file is not valid JSON: {path.name}",
            code=ErrorCode.CONFIG_INVALID,
            details={"path": str(path)},
        ) from exc


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RecordingError(
                    f"invalid JSONL at {path.name}:{line_number}",
                    code=ErrorCode.CONFIG_INVALID,
                ) from exc
    return records


def validate_episode(
    path: str | Path,
    *,
    max_dropped_lowdim: int = 0,
    max_dropped_observations: int = 0,
    max_dropped_actions: int = 0,
    require_calibrated: bool = False,
) -> List[str]:
    """Return a list of consistency issues; empty means valid.

    The default quality gate requires zero recorder drops.  Callers that want to
    inspect a lossy episode can raise the thresholds explicitly; the drops are
    never silently ignored.
    """
    episode = load_episode(path)
    issues: List[str] = []
    metadata = episode.metadata
    if metadata.get("complete") is not True:
        issues.append("metadata.complete is not true")
    counts = metadata.get("counts", {})

    # 1. Schema and raw binary agreement.
    schema = episode.schema
    if int(schema.get("row_count", -1)) != int(episode.lowdim.shape[0]):
        issues.append("schema.row_count does not match raw.bin")
    raw_bin = episode.path / "lowdim" / "raw.bin"
    if raw_bin.exists():
        expected_bytes = int(episode.lowdim.shape[0]) * int(episode.lowdim.dtype.itemsize)
        if int(schema.get("byte_length", -1)) != expected_bytes:
            issues.append("schema.byte_length does not match raw.bin size")
        if raw_bin.stat().st_size != expected_bytes:
            issues.append("raw.bin byte size does not match schema/row count")
    else:
        issues.append("lowdim/raw.bin is missing")

    # 2. All metadata counts must agree with the actual files.
    if counts.get("lowdim_rows") != episode.lowdim.shape[0]:
        issues.append("metadata lowdim_rows does not match raw.bin")
    total_camera_frames = sum(len(records) for records in episode.camera_frames.values())
    if counts.get("camera_frames") != total_camera_frames:
        issues.append("metadata camera_frames does not match frame manifests")
    if counts.get("observation_records") != len(episode.observations):
        issues.append("metadata observation_records does not match observations.jsonl")
    if counts.get("action_records") != len(episode.actions):
        issues.append("metadata action_records does not match actions/raw.jsonl")

    # 3. Lowdim timing.
    if episode.lowdim.shape[0] > 0:
        host = episode.lowdim["host_time_ns"]
        if not np.all(np.diff(host) > 0):
            issues.append("lowdim host_time_ns is not strictly increasing")
        device = episode.lowdim["device_time_ns"]
        if not np.all(np.diff(device) > 0):
            issues.append("lowdim device_time_ns is not strictly increasing")

    # 4. Camera manifests and image references.
    image_index: Dict[tuple, Mapping[str, Any]] = {}
    for camera, frames in episode.camera_frames.items():
        for kind in ("color", "depth"):
            seen: set[tuple] = set()
            subset = [record for record in frames if record.get("kind") == kind]
            if not subset:
                continue
            frame_numbers = [int(record["frame_number"]) for record in subset]
            if any(b <= a for a, b in zip(frame_numbers, frame_numbers[1:])):
                issues.append(f"camera {camera}/{kind}: frame numbers not increasing")
            duplicate = [number for number in frame_numbers if number in seen]
            if duplicate:
                issues.append(f"camera {camera}/{kind}: duplicate frame numbers {duplicate[:3]}")
            seen.update(frame_numbers)
            for record in subset:
                file_path = episode.path / str(record.get("file", ""))
                if not record.get("file") or not file_path.exists():
                    issues.append(f"camera {camera}/{kind}: missing file {record.get('file')!r}")
                image_index[(camera, kind, int(record["frame_number"]))] = record
            host = [int(record["host_time_ns"]) for record in subset]
            if any(b <= a for a, b in zip(host, host[1:])):
                issues.append(f"camera {camera}/{kind}: host timestamps not increasing")
            device = [int(record["device_time_ns"]) for record in subset]
            if any(b <= a for a, b in zip(device, device[1:])):
                issues.append(f"camera {camera}/{kind}: device timestamps not increasing")

    # 5. Observation sequence/timestamp and canonical state.
    if not episode.observations:
        issues.append("no observation records")
    sequences: List[int] = []
    captures: List[int] = []
    observations_by_sequence: Dict[int, Mapping[str, Any]] = {}
    for index, observation in enumerate(episode.observations):
        try:
            sequence = int(observation["sequence_id"])
            capture = int(observation["capture_time_ns"])
        except (KeyError, TypeError, ValueError):
            issues.append(f"observation {index}: missing/invalid sequence or capture time")
            continue
        sequences.append(sequence)
        captures.append(capture)
        observations_by_sequence[sequence] = observation
        state = observation.get("canonical_state")
        if not isinstance(state, Mapping):
            issues.append(f"observation {sequence}: missing canonical_state")
        else:
            try:
                layout = StateLayout.from_dict(state["layout"])
                values = np.asarray(state["values"], dtype=np.float64).reshape(-1)
            except (KeyError, TypeError, ValueError):
                issues.append(f"observation {sequence}: malformed canonical_state")
            else:
                if layout.axes != CANONICAL_STATE_LAYOUT.axes:
                    issues.append(f"observation {sequence}: non-canonical state layout")
                if values.shape != (CANONICAL_STATE_LAYOUT.dim,):
                    issues.append(f"observation {sequence}: canonical_state has wrong length")
                elif not np.all(np.isfinite(values)):
                    issues.append(f"observation {sequence}: canonical_state is non-finite")
                if state.get("capture_time_ns") != capture:
                    issues.append(f"observation {sequence}: state/observation capture mismatch")
                typed = state.get("typed")
                if not isinstance(typed, Mapping):
                    issues.append(f"observation {sequence}: canonical_state typed provenance missing")
                else:
                    required = (
                        "joint_position",
                        "joint_velocity",
                        "tcp_position",
                        "tcp_quaternion_xyzw",
                        "tcp_rotation_6d",
                        "tcp_velocity",
                        "wrench_tcp",
                        "gripper_width",
                    )
                    missing = [name for name in required if name not in typed]
                    if missing:
                        issues.append(
                            f"observation {sequence}: typed state missing {missing}"
                        )
                    else:
                        try:
                            typed_state = RobotState(
                                joint_position=np.asarray(typed["joint_position"], dtype=np.float64),
                                joint_velocity=np.asarray(typed["joint_velocity"], dtype=np.float64),
                                tcp_position=np.asarray(typed["tcp_position"], dtype=np.float64),
                                tcp_quaternion_xyzw=np.asarray(
                                    typed["tcp_quaternion_xyzw"], dtype=np.float64
                                ),
                                tcp_rotation_6d=np.asarray(
                                    typed["tcp_rotation_6d"], dtype=np.float64
                                ),
                                tcp_velocity=np.asarray(typed["tcp_velocity"], dtype=np.float64),
                                wrench_tcp=np.asarray(typed["wrench_tcp"], dtype=np.float64),
                                gripper_width=np.atleast_1d(
                                    np.asarray(typed["gripper_width"], dtype=np.float64)
                                ),
                                tcp_velocity_frame=str(
                                    typed.get("tcp_velocity_frame", "base")
                                ),
                                wrench_frame=str(typed.get("wrench_frame", "tcp")),
                            )
                            encoded = CANONICAL_STATE_LAYOUT.encode(typed_state).astype(np.float32)
                            stored = np.asarray(values, dtype=np.float32)
                            if not np.array_equal(stored, encoded):
                                issues.append(
                                    f"observation {sequence}: canonical values do not match typed provenance"
                                )
                        except Exception as exc:  # noqa: BLE001 - validation boundary
                            issues.append(
                                f"observation {sequence}: invalid typed state ({type(exc).__name__})"
                            )
        images = observation.get("images")
        if not isinstance(images, list) or not images:
            issues.append(f"observation {sequence}: no image references")
        else:
            for entry in images:
                try:
                    key = (str(entry["camera"]), str(entry["kind"]), int(entry["frame_number"]))
                except (KeyError, TypeError, ValueError):
                    issues.append(f"observation {sequence}: malformed image reference")
                    continue
                manifest = image_index.get(key)
                if manifest is None:
                    issues.append(
                        f"observation {sequence}: dangling image reference {key}"
                    )
                elif entry.get("file") and entry.get("file") != manifest.get("file"):
                    issues.append(
                        f"observation {sequence}: image file/provenance mismatch for {key}"
                    )
    if any(b <= a for a, b in zip(sequences, sequences[1:])):
        issues.append("observation sequence_id is not strictly increasing")
    if any(b <= a for a, b in zip(captures, captures[1:])):
        issues.append("observation capture_time_ns is not strictly increasing")

    # 6. Action identity and completeness (no zero-action fabrication).
    action_sequences: List[int] = []
    for index, action in enumerate(episode.actions):
        sequence = action.get("observation_sequence_id")
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            issues.append(f"action {index}: missing/invalid observation_sequence_id")
            continue
        if sequence not in observations_by_sequence:
            issues.append(f"action {index}: dangling observation_sequence_id {sequence}")
        if sequence in action_sequences:
            issues.append(f"action {index}: duplicate action for observation {sequence}")
        action_sequences.append(sequence)
        values = action.get("actions")
        if values is None:
            issues.append(f"action {index}: missing actions vector")
        else:
            array = np.asarray(values, dtype=np.float64).reshape(-1)
            if array.shape != (7,) or not np.all(np.isfinite(array)):
                issues.append(f"action {index}: actions must be a finite 7-vector")
        declared_capture = action.get("observation_capture_time_ns")
        if declared_capture is not None and sequence in observations_by_sequence:
            if int(declared_capture) != int(observations_by_sequence[sequence]["capture_time_ns"]):
                issues.append(f"action {index}: capture time mismatch for observation {sequence}")
    if episode.actions and set(action_sequences) != set(observations_by_sequence):
        issues.append("actions do not cover every observation exactly once")

    # 7. Recorder quality gate (drops are explicit, not silently tolerated).
    recorder = metadata.get("recorder", {})
    dropped_lowdim = int(recorder.get("dropped_lowdim", 0))
    dropped_observations = int(recorder.get("dropped_observations", 0))
    dropped_actions = int(recorder.get("dropped_actions", 0))
    if dropped_lowdim > max_dropped_lowdim:
        issues.append(f"dropped_lowdim {dropped_lowdim} exceeds quality gate {max_dropped_lowdim}")
    if dropped_observations > max_dropped_observations:
        issues.append(
            f"dropped_observations {dropped_observations} exceeds quality gate {max_dropped_observations}"
        )
    if dropped_actions > max_dropped_actions:
        issues.append(f"dropped_actions {dropped_actions} exceeds quality gate {max_dropped_actions}")

    # 8. Calibration gate (optional; uncalibrated data is not training-ready).
    calibration = metadata.get("calibration", {})
    if require_calibrated:
        if calibration.get("version") in (None, "", "uncalibrated"):
            issues.append("calibration version is uncalibrated")
        if not calibration.get("base_intrinsics_sha256"):
            issues.append("calibration base intrinsics hash is missing")

    if not (episode.path / "sync" / "sync_report.json").exists():
        issues.append("sync/sync_report.json missing")
    return issues

def load_episode(path: str | Path) -> RawEpisode:
    episode_path = Path(path)
    if not episode_path.is_dir():
        raise RecordingError(
            "episode path is not a directory",
            code=ErrorCode.CONFIG_INVALID,
            details={"path": str(episode_path)},
        )
    if (episode_path / ".incomplete").exists():
        raise RecordingError(
            "episode is not committed (.incomplete present)",
            code=ErrorCode.CONFIG_INVALID,
            details={"path": str(episode_path)},
        )
    metadata = _read_json(episode_path / "metadata.json")
    if metadata.get("schema_version") != RAW_EPISODE_SCHEMA_VERSION:
        raise RecordingError(
            f"unsupported raw episode schema {metadata.get('schema_version')!r}",
            code=ErrorCode.SCHEMA_MISMATCH,
        )
    if metadata.get("complete") is not True:
        raise RecordingError("episode metadata is not marked complete", code=ErrorCode.CONFIG_INVALID)
    schema = _read_json(episode_path / "lowdim" / "schema.json")
    raw_bin = episode_path / "lowdim" / "raw.bin"
    if not raw_bin.exists():
        raise RecordingError("lowdim/raw.bin is missing", code=ErrorCode.CONFIG_INVALID)
    expected_dtype = np.dtype(
        [(field["name"], field["dtype"], tuple(field["shape"])) for field in schema["fields"]]
    )
    lowdim = np.fromfile(raw_bin, dtype=expected_dtype)
    declared_rows = metadata.get("counts", {}).get("lowdim_rows")
    if declared_rows is not None and int(declared_rows) != lowdim.shape[0]:
        raise RecordingError(
            "lowdim row count does not match metadata",
            code=ErrorCode.INVALID_SYNC,
            details={"metadata": declared_rows, "file": int(lowdim.shape[0])},
        )
    camera_frames: Dict[str, List[Mapping[str, Any]]] = {}
    cameras_root = episode_path / "cameras"
    if cameras_root.exists():
        for camera_dir in sorted(cameras_root.iterdir()):
            if not camera_dir.is_dir():
                continue
            records: List[Mapping[str, Any]] = []
            for kind in ("color", "depth"):
                frames_path = camera_dir / kind / "frames.jsonl"
                if frames_path.exists():
                    for record in _read_jsonl(frames_path):
                        records.append({**record, "kind": kind})
            camera_frames[camera_dir.name] = records
    observations = _read_jsonl(episode_path / "observations.jsonl")
    actions = _read_jsonl(episode_path / "actions" / "raw.jsonl")
    return RawEpisode(
        path=episode_path,
        metadata=metadata,
        lowdim=lowdim,
        camera_frames=camera_frames,
        observations=observations,
        actions=actions,
        schema=schema,
    )


__all__ = ["RawEpisode", "load_episode", "validate_episode"]
