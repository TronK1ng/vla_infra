"""Raw episode writer for M2.

Layout (deliberately close to the Force Policy multi-rate convention, without
copying its HDF5 implementation)::

    episode_0001/
      cameras/<name>/color/000000000001.png
      cameras/<name>/depth/000000000001.png
      cameras/<name>/frames.jsonl
      lowdim/raw.bin
      lowdim/schema.json
      actions/raw.jsonl
      observations.jsonl
      calibration/calibration.json
      sync/sync_report.json
      metadata.json
      .incomplete                 # exists only before atomic commit

The episode is written to a hidden temporary directory and renamed to the final
name only after every file and the metadata are complete.  A crash leaves an
identifiable ``.episode_*.tmp-*`` directory; it is never treated as committed.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

from ..clock import Clock, MonotonicClock
from ..contracts.observation import ImageFrame, RobotStateSample
from ..errors import ErrorCode, VlaInfraError
from .schema import LOWDIM_DTYPE, LOWDIM_FIELDS, samples_to_rows, schema_dict

RAW_EPISODE_SCHEMA_VERSION = "raw-episode/v1"
_TMP_PREFIX = ".episode_"
_TMP_SUFFIX = ".tmp-"


class RecordingError(VlaInfraError):
    code = ErrorCode.DEVICE_FAULT


class DiskLowError(RecordingError):
    code = ErrorCode.DEVICE_FAULT


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(_json_safe(value), indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_tree_dirs(root: Path) -> None:
    for directory in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if directory.is_dir():
            _fsync_dir(directory)
    _fsync_dir(root)


def next_episode_index(root: Path) -> int:
    highest = 0
    if root.exists():
        for child in root.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            if name.startswith("episode_"):
                try:
                    highest = max(highest, int(name.split("_", 1)[1]))
                except ValueError:
                    continue
            elif name.startswith(_TMP_PREFIX):
                try:
                    prefix = name[len(_TMP_PREFIX) :].split(".tmp-", 1)[0]
                    highest = max(highest, int(prefix))
                except ValueError:
                    continue
    return highest + 1


def uncommitted_episodes(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        child
        for child in root.iterdir()
        if child.is_dir() and child.name.startswith(_TMP_PREFIX)
    )


@dataclass(frozen=True)
class RawEpisodeResult:
    path: Path
    episode_index: int
    lowdim_rows: int
    camera_frames: int
    action_records: int
    observation_records: int


class RawEpisodeWriter:
    def __init__(
        self,
        root: str | Path,
        task: str,
        *,
        metadata: Mapping[str, Any],
        camera_names: Iterable[str],
        episode_index: Optional[int] = None,
        clock: Optional[Clock] = None,
        lowdim_dtype: np.dtype = LOWDIM_DTYPE,
    ) -> None:
        if not isinstance(task, str) or not task.strip():
            raise RecordingError("task must be a non-empty string", code=ErrorCode.CONFIG_INVALID)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.task = task
        self._clock = clock or MonotonicClock()
        self._metadata = dict(metadata)
        self._camera_names = tuple(str(name) for name in camera_names)
        if not self._camera_names:
            raise RecordingError("at least one camera is required", code=ErrorCode.CONFIG_INVALID)
        self._lowdim_dtype = lowdim_dtype
        self.episode_index = int(episode_index) if episode_index is not None else next_episode_index(self.root)
        if self.episode_index <= 0:
            raise RecordingError("episode_index must be positive", code=ErrorCode.CONFIG_INVALID)
        self.final_dir = self.root / f"episode_{self.episode_index:04d}"
        if self.final_dir.exists():
            raise RecordingError(
                "episode final directory already exists; refusing to overwrite",
                code=ErrorCode.CONFIG_INVALID,
                details={"path": str(self.final_dir)},
            )
        self.tmp_dir = self.root / f"{_TMP_PREFIX}{self.episode_index:04d}{_TMP_SUFFIX}{uuid.uuid4().hex[:8]}"
        self.tmp_dir.mkdir(parents=True, exist_ok=False)
        (self.tmp_dir / ".incomplete").write_text(
            json.dumps(
                {
                    "episode_index": self.episode_index,
                    "started_ns": self._clock.now_ns(),
                    "reason": "in_progress",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self._started_ns = self._clock.now_ns()
        self._lowdim_path = self.tmp_dir / "lowdim" / "raw.bin"
        self._lowdim_path.parent.mkdir(parents=True, exist_ok=True)
        self._lowdim_file = self._lowdim_path.open("wb")
        self._lowdim_rows = 0
        self._camera_frames = 0
        self._action_records = 0
        self._observations = 0
        self._frame_files: Dict[str, Any] = {}
        self._observation_file = (self.tmp_dir / "observations.jsonl").open("a", encoding="utf-8")
        self._sync_report: Dict[str, Any] = {}
        self._closed = False
        self._committed = False
        self._streams_closed = False
        self._failed_reason: Optional[str] = None

    # -- data ---------------------------------------------------------------
    def append_lowdim(self, samples: Iterable[RobotStateSample]) -> int:
        self._assert_open()
        rows = samples_to_rows(samples)
        if rows.size == 0:
            return 0
        rows.tofile(self._lowdim_file)
        self._lowdim_rows += int(rows.shape[0])
        return int(rows.shape[0])

    def write_image(self, camera_name: str, frame: ImageFrame, *, kind: str = "color") -> str:
        self._assert_open()
        if camera_name not in self._camera_names:
            raise RecordingError(
                f"unknown camera {camera_name!r}",
                code=ErrorCode.CONFIG_INVALID,
                details={"known": list(self._camera_names)},
            )
        if kind not in ("color", "depth"):
            raise RecordingError(
                f"unknown image kind {kind!r}", code=ErrorCode.CONFIG_INVALID
            )
        expected_encoding = "rgb8" if kind == "color" else "depth16"
        if frame.encoding != expected_encoding:
            raise RecordingError(
                f"{kind} frame has encoding {frame.encoding!r}, expected {expected_encoding!r}",
                code=ErrorCode.INVALID_ENCODING,
            )
        from .image_codec import write_png

        directory = self.tmp_dir / "cameras" / camera_name / kind
        filename = f"{frame.frame_number:012d}.png"
        path = directory / filename
        if path.exists():
            raise RecordingError(
                "duplicate camera frame number",
                code=ErrorCode.INVALID_SYNC,
                details={"camera": camera_name, "frame_number": frame.frame_number},
            )
        write_png(path, frame.data, encoding=frame.encoding)
        handle = self._camera_handle(camera_name, kind)
        record = {
            "kind": kind,
            "file": path.relative_to(self.tmp_dir).as_posix(),
            "frame_number": frame.frame_number,
            "device_time_ns": frame.device_time_ns,
            "host_time_ns": frame.host_time_ns,
            "width": frame.width,
            "height": frame.height,
            "encoding": frame.encoding,
        }
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._camera_frames += 1
        return record["file"]

    def write_observation(self, record: Mapping[str, Any]) -> None:
        """Append observation-level provenance (sequence, capture, frame ids)."""
        self._assert_open()
        self._observation_file.write(
            json.dumps(_json_safe(dict(record)), ensure_ascii=False) + "\n"
        )
        self._observations += 1

    def write_action(self, record: Mapping[str, Any]) -> None:
        self._assert_open()
        path = self.tmp_dir / "actions" / "raw.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(dict(record)), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._action_records += 1

    def set_sync_report(self, report: Mapping[str, Any]) -> None:
        self._sync_report = _json_safe(dict(report))

    # -- lifecycle ----------------------------------------------------------
    def commit(
        self,
        *,
        end_reason: str = "duration",
        verdict: str = "unreviewed",
        extra_metadata: Optional[Mapping[str, Any]] = None,
    ) -> RawEpisodeResult:
        self._assert_open()
        self._close_streams()
        schema = schema_dict()
        schema["row_count"] = self._lowdim_rows
        schema["byte_length"] = self._lowdim_rows * self._lowdim_dtype.itemsize
        schema["fields"] = [field.as_dict() for field in LOWDIM_FIELDS]
        _write_json(self.tmp_dir / "lowdim" / "schema.json", schema)
        _write_json(
            self.tmp_dir / "calibration" / "calibration.json",
            self._metadata.get("calibration", {"version": "uncalibrated"}),
        )
        _write_json(self.tmp_dir / "sync" / "sync_report.json", self._sync_report)
        metadata = dict(self._metadata)
        metadata.update(
            {
                "schema_version": RAW_EPISODE_SCHEMA_VERSION,
                "episode_index": self.episode_index,
                "task": self.task,
                "started_ns": self._started_ns,
                "finalized_ns": self._clock.now_ns(),
                "complete": True,
                "durability": "fsync_best_effort",
                "durability_detail": (
                    "files, manifests and temporary/parent directories are fsynced; "
                    "a power loss in the post-rename window remains filesystem-defined"
                ),
                "end_reason": str(end_reason),
                "verdict": str(verdict),
                "counts": {
                    "lowdim_rows": self._lowdim_rows,
                    "camera_frames": self._camera_frames,
                    "action_records": self._action_records,
                    "observation_records": self._observations,
                },
                "sync_report": self._sync_report,
            }
        )
        if extra_metadata:
            metadata.update(_json_safe(dict(extra_metadata)))
        _write_json(self.tmp_dir / "metadata.json", metadata)
        marker = self.tmp_dir / ".incomplete"
        try:
            marker.unlink(missing_ok=True)
            _fsync_tree_dirs(self.tmp_dir)
            if self.final_dir.exists():
                raise RecordingError(
                    "final episode directory appeared during recording; refusing to overwrite",
                    code=ErrorCode.CONFIG_INVALID,
                )
            self.tmp_dir.rename(self.final_dir)
            try:
                _fsync_dir(self.root)
            except OSError:
                # The directory entries are visible and all files were fsynced;
                # a parent-directory fsync failure must not leave a committed
                # episode without a result object.  The durability caveat is
                # documented; data loss in this window is filesystem-defined.
                pass
        except Exception:
            # Restore the crash marker on any commit failure (fsync, collision,
            # rename) so the directory is never mistaken for a complete episode.
            if self.tmp_dir.exists():
                _write_json(
                    marker,
                    {
                        "episode_index": self.episode_index,
                        "reason": "commit_failed",
                        "aborted_ns": self._clock.now_ns(),
                    },
                )
            raise
        self._committed = True
        self._closed = True
        return RawEpisodeResult(
            path=self.final_dir,
            episode_index=self.episode_index,
            lowdim_rows=self._lowdim_rows,
            camera_frames=self._camera_frames,
            action_records=self._action_records,
            observation_records=self._observations,
        )

    def abort(self, reason: str, *, remove: bool = False) -> Path:
        if self._committed:
            raise RecordingError("cannot abort a committed episode", code=ErrorCode.INVALID_ACTION)
        self._failed_reason = str(reason)
        if not self._closed:
            self._close_streams()
        if self.tmp_dir.exists():
            _write_json(
                self.tmp_dir / ".incomplete",
                {
                    "episode_index": self.episode_index,
                    "reason": self._failed_reason,
                    "aborted_ns": self._clock.now_ns(),
                },
            )
        if remove and self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)
        self._closed = True
        return self.tmp_dir

    @property
    def committed(self) -> bool:
        return self._committed

    @property
    def path(self) -> Path:
        return self.final_dir

    @property
    def tmp_path(self) -> Path:
        return self.tmp_dir

    @property
    def lowdim_rows(self) -> int:
        return self._lowdim_rows

    @property
    def camera_frames(self) -> int:
        return self._camera_frames

    @property
    def action_records(self) -> int:
        return self._action_records

    @property
    def observation_records(self) -> int:
        return self._observations

    def check_free_space(self, min_free_bytes: int) -> None:
        usage = shutil.disk_usage(self.root)
        if usage.free < int(min_free_bytes):
            raise DiskLowError(
                "free disk space is below the configured recording minimum",
                details={
                    "free_bytes": int(usage.free),
                    "min_free_bytes": int(min_free_bytes),
                    "root": str(self.root),
                },
            )

    # -- internals ----------------------------------------------------------
    def _assert_open(self) -> None:
        if self._closed:
            raise RecordingError("episode writer is closed", code=ErrorCode.INVALID_ACTION)
        if self._failed_reason is not None:
            raise RecordingError(
                f"episode writer is failed: {self._failed_reason}",
                code=ErrorCode.DEVICE_FAULT,
            )

    def _camera_handle(self, camera_name: str, kind: str):
        key = (camera_name, kind)
        if key not in self._frame_files:
            path = self.tmp_dir / "cameras" / camera_name / kind / "frames.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            self._frame_files[key] = path.open("a", encoding="utf-8")
        return self._frame_files[key]

    def _close_streams(self) -> None:
        if self._streams_closed:
            return
        self._streams_closed = True
        for handle in [self._lowdim_file, self._observation_file, *self._frame_files.values()]:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        self._frame_files.clear()


__all__ = [
    "DiskLowError",
    "RAW_EPISODE_SCHEMA_VERSION",
    "RawEpisodeResult",
    "RawEpisodeWriter",
    "RecordingError",
    "next_episode_index",
    "uncommitted_episodes",
]
