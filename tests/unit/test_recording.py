"""M2 raw recording, atomic commit, async behaviour and LeRobot draft."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from vla_infra.config import load_app_config
from vla_infra.contracts.constants import IMAGE_KEY_BASE_RGB, IMAGE_KEY_WRIST_RGB
from vla_infra.contracts.policy_io import CANONICAL_STATE_LAYOUT
from vla_infra.recording.lerobot_writer import export_lerobot_draft
from vla_infra.recording.raw_writer import RawEpisodeWriter, RecordingError
from vla_infra.recording.reader import load_episode, validate_episode
from vla_infra.recording.recorder import EpisodeRecorder, RecorderConfig
from vla_infra.runtime.collection import CollectionHarness, build_collection_metadata
from vla_infra.sync.synchronizer import ObservationSynchronizer, SyncConfig
from vla_infra.clock import FakeClock
from vla_infra.testing import make_image, make_observation, make_robot_sample, make_sync

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def _app():
    return load_app_config(
        station_path=CONFIGS / "station.example.yaml",
        safety_path=CONFIGS / "safety.example.yaml",
        policy_path=CONFIGS / "policy.example.yaml",
    )


def _recorder_config(**overrides) -> RecorderConfig:
    values = dict(min_free_gb=0.0, queue_size=64, max_lowdim_rows=10000, stop_timeout_s=5.0)
    values.update(overrides)
    return RecorderConfig(**values)


@pytest.fixture(scope="module")
def recorded_episode(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw")
    app = _app()
    harness = CollectionHarness(
        app,
        task="pick the cube",
        output_root=str(root),
        recorder_config=_recorder_config(),
    )
    result = harness.run(duration_s=0.4)
    assert result.recorder.committed, result.recorder.error
    assert result.episode_path is not None
    return Path(result.episode_path)


def test_fake_collection_records_valid_episode(recorded_episode: Path) -> None:
    assert (recorded_episode / "metadata.json").exists()
    metadata = json.loads((recorded_episode / "metadata.json").read_text())
    assert metadata["schema_version"] == "raw-episode/v1"
    assert metadata["complete"] is True
    assert metadata["task"] == "pick the cube"
    assert metadata["counts"]["lowdim_rows"] > 0
    assert metadata["counts"]["camera_frames"] > 0
    assert metadata["counts"]["observation_records"] > 0
    assert metadata["end_reason"] == "duration"
    assert metadata["contracts"]["delta_semantics"] == "cumulative"
    assert metadata["contracts"]["gripper_semantics"] == "closed_0_open_1"
    assert metadata["calibration"]["version"] == "uncalibrated"
    assert metadata["devices"]["cameras"]["base"]["serial"]
    assert (recorded_episode / "sync" / "sync_report.json").exists()
    assert (recorded_episode / "lowdim" / "schema.json").exists()
    assert (recorded_episode / "cameras" / "base" / "color" / "frames.jsonl").exists()
    assert (recorded_episode / "cameras" / "base" / "depth" / "frames.jsonl").exists()
    assert validate_episode(recorded_episode) == []


def test_reader_exposes_schema_and_links(recorded_episode: Path) -> None:
    episode = load_episode(recorded_episode)
    assert episode.lowdim.shape[0] == episode.metadata["counts"]["lowdim_rows"]
    assert set(episode.camera_frames) == {"base", "wrist"}
    assert episode.observations
    schema = episode.schema
    names = {field["name"] for field in schema["fields"]}
    assert {"host_time_ns", "device_time_ns", "joint_position", "wrench_tcp"} <= names
    assert "force_policy_mapping_note" in schema
    state = episode.observations[0]["canonical_state"]
    assert state["layout"]["axes"] == episode.metadata["contracts"]["state_layout"]["axes"]
    assert len(state["values"]) == 22
    assert state["capture_time_ns"] == episode.observations[0]["capture_time_ns"]


def test_lerobot_observation_only_draft_uses_persisted_state(recorded_episode: Path, tmp_path: Path) -> None:
    out = tmp_path / "lerobot"
    result = export_lerobot_draft(recorded_episode, out)
    assert result.format == "jsonl_observation_only_draft"
    assert result.frames > 0
    raw = load_episode(recorded_episode)
    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["format"] == "jsonl_observation_only_draft"
    assert info["action_available"] is False
    assert "action" not in info["features"]
    assert info["total_frames"] == result.frames
    assert info["features"]["observation.images.base"]["dtype"] == "image"
    assert info["features"]["observation.state"]["shape"] == [22]
    lines = [json.loads(line) for line in (out / "data" / "chunk-000" / "episode_000001.jsonl").read_text().splitlines()]
    assert len(lines) == result.frames
    for row, observation in zip(lines, raw.observations):
        assert "action" not in row
        assert row["observation.state"] == observation["canonical_state"]["values"]
    assert set(lines[0]) >= {
        "observation.images.base",
        "observation.images.wrist",
        "observation.state",
        "task",
        "timestamp",
        "episode_index",
        "frame_index",
    }
    assert (out / "meta" / "episodes.jsonl").exists()
    assert (out / "meta" / "tasks.jsonl").exists()


def test_uncommitted_episode_is_identifiable(tmp_path: Path) -> None:
    app = _app()
    writer = RawEpisodeWriter(
        tmp_path,
        "aborted",
        metadata=build_collection_metadata(app, control_source="test"),
        camera_names=("base", "wrist"),
        clock=FakeClock(start_ns=1_000_000_000),
    )
    writer.append_lowdim([make_robot_sample(FakeClock(), host_time_ns=1_000_000_000)])
    writer.abort("simulated crash", remove=False)
    from vla_infra.recording.raw_writer import uncommitted_episodes

    pending = uncommitted_episodes(tmp_path)
    assert len(pending) == 1
    assert (pending[0] / ".incomplete").exists()
    with pytest.raises(RecordingError):
        load_episode(pending[0])


def test_disk_low_refuses_to_start(tmp_path: Path, monkeypatch) -> None:
    app = _app()
    writer = RawEpisodeWriter(
        tmp_path,
        "disk",
        metadata=build_collection_metadata(app, control_source="test"),
        camera_names=("base", "wrist"),
        clock=FakeClock(start_ns=1_000_000_000),
    )
    recorder = EpisodeRecorder(writer, _recorder_config(min_free_gb=1.0))
    monkeypatch.setattr(
        "vla_infra.recording.raw_writer.shutil.disk_usage",
        lambda _path: type("Usage", (), {"total": 1, "used": 1, "free": 0})(),
    )
    from vla_infra.recording.raw_writer import DiskLowError

    with pytest.raises(DiskLowError):
        recorder.start()
    writer.abort("disk low test", remove=True)


def _observation_with_frames(frame_number: int):
    base = make_image(IMAGE_KEY_BASE_RGB, frame_number=frame_number, host_time_ns=1_000_000_000)
    wrist = make_image(IMAGE_KEY_WRIST_RGB, frame_number=frame_number, host_time_ns=1_000_000_000)
    observation = make_observation()
    return replace(
        observation,
        images={IMAGE_KEY_BASE_RGB: base, IMAGE_KEY_WRIST_RGB: wrist},
        sync=make_sync(before_ns=1_000_000_000, after_ns=1_000_000_000, capture_ns=1_000_000_000),
    )


def test_recorder_never_blocks_on_slow_image_encoding(tmp_path: Path, monkeypatch) -> None:
    app = _app()
    writer = RawEpisodeWriter(
        tmp_path,
        "slow disk",
        metadata=build_collection_metadata(app, control_source="test"),
        camera_names=("base", "wrist"),
        clock=FakeClock(start_ns=1_000_000_000),
    )
    original = writer.write_image

    def slow(*args, **kwargs):
        time.sleep(0.02)
        return original(*args, **kwargs)

    monkeypatch.setattr(writer, "write_image", slow)
    recorder = EpisodeRecorder(writer, _recorder_config(queue_size=1, max_lowdim_rows=1000))
    recorder.start()
    start = time.monotonic()
    for index in range(20):
        recorder.record_observation(_observation_with_frames(index))
    elapsed = time.monotonic() - start
    assert elapsed < 0.1  # producer never waits for PNG encoding
    assert recorder.result.dropped_observations > 0
    result = recorder.stop(timeout_s=5.0)
    # Dropping is allowed under backpressure; blocking the capture thread is not.
    assert result.observations >= 0
    assert result.dropped_observations > 0


def test_sync_report_has_distribution() -> None:
    clock = FakeClock(start_ns=1_000_000_000)
    sync = ObservationSynchronizer(
        clock,
        SyncConfig(max_sync_error_ns=50_000_000, max_state_gap_ns=100_000_000, max_camera_skew_ns=50_000_000),
    )
    sync.add_robot_sample(make_robot_sample(clock, host_time_ns=1_000_000_000))
    sync.add_robot_sample(make_robot_sample(clock, host_time_ns=1_010_000_000))
    images = {
        IMAGE_KEY_BASE_RGB: make_image(IMAGE_KEY_BASE_RGB, host_time_ns=1_005_000_000),
        IMAGE_KEY_WRIST_RGB: make_image(IMAGE_KEY_WRIST_RGB, host_time_ns=1_005_000_000),
    }
    sync.build(images, "task")
    report = sync.sync_report()
    assert report["built"] == 1
    assert report["sync_error_ns"]["count"] == 1
    assert report["sync_error_ns"]["p95_ns"] == 5_000_000
    assert report["camera_skew_ns"]["max_ns"] == 0


def _writer(tmp_path: Path, *, task: str = "state test") -> RawEpisodeWriter:
    app = _app()
    return RawEpisodeWriter(
        tmp_path,
        task,
        metadata=build_collection_metadata(app, control_source="test"),
        camera_names=("base", "wrist"),
        clock=FakeClock(start_ns=1_000_000_000),
    )


def _record_observations(tmp_path: Path, observations, actions=None):
    writer = _writer(tmp_path)
    recorder = EpisodeRecorder(writer, _recorder_config())
    recorder.start()
    for observation in observations:
        assert recorder.record_observation(observation)
    for action in actions or []:
        assert recorder.record_action(action)
    return recorder.stop(timeout_s=5.0), writer.path


def _interpolated_observation():
    clock = FakeClock(start_ns=1_000_000_000)
    sync = ObservationSynchronizer(
        clock,
        SyncConfig(
            max_sync_error_ns=50_000_000,
            max_state_gap_ns=100_000_000,
            max_camera_skew_ns=50_000_000,
        ),
    )
    from vla_infra.testing import make_robot_state

    sync.add_robot_sample(
        make_robot_sample(
            clock,
            host_time_ns=1_000_000_000,
            state=make_robot_state(position=(0.5, 0.0, 0.3)),
        )
    )
    sync.add_robot_sample(
        make_robot_sample(
            clock,
            host_time_ns=1_010_000_000,
            state=make_robot_state(position=(0.5, 0.1, 0.3)),
        )
    )
    target = 1_005_000_000
    images = {
        IMAGE_KEY_BASE_RGB: make_image(
            IMAGE_KEY_BASE_RGB, host_time_ns=target, device_time_ns=target - 1_000
        ),
        IMAGE_KEY_WRIST_RGB: make_image(
            IMAGE_KEY_WRIST_RGB, host_time_ns=target, device_time_ns=target - 1_000
        ),
    }
    return sync.build(images, "interpolated")


def test_canonical_state_is_recorded_and_exported_exactly(tmp_path: Path) -> None:
    observation = _interpolated_observation()
    assert observation.sync.interpolation_alpha == pytest.approx(0.5)
    assert observation.robot.tcp_position[1] == pytest.approx(0.05)
    expected = CANONICAL_STATE_LAYOUT.encode(observation.robot).tolist()

    result, episode_path = _record_observations(tmp_path, [observation])
    assert result.committed
    raw = load_episode(episode_path)
    assert raw.observations[0]["canonical_state"]["values"] == pytest.approx(expected)
    assert raw.observations[0]["sync"]["interpolation_alpha"] == pytest.approx(0.5)
    assert raw.observations[0]["canonical_state"]["source"] == "ObservationSynchronizer"
    out = tmp_path / "lerobot"
    exported = export_lerobot_draft(episode_path, out)
    row = json.loads((out / "data" / "chunk-000" / "episode_000001.jsonl").read_text())
    assert row["observation.state"] == pytest.approx(expected)
    assert exported.format == "jsonl_observation_only_draft"


def _two_observations():
    first = _interpolated_observation()
    second = make_observation(
        sequence_id=1,
        capture_time_ns=1_015_000_000,
        robot=first.robot,
        images={
            IMAGE_KEY_BASE_RGB: make_image(
                IMAGE_KEY_BASE_RGB,
                host_time_ns=1_015_000_000,
                device_time_ns=1_014_999_000,
                frame_number=1,
            ),
            IMAGE_KEY_WRIST_RGB: make_image(
                IMAGE_KEY_WRIST_RGB,
                host_time_ns=1_015_000_000,
                device_time_ns=1_014_999_000,
                frame_number=1,
            ),
        },
    )
    return [first, second]


def test_actions_require_explicit_observation_identity(tmp_path: Path) -> None:
    observations = _two_observations()
    action = {
        "observation_sequence_id": 0,
        "observation_capture_time_ns": observations[0].capture_time_ns,
        "actions": [0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "phase": "commanded",
    }
    second_action = {
        "observation_sequence_id": 1,
        "observation_capture_time_ns": observations[1].capture_time_ns,
        "actions": [0.002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "phase": "commanded",
    }
    result, episode_path = _record_observations(tmp_path, observations, [action, second_action])
    assert result.committed
    assert validate_episode(episode_path) == []
    out = tmp_path / "lerobot"
    exported = export_lerobot_draft(episode_path, out, require_calibrated=False)
    info = json.loads((out / "meta" / "info.json").read_text())
    assert exported.action_available is True
    assert info["features"]["action"]["shape"] == [7]
    assert info["action_identity"] == "observation_sequence_id"
    rows = [
        json.loads(line)
        for line in (out / "data" / "chunk-000" / "episode_000001.jsonl").read_text().splitlines()
    ]
    assert rows[0]["action"] == pytest.approx(action["actions"])
    assert rows[1]["action"] == pytest.approx(second_action["actions"])


def test_partial_actions_refuse_training_export(tmp_path: Path) -> None:
    observations = _two_observations()
    action = {
        "observation_sequence_id": 0,
        "actions": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "phase": "commanded",
    }
    _, episode_path = _record_observations(tmp_path, observations, [action])
    with pytest.raises(Exception) as exc:
        export_lerobot_draft(episode_path, tmp_path / "lerobot", require_calibrated=False)
    message = str(exc.value).lower()
    assert "partial action" in message or "do not cover every observation" in message


def test_action_without_identity_is_refused(tmp_path: Path) -> None:
    observations = _two_observations()
    _, episode_path = _record_observations(
        tmp_path,
        observations,
        [{"actions": [0.0] * 7, "phase": "commanded"}],
    )
    with pytest.raises(Exception) as exc:
        export_lerobot_draft(episode_path, tmp_path / "lerobot", require_calibrated=False)
    assert "observation_sequence_id" in str(exc.value)


@pytest.mark.parametrize("failure_kind", ["robot", "camera", "sync"])
def test_collection_exception_aborts_without_committing(tmp_path: Path, monkeypatch, failure_kind: str) -> None:
    from vla_infra.errors import DeviceError

    app = _app()
    harness = CollectionHarness(
        app,
        task="fault injection",
        output_root=str(tmp_path),
        recorder_config=_recorder_config(),
    )
    original_start = harness.start

    def start_then_inject():
        original_start()
        if failure_kind == "robot":
            def boom():
                raise DeviceError("injected robot read failure")

            monkeypatch.setattr(harness._robot, "read_state", boom)
        elif failure_kind == "camera":
            def boom_camera():
                raise DeviceError("injected camera failure")

            monkeypatch.setattr(harness._cameras["base"]["color"], "read", boom_camera)
        else:
            def boom_sync(*args, **kwargs):
                raise RuntimeError("injected sync step failure")

            monkeypatch.setattr(harness._sync, "build", boom_sync)

    monkeypatch.setattr(harness, "start", start_then_inject)
    with pytest.raises(Exception):
        harness.run(duration_s=0.2)

    assert not list(tmp_path.glob("episode_*"))
    from vla_infra.recording.raw_writer import uncommitted_episodes

    pending = uncommitted_episodes(tmp_path)
    assert len(pending) == 1
    marker = json.loads((pending[0] / ".incomplete").read_text())
    assert marker["reason"].startswith("collection_exception")
    result = harness.recorder.result
    assert result.committed is False
    assert "collection_exception" in result.error


def _copy_episode(source: Path, destination: Path) -> Path:
    import shutil

    shutil.copytree(source, destination)
    return destination


def test_validator_detects_dangling_image_and_counts(tmp_path: Path, recorded_episode: Path) -> None:
    episode = _copy_episode(recorded_episode, tmp_path / "ep")
    observations = [
        json.loads(line) for line in (episode / "observations.jsonl").read_text().splitlines()
    ]
    base_file = episode / observations[0]["images"][0]["file"]
    base_file.unlink()
    issues = validate_episode(episode)
    assert any("missing file" in issue or "dangling image" in issue for issue in issues)

    metadata = json.loads((episode / "metadata.json").read_text())
    metadata["counts"]["camera_frames"] += 1
    (episode / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    issues = validate_episode(episode)
    assert any("camera_frames" in issue for issue in issues)


def test_validator_detects_schema_and_sequence_tampering(tmp_path: Path, recorded_episode: Path) -> None:
    episode = _copy_episode(recorded_episode, tmp_path / "ep")
    schema = json.loads((episode / "lowdim" / "schema.json").read_text())
    schema["byte_length"] += 1
    (episode / "lowdim" / "schema.json").write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
    assert any("byte_length" in issue for issue in validate_episode(episode))

    observations = (episode / "observations.jsonl").read_text().splitlines()
    (episode / "observations.jsonl").write_text("\n".join(reversed(observations)) + "\n")
    assert any("sequence" in issue for issue in validate_episode(episode))


def test_validator_detects_missing_canonical_state_and_drop_gate(
    tmp_path: Path, recorded_episode: Path
) -> None:
    episode = _copy_episode(recorded_episode, tmp_path / "ep")
    records = [
        json.loads(line) for line in (episode / "observations.jsonl").read_text().splitlines()
    ]
    records[0].pop("canonical_state")
    (episode / "observations.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n"
    )
    assert any("canonical_state" in issue for issue in validate_episode(episode))

    episode2 = _copy_episode(recorded_episode, tmp_path / "ep2")
    metadata = json.loads((episode2 / "metadata.json").read_text())
    metadata.setdefault("recorder", {})["dropped_observations"] = 1
    (episode2 / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    assert any("dropped_observations" in issue for issue in validate_episode(episode2))
    assert not any(
        "dropped_observations" in issue
        for issue in validate_episode(episode2, max_dropped_observations=1)
    )


def test_commit_rename_failure_restores_incomplete_marker(tmp_path: Path, monkeypatch) -> None:
    writer = _writer(tmp_path)
    from vla_infra.testing import make_robot_sample

    writer.append_lowdim([make_robot_sample(FakeClock(), host_time_ns=1_000_000_000)])

    def fail_rename(self, target):
        raise OSError("injected rename failure")

    monkeypatch.setattr(Path, "rename", fail_rename)
    with pytest.raises(OSError):
        writer.commit()
    assert not writer.final_dir.exists()
    assert writer.tmp_dir.exists()
    marker = json.loads((writer.tmp_dir / ".incomplete").read_text())
    assert marker["reason"] == "commit_failed"
    with pytest.raises(RecordingError):
        load_episode(writer.tmp_dir)
    writer.abort("rename test", remove=True)


def test_commit_fsyncs_data_and_directories(tmp_path: Path, monkeypatch) -> None:
    import os as os_module

    calls = []
    original_fsync = os_module.fsync

    def counting_fsync(fd):
        calls.append(fd)
        return original_fsync(fd)

    monkeypatch.setattr("vla_infra.recording.raw_writer.os.fsync", counting_fsync)
    writer = _writer(tmp_path)
    from vla_infra.testing import make_robot_sample

    writer.append_lowdim([make_robot_sample(FakeClock(), host_time_ns=1_000_000_000)])
    writer.commit()
    assert len(calls) >= 3
    assert writer.path.exists()


def test_concurrent_producers_drop_counters_are_consistent(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    warnings = []
    recorder = EpisodeRecorder(
        writer,
        _recorder_config(max_lowdim_rows=10, queue_size=1),
        on_warning=lambda reason, details: warnings.append(reason),
    )
    recorder.start()
    accepted = 0
    dropped = 0
    lock = __import__("threading").Lock()

    def produce(worker: int) -> None:
        nonlocal accepted, dropped
        for index in range(50):
            sample = make_robot_sample(
                FakeClock(),
                host_time_ns=1_000_000_000 + worker * 1_000 + index,
            )
            ok = recorder.record_robot_sample(sample)
            with lock:
                if ok:
                    accepted += 1
                else:
                    dropped += 1

    threads = [__import__("threading").Thread(target=produce, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    result = recorder.stop(timeout_s=5.0)
    assert accepted + dropped == 400
    assert result.dropped_lowdim == dropped
    assert result.lowdim_rows + result.dropped_lowdim == 400
    assert warnings.count("lowdim_buffer_full") <= 1


def test_exporter_rejects_corrupt_episode_without_replay(tmp_path: Path, recorded_episode: Path) -> None:
    episode = _copy_episode(recorded_episode, tmp_path / "corrupt")
    observations = (episode / "observations.jsonl").read_text().splitlines()
    (episode / "observations.jsonl").write_text("\n".join(reversed(observations)) + "\n")
    with pytest.raises(Exception) as exc:
        export_lerobot_draft(episode, tmp_path / "out")
    assert "validation" in str(exc.value).lower() or "sequence" in str(exc.value).lower()

    episode2 = _copy_episode(recorded_episode, tmp_path / "counts")
    metadata = json.loads((episode2 / "metadata.json").read_text())
    metadata["counts"]["observation_records"] = 0
    (episode2 / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    with pytest.raises(Exception) as exc:
        export_lerobot_draft(episode2, tmp_path / "out2")
    assert "observation_records" in str(exc.value)

    episode3 = _copy_episode(recorded_episode, tmp_path / "drop")
    metadata = json.loads((episode3 / "metadata.json").read_text())
    metadata.setdefault("recorder", {})["dropped_observations"] = 1
    (episode3 / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    with pytest.raises(Exception) as exc:
        export_lerobot_draft(episode3, tmp_path / "out3")
    assert "dropped_observations" in str(exc.value)


def test_observation_only_info_is_not_training_ready(recorded_episode: Path, tmp_path: Path) -> None:
    out = tmp_path / "lerobot"
    export_lerobot_draft(recorded_episode, out)
    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["training_ready"] is False


def _tamper_first_state(episode: Path, *, values: bool = False, typed: bool = False) -> None:
    records = [
        json.loads(line) for line in (episode / "observations.jsonl").read_text().splitlines()
    ]
    state = records[0]["canonical_state"]
    if values:
        state["values"][0] = state["values"][0] + 0.25
    if typed:
        state["typed"]["tcp_position"][0] = state["typed"]["tcp_position"][0] + 0.25
    (episode / "observations.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n"
    )


def test_validator_detects_values_or_typed_tampering(tmp_path: Path, recorded_episode: Path) -> None:
    for name, kwargs in (("values", {"values": True}), ("typed", {"typed": True})):
        episode = _copy_episode(recorded_episode, tmp_path / name)
        _tamper_first_state(episode, **kwargs)
        issues = validate_episode(episode)
        assert any("canonical values do not match typed" in issue for issue in issues)
        with pytest.raises(Exception):
            export_lerobot_draft(episode, tmp_path / f"{name}_out")


def _observation_only_episode(tmp_path: Path):
    writer = _writer(tmp_path)
    return writer, EpisodeRecorder(writer, _recorder_config(queue_size=4))


def test_stop_closes_accepting_atomically_for_observation(tmp_path: Path, monkeypatch) -> None:
    writer, recorder = _observation_only_episode(tmp_path)
    recorder.start()
    entered = threading.Event()
    release = threading.Event()
    original_put = recorder._events.put_nowait

    def blocked_put(item):
        entered.set()
        assert release.wait(2.0)
        return original_put(item)

    monkeypatch.setattr(recorder._events, "put_nowait", blocked_put)
    accepted = []
    producer = threading.Thread(
        target=lambda: accepted.append(recorder.record_observation(_observation_with_frames(0)))
    )
    producer.start()
    assert entered.wait(2.0)
    stopped = []
    stopper = threading.Thread(target=lambda: stopped.append(recorder.stop(timeout_s=2.0)))
    stopper.start()
    time.sleep(0.02)
    assert stopper.is_alive()  # stop is blocked on the same lifecycle lock
    release.set()
    producer.join(2.0)
    stopper.join(2.0)
    result = stopped[0]
    assert accepted == [True]
    assert result.committed is True
    assert result.observations == 1


def test_stop_closes_accepting_atomically_for_action(tmp_path: Path, monkeypatch) -> None:
    writer, recorder = _observation_only_episode(tmp_path)
    recorder.start()
    entered = threading.Event()
    release = threading.Event()
    original_put = recorder._events.put_nowait

    def blocked_put(item):
        entered.set()
        assert release.wait(2.0)
        return original_put(item)

    monkeypatch.setattr(recorder._events, "put_nowait", blocked_put)
    accepted = []
    producer = threading.Thread(
        target=lambda: accepted.append(
            recorder.record_action(
                {
                    "observation_sequence_id": 0,
                    "actions": [0.0] * 7,
                    "phase": "commanded",
                }
            )
        )
    )
    producer.start()
    assert entered.wait(2.0)
    stopped = []
    stopper = threading.Thread(target=lambda: stopped.append(recorder.stop(timeout_s=2.0)))
    stopper.start()
    time.sleep(0.02)
    assert stopper.is_alive()
    release.set()
    producer.join(2.0)
    stopper.join(2.0)
    assert accepted == [True]
    assert stopped[0].committed is True
    assert stopped[0].action_records == 1


def test_events_after_stop_are_rejected_and_counted(tmp_path: Path) -> None:
    writer, recorder = _observation_only_episode(tmp_path)
    recorder.start()
    result = recorder.stop(timeout_s=2.0)
    assert result.committed is True
    assert recorder.record_observation(_observation_with_frames(0)) is False
    assert recorder.record_action({"observation_sequence_id": 0, "actions": [0.0] * 7}) is False
    final = recorder.result
    assert final.dropped_observations == 1
    assert final.dropped_actions == 1


def _wait_thread(recorder, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while recorder._thread is not None and recorder._thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)


def test_stop_timeout_cancels_before_commit(tmp_path: Path, monkeypatch) -> None:
    writer, recorder = _observation_only_episode(tmp_path)
    original_write = writer.write_image
    image_started = threading.Event()

    def slow_write(*args, **kwargs):
        image_started.set()
        time.sleep(0.3)
        return original_write(*args, **kwargs)

    monkeypatch.setattr(writer, "write_image", slow_write)
    recorder.start()
    assert recorder.record_observation(_observation_with_frames(0))
    assert image_started.wait(2.0)
    first = recorder.stop(timeout_s=0.01)
    _wait_thread(recorder)
    final = recorder.result
    assert final.committed is False
    assert not list(tmp_path.glob("episode_*"))
    assert list(tmp_path.glob(".episode_*.tmp-*"))
    assert first.still_stopping is True or first.committed is False


def test_stop_timeout_during_commit_reports_still_stopping(tmp_path: Path, monkeypatch) -> None:
    writer, recorder = _observation_only_episode(tmp_path)
    original_commit = writer.commit
    commit_started = threading.Event()

    def slow_commit(*args, **kwargs):
        commit_started.set()
        time.sleep(0.6)
        return original_commit(*args, **kwargs)

    monkeypatch.setattr(writer, "commit", slow_commit)
    recorder.start()
    first = recorder.stop(timeout_s=0.1)
    assert commit_started.wait(2.0)
    _wait_thread(recorder)
    final = recorder.result
    assert first.still_stopping is True
    assert final.committed is True
    assert final.path is not None and final.path.exists()


def test_collection_harness_waits_for_slow_commit(tmp_path: Path) -> None:
    app = _app()
    harness = CollectionHarness(
        app,
        task="slow commit",
        output_root=str(tmp_path),
        recorder_config=_recorder_config(finalize_timeout_s=3.0),
    )
    harness.start()
    harness.step()
    original_commit = harness._writer.commit

    def slow_commit(*args, **kwargs):
        time.sleep(0.6)
        return original_commit(*args, **kwargs)

    harness._writer.commit = slow_commit
    started = time.monotonic()
    # The finalize deadline is shorter than the commit: once committing the
    # harness must wait for the authoritative result rather than exit early.
    result = harness.stop(timeout_s=0.2, finalize_timeout_s=0.05)
    assert time.monotonic() - started >= 0.5
    assert result.recorder.committed is True
    assert result.recorder.still_stopping is False
    assert result.episode_path is not None
    assert Path(result.episode_path).exists()


def test_collection_harness_finalize_deadline_keeps_temp_incomplete(tmp_path: Path) -> None:
    app = _app()
    harness = CollectionHarness(
        app,
        task="slow png",
        output_root=str(tmp_path),
        recorder_config=_recorder_config(finalize_timeout_s=0.2),
    )
    harness.start()
    harness.step()
    original_write = harness._writer.write_image

    def slow_write(*args, **kwargs):
        time.sleep(0.3)
        return original_write(*args, **kwargs)

    harness._writer.write_image = slow_write
    result = harness.stop(timeout_s=0.01, finalize_timeout_s=0.02)
    _wait_thread(harness._recorder)
    final = harness.recorder.result
    assert final.committed is False
    assert not list(tmp_path.glob("episode_*"))
    assert list(tmp_path.glob(".episode_*.tmp-*"))
    assert result.recorder.committed is False
