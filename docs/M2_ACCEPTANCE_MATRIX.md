# M2 acceptance matrix

| Master-plan M2 requirement | Implementation | Automated evidence |
| --- | --- | --- |
| RealSense adapter | `devices/realsense.py` | `test_realsense.py` with injected SDK |
| RDK read-only state adapter | `devices/flexiv_rdk.py` | `test_flexiv_rdk_readonly.py` |
| Observation sync | `sync/synchronizer.py` | existing sync tests + sync distribution |
| Raw recorder | `recording/raw_writer.py`, `recorder.py` | `test_recording.py` |
| LeRobot export draft | `recording/lerobot_writer.py` | `test_lerobot_observation_only_draft_uses_persisted_state`, action tests, `test_exporter_rejects_corrupt_episode_without_replay` |
| doctor/preflight CLI | `cli/doctor.py --preflight` | config/CLI tests |
| Atomic commit/crash markers | `RawEpisodeWriter` | `test_uncommitted_episode_is_identifiable` |
| commit fsync order | `RawEpisodeWriter` | `test_commit_fsyncs_data_and_directories` |
| exception never commits | `CollectionHarness.run` | `test_collection_exception_aborts_without_committing` |
| canonical observation state | recorder/export | `test_canonical_state_is_recorded_and_exported_exactly` |
| no fabricated zero actions | exporter | observation-only/partial/identity tests |
| producer/stop atomic acceptance | recorder lifecycle | `test_stop_closes_accepting_atomically_*`, `test_events_after_stop_are_rejected_and_counted` |
| stop timeout vs commit | commit authorization | `test_stop_timeout_cancels_before_commit`, `test_stop_timeout_during_commit_reports_still_stopping` |
| CLI/harness finalize wait | `wait_until_final` / `--finalize-timeout` | `test_collection_harness_waits_for_slow_commit`, `test_collection_harness_finalize_deadline_keeps_temp_incomplete`, CLI still-stopping test |
| exporter strict/no sort | `export_lerobot_draft` | `test_exporter_rejects_corrupt_episode_without_replay` |
| canonical values vs typed | reader | `test_validator_detects_values_or_typed_tampering` |
| validator tamper detection | reader | `test_validator_detects_*`, `test_validator_detects_values_or_typed_tampering` |
| Disk slow/full does not block capture | bounded recorder | `test_recorder_never_blocks_on_slow_image_encoding`, disk-low test |
| Frame/timestamp/sequence monotonicity | reader validation | `validate_episode`, adapter tests |
| Sync error distribution | `sync_report()` | `test_sync_report_has_distribution` |
| No motion commands | collection path has no executor | code + fake tests |
| Real 30-minute soak | on-site | **not run** |
| Real serials/intrinsics/calibration | real adapters/metadata | code only; **not run** |
| Real disk slow/full | recorder design | simulated only |

Known gaps: real collection needs GN01 width provider; LeRobot is JSONL draft
(no PyArrow/FFmpeg); real hardware/30-minute acceptance is on-site only.
