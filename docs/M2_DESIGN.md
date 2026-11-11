# M2 design: read-only collection, sync and raw persistence

Scope: implement the M2 data path from the master plan §11 without connecting
real hardware in this round and without entering policy rollout (M4/M5).

## Delivered components

```text
RealSenseCamera (read-only, lazy SDK) ─┐
FlexivRdkReadOnlyState (read-only) ───┼─> ObservationSynchronizer ─> EpisodeRecorder
FakeCamera / FakeRobotArm ────────────┘         (sync report)             │
                                                                   RawEpisodeWriter
                                                                    (temp -> atomic rename)
                                                                           │
                                                         EpisodeRecorder queue/thread
                                                                           │
                                                              raw episode + schema/metadata
                                                                           │
                                                       reader/validator + LeRobot JSONL draft
```

* `devices/realsense.py`: serial-bound RealSense adapter, RGB8 color, aligned
  Z16 depth, intrinsics/depth scale, device-ms→ns preservation, USB3 check.
* `devices/flexiv_rdk.py`: strict read-only RDK state adapter. Only `Robot()` and
  `states()` are called; `tcp_pose` wxyz is converted to canonical xyzw;
  missing wrench/gripper width fields fail closed.
* `recording/schema.py`: explicit lowdim structured schema (units/frames).
* `recording/raw_writer.py`: raw episode layout, PNG color/depth, observation and
  action provenance, atomic directory commit, uncommitted crash markers.  On
  commit, JSON/lowdim/manifest handles are flushed and `fsync`ed, image PNGs are
  `fsync`ed when written, all temporary directories are `fsync`ed before rename,
  and the parent directory after rename; the remaining durability caveat is that
  a power loss in the tiny window after the rename syscall is still filesystem
  defined (documented, not claimed stronger).
* `recording/recorder.py`: bounded asynchronous recorder with an explicit
  lifecycle (`accepting -> closing -> committing -> finished`). Producer
  acceptance and queue insertion happen under the same lock; `stop()` closes
  acceptance before waking the writer. The writer enters `committing` only after
  accepting=false, inflight=0 and buffers/queues are empty. A stop timeout before
  that point cancels the commit; a timeout during the non-cancellable rename
  section returns `still_stopping` instead of a contradictory final result.
  CollectionHarness then polls `wait_until_final()` using the configurable
  `recording.finalize_timeout_s`; the CLI waits and only reports incomplete
  after the deadline. This keeps the daemon writer from being killed mid-commit
  during a 30-minute field run.
  Lowdim uses a bounded buffer; observations/actions use a bounded queue. Disk
  full/slow never blocks the producer; drops are counted. Every observation
  persists its exact canonical state (vector, layout, typed provenance,
  capture/alpha).
* `recording/lerobot_writer.py`: runs strict `validate_episode()` before export,
  preserves the original observation order, omits the action feature for
  observation-only data, requires explicit `observation_sequence_id` coverage
  for every observation, refuses partial labels, and requires calibrated data
  by default for action-bearing training candidates.
* `recording/reader.py`: load/validate counts, timestamps, frame manifests and
  sync report.
* `recording/lerobot_writer.py`: JSONL LeRobot **draft** with canonical feature
  names, image copies, state/action metadata.
* `sync/synchronizer.py`: bounded sync-error/skew distribution report (p50/p95/
  p99/max/drops) written into every episode.
* `runtime/collection.py`: fake or real_readonly collection loop; no policy,
  no executor, no command sink.
* `cli/record.py`, `cli/replay.py`, `cli/doctor.py --preflight`.

## Data schema

Lowdim fields (float64 unless noted): host/device time ns (i8), sequence id (i8),
joint position/velocity (7), TCP position (3), quaternion xyzw (4), rotation-6D
(6), TCP velocity (6), TCP wrench (6), gripper width (1). Camera frames are PNG
with per-kind `frames.jsonl` entries (frame number, device/host time, path).
Observations reference camera frame numbers and record sync provenance.

## Async and failure behaviour

| Condition | Behaviour |
| --- | --- |
| PNG/disk slow | producer only queues; bounded drop counters, no blocking |
| disk below minimum at start | `DiskLowError`, no episode committed |
| disk fills during run | recorder fault, writer aborts/leaves `.incomplete`, capture loop continues |
| process crash | temp `.episode_*.tmp-*` with `.incomplete`, never treated as committed |
| image duplicate/encode failure | recorder fault, temp episode left identifiable |
| sync error/skew | synchronizer drops/counters; report emitted |
| collection step exception | `run()` aborts recorder/quarantine temp, re-raises; no `episode_xxxx` |
| producer/stop race | acceptance + queue insertion under one lock; accepted event cannot be enqueued after close |
| stop timeout before commit | commit cancelled, temporary `.incomplete`, final `committed=false` |
| stop timeout inside commit | result reports `still_stopping=true`; caller must observe final state |
| fsync/rename failure at commit | marker restored as `.incomplete`, no committed episode |

## Safety boundaries

* Collection creates no `ControlAuthority`, no executor and no command sink; it
  cannot move the robot.
* Real adapters are read-only by construction; `FlexivRdkRobot` remains M5.
* Automated tests use fake devices or injected SDK doubles; no real SDK import
  or hardware connection occurs.

## M2 acceptance status

Fake-backend pipeline tests and adapter SDK-double tests pass. The on-site M2
acceptance (30-minute uninterrupted capture, real serial/intrinsics verification,
real sync distribution, disk slow/full on the station) is **not executed** and
must be run by on-site personnel. Real read-only collection additionally needs a
GN01 width provider or an RDK gripper-width field; until then it fails closed.
