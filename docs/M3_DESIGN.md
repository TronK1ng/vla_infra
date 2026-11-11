# M3 design: fake 100 Hz execution + full safety profile

Scope: **fake/simulation only**. M3 does not connect RDK/RealSense, does not
enable/clear/move a real robot, and does not run a real VLA rollout.

## Components

```text
FakeRobotArm / FakeCamera
        │
        ├── 500 Hz state acquisition thread ──> latest state ──> Executor
        │        └── heartbeats robot/camera
        │
        ├── 20 Hz chunk producer thread ─────> ActionInbox (latest-only)
        │        └── heartbeat policy          │
        │                                      ▼
        └──────────────────────────────> 100 Hz executor thread
                                                │
                                  SafetyGate ──> FakeRobotArm commands
                                                │
                                  Supervisor watchdog / HOLD / FAULT
                                                │
                                  ExecutionLog (bounded, async JSONL)
```

- `runtime/m3.py`: `M3Runtime` fake-only composition and metrics.  Startup
  waits (bounded) for real robot/camera/policy/executor readiness before
  granting authority; no heartbeat is fabricated.  Startup failure cleans up
  threads/log/cameras and never writes.
- `runtime/execution_log.py`: bounded audit queue + writer thread; never blocks
  the executor.
- `execution/safety.py`: full M3 profile adds configurable acceleration and
  joint-velocity limits; chunk and per-tick checks reject rather than clip.
- `execution/executor.py`: 100 Hz tick; tracks vector velocities for acceleration
  checks; places the exact state sample used into `ExecutionReport` for audit;
  single-writer `ControlAuthority` unchanged.
- `execution/supervisor.py`: robot/camera/policy/executor watchdogs with
  HOLD/FAULT severity; only `RUNNING` permits writes.
- `execution/interpolation.py`: M3 explicit **reject-only** baseline; automatic
  velocity-based retiming is not implemented and is deferred to a separate
  milestone with its own acceptance.  Safety rejects overspeed instead of
  time-stretching it.

## Safety profile additions

`SafetyProfile` now supports:

- `max_linear_acceleration_m_s2`;
- `max_angular_acceleration_rad_s2`;
- `joint_velocity_max_rad_s` (7 values);
- existing workspace, height, step, velocity, first-target, force/torque,
  gripper, staleness, joint-position and frame/type gates.

All are reject-only. Acceleration is checked at chunk and per-tick level; the
executor records previous tick velocities and passes them to `check_tick`.
`configs/safety.example.yaml` sets a conservative fake profile.

## Runtime timing and logging

- executor period: 10 ms (100 Hz), dedicated thread; first write uses the real
  previous control-tick interval, never a fabricated 1 ns dt;
- state acquisition: 2–5 ms (fake);
- policy chunk production: 50 ms (20 Hz), separate thread;
- log: bounded queue (4096), JSONL writer thread, fsync on close; close returns
  thread-liveness and is part of the stop total deadline; a blocked logger makes
  `STOP_FAILED`, never `STOPPED`;
- `on_tick` callbacks run on a separate bounded `m3-callback` thread with
  capacity 1, so a blocked callback cannot stall the 100 Hz executor.  Callback
  metrics expose `submitted/enqueued/processed/dropped/errors`, last drop/error
  tick and error type; this is a **latest-only notification**, never safety,
  command or complete-audit critical.
- Startup cleanup uses a total deadline for threads and logger close.  If any
  control thread **or the logger** is still alive, lifecycle is
  `START_CLEANUP_FAILED`; cameras/resources are not closed and completion is not
  claimed.  Logger drain uses only the remaining cleanup deadline.
- metrics: tick period, runtime, jitter, overrun, p50/p95/p99/max; log
  each record correlates `raw_action` (request sequence, capture/receive time,
  epoch, frame/type/dt, segment and raw action row), `raw_state`,
  `filtered_target` (SafetyGate output), `commanded_target` (interpolated target
  actually attempted), `commanded_status`
  (`not_attempted|attempted|partial|succeeded`) and `measured_state` (timestamp
  and sequence).  HOLD/reject never reports a commanded write.

## Fault matrix (fake)

| Fault | Injection | Expected |
| --- | --- | --- |
| policy/network stall | `set_policy_enabled(False)` | policy watchdog HOLD; no writes |
| state stale | `set_state_enabled(False)` | executor stale-state HOLD; no writes |
| camera stale | `set_camera_enabled(False)` | camera watchdog HOLD |
| old chunk | `submit_actions(capture_offset_ns<0)` | stale observation rejection; HOLD |
| queue overwrite | rapid `submit_actions` | latest-only inbox replaces; count tracked |
| NaN/Inf | ActionChunk/contract rejection | rejected before executor |
| workspace/velocity/accel | valid chunk with unsafe values | SafetyViolation; plan rejected; HOLD |
| writer exception | inject command sink failure | executor WRITE_FAILED/FAULT; no further writes |
| process stop | `M3Runtime.stop()` | total stop deadline; revoke in a helper thread; timeout returns `stop_error`/`STOP_FAILED`, no resource close/release |
| HOLD recovery | `M3Runtime.rearm()` | no fabricated heartbeat; requires real robot/camera/policy/executor freshness after HOLD; fresh watchdogs, ARMED -> RUNNING |

Watchdogs with `fault` severity latch FAULT and require supervisor reset; `hold`
severity stops at HOLD and can be re-armed by an explicit operator path. M3 tests
cover HOLD rearm; FAULT reset remains an operator/runtime policy.

## Acceptance boundary

M3 automation uses fake devices only. Real 100 Hz RDK performance, GN01,
RealSense and real policy rollout remain field/M4/M5 work; nothing here authorizes
real motion.
