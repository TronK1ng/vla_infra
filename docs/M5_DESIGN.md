# M5 design: real hardware boundary (Round 01 skeleton)

Scope: **code skeleton and injected-SDK tests only**. No real robot connection,
Enable, mode switch or command is authorized by this document. M5 field motion
still requires the M2 on-site read-only gate, M5 code review and the on-site
operator/leader gates.

## Delivered in Round 01

- `devices/flexiv_rdk.py::FlexivRdkNrtCommandSink`:
  - injected SDK / lazy import; `connect()` only constructs `Robot`;
  - explicit `enable_and_switch_nrt(authorization, ...)` requiring an
    HMAC-issued preflight/ACK authorization; lifecycle is
    Enable -> bounded operational wait -> SwitchMode -> actual `mode()` verify;
  - authorization is re-checked before every command; `revoke()`/Stop/failure
    invalidate the epoch and all future writes;
  - Stop revokes the shared capability **before** calling the blocking SDK
    `Stop()`; success sets `physical_stop_confirmed=True`, failure keeps it
    False and still leaves software authority revoked;
  - failure attempts `Stop()` and never ClearFault;
  - `Mode.NRT_CARTESIAN_MOTION_FORCE`, all six force axes false;
  - explicit `NrtLimits` on **every** `SendCartesianMotionForce` call;
  - explicit `base_to_world` 4x4 transform, xyzw -> wxyz conversion and
    shortest-quaternion path;
  - single writer/owner-thread guard;
  - `Stop()` failure/status observable; no automatic ClearFault/recovery.
- `devices/gn01.py::Gn01CommandSink`:
  - requires the **shared RDK Robot handle** (`Gripper(robot)`);
  - `connect()` only constructs the interface; explicit `enable(authorization,
    authority_epoch, name)` requires the shared capability to be **already
    active** with the same auth/epoch (it never grants it itself), then calls
    `Gripper.Enable(name)` and validates configured width/velocity/force are a
    subset of `gripper.params()`;
  - params/Enable failure attempts `Disable()` and records rollback result;
  - every Move checks authorization validity and epoch; `revoke()`/Stop expires
    both arm and gripper capabilities;
  - Stop failure still revokes software authority and records
    `physical_stop_confirmed=False`; Tool.Switch uses the same
    `capability.check(auth, epoch)` as Move;
  - Send/Move exceptions immediately revoke the shared capability;
  - Tool.Switch success marks `ToolReadiness.switched` but not `verified`;
    motion requires explicit `verify_tool(name, tcp_hash, payload_hash)`.
    Tool.Switch failure revokes the shared capability and resets tool readiness.
  - `Tool.Switch(name)` is a separate explicit method; never automatic;
  - `Move(width, velocity, force)` positional API matching v1.9.3;
  - normalized 0..1 -> physical width mapping, dedup/range checks, Stop status.
- `devices/real_robot.py::RealRobotCommandSink`:
  - composite arm + gripper; partial gripper writes observable;
  - composite Stop failure observable.
- `config_m5.py`:
  - strict `m5-station/v1` + `m5-safety/v1` loader;
  - station/safety hashes include the complete safety profile;
  - `allow_real_motion: false` default and env gate helper.
- `devices/preflight.py`:
  - machine-readable `M5PreflightReport` with required sections;
  - strict probe schema (`PASS`/`FAIL` only); malformed/missing/FAIL blocks;
  - fixed enable probe list; empty/short lists fail closed;
  - `PASS_ENABLE_ALLOWED` requires all enable probes + explicit operator ACK;
  - deep-immutable report (canonical JSON) with
    `validate_for_enable()` re-checking schema, blockers, every section/probe,
    verdict and freshness;
  - report carries `generated_at_ns`/`expires_at_ns` and `report_hash`.
- `devices/tool_state.py`:
  - shared `ToolReadiness` gate: switched != verified; motion requires both;
  - optional expected name/TCP/payload hashes are compared on verification.
- `devices/stop_control.py`:
  - `run_stop_with_deadline()` observes a blocking SDK Stop in a controlled
    thread and reports `timed_out`/`thread_alive`; it never claims stopped.
- `runtime/m5.py`:
  - `M5Runtime.compose()` is the single production composition root: it
    creates authority, capability, expected ToolReadiness and the shared
    arm/GN01 factory; non-composed runtime construction validates that both
    sinks are bound to the same ControlAuthority;
  - production constructor has no `production=False` switch; production
    composition rejects placeholder tool profiles;
  - Enable checks `allow_real_motion`, env, immutable config hash and
    report/auth config hash before any Authority grant/SDK call;
  - NEW -> CONNECTED_READONLY -> PREFLIGHT_OK/FAILED ->
    WAITING_OPERATOR_ACK -> AUTHORIZED -> ENABLED/MODE_READY -> ARMED;
  - ACK must echo report or config hash; authorization issued only after
    PASS_ENABLE_ALLOWED; reconnect/invalidation clears ACK/auth;
  - Enable is transactional: any stage failure revokes authority, enters
    STOPPING, stops arm and GN01 independently, then FAULT/STOP_FAILED;
  - Stop starts arm and GN01 Stop observers independently (one blocked device
    cannot prevent the other), aggregates per-device outcomes and only reports
    STOPPED when all confirmed;
  - fault/invalidate while active enter STOPPING and perform both bounded
    Stops; Stop failure prevents a false CONNECTED_READONLY/FAULT claim;
  - stop timeout comes from approved `config.safety.stop_timeout_s`;
  - all lifecycle long operations are serialized by a separate operation
    mutex; the ineffective generation field is removed; state lock is short-held
    for snapshots/commits;
  - rollback/close use `stop_pair_bounded()`: arm and GN01 Stop start in
    parallel under one deadline; a blocked arm cannot prevent GN01 Stop;
  - watchdog terminal Stop acquires the same lifecycle operation mutex and
    re-reads state under it; normal close sets a normal-shutdown intent so a
    watchdog exit cannot overwrite STOPPED with FAULT;
  - runtime/session share `stop_pair_bounded()` for rollback/close;
  - `close()` has separate CLOSING/CLOSED flags: failure never becomes a
    silent successful close; owner is released only after confirmed Stop and
    resource cleanup;
  - `create_composite()` tracks partial arm/GN01 in `_pending_owned`; cleanup
    uses the configured safety timeout, keeps failed pending resources for
    audit, and quarantines on any unconfirmed Stop/close;
  - STOPPING/STOP_FAILED are sticky; only an explicit
    `recover_stop_failed(operator_ack, e_stop_released, physical_stop_confirmed)`
    can leave STOP_FAILED;
  - `start_watchdog()` runs an independent `m5-watchdog` thread using the
    runtime monotonic clock; it actively faults/stops on authorization expiry
    and missing/stale policy/executor/state heartbeats, not only on the next
    command; Enable is refused unless all three heartbeats are fresh and the
    watchdog is alive/healthy; terminal watchdog exceptions revoke + Stop;
    `close()` joins it;
  - tool/TCP/payload production hashes require lowercase SHA-256 hex and config
    hash binding; canonical serialization/hash helper is required before field
    use;
  - a session can claim only one production runtime owner (lock-protected,
    exact token identity); wrong/missing tokens fail closed; compose abort
    revokes/stops/closes before releasing the token; failed Stop quarantines
    the session and retains ownership;
  - all writer factories require the exact runtime owner token once a session
    is claimed; create/abort/close/quarantine run under one session lifecycle
    lock; arm/GN01 share a permanent `SessionSafetyGate`;
  - quarantine is a hard gate across connect/create/preflight/authorize/
    enable/recover/heartbeat/start_watchdog; a quarantined old runtime can never
    recover/re-enable; it exposes `quarantine_reason`, pending resources and
    per-device StopOutcome audit; only process restart/new session can recover;
  - runtime detects session quarantine and commits sticky `QUARANTINED`;
  - closed runtime refuses all mutating APIs except idempotent close;
  - heartbeat accepts only policy/executor/state, rejects future/regressing
    timestamps; production callers do not supply wall time.
- `devices/m5_auth.py`:
  - HMAC-bound `MotionAuthorization` over station/config/report hash, ACK and
    freshness; issuer binds expected station/config and validates the report
    via `validate_for_enable()` before signing; writer requires a verifier, so
    a fabricated dataclass/report is rejected.
- `devices/real_session.py`:
  - one `FlexivRdkSession` creates the single RDK `Robot`, then injects the same
    handle into arm NRT sink, GN01 sink and Tool; tests use this factory rather
    than private fields;
  - **no public raw Robot handle**: `connect()` returns `None` and the class
    exposes only a read-only `state` facade plus controlled sinks; tests inspect
    the fake SDK's recorded handle, not a production accessor;
  - `close()` revokes all owned sinks, attempts Stop/close, and raises
    `STOP_FAILED` if any physical stop cannot be confirmed; it keeps handles on
    failure rather than pretending the session is safely closed.
- `configs/station.real.template.yaml`, `configs/safety.real.template.yaml`.

## Explicit non-goals in Round 01

- No real `flexivrdk` import in tests and no real hardware connection.
- No M5Runtime lifecycle, operator ACK, preflight field execution, disk soak or
  field motion. Those are Round 02 / Field gates.
- No RT 1 kHz control, force-axis enable, wrench/torque policy, automatic
  fault recovery, or `ClearFault`.

## Round 01 tests

- `tests/unit/test_m5_writer.py`: connect-no-enable, explicit enable/NRT,
  quaternion/frame/limits, single writer, SDK/Stop failures, GN01 mapping/dedup,
  composite partial write, real gate.
- `tests/unit/test_m5_config_preflight.py`: strict templates, hash, disabled
  real motion, preflight verdicts/report JSON.

## Stop timeout capability boundary (Round 01)

`stop_motion(timeout_s)` records calls/failures but the RDK `Stop()` API is
blocking and cannot be interrupted from Python.  Round 01 does **not** claim
that `timeout_s` bounds the SDK call.  Round 02 must run Stop in a controlled
Supervisor boundary, observe the deadline, and escalate to `STOP_FAILED` /
E-stop instructions.  This limitation is explicit rather than hidden.

## Field preconditions (not executed)

- M2 read-only 30 min signed, GN01 read provider, serials/calibration/tool/TCP;
- M5 Round 02 lifecycle/ACK/FAULT matrix reviewed;
- on-site safety profile approval, writer lock, E-stop observer.
