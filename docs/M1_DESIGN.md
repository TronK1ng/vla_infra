# M1 design notes and decisions

This document records *why* the M1 implementation looks the way it does, and
where it intentionally stops. The leader's master plan is the source of truth;
section references below are to that plan.

## 1. Scope

Delivered:

* installable package skeleton (`src/vla_infra`, `pyproject.toml`, configs);
* canonical observation / action contracts and validators;
* fake camera, robot and gripper backends;
* bounded ring buffer and observation synchroniser;
* dummy policy adapters, WebSocket server, client wrapper and background worker;
* control authority, latest-only inbox, provisional safety gate, interpolation,
  single-writer executor and supervisor state machine;
* hardware-free shadow harness and CLI (`doctor`, `shadow`);
* unit and integration tests, all offline.

Deliberately not delivered (M2–M6): RealSense/RDK/GN01 adapters, recording,
LeRobot export, official `openpi_client` adapter, Force Policy adapter, real
motion procedure, acceleration/force-control tuning, safety certification.

## 2. Process/threading model

One process, several threads, with interfaces that can be split into processes:

| Component | Thread | Notes |
| --- | --- | --- |
| fake camera capture | control thread (M1 harness) | separate worker in M2 |
| robot state | control thread (M1 harness) | 500–1000 Hz worker in M2 |
| observation sync | control thread | single consumer of ring buffer |
| policy worker | dedicated thread | latest-only input, one inference in flight |
| executor | control thread (harness) | own high-priority thread in M3/M5 |
| supervisor | control thread | sole authority writer |
| recording | absent (M2) | must be asynchronous and bounded |

The fake harness is intentionally single-threaded for the device/sync/execute
path plus one policy thread, so failures are reproducible. The production loop
can move the executor to its own thread without changing interfaces.

## 3. Key decisions

### 3.1 Epoch-bound action chunks

The policy worker reads the control epoch **before** sending the request and
submits the result with that epoch. The inbox rejects a submission whose epoch
is no longer current. Combined with HOLD clearing the inbox and incrementing the
epoch, this closes the classic race:

```text
observe/request → (HOLD happens) → response arrives → (ARM) → stale chunk executes
```

Tested by `test_hold_during_inference_rejects_chunk` and
`test_old_epoch_inbox_entry_is_not_written`.

### 3.2 Synchronous revocation

`ControlAuthority.write_permit` holds the authority lock across the device write.
`revoke()` waits for an in-flight write and no write can start afterwards. This
makes HOLD a real barrier rather than a status flag. The cost is that a slow
device write delays HOLD; real backends must keep NRT writes short and bounded.

### 3.3 Latest-only queues

The policy input slot and action inbox are both capacity 1 with explicit
replacement counters. This matches M1's "whole-chunk replacement" decision and
avoids unbounded latency growth. `ActionInbox` refuses any other capacity until
a replacement policy is designed and tested (M3+).

### 3.4 One target time per observation

The tick target is the newest required RGB host timestamp. Robot samples must
bracket it, so no future sample is used. An alternative is to trigger on the
base camera only; using the newest frame avoids declaring a wrist frame captured
after the tick as "future". This is recorded as an open question.

### 3.5 Provisional safety gate

Implemented: workspace AABB, height bounds, per-step translation/rotation,
first-target deviation, implied velocity, observation/chunk age, frame/type
allow-list, measured wrench force/torque, gripper width, optional joint limits,
per-tick re-check. Default action is rejection, never clipping.

Not implemented: acceleration profiles, an explicit trajectory retiming module,
RDK joint-limit source, contact-force prediction, F/T zeroing state machine, RT
loop. These are M3/M5 items. Per the leader's Round 02 decision, M1 keeps
"reject when nominal speed is infeasible" and the interpolator does **not**
silently stretch policy macro timing.

### 3.6 ARMED is not write-capable

The plan says ARMED/RUNNING may command motion. M1 grants the authority at ARM
but `write_permit` additionally requires RUNNING, so an operator must explicitly
start execution. This is stricter than the plan and fail-safe. It is flagged for
leader confirmation.

### 3.7 Protocol shape

The protocol is structurally OpenPI-compatible: dictionary observation with
`observation/image`, `observation/wrist_image`, `observation/state`, `prompt`,
`meta/*`, and dictionary action results with a chunk. It uses a thin envelope
(`type`, `schema_version`, `request_sequence_id`) and a msgpack codec with a
JSON+base64 fallback. It is **not** byte-compatible with `openpi_client` in M1;
`PolicyTransport` isolates that adapter for M4. The official server also does
not validate robot safety, which is why all safety lives locally.

### 3.8 Capture binding and executor fail-closed

Every response is bound to the request observation: the client stores the
request `meta/capture_time_ns` before sending and requires the echoed chunk
capture time to be identical; the policy worker repeats this check before the
inbox. This prevents a server/adapter bug from relabeling an old chunk with a
fresh timestamp and bypassing safety staleness checks.

The executor treats unexpected exceptions and device write failures as hard
faults: it clears the plan, revokes the authority locally and emits an audit
event **before** asking the supervisor to transition. The runtime maps
`internal_error`, `write_failed` and `write_partial` to FAULT. Pose and gripper
are separate calls; their outcomes are recorded explicitly, and the gripper is
not re-sent every tick when its normalised target is unchanged.

### 3.9 Wire images are RGB8, letterboxed client-side

Images are resized/padded on the robot side to reduce bandwidth. RGB is explicit
in metadata and validated on both sides; BGR is impossible to express silently.
Depth is in the canonical observation but not in the M1 policy payload (the
plan's minimal policy observation has no depth); model adapters can request it in
M4/M6 via extras.

## 4. Failure behaviour

| Condition | Detection | Result |
| --- | --- | --- |
| Policy timeout | client recv timeout | connection closed, supervisor HOLD |
| Policy disconnect / server stop | send/recv error | supervisor HOLD, inbox cleared |
| Reconnect | worker retry with backoff | policy heartbeat resumes, state stays HOLD, new ARM required |
| Stale observation | capture age > limit | chunk rejected, executor event, runtime HOLD |
| Stale chunk | receive age > limit | inbox drops at take, no write |
| Sequence replay/regression | inbox / server tracking | rejected |
| Over-force/torque | measured wrench vs profile | executor blocks plan; runtime HOLD |
| Watchdog expiry | supervisor tick | HOLD (camera/policy) or latched FAULT (robot/executor) |
| Plan deadline missed | executor tick | plan expired, no catch-up jump, runtime HOLD |
| Stale robot state | executor state age | no plan start, local authority revoke + runtime HOLD |
| Mislabelled policy response | request/response capture equality (client + worker) | connection closed / worker FAULT, chunk never enters inbox |
| Executor internal exception | fail-closed wrapper around every tick | active plan cleared, authority revoked, event `internal_error`, supervisor FAULT |
| Pose ok / gripper write failed | separate write outcomes | `write_partial` audit event, authority revoked, no further writes |
| Device raises NotArmedError after permit | permit-acquired flag distinguishes it from a HOLD race | `write_failed`/`write_partial`, local revoke, hard fault |
| New authority epoch, same gripper target | epoch-bound gripper cache | first command of each epoch is sent even if value is unchanged |
| publish() payload/bundle identity mismatch | immediate argument/ payload comparison | `ValidationError` before the latest slot |
| System proxy intercepts loopback | explicit `proxy=None` for loopback | local policy client works regardless of operator proxy env |

The runtime is responsible for mapping executor events to HOLD/FAULT. The M1
harness binds every rejection/expiry/stale event to HOLD and write failures to
FAULT; tests cover each path.

## 5. Concurrency and lock discipline

* Lock order: authority → inbox. Never the reverse.
* Inbox sequence history survives `clear()`, so a delayed response cannot be
  replayed after a HOLD.
* The ring buffer rejects non-increasing host timestamps and never extrapolates.
* Stats counters are updated by explicit dataclass replacement; no shared
  mutable counter arithmetic across threads without a lock.
* Supervisor history is bounded; queue depths are bounded at one.

## 6. Deliberate deviations / questions for the leader

1. **ARMED without write capability** (stricter than §9) — confirm acceptable.
2. **Velocity gate rejects rather than time-stretches** (resolved in Round 02:
   leader decided to keep nominal-timing rejection; any future explicit retiming
   must be a separate module with a dual time axis).
3. **`delta_tcp` is chunk-relative** (first row from measured pose, then
   cumulative). The plan does not define whether rows are independent or
   cumulative. Force Policy/π0.5 adapters must be mapped to this explicitly.
4. **Gripper normalisation direction** `0=closed, 1=open` is an assumption; the
   GN01 adapter must confirm or the contract must be revised before M5.
5. **Absolute orientation as rotation vector** is representation-ambiguous for
   general orientations; M1 allows it only in base frame and tests round-trips,
   but no model adapter is assumed to emit it yet.
6. **Tick target = newest RGB frame** rather than base camera only — confirm.
7. **Preflight/ack semantics**: preflight (`devices_ready`) does not require the
   operator ack; ARM does. M2 will attach the real read-only preflight evidence.
8. **Policy protocol not byte-compatible** with `openpi_client` until M4 —
   confirm the adapter boundary is acceptable.
9. **No recording in M1**, as instructed by §12.

## 7. Security notes

* Policy server defaults to loopback; non-loopback binding requires explicit
  opt-in **and** a bearer token, and is still plaintext (VPN/TLS required).
* Loopback WebSocket connections force `proxy=None`; remote proxy use is an
  explicit `policy.proxy` setting and is logged (credentials redacted). All
  proxy/handshake exceptions are normalized to `PolicyDisconnectedError`.
* Token is sent in an `Authorization: Bearer` header, never in the URL, and is
  redacted from endpoint logs. Config stores only the env-var name.
* No credentials or raw observations are logged by the M1 components.
* Real hardware imports are absent; SDK imports happen only in M2 adapters, and
  the real backend is refused by configuration.
* Error responses from the server never contain tracebacks; unexpected
  exceptions become `internal_error`.

## 8. Next milestones

* **M2**: RealSense + read-only RDK adapters, `doctor` preflight evidence,
  raw recorder with atomic commit, LeRobot draft, 30-minute soak.
* **M3**: complete safety profile (acceleration, time stretching policy, joint
  limits), action queue policy, fault-injection matrix, 100 Hz latency/jitter
  report, full raw/filtered/commanded/measured logging.
* **M4**: official OpenPI client wrapper, remote dummy server, shadow mode,
  reconnect/sequence tests across a real network.
* **M5**: RDK writer + GN01, on-site preflight and low-speed procedure.
* **M6**: Force Policy adapter and shadow comparison before any real rollout.
