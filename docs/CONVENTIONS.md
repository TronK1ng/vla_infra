# M1 conventions (frozen v1)

Changing any of these is a schema change: bump the relevant `*_schema_version`
and record the migration in `exchange/worker/`.

## Clocks and time

| Value | Source | Use |
| --- | --- | --- |
| `host_time_ns` | process `time.monotonic_ns()` | scheduling, age checks, ring buffer ordering |
| `device_time_ns` | camera/controller timestamp | provenance only, never scheduling |
| `server_*_time_ns` | policy server monotonic clock | diagnostics only, never compared with client times |
| `observation_capture_time_ns` | client host monotonic | staleness check in the local safety gate |

* The observation target time is the newest required RGB frame host timestamp in
  the tick. Robot samples must bracket it (before ≤ target ≤ after). No future
  sample is ever used.
* A device timestamp is preserved exactly; file write time is never used as a
  capture time.
* Clock-domain mistakes are rejected: an observation capture time far in the
  future raises `STALE_OBSERVATION`.
* The policy client binds the response to the request: the returned
  `observation_capture_time_ns` must equal the request's
  `meta/capture_time_ns`. A mismatch closes the connection and raises
  `CAPTURE_MISMATCH`; the worker repeats the check before submitting to the
  inbox, so a stale/mislabeled chunk cannot reach execution.
* `request_sequence_id` is a **transport** request/response correlation id.
  `meta/sequence_id` identifies the observation (and is preserved in the
  recorder/metadata). They are not interchangeable and neither is derived from
  the other.
* `PolicyWorker.publish()` validates that the payload's
  `meta/capture_time_ns` and `meta/sequence_id` match the explicit arguments
  before the observation can enter the latest-only slot; mismatches raise
  `CAPTURE_MISMATCH`/`SEQUENCE_MISMATCH`/`SCHEMA_MISMATCH` immediately.

## Frames, rotations and poses

* `tcp_position`/`tcp_quaternion_xyzw` describe the TCP in the **base** frame
  (`R_base_tcp`).
* Quaternions are `xyzw`, unit length (norm tolerance 1e-3). `q` and `-q` are
  equivalent; helpers producing a rotation canonicalise `w ≥ 0`.
* `tcp_rotation_6d` is `[R00, R10, R20, R01, R11, R21]` — the first two
  **columns** of `R_base_tcp`. The contract rejects 6D values inconsistent with
  the quaternion (tolerance 1e-3).
* Action rotation is an axis-angle rotation vector in radians.
* `action_frame="base"`: translation in base axes and rotation left-multiplied.
  `action_frame="tcp"`: translation in tool axes and rotation right-multiplied.
  `action_frame` is mandatory.
* `action_type="delta_tcp"`: the first row is relative to the measured TCP pose;
  each following row is relative to the previous commanded target.
  `action_type="absolute_tcp"`: absolute position and absolute orientation as a
  rotation vector; only allowed with `action_frame="base"` (ambiguous otherwise).

## Actions

```text
[dx, dy, dz, drx, dry, drz, gripper]     # float32, shape (H, 7)
```

* translation: metres; rotation: radians; gripper: normalised `[0,1]`
  (0 = closed, 1 = open). The local gripper adapter owns the physical mapping.
* `H` and `action_dt_s` are declared by the adapter metadata and enforced on
  handshake and per chunk; M1 supports `H ∈ [1, 512]`,
  `action_dt_s ∈ [0.001, 1] s`.
* Force Policy/π0.5-specific data goes in `extras` (e.g. `reference_wrench`,
  `interaction_frame`, `selection_mask`, `policy_mode`) and is never required by
  a lower layer. Extras are **diagnostic/model metadata**: execution, safety and
  supervision must never depend on them. `ActionChunk` recursively freezes
  extras (nested mappings become read-only proxies, lists/tuples become tuples,
  numpy arrays are copied and read-only).
* There is no silent clipping. Values outside bounds are rejected; adapters are
  responsible for producing valid output.

## State layout (policy observation)

The state vector is accompanied by `meta/state_layout` and `meta/state_dim`:
name, ordered axes, units, frame and normalisation. Two layouts ship in M1:

* `canonical_v1` (22): TCP xyz, rotation-6D, TCP velocity, wrench, gripper width;
* `joint_v1` (8): seven joint positions, gripper width.

M1 only permits `normalization: none` (physical units on the wire). A server
rejects a request whose declared layout does not match its adapter metadata.
No code may infer axis meaning from vector length.

## Images

* RGB frames are `uint8[H,W,3]`, encoding `rgb8`; depth is `uint16[H,W]`,
  encoding `depth16`; keys and encodings must match (`base_rgb`, `wrist_rgb`,
  optional `base_depth`, `wrist_depth`).
* The policy client letterboxes RGB to a square `image_size` (centred, zero
  padding) and declares `meta/image_encoding="rgb8"` with the source size.
* Silent BGR is forbidden: a wrong encoding is rejected before the wire and on
  the server.

## Synchronisation

* Continuous state (joints, TCP position/velocity) is linearly interpolated.
* Orientation uses shortest-path SLERP and the 6D representation is recomputed
  from the interpolated quaternion.
* Wrench and gripper width use zero-order hold by default (contact signals are
  not smeared). Both can be configured as `linear`, explicitly.
* Rejection rules: no bracketing sample (`robot_stale`), state gap too large
  (`state_gap`), camera skew too large (`camera_skew`), target offset too large
  (`sync_error`), any image newer than the target (`future_image`). Rejected
  observations are counted and never silently built from stale data.
* `sync.max_age_ns` is the largest offset between the target and the two
  bracketing samples; it must equal that value (no under-reporting).

## Control lifecycle

```text
OFFLINE → CONNECTED → DEVICES_READY → HOLD → ARMED → RUNNING → HOLD
                                                   ↓
                                                 FAULT → (operator reset) OFFLINE
```

* Only `RUNNING` allows device writes in M1. `ARMED` is a grace state: the
  authority exists but `write_permit` refuses writes until `start_running`.
* HOLD/FAULT revoke the epoch, clear the inbox and block new writes. Revocation
  waits for an in-flight write to finish; no write begins afterwards.
* Reconnects, policy switches, watchdog expiry and operator stops enter HOLD.
  FAULT is latched and requires an explicit operator reset, which returns to
  OFFLINE and requires a new preflight.
* Policy chunks are bound to the epoch that was current **before** inference.
  A chunk inferred across a HOLD can never execute after a later ARM.
* Every transition is recorded with monotonic time, from/to state, reason and
  epoch; history is bounded.
* Executor internal exceptions and pose/gripper write failures are fail-closed
  **inside the executor**: the active plan is cleared and the authority is
  revoked before the event is emitted. The runtime must additionally map
  `internal_error`/`write_failed`/`write_partial` to supervisor FAULT; the local
  revocation is the minimum barrier if that callback fails.

## Recording (M2)

* Raw layout: `episode_xxxx/{cameras/<name>/color|depth/*.png,
  cameras/<name>/<kind>/frames.jsonl, lowdim/raw.bin, lowdim/schema.json,
  observations.jsonl, actions/raw.jsonl, calibration/calibration.json,
  sync/sync_report.json, metadata.json}`.
* Lowdim is an append-only numpy structured array.  Fields, dtype, shape, units
  and frames are written in `lowdim/schema.json`; no consumer may infer them from
  length.  Force Policy HDF5 conversion is a separate mapping step, documented in
  the schema note and not attempted in M2.
* Episode directory is built under `.episode_xxxx.tmp-*` and renamed only after
  metadata/schema are complete.  `.incomplete` exists while uncommitted; crash
  remnants are identifiable and `load_episode` refuses them.  Commit flushes and
  `fsync`s JSON/lowdim/manifest files, `fsync`s image PNGs when written, fsyncs
  temporary directories before rename and the parent after rename; a power loss
  in the post-rename window remains filesystem-defined.
* Recorder lifecycle is explicit: `stop()` closes producer acceptance under the
  same lock used for queue insertion; the writer only commits after
  accepting=false, inflight=0 and all buffers are empty.  A timeout before the
  commit authorization aborts; a timeout during the rename section reports
  `still_stopping` rather than a false final failure. CollectionHarness polls
  until finished/aborted with `recording.finalize_timeout_s` (default 300 s);
  the CLI waits for this deadline and returns 3 rather than exiting during a
  live commit.
* Capture/control threads only append to bounded buffers.  PNG encoding and disk
  I/O happen on the recorder thread; queue/buffer overflow is dropped and
  counted, never blocked.  Disk free is checked before start and periodically;
  low disk aborts the uncommitted episode and leaves the marker.
* Every `observations.jsonl` record persists `canonical_state` (canonical 22-D
  values, layout, capture time, typed source and sync interpolation provenance).
  LeRobot export uses that state directly; it never re-selects raw lowdim rows.
  The validator reconstructs the typed `RobotState`, re-encodes it with the
  canonical layout and requires exact float32 equality with `values`.
  If an episode has no action labels, the LeRobot draft omits `action`; partial
  action coverage or an action without `observation_sequence_id` is refused.
  Action identity is episode-local: `(episode_index, observation_sequence_id)`,
  with the observation capture timestamp checked when the action declares one.
* Metadata records observation/action/state contracts, `delta_semantics`,
  `gripper_semantics`, camera/robot serials, calibration version and hashes
  (null when not yet calibrated), RDK/TDK/firmware fields, rates, sync
  distribution, control source, end reason and verdict.
* LeRobot export is a JSONL draft with the canonical feature names; it is marked
  `format: jsonl_draft` and does not claim Parquet/video compatibility.

## Real read-only devices

`backend: fake` is the default. `backend: real_readonly` is accepted only with
`on_site_readonly_ack: true`, real serials, and `VLA_INFRA_ALLOW_REAL_HARDWARE=1`;
`backend: real` (writer) remains refused.  The real adapters import their SDK only
on `start()` and expose no write methods: RealSense uses color rgb8 + aligned
depth z16; RDK uses only `Robot(...)` and `states()` (never enable/mode/set
target). Real collection additionally needs a GN01 width provider or an RDK
gripper-width field; otherwise it fails closed.

## Network and proxy policy

* Policy endpoint proxy policy is explicit configuration (`policy.proxy`):
  `none` (direct), `env` (inherit and record the environment proxy), or a proxy
  URL. It is never implicitly inherited for a remote endpoint.
* Loopback endpoints (`localhost`, `127.0.0.0/8`, `::1`) always bypass proxies.
  The client passes `proxy=None` to `websockets` even if the operator's shell
  has `all_proxy`/`http_proxy` set; this keeps local dummy servers and tests
  reproducible.
* Connection/proxy/handshake exceptions are normalized to
  `PolicyDisconnectedError`; a local circular/contract error closes the socket
  and becomes `PolicyContractError`. The token is sent in an `Authorization`
  header, never in the URL.

## Threading, queues and lock order

* `RobotStateWorker` → ring buffer → single observation thread → latest-only
  policy slot → background policy worker → capacity-1 action inbox.
* The executor is the only writer to the command sink and claims it with
  `SingleWriterGuard`.
* Pose and gripper are separate device calls. Failure events carry
  `pose_ok`/`gripper_ok`/`gripper_sent`, so a pose-success/gripper-failure tick is
  auditable as `write_partial`; the executor then revokes authority and stops.
* Only a `NotArmedError` raised while **acquiring** the write permit is a benign
  HOLD/ARM race. Once the permit is held, any exception from a device method —
  including a device-level `NotArmedError` — is a write failure/partial write and
  triggers local revocation and hard fault.
* The gripper command is suppressed only when the normalised target is unchanged
  within `GRIPPER_DEDUP_TOLERANCE = 1e-4` **and within the same authority
  epoch**. A new epoch always sends its first gripper command; fail-closed and
  HOLD boundaries invalidate the cache.
* Lock order is authority → inbox. The authority lock is held across one device
  write so revocation is synchronous; real backends must keep writes short.
* Recording/visualisation must use bounded queues and must not block the control
  loop (M2+).
