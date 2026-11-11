# M4 design: OpenPI-shaped remote dummy closed loop (fake only)

Scope: **fake/simulation only**. M4 connects the local fake runtime to an
OpenPI-shaped dummy policy endpoint over WebSocket; it does not import/use RDK,
enable a robot, send real motion, or run a real VLA policy.

## Dependency

```bash
pip install -e '.[m4]'   # openpi-client==0.1.2
```

The package is optional; the rest of the repository does not require it. If it
is absent, official-adapter tests skip and the adapter raises
`OpenPIUnavailableError` explicitly.

## Components

- `policy/openpi.py` (custom protocol shim, **not** official OpenPI):
  - `CustomPolicyClient` / `CustomDummyPolicyServer`: project's validated
    JSON/msgpack WebSocket protocol.
  - `OpenPIClient`/`OpenPIServer` remain aliases only for compatibility and are
    not official-OpenPI evidence.
- `policy/openpi_official.py` (official adapter, pinned):
  - constructs the real `openpi_client.websocket_client_policy.
    WebsocketClientPolicy` from `openpi-client==0.1.2`;
  - strict observation conversion: all of `state`, `image`, `wrist_image`,
    `prompt`/`task`, capture time and sequence are required; state is checked
    against the declared layout (float32, shape, finite), both images against
    HWC/RGB/uint8/non-empty/metadata resolution, and capture/sequence ranges;
    there is **no** synthetic state/image/prompt/capture fallback;
  - explicit action contract: actual float32 is required (int/float64 rejected),
    then `(H,7)`, finite checks; no silent dtype coercion;
  - bounded connect via an explicitly managed websocket connection with
    `open_timeout`/`proxy` and a preconnected official client subclass; no
    process-global function monkeypatch; infer timeout closes the connection
    and joins the operation thread; a thread still alive raises
    `PolicyCleanupError` (non-retryable lifecycle fault);
  - loopback bypass is explicit `proxy=None`; no runtime mutation of
    `os.environ`;
  - request sequence must equal the observation sequence; duplicate/regression
    and type errors are rejected before any network request;
  - actions must strictly match `(metadata.horizon, metadata.action_dim)`;
  - metadata safety fields (`horizon`, `action_dt_s`, `action_frame`,
    `action_type`, `delta_semantics`, `state_layout`, `image_size`,
    `action_dim`) must be declared by the server or supplied by an explicit
    deployment profile; no silent defaults;
  - cleanup failure after an infer timeout is a dedicated
    `PolicyCleanupError`; PolicyWorker faults and stops reconnecting;
  - a temporary official connection is owned only after metadata parsing and
    compatibility succeed; any metadata parse/compatibility failure closes it
    before propagating;
  - all remote response contract failures (missing actions, dtype, shape,
    horizon, finite, extras) increment `protocol_errors`;
  - explicit disconnect/reconnect; metadata handshake and fail-closed errors;
  - `OfficialOpenPIServer` is a minimal official-wire-compatible dummy server
    using `openpi_client.msgpack_numpy` for loopback tests.  `stop()` joins the
    server and active handler threads; on timeout it raises
    `PolicyTimeoutError` and keeps the handles/state instead of reporting
    stopped.
- `policy/wire.py`: JSON+base64 or msgpack codec, ndarray dtype/shape preserving.
- `policy/dummy_server.py`: OpenPI-shaped request/result/error contracts,
  metadata handshake, sequence validation, bounded message size.
- `policy/client.py`: endpoint/proxy handling, connect timeout, RTT, codec,
  metadata compatibility, capture/sequence binding.
- `policy/worker.py`: latest-only observation slot, async one-in-flight
  inference, epoch snapshot, timeout/disconnect handling, reconnect does not
  auto-resume.
- `runtime/shadow.py`: full fake camera/robot -> sync -> WebSocket -> inbox ->
  safety/interpolation -> fake executor loop.  `transport_kind="custom"` or
  `"official"` selects the transport via an injectable factory; no client class
  is hard-coded in the loop.  Disconnect holds; reconnect does not resume
  without a new arm.

## M4 boundaries

- Zero-action closed loop is the first acceptance target; fixed actions may only
  reach the fake backend.
- No real policy rollout and no real robot writer.
- VPN/TLS is not implemented in this fake M4; non-loopback servers are gated by
  explicit opt-in + auth token and remain outside production scope.
- Callback/telemetry paths are latest-only notifications and never safety,
  command or complete-audit critical.
- M4 lifecycle uses the M3 fail-closed `START_FAILED`/`START_CLEANUP_FAILED`/
  `STOP_FAILED` rules.

## Relevant tests

- `tests/unit/test_openpi_wrapper.py`: custom-protocol zero action/RTT and
  duplicate/regression with structured errors.
- `tests/unit/test_openpi_official.py`: official package identity/version,
  zero action, strict missing/wrong-shape/wrong-dtype/NaN input matrix (server
  request count unchanged), action dtype/shape/finite matrix, connect/infer
  timeout with no residual threads, repeatable connect failures, disconnect/
  reconnect, duplicate/regression, blocked-server stop failure, proxy-env
  invariance.
- `tests/integration/test_m4_openpi_loop.py`: zero-action fake closed loop for
  both custom and official transport; official path asserts package/version/
  client/server classes in `result.transport`.
- `tests/integration/test_shadow_loop.py`: disconnect HOLD, reconnect no auto
  RUNNING.
- `tests/unit/test_policy_server.py` / `test_policy_worker.py`: timeout,
  disconnect, sequence, contract, epoch.
