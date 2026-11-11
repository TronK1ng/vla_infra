# M2 on-site acceptance checklist

M2 automated code acceptance is complete for fake devices / injected SDK doubles.
The following steps are **not executed** by automated tests and remain required
before M2 can be called field-complete. No step here authorizes robot motion.

## 0. Hard gates

- [ ] GN01 read-only width source is available (vendor SDK/topic/service or an
      RDK field). Until then real collection fails closed and the CLI refuses
      `backend: real_readonly`.
- [ ] Real camera and robot serials are recorded in station config (not
      `REPLACE_ME`).
- [ ] Calibration file is tracked; `calibration_version` and intrinsics/extrinsics
      hashes in episode metadata point to that file. `uncalibrated`/null data is
      not training data.
- [ ] Station is authorized for read-only operations only; no enable/mode/motion
      command, no policy rollout.

## 1. Preflight (no motion, read-only)

```bash
PYTHONPATH=src python3 -m vla_infra.cli.doctor --preflight --json
```

Check:

- recording root free space >= `recording.min_free_gb`;
- 相机配置为 `driver: realsense` 时 `pyrealsense2` 可导入，配置为
  `driver: orbbec` 时 `pyorbbecsdk` 可导入；`flexivrdk` 可导入；
- Gemini 336L 已安装 Orbbec udev 规则并连接 USB 3.x；USB 2.x 必须拒绝；
- serials and calibration version/hash present;
- `recording.finalize_timeout_s` is long enough for the station disk (default
  300 s).

## 2. Fake dry run (mandatory before touching hardware)

```bash
PYTHONPATH=src python3 -m vla_infra.cli.record \
  --duration 5 --task dry-run --output /tmp/m2_dry --min-free-gb 0 --json
PYTHONPATH=src python3 -m vla_infra.cli.replay /tmp/m2_dry/episode_0001 --json
PYTHONPATH=src python3 -m vla_infra.cli.replay /tmp/m2_dry/episode_0001 \
  --lerobot-out /tmp/m2_dry_lerobot
```

Acceptance: `issues=[]`, no drops, no zero action labels; observation-only info
has `training_ready=false`.

## 3. Real read-only 30-minute soak

**Status: NON-EXECUTABLE TEMPLATE.**

The current `vla-record` CLI deliberately refuses every
`backend: real_readonly` because no GN01 read-only width provider is wired into
the CLI/config yet. The command below is therefore **not an executable
acceptance step**. Do not copy it onto the station until the GN01 SDK/topic/
service or RDK width field is confirmed and a provider adapter is implemented
and reviewed.

```bash
# TEMPLATE ONLY — current CLI returns CONFIG ERROR and does not run.
# Requires: GN01 provider implementation, on_site_readonly_ack, real serials,
# VLA_INFRA_ALLOW_REAL_HARDWARE=1, and separate on-site authorization.
# PYTHONPATH=src python3 -m vla_infra.cli.record \
#   --duration 1800 --task m2-onsite --finalize-timeout 300 --json
```

Monitor during the run:

- both cameras produce frames; no USB dropout;
- `dropped_observations=0`, `dropped_lowdim=0`, `dropped_actions=0`;
- sync p95/max and camera skew are within station limits;
- temp directory remains until commit; stop waits for finalize.

After the run:

```bash
PYTHONPATH=src python3 -m vla_infra.cli.replay <episode> --json
```

Acceptance: `issues=[]`, `complete=true`, `durability=fsync_best_effort`, no
stale `.incomplete`; metadata counts match files; replay/validator no tamper.

## 4. Failure injection (read-only, on-site)

- [ ] stop with a short recorder timeout, verify `still_stopping` is consumed by
      CollectionHarness/CLI and no half-committed formal episode is reported;
- [ ] camera/USB disconnect, verify no formal `episode_xxxx`, temp `.incomplete`
      remains with failure reason;
- [ ] disk pressure, verify collection does not block and recorder aborts cleanly;
- [ ] process kill during recording, verify only temp `.incomplete` remains;
- [ ] validator/export rejects corrupted counts/order/state.

## 5. Sign-off

Record operator, station, serials, calibration version/hash, SDK versions, disk
free bytes, run duration, drops, sync percentiles, and validator/export results.
M2 field acceptance is not implied by the automated suite.
