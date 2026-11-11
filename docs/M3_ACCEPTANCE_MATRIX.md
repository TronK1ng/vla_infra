# M3 acceptance matrix

| Master-plan M3 requirement | Implementation | Automated evidence |
| --- | --- | --- |
| action inbox/queue | existing `execution/inbox.py` latest-only | `test_m3_latest_only_inbox_replaces_chunks`, inbox tests |
| full safety profile | `SafetyProfile` acceleration/joint velocity | `test_chunk_acceleration_limit`, vector direction/axis tests, `test_tick_*vector*`, `test_first_write_runs_full_per_tick_gate`, `test_measured_joint_velocity_limit` |
| interpolation | `execution/interpolation.py` | existing tests; **reject-only baseline**, no automatic retiming claimed |
| fake executor 100 Hz | `runtime/m3.py` dedicated thread | `test_m3_100hz_metrics_and_audit_log` |
| watchdog integration | Supervisor robot/camera/policy/executor | `test_m3_policy_watchdog_hold`, `test_m3_camera_continuous_none_hold`, `test_m3_camera_exception_hold`, `test_m3_duplicate_old_camera_frame_is_not_heartbeat`, `test_m3_stale_state_hold` |
| fault matrix | `M3Runtime` injection hooks/tests | M3 tests + existing safety/executor tests |
| raw/filtered/commanded/measured log | `ExecutionLog` / M3 runtime | `test_m3_four_stage_action_audit_correlates_and_differs`; raw action/filtered/commanded are distinct |
| latency/jitter report | `M3Result.metrics` | `test_m3_100hz_metrics_and_audit_log`, `vla-m3 --json` |
| no real hardware | fake-only guard | M3 runtime rejects non-fake; no SDK import |
| docs/commands | this file + M3_DESIGN + CLI | `vla-m3 --duration ...` |
| real 100 Hz RDK | out of scope | **not run** |
| real policy rollout | out of scope | **not run** |

## Measured fake latency (2 s run)

`vla-m3 --duration 2.0 --json` on this Linux container:

```text
ticks=203, writes=198, overruns=0
tick period: p50=9.999769 ms, p95=10.222427 ms, p99=10.654124 ms, max=10.986224 ms
tick runtime: p50=0.405854 ms, p95=0.865419 ms, p99=2.595766 ms, max=2.988910 ms
jitter: p50=0.006726 ms, p95=0.515808 ms, p99=0.732948 ms, max=1.205432 ms
log_records=203, log_dropped=0
```

Exact values are in the M3 evidence `M3_LATENCY_REPORT.json`; this is a
fake/simulation measurement, not an RDK real-time claim.

## Reproduce

```bash
PYTHONPATH=src python3 -m vla_infra.cli.m3 --duration 2 --output /tmp/m3.jsonl --json
# fault examples (fake only)
PYTHONPATH=src python3 -m vla_infra.cli.m3 --duration 1.5 --fault policy_stall --output /tmp/m3_policy.jsonl --json
PYTHONPATH=src python3 -m vla_infra.cli.m3 --duration 1.5 --fault writer_error --output /tmp/m3_writer.jsonl --json
```

## Known gaps

- no RDK/RealSense/real-time OS measurements;
- no real 1 kHz RT path (M3 uses fake NRT loop);
- no real GN01/force/contact validation;
- no real policy/network latency;
- velocity-based automatic retiming is **not implemented**; the reject-only
  baseline is approved for M3 fake, and automatic retiming requires a separate
  future milestone;
- M3 results are fake/simulation only.  M4 preparation is approved, but every
  M4 action remains fake-only and this does not authorize real rollout.
