# M1 acceptance matrix

Maps the master plan's M1 deliverables/acceptance criteria and the §14 review
checklist to implementation and tests. Test names refer to `tests/`.

## M1 deliverables (§11, §12)

| Requirement | Implementation | Evidence |
| --- | --- | --- |
| Package skeleton | `pyproject.toml`, `src/vla_infra`, CLI entry points | `test_config.py`, doctor CLI |
| Observation/action dataclasses + validation | `contracts/` | `test_contracts_observation.py`, `test_contracts_action.py`, `test_geometry.py`, `test_policy_io.py` |
| Fake camera/robot/gripper | `devices/fake.py` | `test_fake_devices.py` |
| Ring buffer + synchroniser | `sync/` | `test_ring_buffer.py`, `test_synchronizer.py` |
| Dummy policy server/client | `policy/` | `test_policy_server.py`, `test_wire.py`, `test_policy_worker.py` |
| Supervisor state machine | `execution/supervisor.py` | `test_supervisor.py`, `test_inbox.py`, `test_authority.py` |
| Unit tests | `tests/unit` (200+ cases) | `scripts/run_tests.sh` |
| No hardware/network external deps | fake devices, loopback server, stubs | `test_no_real_hardware.py` |

## M1 acceptance (§11)

| Criterion | How it is enforced | Evidence |
| --- | --- | --- |
| Runs without hardware or external network | fake devices + loopback ephemeral port | entire suite; `test_shadow_harness_executes_on_fake_backend` |
| Shape/dtype/frame/NaN/sequence errors tested | constructors + validators + wire checks | observation/action/wire/server/worker tests |
| Disconnect/timeout enters HOLD | client → worker → supervisor | `test_timeout_forces_hold_and_reconnect_does_not_resume`, `test_disconnect_holds_and_reconnect_does_not_resume` |
| Tests never connect real hardware | stubs raise; config refuses `backend: real`; no SDK imports | `test_no_real_hardware.py`, `test_config.py` |

## §14 review checklist

### Architecture

| Check | Status |
| --- | --- |
| Model/network/hardware/safety decoupled | Server has no RDK/camera import; executor owns all writes; policy worker has no device reference |
| No network callback writes RDK | Writes only via `Executor.tick` + `write_permit`; enforced by `SingleWriterGuard` |
| Single writer | Thread-affine fake sink test + executor claims writer |
| Bounded queues/backpressure | Latest-only policy slot (capacity 1), action inbox capacity 1, bounded history/ring buffer |
| Fake/real same interface | `CameraSource`, `RobotStateSource`, `RobotCommandSink` protocols; real stubs fail closed |
| Loopback never uses system proxy | explicit `proxy=None` + proxy policy tests |
| Response bound to observation | client capture equality + worker inbox guard |
| Executor fails closed locally | tick wrapper + partial-write audit + local authority revoke |

### Contracts

| Check | Status |
| --- | --- |
| Axis order explicit | `constants.py`, `StateLayout`, metadata on the wire; no length-based inference |
| Units/frame/dtype/shape validated | Frozen dataclasses + validators + wire checks; tests for each |
| RGB/BGR, rotation-6D, quaternion | RGB8 declared/validated; 6D column convention tested; q/-q tested |
| Sequence/timestamp end-to-end | request sequence, server monotonic sequence, observation sequence, capture/receive times preserved |
| Model-specific fields optional | `extras` mapping; Force Policy keys optional and tested; extras deeply frozen and diagnostic-only |

### Concurrency/timing

| Check | Status |
| --- | --- |
| Monotonic clock | `clock.py`; device timestamps never used for scheduling |
| Thread-safe ring buffer | lock + concurrency test |
| No races/unbounded growth/blocking I/O | bounded structures, lock order documented; policy/network isolated in worker |
| Reconnect cannot execute old chunk | epoch snapshot + inbox clear + tests |
| No future/stale sample in sync | bracket-only interpolation + rejection tests |

### Safety

| Check | Status |
| --- | --- |
| Fail-safe default | reject/no-write; FAULT latched; real backend refused |
| HOLD/FAULT block commands | authority revocation barrier + tests |
| NaN/Inf, bounds, over-force, watchdog | covered by validators/safety/supervisor tests |
| Unknown executor exception blocks writes | local revoke + FAULT mapping tests |
| Missing config refuses start | strict YAML loader tests |
| Tests cannot connect real hardware | stubs, config guard, import check |

### Testing/maintainability

| Check | Status |
| --- | --- |
| Deterministic clocks/fixtures | `FakeClock`, `vla_infra.testing` builders |
| Error paths tested | extensive `pytest.raises` coverage |
| Clear types/logs/error codes | `ErrorCode` enum; structured transition/watchdog logs |
| Third-party licenses | worker report §10; no Force Policy code copied |
| No unrelated heavy deps/code copies | deps: numpy, PyYAML, websockets (+optional msgpack) |

## Known gaps (not M1 blockers)

Acceleration limits, explicit trajectory retiming (M1 deliberately rejects at
nominal timing instead), F/T zeroing, RDK joint limits, 1 kHz RT loop,
recording/LeRobot, real adapters, official OpenPI client, TLS/VPN deployment,
full latency/jitter acceptance report (M3).
