# M5 Round 01 acceptance matrix

| Requirement | Implementation | Evidence |
| --- | --- | --- |
| injected RDK/GN01 interfaces | `FlexivRdkNrtCommandSink`, `Gn01CommandSink` | `test_m5_writer.py` |
| preflight fail closed | strict probe/result/ACK + deep-immutable validate | `test_m5_config_preflight.py`, `test_m5_auth.py` |
| GN01 v1.9.3 API | shared Robot, no self-grant, auth/epoch, params/Disable rollback | `test_gn01_*`, `test_gn01_cannot_self_grant_without_arm_lifecycle` |
| Tool readiness | switch != verified; failure revokes | `test_tool_switch_failure_revokes_and_requires_verification` |
| session API surface | connect() returns None; no raw Robot; read-only facade | `test_session_does_not_expose_raw_robot` |
| Enable lifecycle | operational wait, mode verify, Stop on failure | `test_enable_*` |
| authorization required | HMAC verifier + expected station/config | `test_enable_requires_verifier_authorization`, `test_issuer_*` |
| shared session/factory | one Robot handle | `test_shared_session_factory_uses_one_robot_handle` |
| command epoch/heartbeat | epoch + inter-command gap | `test_command_epoch_and_inter_command_gap_are_bound` |
| connect != Enable | `connect()` only constructs SDK | `test_connect_does_not_enable_or_switch_mode` |
| explicit NRT limits | `NrtLimits` + per-command positional limits | `test_explicit_enable_and_nrt_pose_conversion` |
| frame/quaternion conversion | explicit base->world, xyzw->wxyz | `test_base_to_world_transform_and_quaternion_order` |
| single writer | `SingleWriterGuard` | `test_writer_requires_enable_and_single_writer` |
| SDK/Stop failure | typed `DeviceError`, status | `test_writer_sdk_failure_and_stop_failure_are_observable` |
| no default real SDK | env gate/lazy import | `test_writer_without_sdk_requires_real_gate` |
| GN01 normalized mapping/dedup | `Gn01Config`/sink | `test_gn01_normalized_width_dedup_range_and_stop` |
| GN01 params rollback | Disable on Enable failure | `test_gn01_tool_switch_is_explicit_and_params_subset_enforced` |
| shared revocation | arm revoke blocks gripper | `test_arm_revocation_blocks_shared_gripper` |
| Stop failure revokes | software authority revoked before/after Stop | `test_arm_stop_failure_still_revokes_capability`, `test_gn01_stop_failure_still_revokes_and_move_rejected` |
| Tool.Switch identity | capability auth+epoch required | `test_tool_switch_rejects_old_authorization_after_regrant` |
| session close | stop/revoke or STOP_FAILED | `test_session_close_stops_and_revokes_or_fails_loudly` |
| capability concurrency | lock-protected | `test_motion_capability_concurrent_grant_revoke_is_safe` |
| runtime composition authority | production identity validation | `test_runtime_production_rejects_mismatched_authority` |
| motion gates | allow_real_motion/env/config/ACK | `test_runtime_enable_requires_motion_gate_before_sdk` |
| fault/invalidate Stop | active states physically Stop | `test_fault_and_running_invalidate_stop_both_devices` |
| placeholder tool profile | reject REPLACE_ME/short hashes | `test_m5_templates_load_strictly...` |
| lifecycle serialization | operation mutex/generation | `test_enable_and_stop_are_serialized_by_lifecycle_mutex` |
| sticky STOP_FAILED | recover only explicit | `test_stop_failed_is_sticky_until_explicit_recovery` |
| watchdog active Stop | auth expiry/heartbeat stale | `test_watchdog_staleness_and_authorization_expiry_stop_devices` |
| single runtime ownership | session claim lock/rollback | `test_session_allows_only_one_runtime_owner`, `test_concurrent_session_claim_has_exactly_one_winner` |
| compose tool identity | shared expected ToolReadiness | `test_compose_uses_expected_shared_tool_readiness` |
| active watchdog | independent thread, no external poll | `test_watchdog_stops_without_external_poll_call` |
| zero-heartbeat gate | Enable rejects missing/stale | `test_enable_rejects_zero_or_missing_heartbeat` |
| owner token | identity-bound release | `test_session_allows_only_one_runtime_owner` |
| compose rollback | abort/quarantine before release | `test_abort_stop_failure_quarantines_session` |
| watchdog terminal | exception -> Stop | `test_watchdog_terminal_exception_revokes_and_stops` |
| bounded session abort/close | arm block still stops GN01 | `test_abort_pair_does_not_skip_gripper_when_arm_blocks` |
| tool identity behavior | wrong name/TCP/payload blocks command | `test_tool_identity_mismatch_blocks_command` |
| quarantine protocol | process restart/new session + reason | docs + `test_abort_stop_failure_quarantines_session` |
| close CLOSED semantics | failure not silent | `test_close_failure_is_not_silent_and_owner_remains` |
| terminal no state overwrite | re-read under op lock | `test_terminal_failure_does_not_overwrite_stopped` |
| partial construction | pending cleanup/quarantine/owner retained | `test_partial_construction_blocking_stop_quarantines_and_keeps_pending` |
| quarantine hard gate | old runtime cannot re-enable | `test_quarantined_session_blocks_recovery_and_enable` |
| writer factory owner | exact token, quarantine gate | `test_session_quarantine_gate_blocks_existing_sink` |
| shared sink gate | quarantine blocks Send | `test_session_quarantine_gate_blocks_existing_sink` |
| QUARANTINED state | sticky state commit | `test_quarantined_session_blocks_recovery_and_enable` |
| heartbeat validation | source/future/regression | heartbeat tests |
| transactional Enable | failure stops both, FAULT/STOP_FAILED | `test_enable_failure_is_transactional_and_stops_both_devices` |
| parallel Stop | GN01 not skipped when arm blocks | `test_parallel_stop_does_not_skip_gripper_when_arm_blocks` |
| connect failure state | FAULT not false connected | `test_connect_failure_marks_fault` |
| stop helper default | missing result is failure | `test_stop_helper_missing_result_is_failure` |
| forged report rejected | deep-immutable/validate | `test_issuer_*` |
| composite partial write | `RealRobotCommandSink` | `test_composite_partial_write_and_stop_are_observable` |
| strict config + hash | `config_m5.py` templates | `test_m5_templates_load_strictly...` |
| preflight JSON | `M5PreflightReport` | `test_preflight_*` |
| no real hardware | injected-only tests | full suite imports no `flexivrdk` |

## Not yet accepted

- M5Runtime/state machine/operator ACK (Round 02);
- real GN01 read/write provider and serial/tool/TCP/calibration;
- field read-only soak, disk stress, E-stop/Stop observation;
- any real motion, remote policy rollout, M5 Field 02+.

This Round 01 artifact is a code skeleton only and does not authorize motion.
