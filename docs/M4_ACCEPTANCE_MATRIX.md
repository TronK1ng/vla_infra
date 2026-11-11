# M4 acceptance matrix

| M4 requirement | Implementation | Evidence |
| --- | --- | --- |
| custom protocol shim (not official) | `policy/openpi.py::CustomPolicyClient` | `test_openpi_wrapper.py`; explicitly not official evidence |
| official OpenPI client adapter | `policy/openpi_official.py::OfficialOpenPIClient` | `test_openpi_official.py`; actual `openpi-client==0.1.2`; identity in report |
| official-wire dummy server | `policy/openpi_official.py::OfficialOpenPIServer` | `test_openpi_official.py` |
| official transport injection | `ShadowHarness(transport_kind="official")` | `test_m4_official_openpi_zero_action_fake_closed_loop` |
| strict observation contract | no synthetic fallback | `test_official_observation_missing_or_invalid_never_reaches_server` |
| action dtype contract | actual float32 only | `test_official_action_dtype_shape_and_finite_are_enforced` |
| bounded connect/infer | no residual threads | `test_official_connect_timeout_leaves_no_threads_and_is_repeatable`, `test_official_infer_timeout_leaves_no_threads` |
| server stop failure observable | raises, retains live state | `test_official_server_stop_does_not_false_report_stopped` |
| proxy env invariance | explicit proxy only | `test_official_operation_does_not_mutate_proxy_environment` |
| sequence binding | request == observation sequence, type/range | `test_official_request_and_observation_sequence_are_bound` |
| horizon binding | `(metadata.horizon, action_dim)` exact | `test_official_action_dtype_shape_and_finite_are_enforced` short/long |
| cleanup failure non-retry | `PolicyCleanupError` -> FAULT | `test_official_infer_cleanup_failure_is_non_retryable`, `test_cleanup_failure_faults_and_never_retries` |
| metadata safety fields | required or explicit profile | `test_official_metadata_missing_safety_fields_fail_closed` |
| metadata failure no leak | temporary connection closed | `test_official_metadata_failure_closes_temporary_connection_repeatedly` |
| response contract metrics | all failures increment protocol_errors | action/missing-actions protocol tests |
| worker contract classification | ValidationError -> FAULT, not internal | `test_action_contract_validation_is_not_internal_error` |
| official exception matrix | connect/infer timeout, disconnect, reconnect, duplicate/regression | `test_openpi_official.py` specifics |
| zero action first closure | `ZeroActionAdapter` + `ShadowHarness` | `test_m4_zero_action_fake_closed_loop` |
| official zero-action closure | `OfficialOpenPIClient/Server` + shadow | `test_m4_official_openpi_zero_action_fake_closed_loop` |
| fixed action fake-only | existing `FixedDeltaAdapter` + shadow tests | `test_shadow_harness_executes_on_fake_backend` |
| RTT | `ClientStats.rtt_*` | wrapper + shadow tests |
| timeout | client/worker timeout paths | `test_policy_worker.py` timeout tests |
| disconnect -> HOLD | shadow integration | `test_disconnect_holds_and_reconnect_does_not_resume` |
| reconnect no auto RUNNING | shadow integration | same test: explicit new arm required |
| sequence regression/duplicate | server validation | `test_duplicate_sequence_rejected`, wrapper sequence test |
| no real hardware | fake-only shadow/wrapper | no SDK import; tests only fake |
| transport identity report | `ShadowHarness.transport` | `test_m4_official...` asserts package/version/classes |
| VPN/TLS | **not implemented** | production boundary outside fake M4 |
| real policy rollout | **not implemented / forbidden** | — |

## Strictly fake

M4 automation is loopback/fake only. It does not authorize real robot motion,
real RDK writer use, or real policy rollout.
