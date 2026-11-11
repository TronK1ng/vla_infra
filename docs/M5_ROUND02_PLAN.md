# M5 Round 02 plan (first items)

This plan is a prerequisite for any interface to the real robot; no real
Enable/motion is authorized by Round 01R4.

1. **Authority permit over check-to-write**: Executor/Authority write permit
   must wrap `MotionCapability.check()` through the final SDK write, so a
   concurrent revoke cannot race between check and command. Add a revoke-during
   write fault test.
2. **Bounded shutdown**: run SDK `Stop()` in a controlled Supervisor boundary
   with a deadline, `STOP_FAILED` escalation and E-stop instructions; never
   block the only control thread indefinitely.
3. **Runtime composition**: M5Runtime state machine
   (CONNECTED_READONLY → PREFLIGHT_OK → WAITING_OPERATOR_ACK → ENABLED →
   MODE_READY → ARMED → RUNNING), one-shot ACK bound to station/config/report
   hashes, reconnect re-ACK, HOLD/FAULT.
4. **Tool/TCP/payload verification**: preflight binds tool name, TCP and
   payload hashes to config hash before `ToolReadiness.verify`; no motion from
   Switch alone.
5. **GN01 read provider** and actual RDK/GN01 integration tests on the station.
6. **Fault matrix**: Stop failure, Send/Move partial, GN jam, stale state,
   policy disconnect, process kill, cleanup thread alive.
7. **Field read-only**: 30 min soak, disk stress, time sync; no Enable/Move.

Round 01R4 continues to use injected fake SDK only.
