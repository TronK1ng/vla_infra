"""``vla-shadow``: hardware-free end-to-end shadow / fake-execution harness.

Default mode is shadow: policy chunks are produced, logged and discarded; the
supervisor stays in HOLD and no device write occurs.  ``--execute-fake`` is the
operator acknowledgement that enables motion **on the fake backend only**.
It cannot enable real hardware: ``backend: real`` is refused during config
loading.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from ..config import load_app_config
from ..errors import ConfigError, VlaInfraError
from ..policy.client import sanitize_endpoint
from ..runtime.shadow import ShadowHarness

DEFAULT_STATION = "configs/station.example.yaml"
DEFAULT_SAFETY = "configs/safety.example.yaml"
DEFAULT_POLICY = "configs/policy.example.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the M1 fake-device shadow loop (no real hardware)."
    )
    parser.add_argument("--station", default=DEFAULT_STATION)
    parser.add_argument("--safety", default=DEFAULT_SAFETY)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--duration", type=float, default=2.0, help="seconds to run")
    parser.add_argument(
        "--execute-fake",
        action="store_true",
        help="acknowledge and enable execution on the FAKE backend (no real robot)",
    )
    parser.add_argument("--task", default="shadow run: move slowly")
    parser.add_argument("--json", action="store_true", help="emit a JSON summary")
    parser.add_argument(
        "--external-server",
        action="store_true",
        help="do not start an embedded dummy server; connect to policy.endpoint",
    )
    return parser


def _resolve_token(policy) -> Optional[str]:
    if not policy.auth_token_env:
        return None
    token = os.environ.get(policy.auth_token_env)
    if not token:
        raise ConfigError(
            f"policy.auth_token_env is set but {policy.auth_token_env} is not in the environment",
            details={"env": policy.auth_token_env},
        )
    return token


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        app = load_app_config(
            station_path=Path(args.station),
            safety_path=Path(args.safety),
            policy_path=Path(args.policy),
        )
        token = _resolve_token(app.policy)
    except VlaInfraError as exc:
        print(f"CONFIG ERROR {exc.code.value}: {exc.message}")
        return 1

    if not args.json:
        mode = "SHADOW (no motion)" if not args.execute_fake else "FAKE-BACKEND EXECUTION"
        print(f"mode: {mode}")
        print(f"policy: {sanitize_endpoint(app.policy.endpoint)} adapter={app.policy.server.adapter}")
        if args.execute_fake:
            print("NOTE: --execute-fake only moves the fake robot; real hardware is disabled in M1.")

    harness = ShadowHarness(
        app,
        embedded_server=not args.external_server,
        task=args.task,
        auth_token=token,
        on_chunk=(
            None
            if args.json
            else (lambda chunk: print(f"chunk seq={chunk.request_sequence_id} H={chunk.horizon} first={chunk.actions[0].tolist()}"))
        ),
    )
    try:
        result = harness.run(duration_s=args.duration, operator_ack=args.execute_fake)
    except VlaInfraError as exc:
        print(f"RUNTIME ERROR {exc.code.value}: {exc.message}")
        if exc.details:
            print(json.dumps(exc.details, indent=2, ensure_ascii=False))
        return 2
    except Exception as exc:  # CLI boundary: never dump a bare traceback to an operator
        print(f"RUNTIME ERROR: {type(exc).__name__}: {exc}")
        return 2
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("interrupted; stopping fake harness")
        harness.stop()
        return 130

    summary = result.as_dict()
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    else:
        print(
            f"final state: {result.final_state}  armed={result.armed}  "
            f"iterations={result.iterations}  observations={result.observations}"
        )
        print(
            f"policy chunks={result.policy_chunks}  writes={result.writes}  "
            f"state={result.supervisor.get('state')} epoch={result.supervisor.get('epoch')}"
        )
        print(f"executor: {result.executor}")
        print(f"worker: {result.worker}")
        print(f"client: {result.client}")
        print(f"server: {result.server}")
        print(f"sync drops: {result.sync_dropped}")
        if result.supervisor.get("latched_fault"):
            print(f"LATCHED FAULT: {result.supervisor['latched_fault']}")
    if result.final_state == "FAULT":
        return 2
    if args.execute_fake and result.writes == 0:
        print("FAILED: execution was requested but no fake device write occurred")
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
