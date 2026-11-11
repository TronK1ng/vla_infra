"""``vla-m3``: fake/simulation 100 Hz execution profile and fault injection."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import load_app_config
from ..runtime.m3 import M3Config, M3Runtime

DEFAULT_STATION = "configs/station.example.yaml"
DEFAULT_SAFETY = "configs/safety.example.yaml"
DEFAULT_POLICY = "configs/policy.example.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fake M3 100 Hz execution profile (no hardware)."
    )
    parser.add_argument("--station", default=DEFAULT_STATION)
    parser.add_argument("--safety", default=DEFAULT_SAFETY)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--output", default="data/m3/execution.jsonl")
    parser.add_argument(
        "--fault",
        choices=("none", "policy_stall", "state_stall", "invalid_chunk", "writer_error"),
        default="none",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.duration <= 0:
        print("CONFIG ERROR: --duration must be positive")
        return 1
    try:
        app = load_app_config(
            station_path=Path(args.station),
            safety_path=Path(args.safety),
            policy_path=Path(args.policy),
        )
    except Exception as exc:
        print(f"CONFIG ERROR: {type(exc).__name__}: {exc}")
        return 1
    runtime = M3Runtime(app, output=args.output, config=M3Config())
    runtime.start()
    try:
        time.sleep(min(0.2, args.duration / 2.0))
        if args.fault == "policy_stall":
            runtime.set_policy_enabled(False)
        elif args.fault == "state_stall":
            runtime.set_state_enabled(False)
        elif args.fault == "invalid_chunk":
            runtime.set_policy_enabled(False)
            bad = np.zeros((1, 7), dtype=np.float32)
            bad[0, 0] = 10.0
            runtime.submit_actions(bad)
        elif args.fault == "writer_error":
            def boom(*_args, **_kwargs):
                raise RuntimeError("injected writer failure")

            runtime._robot.command_pose = boom
        time.sleep(max(0.0, args.duration - min(0.2, args.duration / 2.0)))
    finally:
        result = runtime.stop()
    if args.json:
        print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False, default=str))
    else:
        print(f"final_state: {result.final_state}")
        print(f"ticks={result.ticks} writes={result.writes} policy_chunks={result.policy_chunks}")
        print(f"log: {result.log_path} records={result.log_records} dropped={result.log_dropped}")
        print("tick period:", json.dumps(result.metrics["tick_period_ns"]))
        print("runtime:", json.dumps(result.metrics["runtime_ns"]))
        print("jitter:", json.dumps(result.metrics["jitter_ns"]))
    return 4 if result.stop_error else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
