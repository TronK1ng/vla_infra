"""``vla-record``: M2 collection CLI (fake backend by default).

No policy client, no executor and no command sink are created.  Real read-only
collection is only possible with ``backend: real_readonly`` plus the explicit
on-site gate in configuration/environment; the adapter then still never writes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from ..config import load_app_config
from ..errors import VlaInfraError
from ..recording.lerobot_writer import export_lerobot_draft
from ..recording.recorder import RecorderConfig
from ..runtime.collection import CollectionHarness

DEFAULT_STATION = "configs/station.example.yaml"
DEFAULT_SAFETY = "configs/safety.example.yaml"
DEFAULT_POLICY = "configs/policy.example.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record a raw episode using read-only devices (fake by default)."
    )
    parser.add_argument("--station", default=DEFAULT_STATION)
    parser.add_argument("--safety", default=DEFAULT_SAFETY)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--task", default="collection task")
    parser.add_argument("--output", default=None, help="override station.recording.root")
    parser.add_argument("--control-source", default="operator")
    parser.add_argument("--lerobot-out", default=None, help="also export a JSONL LeRobot draft")
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=None,
        help="override station.recording.min_free_gb for this run",
    )
    parser.add_argument(
        "--finalize-timeout",
        type=float,
        default=None,
        help="override station.recording.finalize_timeout_s for drain/commit/fsync wait",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        app = load_app_config(
            station_path=Path(args.station),
            safety_path=Path(args.safety),
            policy_path=Path(args.policy),
        )
    except VlaInfraError as exc:
        print(f"CONFIG ERROR {exc.code.value}: {exc.message}")
        return 1

    # Explicit check so a real_readonly config does not accidentally run without
    # the on-site environment gate (load_station already checked, defense in depth).
    if app.station.backend == "real_readonly" and os.environ.get("VLA_INFRA_ALLOW_REAL_HARDWARE") != "1":
        print("CONFIG ERROR: real_readonly requires VLA_INFRA_ALLOW_REAL_HARDWARE=1")
        return 1
    if app.station.backend == "real_readonly":
        print(
            "CONFIG ERROR: real_readonly collection needs a read-only GN01 width provider, "
            "which is not implemented in this round; use backend: fake for pipeline tests"
        )
        return 1

    min_free_gb = (
        app.station.recording.min_free_gb if args.min_free_gb is None else args.min_free_gb
    )
    finalize_timeout_s = (
        app.station.recording.finalize_timeout_s
        if args.finalize_timeout is None
        else args.finalize_timeout
    )
    if min_free_gb < 0 or finalize_timeout_s <= 0:
        print("CONFIG ERROR: --min-free-gb must be >= 0 and --finalize-timeout must be > 0")
        return 1
    harness = CollectionHarness(
        app,
        task=args.task,
        control_source=args.control_source,
        output_root=args.output,
        recorder_config=RecorderConfig(
            min_free_gb=min_free_gb,
            finalize_timeout_s=finalize_timeout_s,
        ),
    )
    try:
        result = harness.run(duration_s=args.duration)
    except VlaInfraError as exc:
        print(f"RECORD ERROR {exc.code.value}: {exc.message}")
        if exc.details:
            print(json.dumps(exc.details, indent=2, ensure_ascii=False))
        return 2
    except Exception as exc:  # CLI boundary
        print(f"RECORD ERROR: {type(exc).__name__}: {exc}")
        return 2

    if result.recorder.still_stopping:
        print(
            "FINALIZE TIMEOUT: recorder is still committing; the temporary episode must be "
            "treated as incomplete (no committed episode is reported)."
        )
        if result.recorder.path is not None:
            print(f"published path: {result.recorder.path}")
        return 3

    summary = result.as_dict()
    if args.lerobot_out:
        if not result.recorder.committed or result.episode_path is None:
            print("LEROBOT SKIPPED: episode was not committed")
        else:
            try:
                exported = export_lerobot_draft(result.episode_path, args.lerobot_out)
                summary["lerobot"] = {
                    "path": str(exported.path),
                    "frames": exported.frames,
                    "images": exported.images,
                    "format": exported.format,
                }
            except VlaInfraError as exc:
                print(f"LEROBOT ERROR {exc.code.value}: {exc.message}")
                return 2

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    else:
        print(f"episode: {result.episode_path}")
        print(f"observations: {result.observations}  robot_samples: {result.robot_samples}")
        print(f"recorder: {result.recorder.as_dict()}")
        if args.lerobot_out:
            print(f"lerobot draft: {args.lerobot_out}")
    return 0 if result.recorder.committed else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
