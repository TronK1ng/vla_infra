"""``vla-replay``: inspect/validate a raw episode and optionally export a draft."""

from __future__ import annotations

import argparse
import json
from typing import Optional

from ..errors import VlaInfraError
from ..recording.lerobot_writer import export_lerobot_draft
from ..recording.reader import load_episode, validate_episode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate/replay an M2 raw episode (no hardware).")
    parser.add_argument("episode", help="path to episode_xxxx")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--lerobot-out", default=None)
    parser.add_argument("--no-copy-images", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        issues = validate_episode(args.episode)
    except VlaInfraError as exc:
        print(f"REPLAY ERROR {exc.code.value}: {exc.message}")
        return 2
    episode = load_episode(args.episode)
    summary = {
        "episode": str(episode.path),
        "metadata": {
            "episode_index": episode.metadata.get("episode_index"),
            "task": episode.metadata.get("task"),
            "complete": episode.metadata.get("complete"),
            "end_reason": episode.metadata.get("end_reason"),
            "verdict": episode.metadata.get("verdict"),
        },
        "lowdim_rows": int(episode.lowdim.shape[0]),
        "camera_frames": {name: len(records) for name, records in episode.camera_frames.items()},
        "observations": len(episode.observations),
        "actions": len(episode.actions),
        "sync_report": episode.metadata.get("sync_report", {}),
        "issues": issues,
    }
    if args.lerobot_out:
        try:
            exported = export_lerobot_draft(
                episode.path,
                args.lerobot_out,
                copy_images=not args.no_copy_images,
            )
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
        print(f"episode: {episode.path}")
        print(f"lowdim rows: {summary['lowdim_rows']}  observations: {summary['observations']}")
        print(f"camera frames: {summary['camera_frames']}")
        print(f"sync: {json.dumps(summary['sync_report'], ensure_ascii=False)}")
        print(f"issues: {issues if issues else 'none'}")
    return 0 if not issues else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
