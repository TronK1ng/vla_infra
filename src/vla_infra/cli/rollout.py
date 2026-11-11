"""Rollout CLI placeholder (M5)."""

from __future__ import annotations

import argparse
import sys
from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Policy rollout (M5, not implemented).")
    parser.add_argument("--station", default="configs/station.example.yaml")
    parser.add_argument("--safety", default="configs/safety.example.yaml")
    parser.add_argument("--policy", default="configs/policy.example.yaml")
    parser.parse_args(argv)
    print(
        "NOT IMPLEMENTED: real-robot policy rollout is milestone M5 and requires an "
        "on-site operator, the M3 safety gate and an M6 shadow review. "
        "M1 never connects real hardware.",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
