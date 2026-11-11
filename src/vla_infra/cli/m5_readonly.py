"""M5 Field-01 read-only Flexiv connectivity probe.

This command intentionally does not reuse the M5 production runtime: doing so
would construct command sinks.  The only SDK calls made after construction are
``states()``, ``fault()`` and ``operational()``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Callable, Optional


_ALLOW_ENV = "VLA_INFRA_ALLOW_REAL_HARDWARE"
_ACK = "I_ACKNOWLEDGE_FLEXIV_READONLY_NO_MOTION"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Flexiv M5 Field-01 read-only probe (never enables or commands)")
    parser.add_argument("--robot-serial", required=True)
    parser.add_argument("--network-interface", default=None)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--ack", required=True, help=f"must equal {_ACK}")
    parser.add_argument("--json", action="store_true")
    return parser


def run_probe(
    *,
    sdk: Any,
    serial: str,
    network_interface: Optional[str],
    duration_s: float,
    rate_hz: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run the strictly read-only SDK probe; ``sdk`` is injectable for tests."""
    if not serial or "REPLACE_ME" in serial:
        raise ValueError("a real robot serial is required")
    if duration_s <= 0 or rate_hz <= 0:
        raise ValueError("duration and rate must be positive")

    robot = sdk.Robot(serial, [network_interface]) if network_interface else sdk.Robot(serial)
    started = monotonic()
    deadline = started + duration_s
    period = 1.0 / rate_hz
    samples = 0
    first_timestamp = None
    last_timestamp = None
    field_lengths: dict[str, int] = {}
    fault_seen = False
    operational_seen = False

    while monotonic() < deadline:
        state = robot.states()
        fault_seen = fault_seen or bool(robot.fault())
        operational_seen = operational_seen or bool(robot.operational())
        for name, expected in (
            ("q", 7),
            ("dq", 7),
            ("tcp_pose", 7),
            ("tcp_vel", 6),
            ("ext_wrench_in_tcp", 6),
        ):
            value = getattr(state, name)
            length = len(value)
            if length != expected:
                raise RuntimeError(f"unexpected {name} length: {length}, expected {expected}")
            field_lengths[name] = length
        timestamp = getattr(state, "timestamp", None)
        if timestamp is not None:
            current = list(timestamp) if not isinstance(timestamp, (int, float)) else timestamp
            if first_timestamp is None:
                first_timestamp = current
            last_timestamp = current
        samples += 1
        sleep(period)

    elapsed = max(monotonic() - started, 0.0)
    return {
        "schema_version": "m5-field01-readonly/v1",
        "serial": serial,
        "sdk_version": str(getattr(sdk, "__version__", "unknown")),
        "duration_s": elapsed,
        "samples": samples,
        "observed_rate_hz": samples / elapsed if elapsed else 0.0,
        "fault_seen": fault_seen,
        "operational_seen": operational_seen,
        "field_lengths": field_lengths,
        "first_device_timestamp": first_timestamp,
        "last_device_timestamp": last_timestamp,
        "write_calls": 0,
    }


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if os.environ.get(_ALLOW_ENV) != "1":
        print(f"REFUSED: set {_ALLOW_ENV}=1 at the robot station")
        return 2
    if args.ack != _ACK:
        print(f"REFUSED: --ack must exactly equal {_ACK}")
        return 2
    try:
        import flexivrdk  # type: ignore

        report = run_probe(
            sdk=flexivrdk,
            serial=args.robot_serial,
            network_interface=args.network_interface,
            duration_s=args.duration,
            rate_hz=args.rate_hz,
        )
    except Exception as exc:  # SDK/CLI boundary
        print(f"READONLY PROBE FAILED: {type(exc).__name__}: {exc}")
        return 1
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(
            f"READONLY PASS samples={report['samples']} rate={report['observed_rate_hz']:.1f}Hz "
            f"fault_seen={report['fault_seen']} operational_seen={report['operational_seen']}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
