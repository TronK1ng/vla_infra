"""``vla-doctor``: configuration and environment validation.

This command never opens a camera, a robot or the policy network.  It only
validates files, imports and dependency versions, and refuses to start on
``backend: real`` in M1.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Optional

from .. import __version__
from ..config import load_app_config
from ..errors import VlaInfraError
from ..policy.client import sanitize_endpoint

DEFAULT_STATION = "configs/station.example.yaml"
DEFAULT_SAFETY = "configs/safety.example.yaml"
DEFAULT_POLICY = "configs/policy.example.yaml"


def _dependency_versions() -> dict:
    versions: dict[str, str] = {}
    for name in ("numpy", "yaml", "websockets", "msgpack"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[name] = "not installed"
    return versions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate VLA infrastructure configuration (no hardware access).")
    parser.add_argument("--station", default=DEFAULT_STATION, help="station YAML path")
    parser.add_argument("--safety", default=DEFAULT_SAFETY, help="safety YAML path")
    parser.add_argument("--policy", default=DEFAULT_POLICY, help="policy YAML path")
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="fail if disk/SDK prerequisites for the selected backend are not met",
    )
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
        if exc.details:
            print(json.dumps(exc.details, indent=2, ensure_ascii=False))
        return 1
    recording_root = Path(app.station.recording.root)
    probe = recording_root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    disk = shutil.disk_usage(probe)
    min_free_bytes = int(app.station.recording.min_free_gb * 1024**3)
    report = {
        "ok": True,
        "vla_infra_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": _dependency_versions(),
        "station": {
            "station_id": app.station.station_id,
            "backend": app.station.backend,
            "cameras": {
                name: {
                    "width": camera.width,
                    "height": camera.height,
                    "fps": camera.fps,
                    "depth": camera.depth,
                    "serial_configured": bool(camera.serial and "REPLACE_ME" not in camera.serial),
                }
                for name, camera in app.station.cameras.items()
            },
            "robot_serial_configured": bool(
                app.station.robot.serial and "REPLACE_ME" not in app.station.robot.serial
            ),
            "recording_root": app.station.recording.root,
            "min_free_gb": app.station.recording.min_free_gb,
            "finalize_timeout_s": app.station.recording.finalize_timeout_s,
            "calibration_version": app.station.calibration_version,
        },
        "recording": {
            "free_bytes": int(disk.free),
            "min_free_bytes": min_free_bytes,
            "disk_ok": disk.free >= min_free_bytes,
        },
        "sdks": {
            "pyrealsense2": importlib.util.find_spec("pyrealsense2") is not None,
            "flexivrdk": importlib.util.find_spec("flexivrdk") is not None,
        },
        "sync": {
            "max_sync_error_ms": app.station.sync.max_sync_error_ns / 1e6,
            "max_state_gap_ms": app.station.sync.max_state_gap_ns / 1e6,
            "max_camera_skew_ms": app.station.sync.max_camera_skew_ns / 1e6,
            "wrench_policy": app.station.sync.wrench_policy,
            "gripper_policy": app.station.sync.gripper_policy,
        },
        "safety": {
            "workspace_min": list(app.safety.workspace_min),
            "workspace_max": list(app.safety.workspace_max),
            "tcp_height_m": [app.safety.tcp_height_min_m, app.safety.tcp_height_max_m],
            "max_step_translation_m": app.safety.max_step_translation_m,
            "max_step_rotation_rad": app.safety.max_step_rotation_rad,
            "max_linear_velocity_m_s": app.safety.max_linear_velocity_m_s,
            "max_angular_velocity_rad_s": app.safety.max_angular_velocity_rad_s,
            "max_observation_age_ms": app.safety.max_observation_age_ns / 1e6,
            "max_chunk_age_ms": app.safety.max_chunk_age_ns / 1e6,
            "max_force_n": app.safety.max_force_n,
            "max_torque_nm": app.safety.max_torque_nm,
            "gripper_width_m": [app.safety.gripper_width_min_m, app.safety.gripper_width_max_m],
            "allowed_action_frames": list(app.safety.allowed_action_frames),
            "allowed_action_types": list(app.safety.allowed_action_types),
            "joint_limits_configured": app.safety.joint_position_min_rad is not None,
        },
        "policy": {
            "endpoint": sanitize_endpoint(app.policy.endpoint),
            "server": {
                "host": app.policy.server.host,
                "port": app.policy.server.port,
                "adapter": app.policy.server.adapter,
            },
            "state_layout": app.policy.state_layout.name,
            "state_dim": app.policy.state_layout.dim,
            "image_size": app.policy.image_size,
            "expected_horizon": app.policy.expected_horizon,
            "expected_action_dt_s": app.policy.expected_action_dt_s,
            "expected_action_frame": app.policy.expected_action_frame,
            "expected_action_type": app.policy.expected_action_type,
            "auth_token_env": app.policy.auth_token_env,
            "proxy": app.policy.proxy,
        },
        "hardware_access": "none (backend must be fake in M1)",
    }
    if args.preflight:
        problems = []
        if disk.free < min_free_bytes:
            problems.append(
                f"disk free {disk.free} < required {min_free_bytes} bytes"
            )
        if app.station.backend == "real_readonly":
            for sdk_name in ("pyrealsense2", "flexivrdk"):
                if not report["sdks"][sdk_name]:
                    problems.append(f"{sdk_name} is not importable")
        if problems:
            report["ok"] = False
            report["preflight_problems"] = problems
            if args.json:
                print(json.dumps(report, indent=2, ensure_ascii=False))
            else:
                for problem in problems:
                    print(f"PREFLIGHT FAIL: {problem}")
            return 1
        report["preflight"] = "ok"
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    print(f"vla_infra {__version__}  python {report['python']}  {report['platform']}")
    print(f"dependencies: {report['dependencies']}")
    print(f"station: {app.station.station_id} backend={app.station.backend}")
    for name, camera in app.station.cameras.items():
        serial_state = "configured" if report["station"]["cameras"][name]["serial_configured"] else "PLACEHOLDER"
        print(f"  camera {name}: {camera.width}x{camera.height}@{camera.fps} depth={camera.depth} serial={serial_state}")
    print(f"  robot serial: {'configured' if report['station']['robot_serial_configured'] else 'PLACEHOLDER'}")
    print(
        "safety: workspace "
        f"{app.safety.workspace_min} .. {app.safety.workspace_max}, "
        f"v_max={app.safety.max_linear_velocity_m_s} m/s, "
        f"force_max={app.safety.max_force_n} N"
    )
    print(
        f"policy: {report['policy']['endpoint']} adapter={app.policy.server.adapter} "
        f"layout={app.policy.state_layout.name} dim={app.policy.state_layout.dim}"
    )
    print("OK: configuration valid; no hardware was contacted.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
