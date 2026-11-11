#!/usr/bin/env bash
# Configuration check + short fake shadow run.  Never opens hardware.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
python3 -m vla_infra.cli.doctor --json >/dev/null
python3 -m vla_infra.cli.shadow --duration 0.5 --json
