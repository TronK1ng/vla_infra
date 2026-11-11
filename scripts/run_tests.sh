#!/usr/bin/env bash
# Run the hardware-free M1 test suite.
#
# The station image ships pytest 6.2.5 with an anyio plugin that requires
# pytest >= 7; disable third-party plugin autoloading to keep the run offline
# and deterministic.  Pass pytest arguments through, e.g.:
#   ./scripts/run_tests.sh -q tests/unit/test_supervisor.py
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
# Test isolation only; the client itself forces direct connections for loopback.
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="$NO_PROXY"
exec python3 -m pytest -ra "$@"
