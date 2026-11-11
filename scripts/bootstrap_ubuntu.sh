#!/usr/bin/env bash
# Create an isolated Python environment and install vla_infra.
# This script never installs/starts system services and never contacts hardware.
set -euo pipefail

cd "$(dirname "$0")/.."

extra=""
if [[ "${1:-}" == "--dev" ]]; then
  extra="[dev]"
elif [[ -n "${1:-}" ]]; then
  echo "usage: $0 [--dev]" >&2
  exit 2
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required (Ubuntu 22.04 / Python 3.10 recommended)" >&2
  exit 1
fi

python3 - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python >= 3.10 is required")
print(f"using Python {sys.version.split()[0]}")
PY

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -e ".${extra}"

echo "installation complete"
echo "next: source .venv/bin/activate && ./scripts/smoke.sh"
