#!/usr/bin/env bash
# Fail-fast checks before publishing the repository.
set -euo pipefail

cd "$(dirname "$0")/.."

for path in configs/local data artifacts evidence .openpi_vendor .vlawork reference/ForcePolicy; do
  if [[ -e "$path" ]]; then
    echo "INFO: local-only path exists and must remain ignored: $path"
  fi
done

large_files="$(find . -path './.git' -prune -o -type f -size +20M -print)"
if [[ -n "$large_files" ]]; then
  echo "ERROR: files larger than 20 MiB found:" >&2
  echo "$large_files" >&2
  exit 1
fi

real_configs="$(find configs -maxdepth 2 -type f \( -path 'configs/local/*' -o -name '*.real.yaml' \) -print)"
if [[ -n "$real_configs" ]]; then
  echo "ERROR: local/real station configuration found under configs" >&2
  exit 1
fi

PYTHONPATH=src python3 -m vla_infra.cli.doctor --json >/dev/null
PYTHONPATH=src python3 -m vla_infra.cli.m5_readonly --help >/dev/null

echo "release check passed (this does not run the full test suite)"
