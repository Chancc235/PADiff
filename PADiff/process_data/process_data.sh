#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "Usage: $0 {PP4a|LBF|overcooked}" >&2
  exit 2
fi

env_name="$1"
case "$env_name" in
  PP4a|LBF|overcooked) ;;
  *) echo "Unknown environment: $env_name" >&2; exit 2 ;;
esac

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p data
"${PYTHON:-python3}" "process_data/process_raw_${env_name}.py"
"${PYTHON:-python3}" process_data/compute_rtg.py --env "$env_name"
