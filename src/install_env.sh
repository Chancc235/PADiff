#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python3}"

"$python_bin" -m pip install -r "$repo_root/requirements.txt"
"$python_bin" -m pip install -e "$repo_root/src/envs/overcooked/overcooked_ai"
"$python_bin" -m pip install -e "$repo_root/src/envs/lbf/lb-foraging"
