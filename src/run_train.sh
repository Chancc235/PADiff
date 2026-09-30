#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_name="${1:-stage1_PP}"
if (( $# > 0 )); then shift; fi
cd "$repo_root"
exec "${PYTHON:-python3}" src/meta_main.py "--config=$config_name" "$@"
