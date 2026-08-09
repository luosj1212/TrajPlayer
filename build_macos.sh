#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "TrajPlayer's macOS package must be built on macOS."
    exit 1
fi

venv_dir="${TRAJPLAYER_BUILD_VENV:-.venv-build-macos}"
python3 -m venv "$venv_dir"
source "$venv_dir/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/check_release.py
python scripts/build_release.py
