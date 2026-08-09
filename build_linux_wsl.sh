#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_root="$(mktemp -d -t trajplayer-linux-build-XXXXXXXX)"
trap 'rm -rf "$build_root"' EXIT

for file in \
    .gitattributes \
    .gitignore \
    app.py \
    build_exe.bat \
    build_linux.sh \
    build_linux_wsl.sh \
    CHANGELOG.md \
    CONTRIBUTING.md \
    DISTRIBUTION_README.txt \
    LICENSE \
    pyproject.toml \
    setup.py \
    README.md \
    requirements-dev.txt \
    requirements-linux.txt \
    requirements.txt \
    run_app.bat \
    SECURITY.md \
    THIRD_PARTY_NOTICES.md \
    TrajPlayer.spec; do
    cp "$project_dir/$file" "$build_root/"
done
for directory in .github docs scripts tests trajplayer; do
    cp -a "$project_dir/$directory" "$build_root/"
done
mkdir -p "$build_root/examples"
cp "$project_dir"/examples/*.extxyz "$build_root/examples/"

export TRAJPLAYER_BUILD_VENV="${XDG_CACHE_HOME:-$HOME/.cache}/trajplayer/build-venv"
bash "$build_root/build_linux.sh"
archive="$(find "$build_root" -maxdepth 1 -name 'TrajPlayer-Linux-x86_64-*.tar.gz' -print -quit)"
if [[ -z "$archive" ]]; then
    echo "Linux archive was not produced."
    exit 1
fi
cp "$archive" "$project_dir/"
if [[ -f "$archive.sha256" ]]; then
    cp "$archive.sha256" "$project_dir/"
fi
echo "Copied Linux package to: $project_dir/$(basename "$archive")"
