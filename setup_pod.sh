#!/usr/bin/env bash
# Run inside the workspace pod. Python dependencies/cache stay outside source sync.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${HOME}/venvs/ftn-bc}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${HOME}/.cache/uv}"

if ! ldconfig -p 2>/dev/null | grep 'libEGL.so.1' >/dev/null; then
    if ! command -v apt-get >/dev/null; then
        echo 'Install the EGL/OpenGL system runtime for this pod image, then rerun setup.' >&2
        exit 1
    fi
    elevate=()
    if [ "$(id -u)" -ne 0 ]; then elevate=(sudo); fi
    "${elevate[@]}" apt-get update
    "${elevate[@]}" apt-get install -y --no-install-recommends libegl1 libgl1
fi
uv sync --locked
printf '\nPod environment ready. Keep this variable set for training and visualization:\nexport UV_PROJECT_ENVIRONMENT=%q\n' "$UV_PROJECT_ENVIRONMENT"
