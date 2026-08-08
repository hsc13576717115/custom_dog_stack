#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${project_root}/scripts/activate_env.sh"

python -m pip install --no-deps --editable "${project_root}"

if ! git lfs version >/dev/null 2>&1; then
    echo "WARNING: git-lfs is not installed. Install it before the first commit containing STL or ONNX files."
fi

echo "Custom Dog development environment initialized at ${project_root}"
