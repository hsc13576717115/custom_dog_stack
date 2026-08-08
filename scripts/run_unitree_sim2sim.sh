#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 deploy/candidates/model_N" >&2
    exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
controller_pid=""

cleanup() {
    if [[ -n "${controller_pid}" ]] && kill -0 "${controller_pid}" 2>/dev/null; then
        kill "${controller_pid}" 2>/dev/null || true
        wait "${controller_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

"${project_root}/scripts/run_unitree_policy.sh" "$1" &
controller_pid=$!

"${project_root}/scripts/run_unitree_mujoco.sh"
