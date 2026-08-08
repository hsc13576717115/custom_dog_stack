#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
    echo "Usage: $0 deploy/candidates/model_N [vx vy yaw]" >&2
    exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
candidate="$(realpath "$1")"
vx="${2:-0.3}"
vy="${3:-0.0}"
yaw="${4:-0.0}"

exec "${project_root}/scripts/run_sim2sim.sh" \
    --policy "${candidate}/exported/policy.onnx" \
    --deploy-yaml "${candidate}/params/deploy.yaml" \
    --command "${vx}" "${vy}" "${yaw}" \
    --duration "${CUSTOM_DOG_VIEW_DURATION:-60}" \
    --viewer
