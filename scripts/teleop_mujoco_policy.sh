#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
candidate="${1:-${project_root}/deploy/candidates/model_4500_yaw_straight}"
candidate="$(realpath "${candidate}")"

if [[ ! -f "${candidate}/exported/policy.onnx" || ! -f "${candidate}/params/deploy.yaml" ]]; then
    echo "Candidate must contain exported/policy.onnx and params/deploy.yaml: ${candidate}" >&2
    exit 2
fi

exec "${project_root}/scripts/run_sim2sim.sh" \
    --policy "${candidate}/exported/policy.onnx" \
    --deploy-yaml "${candidate}/params/deploy.yaml" \
    --command 0.0 0.0 0.0 \
    --duration 0 \
    --viewer \
    --interactive
