#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
collision_report="${CUSTOM_DOG_COLLISION_REPORT:-${project_root}/reports/selective_collision_isaac_runtime.json}"

python3 "${project_root}/scripts/check_selective_collision_report.py" \
    "${collision_report}" \
    --asset "${project_root}/ros2/src/custom_dog_description/urdf/custom_dog_selective_collision.urdf"

echo "Selective-collision gate passed; starting the R0 runtime smoke test."
CUSTOM_DOG_NUM_ENVS=64 \
CUSTOM_DOG_MAX_ITERATIONS=1 \
CUSTOM_DOG_RUN_NAME=self_righting_r0_contract_smoke \
"${project_root}/scripts/train_self_righting.sh" R0

echo "R0 smoke passed; starting formal folded-belly recovery training."
CUSTOM_DOG_NUM_ENVS="${CUSTOM_DOG_NUM_ENVS:-4096}" \
CUSTOM_DOG_MAX_ITERATIONS="${CUSTOM_DOG_MAX_ITERATIONS:-1500}" \
CUSTOM_DOG_RUN_NAME="${CUSTOM_DOG_RUN_NAME:-self_righting_r0_seed42}" \
"${project_root}/scripts/train_self_righting.sh" R0

mapfile -t runs < <(
    find "${project_root}/logs/rsl_rl/custom_dog_velocity" \
        -mindepth 1 -maxdepth 1 -type d -name '*_self_righting_r0_seed42' \
        -printf '%T@ %p\n' | sort -n
)
if [[ ${#runs[@]} -eq 0 ]]; then
    echo "Could not locate the completed R0 run." >&2
    exit 1
fi
"${project_root}/scripts/evaluate_self_righting_run.sh" R0 "${runs[-1]#* }"
