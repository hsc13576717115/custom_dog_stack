#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_name="${CUSTOM_DOG_RUN_NAME:-2026-08-14_18-57-35_closed_loop_cross_physics_from_a2_1350}"
run_dir="${project_root}/logs/rsl_rl/custom_dog_velocity/${run_name}"
task="CustomDog-Velocity-OmniTrot-ClosedLoopCrossPhysics-v1"
baseline="${project_root}/logs/rsl_rl/custom_dog_velocity/2026-08-14_17-53-06_closed_loop_a2_yaw_boundary_standing_band/evaluation/candidates/model_1350"

checkpoints=(
    model_1400.pt
    model_1460.pt
    model_1520.pt
    model_1580.pt
    model_1620.pt
    model_1649.pt
)

"${project_root}/scripts/export_checkpoint_candidates.sh" \
    "${run_dir}" "${task}" "${checkpoints[@]}"

python="${CUSTOM_DOG_MUJOCO_PYTHON:-${HOME}/miniconda3/envs/custom_dog_mujoco/bin/python}"
"${python}" "${project_root}/scripts/evaluate_mujoco_grid.py" \
    --candidate "A2_1350=${baseline}" \
    --candidate "A3_1400=${run_dir}/evaluation/candidates/model_1400" \
    --candidate "A3_1460=${run_dir}/evaluation/candidates/model_1460" \
    --candidate "A3_1520=${run_dir}/evaluation/candidates/model_1520" \
    --candidate "A3_1580=${run_dir}/evaluation/candidates/model_1580" \
    --candidate "A3_1620=${run_dir}/evaluation/candidates/model_1620" \
    --candidate "A3_1649=${run_dir}/evaluation/candidates/model_1649" \
    --baseline-label A2_1350 \
    --duration 10 \
    --warmup 2 \
    --output-csv "${run_dir}/evaluation/cross_physics_grid_10s.csv" \
    --output-json "${run_dir}/evaluation/cross_physics_grid_10s.json"
