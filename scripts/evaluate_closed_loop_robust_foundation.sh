#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_name="${CUSTOM_DOG_RUN_NAME:-2026-08-14_19-59-12_closed_loop_robust_foundation_seed42}"
run_dir="${CUSTOM_DOG_RUN_DIR:-${project_root}/logs/rsl_rl/custom_dog_velocity/${run_name}}"
task="CustomDog-Velocity-OmniTrot-ClosedLoopRobustFoundation-v1"
baseline="${project_root}/logs/rsl_rl/custom_dog_velocity/2026-08-14_17-53-06_closed_loop_a2_yaw_boundary_standing_band/evaluation/candidates/model_1350"

checkpoints=(
    model_400.pt
    model_550.pt
    model_700.pt
    model_800.pt
    model_900.pt
    model_999.pt
)

"${project_root}/scripts/export_checkpoint_candidates.sh" \
    "${run_dir}" "${task}" "${checkpoints[@]}"

python="${CUSTOM_DOG_MUJOCO_PYTHON:-${HOME}/miniconda3/envs/custom_dog_mujoco/bin/python}"
"${python}" "${project_root}/scripts/evaluate_mujoco_grid.py" \
    --candidate "A2_1350=${baseline}" \
    --candidate "RF_400=${run_dir}/evaluation/candidates/model_400" \
    --candidate "RF_550=${run_dir}/evaluation/candidates/model_550" \
    --candidate "RF_700=${run_dir}/evaluation/candidates/model_700" \
    --candidate "RF_800=${run_dir}/evaluation/candidates/model_800" \
    --candidate "RF_900=${run_dir}/evaluation/candidates/model_900" \
    --candidate "RF_999=${run_dir}/evaluation/candidates/model_999" \
    --baseline-label A2_1350 \
    --duration 10 \
    --warmup 2 \
    --output-csv "${run_dir}/evaluation/robust_foundation_grid_10s.csv" \
    --output-json "${run_dir}/evaluation/robust_foundation_grid_10s.json"

python3 "${project_root}/scripts/select_mujoco_candidate.py" \
    "${run_dir}/evaluation/robust_foundation_grid_10s.json" \
    --output "${run_dir}/evaluation/robust_foundation_selection.json"
