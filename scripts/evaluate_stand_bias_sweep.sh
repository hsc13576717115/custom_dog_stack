#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="${project_root}/logs/rsl_rl/custom_dog_velocity/2026-08-14_22-32-41_stand_height_hip_calibrated_from319"
source="${run_dir}/evaluation/candidates/model_418"

candidate_args=(--candidate "STAND_CAL3_418=${source}")
for millirad in 20 40 60 80; do
    bias="$(awk -v value="${millirad}" 'BEGIN { printf "%.3f", value / 1000.0 }')"
    candidate="${run_dir}/evaluation/bias_candidates/bias_${millirad}"
    python3 "${project_root}/scripts/build_stand_bias_candidate.py" \
        "${source}" "${candidate}" --hip-bias "${bias}"
    candidate_args+=(--candidate "STAND_BIAS_${millirad}=${candidate}")
done

python="${CUSTOM_DOG_MUJOCO_PYTHON:-${HOME}/miniconda3/envs/custom_dog_mujoco/bin/python}"
"${python}" "${project_root}/scripts/evaluate_mujoco_grid.py" \
    "${candidate_args[@]}" \
    --baseline-label STAND_CAL3_418 \
    --absolute-only \
    --command 0 0 0 \
    --duration 15 \
    --warmup 3 \
    --output-csv "${run_dir}/evaluation/stand_bias_grid_15s.csv" \
    --output-json "${run_dir}/evaluation/stand_bias_grid_15s.json"

python3 "${project_root}/scripts/select_stand_bias_candidate.py" \
    "${run_dir}/evaluation/stand_bias_grid_15s.json" \
    --output "${run_dir}/evaluation/stand_selection.json"
