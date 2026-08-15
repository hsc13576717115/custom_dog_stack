#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_root="${project_root}/logs/rsl_rl/custom_dog_velocity"
mapfile -t selections < <(
    find "${log_root}" -mindepth 3 -maxdepth 3 -type f \
        -path '*/evaluation/stand_selection.json' -printf '%T@ %p\n' \
        | sort -n
)
if [[ ${#selections[@]} -eq 0 ]]; then
    echo "No stand-expert run was found; routed Stage A will not be built."
    exit 0
fi
stand_selection="${selections[-1]#* }"
stand_run="$(dirname "$(dirname "${stand_selection}")")"
if ! jq -e \
    '.accepted == true and (.selected | startswith("STAND_"))' \
    "${stand_selection}" >/dev/null; then
    echo "Stand expert did not pass its zero-command gate; skipping routed Stage A."
    exit 0
fi

locomotion="${log_root}/2026-08-14_19-59-12_closed_loop_robust_foundation_seed42/evaluation/candidates/model_700"
stand="$(jq -r '.selected_candidate' "${stand_selection}")"
routed="${project_root}/deploy/candidates/closed_loop_stage_a_routed_seed42"
python3 "${project_root}/scripts/build_routed_candidate.py" \
    "${locomotion}" "${stand}" "${routed}"

python="${CUSTOM_DOG_MUJOCO_PYTHON:-${HOME}/miniconda3/envs/custom_dog_mujoco/bin/python}"
"${python}" "${project_root}/scripts/evaluate_mujoco_grid.py" \
    --candidate "RF_700=${locomotion}" \
    --candidate "ROUTED_A=${routed}" \
    --baseline-label RF_700 \
    --absolute-only \
    --duration 10 \
    --warmup 2 \
    --output-csv "${stand_run}/evaluation/routed_stage_a_grid_10s.csv" \
    --output-json "${stand_run}/evaluation/routed_stage_a_grid_10s.json"

python3 "${project_root}/scripts/select_mujoco_candidate.py" \
    "${stand_run}/evaluation/routed_stage_a_grid_10s.json" \
    --output "${stand_run}/evaluation/routed_stage_a_selection.json"
