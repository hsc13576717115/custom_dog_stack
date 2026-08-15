#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_root="${project_root}/logs/rsl_rl/custom_dog_velocity"
recovery_run="${1:?Usage: $0 R2_RUN_DIR}"

mapfile -t r2_selections < <(
    find "${recovery_run}" -mindepth 2 -maxdepth 2 -type f \
        -name 'self_righting_r2_selection.json' -printf '%T@ %p\n' | sort -n
)
mapfile -t d_selections < <(
    find "${log_root}" -mindepth 3 -maxdepth 3 -type f \
        -path '*_closed_loop_stage_d_seed42/evaluation/closed_loop_stage_d_selection.json' \
        -printf '%T@ %p\n' | sort -n
)
mapfile -t stand_selections < <(
    find "${log_root}" -mindepth 3 -maxdepth 3 -type f \
        -path '*/evaluation/stand_selection.json' -printf '%T@ %p\n' | sort -n
)
if [[ ${#r2_selections[@]} -eq 0 || ${#d_selections[@]} -eq 0 || ${#stand_selections[@]} -eq 0 ]]; then
    echo "R2, Stage-D, and stand selections are required; blocking recovery handoff gate." >&2
    exit 1
fi
r2_selection="${r2_selections[-1]#* }"
d_selection="${d_selections[-1]#* }"
stand_selection="${stand_selections[-1]#* }"
if ! jq -e '.accepted == true and .selected_candidate != null' "${r2_selection}" >/dev/null \
    || ! jq -e '.accepted == true and (.selected | startswith("D_")) and .selected_candidate != null' "${d_selection}" >/dev/null \
    || ! jq -e '.accepted == true and .selected_candidate != null' "${stand_selection}" >/dev/null; then
    echo "R2, Stage-D, or stand selection did not pass; blocking recovery handoff gate." >&2
    exit 1
fi

recovery_candidate="$(jq -r '.selected_candidate' "${r2_selection}")"
locomotion_candidate="$(jq -r '.selected_candidate' "${d_selection}")"
stand_candidate="$(jq -r '.selected_candidate' "${stand_selection}")"
selective_mjcf="${recovery_run}/evaluation/custom_dog_selective.xml"
if [[ ! -f "${selective_mjcf}" ]]; then
    python3 "${project_root}/scripts/generate_selective_mujoco.py" --output "${selective_mjcf}"
fi
python="${CUSTOM_DOG_MUJOCO_PYTHON:-${HOME}/miniconda3/envs/custom_dog_mujoco/bin/python}"
"${python}" "${project_root}/scripts/evaluate_recovery_handoff.py" \
    --recovery-candidate "${recovery_candidate}" \
    --locomotion-candidate "${locomotion_candidate}" \
    --stand-candidate "${stand_candidate}" \
    --mjcf "${selective_mjcf}" \
    --runner "${project_root}/sim2sim/custom_dog/run_sim2sim.py" \
    --python "${python}" \
    --output "${recovery_run}/evaluation/recovery_handoff_r2.json"
