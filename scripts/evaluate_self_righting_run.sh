#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 R0|R1|R2 RUN_DIR" >&2
    exit 2
fi
stage="${1^^}"
run_dir="$(realpath "$2")"
case "${stage}" in R0|R1|R2) ;; *) echo "Expected R0, R1, or R2" >&2; exit 2 ;; esac

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task="CustomDog-SelfRighting-${stage}-v2"
mapfile -t available < <(
    find "${run_dir}" -mindepth 1 -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' \
        | sort -V
)
if [[ ${#available[@]} -lt 5 ]]; then
    echo "Expected at least five checkpoints in ${run_dir}, found ${#available[@]}" >&2
    exit 1
fi
declare -A selected_names=()
for numerator in 1 2 3 4 5; do
    index=$((numerator * (${#available[@]} - 1) / 5))
    checkpoint="${available[${index}]}"
    if [[ -z "${selected_names[${checkpoint}]:-}" ]]; then
        checkpoints+=("${checkpoint}")
        selected_names["${checkpoint}"]=1
    fi
done

"${project_root}/scripts/export_checkpoint_candidates.sh" \
    "${run_dir}" "${task}" "${checkpoints[@]}"

mujoco_python="${CUSTOM_DOG_MUJOCO_PYTHON:-${HOME}/miniconda3/envs/custom_dog_mujoco/bin/python}"
selective_mjcf="${run_dir}/evaluation/custom_dog_selective.xml"
python3 "${project_root}/scripts/generate_selective_mujoco.py" \
    --output "${selective_mjcf}"
"${mujoco_python}" "${project_root}/scripts/validate_mujoco_self_collision.py" \
    "${selective_mjcf}"
result_args=()
for checkpoint in "${checkpoints[@]}"; do
    label="${checkpoint%.pt}"
    candidate="${run_dir}/evaluation/candidates/${label}"
    result="${run_dir}/evaluation/${label}_recovery_${stage,,}.json"
    if "${mujoco_python}" "${project_root}/scripts/evaluate_self_righting_mujoco.py" \
        "${candidate}" \
        --stage "${stage}" \
        --mjcf "${selective_mjcf}" \
        --output "${result}" >/dev/null; then
        echo "${label}: canonical recovery gate passed"
    else
        echo "${label}: canonical recovery gate failed"
    fi
    result_args+=(--result "${label}=${result}")
done

python3 "${project_root}/scripts/select_self_righting_candidate.py" \
    "${result_args[@]}" \
    --output "${run_dir}/evaluation/self_righting_${stage,,}_selection.json"
