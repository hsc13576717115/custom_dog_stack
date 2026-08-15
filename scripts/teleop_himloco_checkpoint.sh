#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
himloco_root="${CUSTOM_DOG_HIMLOCO_ROOT:-${project_root}/../himloco_lab}"
run_dir="${1:-${himloco_root}/logs/himloco_rsl_rl/go2_rough/2026-08-15_14-38-13_custom_dog_default}"
run_dir="$(realpath "${run_dir}")"
checkpoint="${2:-}"

if [[ ! -d "${run_dir}" || ! -f "${run_dir}/params/deploy.yaml" ]]; then
    echo "HimLoco run must contain params/deploy.yaml: ${run_dir}" >&2
    exit 2
fi

if [[ -n "${checkpoint}" ]]; then
    checkpoint="$(realpath "${checkpoint}")"
else
    latest_iteration=-1
    shopt -s nullglob
    for candidate in "${run_dir}"/model_*.pt; do
        filename="${candidate##*/}"
        iteration="${filename#model_}"
        iteration="${iteration%.pt}"
        if [[ "${iteration}" =~ ^[0-9]+$ ]] && (( iteration > latest_iteration )); then
            latest_iteration="${iteration}"
            checkpoint="${candidate}"
        fi
    done
    shopt -u nullglob
fi

if [[ -z "${checkpoint}" || ! -f "${checkpoint}" ]]; then
    echo "No HimLoco model_*.pt checkpoint found in ${run_dir}" >&2
    exit 2
fi

conda_exe="${CONDA_EXE:-$(command -v conda 2>/dev/null || true)}"
if [[ -z "${conda_exe}" && -x "${HOME}/miniconda3/bin/conda" ]]; then
    conda_exe="${HOME}/miniconda3/bin/conda"
fi
if [[ -z "${conda_exe}" ]]; then
    echo "Conda was not found. Set CONDA_EXE." >&2
    exit 1
fi

echo "Exporting HimLoco checkpoint: ${checkpoint}"
(
    cd "${himloco_root}"
    OMNI_KIT_ACCEPT_EULA=YES "${conda_exe}" run --no-capture-output -n "${CUSTOM_DOG_ISAAC_ENV:-env_isaaclab}" \
        python scripts/himloco_rsl_rl/play.py \
        --task Unitree-Go2-Velocity \
        --checkpoint "${checkpoint}" \
        --num_envs 1 \
        --headless \
        --export-only
)

exec "${project_root}/scripts/teleop_mujoco_policy.sh" "${run_dir}"
