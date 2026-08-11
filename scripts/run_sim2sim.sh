#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda_exe="${CONDA_EXE:-$(command -v conda 2>/dev/null || true)}"
env_name="${CUSTOM_DOG_MUJOCO_ENV:-custom_dog_mujoco}"

if [[ -z "${conda_exe}" ]]; then
    echo "Conda was not found. Set CONDA_EXE or run setup_ubuntu2204.sh." >&2
    exit 1
fi

exec "${conda_exe}" run --no-capture-output -n "${env_name}" \
    python "${project_root}/sim2sim/custom_dog/run_sim2sim.py" "$@"
