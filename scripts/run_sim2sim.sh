#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda_exe="${CONDA_EXE:-/home/hsc/miniconda3/bin/conda}"
env_name="${CUSTOM_DOG_MUJOCO_ENV:-custom_dog_mujoco}"

exec "${conda_exe}" run --no-capture-output -n "${env_name}" \
    python "${project_root}/sim2sim/custom_dog/run_sim2sim.py" "$@"
