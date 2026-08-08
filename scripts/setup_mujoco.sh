#!/usr/bin/env bash
set -euo pipefail

conda_exe="${CONDA_EXE:-/home/hsc/miniconda3/bin/conda}"
env_name="${CUSTOM_DOG_MUJOCO_ENV:-custom_dog_mujoco}"

if [[ ! -x "${conda_exe}" ]]; then
    echo "Conda executable not found: ${conda_exe}" >&2
    exit 1
fi

if ! "${conda_exe}" env list | awk '{print $1}' | grep -Fx "${env_name}" >/dev/null; then
    "${conda_exe}" create -y -n "${env_name}" python=3.10 pip
fi

"${conda_exe}" run -n "${env_name}" python -m pip install \
    "mujoco==3.11.0" \
    "numpy==2.2.6" \
    "onnxruntime==1.23.2" \
    "PyYAML>=6.0,<7" \
    "urdf-to-mjcf==0.1.1"

"${conda_exe}" run --no-capture-output -n "${env_name}" python -c \
    'import mujoco, numpy, onnxruntime, yaml; print(f"MuJoCo: {mujoco.__version__}"); print(f"NumPy: {numpy.__version__}"); print(f"ONNX Runtime: {onnxruntime.__version__}"); print(f"PyYAML: {yaml.__version__}")'
