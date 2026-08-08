#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${project_root}/scripts/activate_env.sh"

for dependency_dir in "${ISAACLAB_ROOT}" "${UNITREE_RL_LAB_ROOT}" "${UNITREE_MUJOCO_ROOT}"; do
    if [[ ! -d "${dependency_dir}" ]]; then
        echo "Missing dependency directory: ${dependency_dir}" >&2
        exit 1
    fi
done

if [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
    /usr/lib/wsl/lib/nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "WARNING: WSL nvidia-smi is unavailable."
fi

python - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
else:
    raise SystemExit("CUDA is required for the configured PPO device.")
PY
