#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 -m unittest discover -s "${project_root}/tests" -v
python3 -m py_compile \
    "${project_root}/rl/src/custom_dog_rl/assets/custom_dog.py" \
    "${project_root}/rl/src/custom_dog_rl/tasks/locomotion/velocity_env_cfg.py" \
    "${project_root}/rl/src/custom_dog_rl/agents/ppo_cfg.py" \
    "${project_root}/rl/scripts/train.py" \
    "${project_root}/rl/scripts/play.py"

echo "Static validation passed."
