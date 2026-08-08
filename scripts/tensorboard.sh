#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${project_root}/scripts/activate_env.sh"

exec tensorboard \
    --logdir "${project_root}/logs/rsl_rl/custom_dog_velocity" \
    --host "${TENSORBOARD_HOST:-0.0.0.0}" \
    --port "${TENSORBOARD_PORT:-6006}"
