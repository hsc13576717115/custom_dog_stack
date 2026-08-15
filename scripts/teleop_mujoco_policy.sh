#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
candidate="${1:-${project_root}/deploy/candidates/model_800_omni_stability_calibrated}"
candidate="$(realpath "${candidate}")"

if [[ ! -f "${candidate}/exported/policy.onnx" || ! -f "${candidate}/params/deploy.yaml" ]]; then
    echo "Candidate must contain exported/policy.onnx and params/deploy.yaml: ${candidate}" >&2
    exit 2
fi

initial_state="${CUSTOM_DOG_INITIAL_STATE:-home}"
recovery_ramp="${CUSTOM_DOG_RECOVERY_RAMP:-2.0}"
recovery_hold="${CUSTOM_DOG_RECOVERY_HOLD:-1.0}"
camera_mode="${CUSTOM_DOG_CAMERA_MODE:-tracking}"

case "${initial_state}" in
    home|prone) ;;
    *)
        echo "CUSTOM_DOG_INITIAL_STATE must be home or prone" >&2
        exit 2
        ;;
esac

case "${camera_mode}" in
    tracking|free) ;;
    *)
        echo "CUSTOM_DOG_CAMERA_MODE must be tracking or free" >&2
        exit 2
        ;;
esac

routed_args=()
encoder_args=()
if [[ -f "${candidate}/exported/encoder.onnx" ]]; then
    encoder_args=(--encoder "${candidate}/exported/encoder.onnx")
fi
if [[ -f "${candidate}/exported/stand_policy.onnx" || -f "${candidate}/params/stand_deploy.yaml" ]]; then
    if [[ ! -f "${candidate}/exported/stand_policy.onnx" || ! -f "${candidate}/params/stand_deploy.yaml" ]]; then
        echo "Routed candidate must contain both stand_policy.onnx and stand_deploy.yaml" >&2
        exit 2
    fi
    routed_args=(
        --stand-policy "${candidate}/exported/stand_policy.onnx"
        --stand-deploy-yaml "${candidate}/params/stand_deploy.yaml"
    )
fi

exec "${project_root}/scripts/run_sim2sim.sh" \
    --policy "${candidate}/exported/policy.onnx" \
    "${encoder_args[@]}" \
    --deploy-yaml "${candidate}/params/deploy.yaml" \
    "${routed_args[@]}" \
    --command 0.0 0.0 0.0 \
    --initial-state "${initial_state}" \
    --recovery-ramp "${recovery_ramp}" \
    --recovery-hold "${recovery_hold}" \
    --duration 0 \
    --viewer \
    --camera-mode "${camera_mode}" \
    --interactive
