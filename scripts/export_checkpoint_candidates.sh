#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 RUN_DIR TASK MODEL_N.pt [MODEL_M.pt ...]" >&2
    exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="$(realpath "$1")"
task="$2"
shift 2
export_attempts="${CUSTOM_DOG_EXPORT_ATTEMPTS:-2}"
export_retry_delay="${CUSTOM_DOG_EXPORT_RETRY_DELAY:-15}"

if ! [[ "${export_attempts}" =~ ^[1-9][0-9]*$ ]]; then
    echo "CUSTOM_DOG_EXPORT_ATTEMPTS must be a positive integer." >&2
    exit 2
fi
if ! [[ "${export_retry_delay}" =~ ^[0-9]+$ ]]; then
    echo "CUSTOM_DOG_EXPORT_RETRY_DELAY must be a non-negative integer." >&2
    exit 2
fi

for checkpoint_name in "$@"; do
    checkpoint="${run_dir}/${checkpoint_name}"
    if [[ ! -f "${checkpoint}" ]]; then
        echo "Missing checkpoint: ${checkpoint}" >&2
        exit 2
    fi

    exported_policy="${run_dir}/exported/policy.onnx"
    export_succeeded=0
    for ((attempt = 1; attempt <= export_attempts; attempt++)); do
        if [[ -f "${exported_policy}" ]]; then
            export_signature_before="$(stat -c '%y:%s:%i' "${exported_policy}")"
        else
            export_signature_before=missing
        fi
        if OMNI_KIT_ACCEPT_EULA=YES \
            CUSTOM_DOG_TASK="${task}" \
            CUSTOM_DOG_PLAY_STEPS=2 \
            CUSTOM_DOG_SIM_DEVICE="${CUSTOM_DOG_SIM_DEVICE:-cuda:0}" \
            CUSTOM_DOG_RL_DEVICE="${CUSTOM_DOG_RL_DEVICE:-cuda:0}" \
                "${project_root}/scripts/play_export.sh" "${checkpoint}"; then
            export_status=0
        else
            export_status=$?
        fi

        if [[ ${export_status} -eq 0 && -f "${exported_policy}" ]]; then
            export_signature_after="$(stat -c '%y:%s:%i' "${exported_policy}")"
            if [[ "${export_signature_after}" != "${export_signature_before}" ]]; then
                export_succeeded=1
                break
            fi
        fi
        if [[ ${attempt} -lt ${export_attempts} ]]; then
            echo "Export attempt ${attempt}/${export_attempts} failed for ${checkpoint_name} " \
                "(status=${export_status}); retrying in ${export_retry_delay}s." >&2
            sleep "${export_retry_delay}"
        fi
    done
    if [[ ${export_succeeded} -ne 1 ]]; then
        echo "Failed to produce a fresh ONNX export for ${checkpoint_name}." >&2
        exit 1
    fi

    label="${checkpoint_name%.pt}"
    candidate_dir="${run_dir}/evaluation/candidates/${label}"
    install -d "${candidate_dir}/exported" "${candidate_dir}/params"
    install -m 0644 "${exported_policy}" "${candidate_dir}/exported/policy.onnx"
    install -m 0644 "${run_dir}/params/deploy.yaml" "${candidate_dir}/params/deploy.yaml"
done
