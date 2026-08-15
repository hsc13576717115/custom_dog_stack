#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 SERVICE COMMAND [ARGUMENT ...]" >&2
    exit 2
fi

dependency="$1"
shift
while systemctl --user is-active --quiet "${dependency}"; do
    sleep 15
done

result="$(systemctl --user show "${dependency}" --property=Result --value)"
if [[ "${result}" != "success" ]]; then
    echo "Dependency ${dependency} did not succeed (Result=${result})." >&2
    exit 1
fi

exec "$@"
