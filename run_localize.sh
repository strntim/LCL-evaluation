#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
install="${root}/implementations/localize/install"

if [[ "${LOCALIZE_RUNNER_IN_ENV:-}" != "1" ]]; then
    if [[ ! -d "${install}" ]]; then
        echo "LOCALIZE install not found: ${install}" >&2
        exit 1
    fi

    bash "${root}/tools/sync_localize_configs.sh"
    bash "${root}/tools/link_resources.sh"
    exec mamba run -n nancy env LOCALIZE_RUNNER_IN_ENV=1 bash "$0" "$@"
fi

run_pipeline() {
    local pipeline="$1"
    shift
    cd "${install}/configs/${pipeline}"
    dvc repro "$@"
}

if [[ "${1:-}" == "--all" ]]; then
    shift
    pids=()
    for config in "${install}/configs"/*; do
        pipeline="$(basename "${config}")"
        if [[ "${pipeline}" == "Benchmarking" || ! -f "${config}/dvc.yaml" ]]; then
            continue
        fi
        (run_pipeline "${pipeline}" "$@") &
        pids+=("$!")
    done

    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            failed=1
        fi
    done
    if [[ "${failed}" -ne 0 ]]; then
        exit 1
    fi

    run_pipeline Benchmarking "$@"
    exit 0
fi

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 PIPELINE [DVC_REPRO_OPTIONS...] | $0 --all [DVC_REPRO_OPTIONS...]" >&2
    exit 1
fi

pipeline="$1"
shift
if [[ ! -f "${install}/configs/${pipeline}/dvc.yaml" ]]; then
    echo "Pipeline not found: ${pipeline}" >&2
    exit 1
fi

run_pipeline "${pipeline}" "$@"
