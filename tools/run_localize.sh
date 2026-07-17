#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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

run_sequential() {
    local config pipeline
    for config in "${install}/configs"/*; do
        pipeline="$(basename "${config}")"
        if [[ "${pipeline}" == "Benchmarking" || "${pipeline}" == "Scalability" || ! -f "${config}/dvc.yaml" ]]; then
            continue
        fi
        run_pipeline "${pipeline}" "$@"
    done

    run_pipeline Benchmarking "$@"
}

if [[ $# -eq 0 ]]; then
    run_sequential
    exit 0
fi

if [[ "${1}" == "-p" || "${1}" == "--parallel" ]]; then
    echo "Parallel LOCALIZE runs are not supported." >&2
    exit 1
fi

if [[ "${1}" == "--help" ]]; then
    echo "Usage: $0 [DVC_REPRO_OPTIONS...] | $0 PIPELINE [DVC_REPRO_OPTIONS...]"
    echo "       $0 Scalability --scale {1|5|10} [DVC_REPRO_OPTIONS...]"
    exit 0
fi

if [[ "${1}" == -* ]]; then
    run_sequential "$@"
    exit 0
fi

pipeline="$1"
shift
if [[ ! -f "${install}/configs/${pipeline}/dvc.yaml" ]]; then
    echo "Pipeline not found: ${pipeline}" >&2
    exit 1
fi

if [[ "${pipeline}" == "Scalability" ]]; then
    scale=""
    options=()
    while [[ $# -gt 0 ]]; do
        if [[ "${1}" == "--scale" ]]; then
            if [[ $# -lt 2 ]]; then
                echo "--scale requires a value." >&2
                exit 1
            fi
            scale="$2"
            shift 2
            continue
        fi
        options+=("$1")
        shift
    done
    if [[ "${scale}" != "1" && "${scale}" != "5" && "${scale}" != "10" ]]; then
        echo "Scalability requires --scale 1, 5, or 10." >&2
        exit 1
    fi
    SCALE_FACTOR="${scale}" run_pipeline "${pipeline}" "${options[@]}"
    exit 0
fi

run_pipeline "${pipeline}" "$@"
