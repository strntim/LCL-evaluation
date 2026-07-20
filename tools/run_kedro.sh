#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project="${root}/implementations/kedro"
experiments=(
    00-Initial
    01-Changed_and_added_model
    02-Changed_dataset_to_logatec
    03-Added_split_and_metric
    04-Added_automl_model
    Benchmarking
)

if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: $0 [EXPERIMENT [KEDRO_RUN_OPTIONS...]]"
    echo "       $0 [KEDRO_RUN_OPTIONS...]"
    exit 0
fi

if [[ "${KEDRO_RUNNER_IN_ENV:-}" != "1" ]]; then
    bash "${root}/tools/link_resources.sh"
    exec mamba run -n kedro env \
        KEDRO_RUNNER_IN_ENV=1 \
        KEDRO_DISABLE_TELEMETRY=1 \
        PYTHONHASHSEED=42 \
        TF_DETERMINISTIC_OPS=1 \
        bash "$0" "$@"
fi

pipeline_for() {
    echo "$1"
}

run_experiment() {
    local experiment="$1"
    shift
    local pipeline
    pipeline="$(pipeline_for "${experiment}")"
    cd "${project}"
    if [[ "${experiment}" == "Benchmarking" ]]; then
        KEDRO_BENCHMARK_DIR="${project}/artifacts/Benchmarking/reports/usage" \
            kedro run --env "${experiment}" --pipelines "${pipeline}" "$@"
    else
        kedro run --env "${experiment}" --pipelines "${pipeline}" "$@"
    fi
}

known_experiment() {
    local candidate="$1"
    local experiment
    for experiment in "${experiments[@]}"; do
        [[ "${candidate}" == "${experiment}" ]] && return 0
    done
    return 1
}

if [[ $# -eq 0 || "${1}" == -* ]]; then
    options=("$@")
    for experiment in "${experiments[@]}"; do
        run_experiment "${experiment}" "${options[@]}"
    done
    exit 0
fi

experiment="$1"
shift
if ! known_experiment "${experiment}"; then
    echo "Unknown Kedro experiment: ${experiment}" >&2
    exit 1
fi
run_experiment "${experiment}" "$@"
