#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
datasets="${root}/datasets"
install="${root}/implementations/localize/install"
configs="${root}/implementations/localize/configs"

if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: $0"
    echo "Create the shared dataset and benchmarking symlinks."
    exit 0
fi

if [[ $# -ne 0 ]]; then
    echo "Usage: $0" >&2
    exit 1
fi

link() {
    local target="$1"
    local source="$2"

    if [[ -e "${target}" && ! -L "${target}" ]]; then
        echo "Cannot replace non-symlink: ${target}" >&2
        exit 1
    fi
    ln -sfn "${source}" "${target}"
}

if [[ ! -d "${datasets}" ]]; then
    echo "Datasets not found: ${datasets}" >&2
    exit 1
fi

link "${root}/implementations/jupyter/datasets" "../../datasets"
link "${root}/implementations/kedro/datasets" "../../datasets"
link "${root}/implementations/kedro/src/benchmarking" "../../../src/benchmarking"

if [[ ! -d "${install}" ]]; then
    exit 0
fi

link "${install}/src/benchmarking" "../../../../src/benchmarking"

for config in "${configs}"/*; do
    dvc_file="${config}/dvc.yaml"
    [[ -f "${dvc_file}" ]] || continue

    archive="$(sed -nE 's|.*raw/([^[:space:]]+\.zip).*|\1|p' "${dvc_file}" | head -n 1)"
    [[ -n "${archive}" ]] || continue

    dataset="${archive%.zip}"
    source="${datasets}/${dataset}/${archive}"
    if [[ ! -f "${source}" ]]; then
        echo "Dataset archive not found: ${source}" >&2
        exit 1
    fi

    artifacts="${install}/artifacts/$(basename "${config}")"
    mkdir -p \
        "${artifacts}/data/raw" \
        "${artifacts}/data/base" \
        "${artifacts}/data/interim" \
        "${artifacts}/data/prepared" \
        "${artifacts}/data/splits" \
        "${artifacts}/models" \
        "${artifacts}/reports"
    raw="${artifacts}/data/raw"
    link "${raw}/${archive}" "../../../../../../../datasets/${dataset}/${archive}"
done
