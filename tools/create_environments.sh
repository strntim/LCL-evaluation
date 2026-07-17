#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
localize_environment="${root}/implementations/localize/install/environment.yaml"

if ! command -v mamba >/dev/null 2>&1; then
    echo "mamba is required but was not found on PATH." >&2
    exit 1
fi

if [[ ! -f "${localize_environment}" ]]; then
    echo "LOCALIZE is not installed: ${localize_environment}" >&2
    exit 1
fi

environment_exists() {
    mamba env list | awk -v expected="$1" '$1 == expected { found = 1 } END { exit !found }'
}

create_environment() {
    local name="$1"
    local file="$2"

    if environment_exists "${name}"; then
        echo "Environment already exists: ${name}"
        return
    fi
    mamba env create -f "${file}"
}

create_environment nancy "${localize_environment}"
create_environment jupyter "${root}/implementations/jupyter/environment.yaml"
mamba run -n jupyter python -m ipykernel install --user --name jupyter --display-name "Python (jupyter)"
create_environment kedro "${root}/implementations/kedro/environment.yaml"
