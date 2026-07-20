#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project="${root}/implementations/kedro"

if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: $0"
    echo "Install the evaluation Kedro project into the kedro environment."
    exit 0
fi

if [[ $# -ne 0 ]]; then
    echo "Usage: $0" >&2
    exit 1
fi

if ! mamba run -n kedro python -c "import sys" >/dev/null 2>&1; then
    echo "The kedro environment does not exist. Run tools/create_environments.sh first." >&2
    exit 1
fi

mamba run -n kedro python -m pip install --no-build-isolation --no-deps --editable "${project}"
