#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
install="${root}/implementations/localize/install"

if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: $0 [LOCALIZE_REPOSITORY]"
    exit 0
fi

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [LOCALIZE_REPOSITORY]" >&2
    exit 1
fi

if [[ ! -d "${install}" ]]; then
    if [[ $# -eq 1 ]]; then
        bash "${root}/tools/install_localize.sh" "$1"
    else
        bash "${root}/tools/install_localize.sh"
    fi
else
    echo "LOCALIZE install already exists: ${install}"
fi

bash "${root}/tools/create_environments.sh"
bash "${root}/tools/install_kedro.sh"
bash "${root}/tools/link_resources.sh"
