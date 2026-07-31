#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v unzip >/dev/null 2>&1; then
    echo "unzip is required but was not found on PATH." >&2
    exit 1
fi

extract() {
    local archive="$1"
    shift

    for expected in "$@"; do
        [[ -f "${expected}" ]] || {
            unzip -q -n "${archive}" -d "$(dirname "${archive}")"
            return
        }
    done
}

extract "${root}/datasets/umu/umu.zip" \
    "${root}/datasets/umu/umu/tcp_nokia_20240325.xlsx"
extract "${root}/datasets/logatec/logatec.zip" \
    "${root}/datasets/logatec/spring_data.json" \
    "${root}/datasets/logatec/winter_data.json"
