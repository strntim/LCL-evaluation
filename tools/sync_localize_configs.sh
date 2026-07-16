#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_configs="${root}/implementations/localize/configs"
install="${root}/implementations/localize/install"

for config in "${install}/configs"/*; do
    [[ -d "${config}" ]] || continue
    name="$(basename "${config}")"
    [[ -d "${source_configs}/${name}" ]] && continue
    rm -rf "${install}/artifacts/${name}"
done

rsync -a --delete "${source_configs}/" "${install}/configs/"
