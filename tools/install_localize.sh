#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_repo="${1:-https://github.com/SensorLab/LOCALIZE.git}"
install="${root}/implementations/localize/install"

if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: $0 [LOCALIZE_REPOSITORY]"
    echo "Clone LOCALIZE and prepare it for this evaluation."
    exit 0
fi

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [LOCALIZE_REPOSITORY]" >&2
    exit 1
fi

if [[ -e "${install}" ]]; then
    echo "LOCALIZE install already exists: ${install}" >&2
    exit 1
fi

if [[ -d "${source_repo}" ]]; then
    git clone --no-hardlinks "${source_repo}" "${install}"
else
    git clone "${source_repo}" "${install}"
fi

rsync -a --delete "${root}/implementations/localize/configs/" "${install}/configs/"
