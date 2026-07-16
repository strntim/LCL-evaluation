#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec mamba run -n jupyter python "${root}/implementations/jupyter/run_notebooks.py" "$@"
