#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${EVALUATION_RUNNER_IN_ENV:-}" != "1" ]]; then
    exec mamba run -n jupyter env EVALUATION_RUNNER_IN_ENV=1 bash "$0" "$@"
fi

exec python "${root}/tools/run_evaluation.py" "$@"
