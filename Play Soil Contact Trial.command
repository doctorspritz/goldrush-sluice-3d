#!/bin/bash
set -euo pipefail
task_repo="$(cd "$(dirname "$0")" && pwd)"
cd "$task_repo"
export SLUICE_SAVE_PATH="${SLUICE_SAVE_PATH:-$task_repo/saves/soil-contact-trial.json}"
exec bash scripts/play-mining.sh --new-shift --articulated-machine --active-soil "$@"
