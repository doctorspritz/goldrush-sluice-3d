#!/usr/bin/env bash
set -euo pipefail
task_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_repo"
task_git="$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
task_target="${SLUICE_TARGET_DIR:-${task_git%/.git}/target}"
if [[ -z "$task_git" ]]; then task_target="${SLUICE_TARGET_DIR:-$task_repo/target}"; fi
task_toolchain="stable"
if rustup run nightly rustc --version >/dev/null 2>&1; then task_toolchain="nightly"; fi
cargo "+$task_toolchain" build -p mining --profile playable --jobs "${SLUICE_BUILD_JOBS:-1}" --locked --target-dir "$task_target"
exec "$task_target/playable/mining" "$@"
