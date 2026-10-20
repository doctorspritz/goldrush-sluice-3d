#!/usr/bin/env bash
# Build a local app bundle with embedded fonts and procedural scene assets.
set -euo pipefail
if [[ "$(uname -s)" != Darwin ]]; then
  echo "This packager requires macOS." >&2
  exit 1
fi
task_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_repo"
task_git="$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
task_target="${SLUICE_TARGET_DIR:-${task_git%/.git}/target}"
if [[ -z "$task_git" ]]; then task_target="${SLUICE_TARGET_DIR:-$task_repo/target}"; fi
task_toolchain=stable
if rustup run nightly rustc --version >/dev/null 2>&1; then task_toolchain=nightly; fi
cargo "+$task_toolchain" build -p mining --profile playable --jobs "${SLUICE_BUILD_JOBS:-1}" --locked --target-dir "$task_target"
task_output="${1:-$task_target/bundles}"
task_app="$task_output/First Bend.app"
mkdir -p "$task_app/Contents/MacOS"
cp "$task_target/playable/mining" "$task_app/Contents/MacOS/mining"
chmod +x "$task_app/Contents/MacOS/mining"
cat > "$task_app/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleExecutable</key><string>mining</string>
<key>CFBundleIdentifier</key><string>com.doctorspritz.sluicemoose.firstbend</string>
<key>CFBundleName</key><string>First Bend</string>
<key>CFBundleDisplayName</key><string>The Sluice Moose — First Bend</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>0.1.0</string>
<key>CFBundleVersion</key><string>1</string>
<key>LSMinimumSystemVersion</key><string>11.0</string>
<key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
codesign --force --sign - "$task_app"
codesign --verify --deep --strict "$task_app"
printf '%s\n' "$task_app"
