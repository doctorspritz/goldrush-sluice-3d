# Flowing-water demo and controls

Approved scope: blend actual FLIP particles into a coherent rendered water surface, repair awkward controls and open the demo with paydirt already flowing. Continue codex/direct-mining without touching the original dirty checkout or changing the physical stages.

- [x] Render particle depth, bilateral smoothing and a single water composite with scene occlusion; retain detached droplets and bounded reusable buffers.
- [x] Align movement/cab turning with camera screen directions, focus/cursor/help handling and movement collision; exercise the actual Bevy input path.
- [x] Generate a finite solver-warmed demo save and embed it; default the demo app to that state and provide a fresh-shift option.
- [x] Sequential targeted tests, strict lint/build, native screenshots/control replay, conservation and live restore.
- [x] Package the app, scoped cleanup, update Beads/evidence, commit, push and verify backup/remote.

Only one heavy workload runs at a time and builds use --jobs 1. New WindowServer/panic reports stop further heavy work; existing reports are preserved. Personal browsers and workstation safety configuration are outside scope.

Verified result: 57.43 render FPS with 124,168 live water particles, 22 CPU regressions and 18 synthetic native input checks. Evidence: docs/playtests/2026-10-20-flowing-demo.md. Physics throughput/calibration and legacy CI formatting remain separately tracked.
