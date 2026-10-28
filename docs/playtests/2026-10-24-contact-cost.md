The active-soil mechanics remain far below real time. Profiling the actual 1,040-clod bank places about 94% of sampled main-thread work inside `PhysicsWorld::step_with_events`. Validation, world construction and save serialization are small in this sample. This checkpoint adds a reproducible CPU command recorder/replayer and preserves the measured negative optimization results. **No faster solver or compiler change is shipped.**

The five-second sample targeted only the task-owned process. One simulated second of observational counters reaches 4,128,768 top-level contact calls, including 2,890,027 six-vertex hull pairs; 817,425 have a face-distance bound beyond prediction. Those are the last printed cumulative counters, not exact end-of-window totals. A rejection trial changes contact history: the one-second endpoint has 13,385 differing serialized leaves, including contact count 6,544 → 6,609 and normal-force feedback 112,441.47 → 112,294.26 N. It also establishes no gain. The trial is reverted.

The tangent-call observation reaches 11,534,336 calls, of which 7,674,268 have both zero admissible radius and zero stored impulse. Skipping that update passes 86 candidate contact unit tests, independent warm-start/two-mass/mixed-lane controls, and complete idle-state comparisons. Alternating timing runs establish no reliable benefit. A separate compiler optimization-level-3 trial also retains the complete idle states but has no reliable benefit. Both are reverted; every one of the 740 previously tracked Rust/shader/build sources remains byte-identical to `dc225bb`.

| CPU step seconds for one simulated second, median of two alternating runs; serialization excluded | Time |
|---|---:|
| Original engine | 12.35046 s |
| Inactive-friction shortcut | 12.31732 s |
| Higher contact compiler optimization | 12.37137 s |

The shortcut's 0.27% difference is smaller than observed variation and its paired direction reverses. The higher compiler setting is 0.17% slower by this measure. No throughput win is claimed. All 24 complete state snapshots across these six runs match the pristine reference. Neither result establishes native FPS, real-time gameplay or physical calibration.

The maintained [`soil_contact_replay` example](../../crates/mining-sim/examples/soil_contact_replay.rs) records the complete initial save and every actuator command. Replay uses that fixed tape, with the original 240 Hz mechanics and 60 Hz control boundaries, and writes full canonical snapshots each simulated second. It rejects runs beyond 3,600 ticks and refuses existing output directories. The compressed [18-second fixture](fixtures/2026-10-24-soil-stroke.commands.json.gz) preserves 1,080 commands; its starting save also preserves the physical configuration. This harness runs CPU mechanics with static terrain, and does not update GPU terrain/water or represent manual OS input.

The final harness reproduces **all 19 complete reference states**, retains all 1,040 clods and identical material inventory, reaches the same 4.2187501676 L centroid-based held-load estimate, and empties completely. The reference executable is a statically linked observational build with unchanged contact equations; its idle snapshots also match the pristine pre-trial executable. The final full CPU replay takes 168.85 seconds of stepping for 18.00000094 simulated seconds. That separate episode is not a paired speedup measurement or a full native benchmark. Held volume is an existing geometric estimate, not a calibrated bucket-yield measurement.

Record/replay roundtrip, compressed-fixture input, existing-output preservation and tick-bound refusal execute successfully. Strict scoped example Clippy, formatting and build pass. The negative guard checks intentionally exit 1 without overwriting earlier evidence. Full-workspace checks, a new native visual review and new native performance measurements are not claimed; game/renderer sources are unchanged. The [receipt](2026-10-24-contact-cost.json) records hashes, counters, complete-state digests and timing results.

Reproduce from the repository root, choosing a fresh output directory:

```bash
CARGO_TARGET_DIR="$(git rev-parse --path-format=absolute --git-common-dir)"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR%/.git}/target"
cargo +nightly build -p mining-sim --example soil_contact_replay --profile playable --locked --jobs 1
gzip -dc docs/playtests/fixtures/2026-10-24-soil-stroke.commands.json.gz | "$CARGO_TARGET_DIR/playable/examples/soil_contact_replay" replay - /tmp/soil-contact-replay-new
```

For a new tape, use `soil_contact_replay record idle|stroke OUTPUT_DIR TICKS`. Sampling and compiler/profiling jobs run sequentially; the example itself launches no browser, renderer or background worker. Scoped cleanup leaves zero owned browser/test/game jobs, with 14 shared Playwright MCP and 56 Node processes untouched. No new panic-full/WindowServer diagnostic appeared. The original dirty checkout and user stash remain preserved.

`sluice-contact-history-20261024` now tracks investigation of persistent contact/world state, including explicit saved numerical history, independent contact/energy controls, loaded timestep convergence, old-save compatibility and measured native throughput. Persistent warm starts have not been proven equivalent to the rebuilt model and are not enabled here. Real-time physics, calibrated bulk digging, wet soil and all remaining operation stages stay open.
