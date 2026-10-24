The energy-admissible rigid warm-start candidate is **rejected as a full-bank stability fix**. It passes local analytic controls, but a complete passive bank still violates the unchanged 0.01 J per-step energy bound. Its runtime flag, solver wiring, persistence guard and application harness are removed from normal source. The retained initializer and eight analytic fixtures compile only under `cfg(test)` in the vendor contact module. Ordinary gameplay, numerical parameters and save format remain unchanged.

The experimental initializer captured the pre-warm velocities, ran the existing cached warm start and unbiased projection, then scaled the complete current normal/tangent impulse vector for each manifold. It reconstructed physical mass/inertia at the current solver pose, included all response cross terms, retained an already dissipative guess and otherwise used the interior energy minimum along its feasible ray. Rounded physical energy, friction cones, bounds, padding and shared prescribed-motion slots were checked before committing that numerical guess. Historical accumulators, contact geometry, physical cadence and all later biased/unbiased passes were preserved. This never clamped the final physical velocity or energy.

| Full 1,040-clod / 1,045-body bank | Existing checked policy | Candidate |
|---|---:|---:|
| Known saved failing step: energy gain | 57.216923 J | 40.640364 J |
| Complete passive trajectory | 300 / 300 at 480 Hz | 300 / 300 at 480 Hz |
| Steps above unchanged 0.01 J bound | 56 | 58 |
| Maximum one-step gain | 57.216923 J | 55.070875 J |
| Net trajectory energy change | -231.618659 J | -242.883904 J |
| Full passivity gate | Fail | Fail |

The existing complete trace is the earlier checked-policy witness; this is not a matched timing benchmark. The candidate trace takes 34.874 seconds for 0.625 physical seconds. No real-time or performance improvement is established. Every accepted candidate step retains the full material stock, dynamic mass, 1,040 clods, 1,045 dynamic bodies, physical parameters and cadence. Actuator motor rows, pump work, elastic energy and engine input are zero. More net dissipation does not compensate for individual positive-energy violations.

The saved default branch reproduces both original numerical boundaries and the complete after-reference exactly. Its physical energy agrees with the phase ledger within 3.063 microjoules; candidate agreement is 1.676 microjoules. The candidate deliberately refuses persistence because its initializer has no admitted versioned policy. That guard exists in the archived experiment only; no new policy is admitted to Session saves.

| Saved-step stage kinetic change | Existing policy | Candidate |
|---|---:|---:|
| Force/gyroscopic increments | -5.998846 J | -5.507318 J |
| Warm start plus local projection | +169.204644 J | +8.706629 J |
| Biased contact passes | -110.538164 J | +33.679724 J |
| Position integration, kinetic component | +0.026548 J | +0.016611 J |
| Unbiased contact passes | -2.777754 J | -2.792947 J |

Warm-stage work falls by 160.498014 J while biased-contact work rises by 144.217888 J. The initialization largely changes when correction momentum is supplied; it does not establish warm history as the root cause. The table excludes gravitational position work, which is retained in the detailed phase evidence. The candidate's individual generic-contact work rows were not retained, so its generic contribution is unmeasured; baseline generic dissipation must not be used to infer that sign.

Next close positive **rigid biased-contact blocks** against current physical mass and actual applied impulse response. Separate normal and tangential-anchor work, rank the largest positive blocks and connect them to their gap, bias target and independent geometric owner. The baseline geometry audit left 374 clustered manifolds outside its primitive certification. Neither zero certified separated pairs nor the candidate's altered path certifies those clustered unions. Do not call this unavoidable genuine-penetration work until that ownership and physical gap are independently checked. Keep all contacts, stocks, physical cadence, bias correction and the original penetration/energy bounds. Loaded digging and timestep refinement remain required after passive stability.

The eight final vendor controls cover the exact 0.1875 J transient projection counterexample from rest, rejection/interior/unchanged acceptance, mixed lanes and multiple points, padding, prescribed motion, a shared kinematic slot, pre-gather mapping/bounds checks and current-pose anisotropic mass. The first run caught a fixture error: body insertion resets its internal solver IDs. That fixture was corrected and rerun; its failed log is preserved. The rejected runtime prototype also hit a vendor lint finding on a constant assertion; that wiring is removed, not reported as a lint pass. Final checkpoint verification is recorded in the receipt.

Reproduce the isolated analytic controls with one compiler job:

```bash
cargo +nightly test --manifest-path vendor/rapier3d/Cargo.toml \
  --features contact-work-audit,enhanced-determinism --lib rigid_warmstart_admissibility \
  --profile playable --locked --jobs 1 -- --test-threads=1 --nocapture
```

To reproduce the **rejected** complete experiment, use a disposable checkout of this checkpoint and apply [the exact patch](experiments/rigid-warmstart-20261024.patch). It includes the temporary wiring, save refusal and ignored full-bank harness. `git apply --check` passes against the final test-only source. Existing numerical inputs remain at `contact-geometry-20261024/fresh-verified-final/worst-before.bin` and `worst-after.bin` under the local artifact root. Regenerate them using the full checked witness in [the contact-work audit](contact-work-audit.md) if unavailable.

```bash
git apply docs/physics/experiments/rigid-warmstart-20261024.patch
SLUICE_ENERGY_BOUNDARY=/path/to/worst-before.bin \
SLUICE_ENERGY_REFERENCE=/path/to/worst-after.bin \
SLUICE_WARMSTART_TRIAL=/tmp/fresh-warm-trial.json \
  cargo +nightly test -p mining-sim --features contact-work-audit --lib \
  saved_bank_energy_admissible_warmstart_trial --profile playable --locked --jobs 1 \
  -- --ignored --test-threads=1 --nocapture
SLUICE_WARMSTART_TRAJECTORY=/tmp/fresh-warm-trajectory.json \
  cargo +nightly test -p mining-sim --features contact-work-audit --lib \
  complete_bank_energy_admissible_warmstart_trajectory --profile playable --locked --jobs 1 \
  -- --ignored --test-threads=1 --nocapture
```

Diagnostic command success means its observations are valid. It does not mean the separate physical gate passed. The root application uses its tracked `wide 1.7.1` lock; standalone vendor unit controls use their tracked `wide 1.6.1` lock. The [receipt](../playtests/2026-10-24-rigid-warmstart.json) preserves complete compact saved-step phases and all 300 candidate energy/inventory observations, source/input hashes and final checks. The latest signed crest/river build is preserved; no new native or FPS claim is made for this test-only checkpoint.

Final verification: eight test-only vendor controls and23 ordinary library tests pass, plus the explicitly executed exact legacy replay. Scoped vendor library/test Clippy passes with only its two existing `let_and_return` findings allowed. Both changed Rust files pass final formatting; the last cleanup reorders an existing import without changing behavior. Full application/soil/gameplay suites and new native rendering are not claimed. Cleanup leaves no owned app/build/encoder or test-browser process; shared services remain untouched and no new panic-full/WindowServer report appears. The original dirty checkout, stash, input boundaries and signed render build are intact. Only27 established regenerable test executables were removed for disk headroom;15 old byte-identical generated comparison copies were replaced with independent APFS clones without changing paths or bytes. No safety configuration or personal browser was touched. Installed Beads rejects `bd sync`; touched project records are exported explicitly before committing.
