---
title: "FLIP end-of-step divergence re-injection + hydrostatic column churn"
category: simulation-issues
component: game/gpu/flip_3d
status: unresolved-characterized
severity: high
symptoms:
  - test_flip_divergence_free fails (post-correction divergence 0.67 vs 0.01 threshold)
  - test_hydrostatic_equilibrium fails (max velocity pinned at 1.0 m/s clamp)
  - test_dam_break_scenario fails (water drifts wrong direction)
  - static hydrostatic water column slowly inflates upward
tags: [flip, pressure, divergence, density-projection, free-surface, investigation]
date: 2026-06-05
---

# FLIP end-of-step divergence re-injection + hydrostatic column churn

Characterized, NOT fixed. Read this before touching the pressure solver — it
rules out two plausible-but-wrong "fixes" that were each disproven by verification.

## Verified facts (evidence-backed)

1. **The pressure solve is correct.** Instrumenting `step_internal` to call
   `compute_post_correction_divergence` immediately after the pressure+gradient
   submit shows **max_div ≈ 0.00001 every step**. The projection drives the
   field divergence-free right after the gradient.

2. **End-of-step divergence is 0.67.** The same function, called after `step()`
   returns (as the tests do), reports **0.670224**. So something between
   "just after the gradient" and "end of step" leaves the field divergent — a
   real re-injection. (`headless_harness.rs::test_flip_divergence_free`.)

3. **The post-pressure BC pass is NOT the re-injector.** Instrumentation right
   after the post-pressure BC submit still showed ≈0.00001. The only stages after
   that for these configs are porosity-drag (skipped when `sediment_porosity_drag==0`)
   and G2P (reads grid, does not write it). The exact mechanism that turns
   1e-5 into 0.67 was NOT identified by static analysis.

4. **Hydrostatic churn = upward column inflation.** In `test_hydrostatic_equilibrium`
   (`configure_for_hydrostatic_equilibrium`, pure PIC `flip_ratio=0`, density
   projection on, open +Y top): the free surface rises Y 0.50→0.80, center-of-mass
   rises, vel_max climbs to the 1.0 clamp, pre-pressure divergence grows to ~5-9,
   while **volume is perfectly conserved (0.0000%)**. The column expands up the open
   boundary — a density-projection / free-surface interaction, not the pressure solve.

## Ruled out (do NOT re-attempt)

- **"Pressure stencil divides by 6 instead of non-solid face count."** FALSE.
  `pressure_3d.wgsl` returns the cell's own pressure for solid (Neumann) neighbors
  and `6 - n_solid = n_nonsolid`, so the `/6` form is algebraically identical to
  `/n_nonsolid` at the fixed point. Changing it is a no-op or a regression.
  Confirmed by fact (1): the solve already reaches 1e-5.

- **"The divergence test asserts a stale buffer (test bug)."** FALSE/incomplete.
  The test was changed to assert `post_div` (the re-run value) and it STILL fails
  at 0.67 — so end-of-step divergence is genuinely 0.67, not a measurement artifact.

## Correct next step (needs a different tool)

Static code-reading hit its limit. Find the re-injection empirically:
add a per-stage buffer-dump harness (or RenderDoc) that records max |divergence|
after EVERY stage of `step_internal` (P2G → density-projection → gravity →
pre-BC → pressure → post-BC → porosity-drag → G2P → sediment-projection), for the
divergence-free config. The stage where 1e-5 becomes 0.67 is the culprit. Only then
propose a fix. Do not patch the pressure solver — fact (1) proves it is not the bug.
