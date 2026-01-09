# FLIP/Grid Module Refactor Plan

## Documentation Index

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | This file - overview and phase descriptions |
| [METHOD_INVENTORY.md](METHOD_INVENTORY.md) | Explicit method-to-module mappings for each agent |
| [PARALLEL_PROTOCOL.md](PARALLEL_PROTOCOL.md) | Git workflow, conflict prevention, communication |
| [PHASE0_SCAFFOLDING.md](PHASE0_SCAFFOLDING.md) | Step-by-step scaffolding instructions |
| [STATUS.md](STATUS.md) | Live status tracking (update as work progresses) |

---

## Overview
Split `crates/sim/src/flip.rs` (~3000 lines) and `crates/sim/src/grid.rs` (~2000 lines) into smaller modules while preserving behavior and API. The primary goal is maintainability and safer parallel development.

Key decisions:
- SDF remains a solid-terrain SDF (distance to solid), used for collision/boundaries.
- Methods stay inherent on `Grid`/`FlipSimulation` (no extension traits or free-function exports).
- Default visibility is private; use `pub(crate)` or `pub` only when required.
- Tests are deterministic by default; randomized coverage uses fixed seeds or `#[ignore]` stress tests.
- File size <= 500 lines is a soft goal, not a strict requirement.
- Shared interpolation helpers live under `grid` (e.g., `grid/interp.rs`) and are re-exported from `grid/mod.rs`.

## Target Structure (High Level)
- `crates/sim/src/grid/` with focused modules (cell_types, sdf, velocity, pressure, extrapolation, vorticity, interp)
- `crates/sim/src/flip/` with focused modules (spawning, diagnostics, transfer, advection, sediment, pile)
- `crates/sim/src/grid/mod.rs` and `crates/sim/src/flip/mod.rs` as orchestrators

## Parallelization Strategy (Swarm)
- Agents only touch their new module file(s) and tests.
- Lead owns `grid/mod.rs`, `flip/mod.rs`, and `lib.rs` cleanup (removing moved methods).
- No reformatting or refactors during moves; keep diffs tight.
- Merge order follows phases; high-risk modules land later.

## Phase 0: Scaffolding (Lead)

PRD:
- Create module scaffolding so agents can work in parallel without conflicts.
- Avoid any behavior change.
- Establish shared helper placement for interpolation math.

To-dos:
- Move `crates/sim/src/grid.rs` -> `crates/sim/src/grid/mod.rs` (no behavior changes).
- Move `crates/sim/src/flip.rs` -> `crates/sim/src/flip/mod.rs` (no behavior changes).
- Add empty module files for planned splits.
- Add `crates/sim/src/grid/interp.rs` for `quadratic_bspline`, `quadratic_bspline_1d`, `apic_d_inverse`.
- Declare all modules in `grid/mod.rs` and `flip/mod.rs`.
- Run `cargo test -p sim`.

Acceptance Criteria:
- Build and tests pass with no functional changes.
- New module stubs compile (even if empty).
- Helpers in `grid/interp.rs` are re-exported from `grid/mod.rs`.

## Phase 1: Grid Split (Parallel)

PRD:
- Split `Grid` functionality into cohesive files.
- Keep SDF as solid-terrain distance (not fluid distance).
- Ensure boundary cells are treated as solid for SDF.

To-dos:
- `grid/cell_types.rs`: move `CellType`, `DepositedCell`, and impls.
- `grid/sdf.rs`: move `compute_sdf`, `sample_sdf`, `sdf_gradient`, bed height helpers.
  - Update `compute_sdf` to read `solid` directly (not `cell_type`).
  - Ensure boundaries are treated as solid (mark boundaries in `solid` or handle in `compute_sdf`).
  - Tests use `set_solid` (not `cell_type`) and assert negative inside solids.
- `grid/velocity.rs`: sampling, gravity, viscosity, interpolation helpers usage.
- `grid/pressure.rs`: divergence, solve, gradient application.
- `grid/extrapolation.rs`: velocity extrapolation variants.
- `grid/vorticity.rs`: vorticity, enstrophy, confinement.
- Lead removes moved methods from `grid/mod.rs` and resolves imports.
- Add unit tests per module (>= 3, deterministic). Use fixed seeds for any randomness.

Acceptance Criteria:
- `cargo test -p sim` passes.
- `compute_sdf` depends on `solid`, not `cell_type`.
- All grid functions remain inherent methods on `Grid`.
- No new warnings.

## Phase 2: FLIP Split (Parallel)

PRD:
- Split `FlipSimulation` functionality into cohesive files.
- Maintain simulation behavior and API surface.
- Keep deterministic tests; add seeded randomized coverage where useful.

To-dos:
- `flip/spawning.rs`: `is_spawn_safe`, spawn functions.
- `flip/diagnostics.rs`: diagnostics, profiled update, isolated test cycles.
- `flip/transfer.rs` (high risk): P2G, G2P, old velocities, pressure gradient coupling.
- `flip/advection.rs`: advection, spatial hash, neighbor counts.
- `flip/sediment.rs` (high risk): sediment forces, deposition, entrainment, state updates.
- `flip/pile.rs`: pile heightfield and constraints.
- Lead removes moved methods from `flip/mod.rs` and resolves imports.
- Tests: avoid exact count/velocity assertions on random spawns unless seeded; prefer bounds/invariants.

Acceptance Criteria:
- `cargo test -p sim` passes.
- All FLIP functions remain inherent methods on `FlipSimulation`.
- Deterministic tests pass; randomized tests use fixed seeds or are `#[ignore]`.

## Phase 3: Integration and Exports (Lead)

PRD:
- Re-export modules from `lib.rs` consistently with previous public API.
- Ensure external tests and examples compile without change or with documented import updates.

To-dos:
- Update `crates/sim/src/lib.rs` module exports.
- Fix imports across the crate if needed.
- Run `cargo test -p sim` and `cargo build --release`.

Acceptance Criteria:
- Full build passes and tests are green.
- Public API remains intact or documented.

## Swarm Assignment Map (Suggested)
- G1: `grid/cell_types.rs`, `grid/sdf.rs`
- G2: `grid/velocity.rs`, `grid/vorticity.rs`
- G3: `grid/pressure.rs`, `grid/extrapolation.rs`
- F1: `flip/spawning.rs`, `flip/pile.rs`
- F2: `flip/diagnostics.rs`
- F3: `flip/advection.rs`
- F4: `flip/transfer.rs`, `flip/pressure.rs` (high risk)
- F5: `flip/sediment.rs` (high risk)
- Lead: scaffolding + integration cleanup

## Agent Checklist Template

Use this in each agent PR/branch summary to keep merges clean and consistent.

```
Agent:
Phase:
Modules:

Moved Methods (cut list):
- ...

Files Added/Updated:
- ...

Tests Added:
- ...

Notes/Risks:
- ...
```

## Risks and Mitigations
- Merge conflicts in `grid/mod.rs` and `flip/mod.rs`.
  - Mitigation: lead owns deletions; agents only add new files.
- Flaky tests from randomization.
  - Mitigation: fixed seeds or `#[ignore]` stress tests.
- Behavior drift from SDF source change.
  - Mitigation: explicit note and targeted tests for solid SDF.

## Success Criteria (Overall)
- All tests pass, no new warnings.
- Solid SDF remains the authoritative collision/boundary distance.
- Modules are split as planned with maintainable file sizes.
- Simulation runs correctly in the visual/game environment.

---

## Regression Test Suite

A golden-value regression test suite guards against accidental behavior changes:

```bash
# Run BEFORE and AFTER every change
cargo test -p sim --test refactor_regression
```

**Test Files:**
- `crates/sim/tests/refactor_regression.rs` - The actual tests (10 total)
- `crates/sim/tests/capture_golden_values.rs` - Regenerate golden values if needed

| Test | Guards | Module |
|------|--------|--------|
| `regression_p2g_transfer` | P2G velocity transfer | transfer |
| `regression_pressure_solve` | Pressure solver | pressure |
| `regression_flip_cycle` | P2G→G2P roundtrip | transfer |
| `regression_extrapolation` | Velocity extrapolation | extrapolation |
| `regression_vorticity` | Vorticity/enstrophy | vorticity |
| `regression_sdf_computation` | SDF distance field | sdf |
| `regression_interpolation_kernels` | B-spline kernels (sum=1) | interp |
| `regression_full_step` | Complete update() | ALL |
| `regression_sediment_settling` | Deposition | sediment |
| `regression_stability_100_steps` | NaN/Inf check | ALL |

**If a test fails after your change: STOP. You broke something. Fix it before continuing.**

---

## Quick Start for Agents

1. **Read the plan documents** in order:
   - This README for overview
   - METHOD_INVENTORY.md for your specific methods
   - PARALLEL_PROTOCOL.md for workflow rules

2. **Claim your module** in STATUS.md

3. **Follow Phase 0** (PHASE0_SCAFFOLDING.md) if you're the Lead

4. **For each method extraction:**
   - Copy method to new module file
   - Add `use super::*;` for parent access
   - Run `cargo build -p sim && cargo test -p sim`
   - Commit when green

5. **Update STATUS.md** when complete

---

## Critical Reminders

From CLAUDE.md project rules:

> **NEVER run destructive git commands without EXPLICIT user approval**
> - No `git reset --hard`
> - No `git checkout -- <file>`
> - Always `git status` first
> - Uncommitted changes may be the user's working solution

> **NO PATCH FIXING - DEBUG NEW CODE**
> - When new code breaks: problem is in NEW code
> - Do NOT touch working systems to fix issues in new code
