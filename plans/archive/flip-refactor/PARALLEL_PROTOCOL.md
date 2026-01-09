# Parallel Agent Refactoring Protocol

This document specifies the coordination protocol for parallel agent refactoring of flip.rs and grid.rs.

---

## 1. Git Workflow & Branching Strategy

### Branch Naming Convention

```
refactor/<file>-<module>-<agent>
```

**Examples:**
```
refactor/grid-cell-types-G1
refactor/grid-sdf-G1
refactor/grid-velocity-G2
refactor/grid-vorticity-G2
refactor/grid-pressure-G3
refactor/grid-extrapolation-G3
refactor/flip-spawning-F1
refactor/flip-pile-F1
refactor/flip-diagnostics-F2
refactor/flip-advection-F3
refactor/flip-transfer-F4
refactor/flip-sediment-F5
```

### Worktree Strategy

Each agent works in an isolated git worktree to prevent conflicts:

```bash
# Lead agent creates worktrees for all agents
git worktree add ../refactor-G1 -b refactor/grid-cell-types-G1
git worktree add ../refactor-G2 -b refactor/grid-velocity-G2
git worktree add ../refactor-G3 -b refactor/grid-pressure-G3
git worktree add ../refactor-F1 -b refactor/flip-spawning-F1
git worktree add ../refactor-F2 -b refactor/flip-diagnostics-F2
git worktree add ../refactor-F3 -b refactor/flip-advection-F3
git worktree add ../refactor-F4 -b refactor/flip-transfer-F4
git worktree add ../refactor-F5 -b refactor/flip-sediment-F5
```

### Merge Order

Branches must merge in this strict order:

```
Phase 0: Scaffolding (Lead)
    refactor/scaffold-grid    # Creates grid/mod.rs + empty submodule files
    refactor/scaffold-flip    # Creates flip/mod.rs + empty submodule files

Phase 1: Grid Split (Parallel after scaffolding merges)
    1. refactor/grid-interp-G3        # First: interp functions used by others
    2. refactor/grid-cell-types-G1    # Second: types used by others
    3. refactor/grid-sdf-G1
    4. refactor/grid-velocity-G2
    5. refactor/grid-vorticity-G2
    6. refactor/grid-pressure-G3
    7. refactor/grid-extrapolation-G3

Phase 2: FLIP Split (Parallel after Phase 1 merges)
    1. refactor/flip-spawning-F1
    2. refactor/flip-pile-F1
    3. refactor/flip-diagnostics-F2
    4. refactor/flip-advection-F3
    5. refactor/flip-transfer-F4      # High risk
    6. refactor/flip-sediment-F5      # High risk

Phase 3: Integration (Lead)
    refactor/integration-cleanup
```

---

## 2. Conflict Prevention Rules

### Rule 1: Agents Only Add, Never Delete From Original

During extraction, agents:
- ADD new module files (grid/velocity.rs, flip/transfer.rs)
- ADD module declarations to mod.rs
- DO NOT delete methods from grid.rs or flip.rs

Lead agent handles deletions in Phase 3.

### Rule 2: Each Agent Owns Specific Line Ranges

Before starting, agents claim specific methods by name (see METHOD_INVENTORY.md). No two agents claim the same method.

### Rule 3: No Formatting Changes

Agents do NOT:
- Run `cargo fmt` on files they didn't create
- Reorder imports in original files
- Add/remove blank lines in original files

This keeps diffs minimal and reduces conflicts.

### Rule 4: Parallel Change Pattern

Use expand-migrate-contract:

```rust
// Phase 1: EXPAND - Add new module with copy of methods
// grid/velocity.rs has Grid::sample_velocity()

// Phase 2: MIGRATE - Original delegates (optional)
// grid.rs could delegate, but we skip this for simplicity

// Phase 3: CONTRACT - Lead removes methods from original
// Lead deletes sample_velocity() from grid.rs
```

---

## 3. Verification Protocol

### Regression Test Suite

A golden-value regression suite exists to catch accidental behavior changes:

```bash
# BEFORE starting ANY refactor work - verify tests pass
cargo test -p sim --test refactor_regression

# If tests fail BEFORE you start, DO NOT proceed
# The golden values may need regenerating first
```

**Test Coverage by Module (10 tests):**

| Test | Catches Changes To | Modules Affected |
|------|-------------------|------------------|
| `regression_p2g_transfer` | `particles_to_grid()` | transfer |
| `regression_pressure_solve` | divergence, solve, gradient | pressure |
| `regression_flip_cycle` | P2G → store_old → G2P roundtrip | transfer |
| `regression_extrapolation` | `extrapolate_velocities()` | extrapolation |
| `regression_vorticity` | vorticity, enstrophy | vorticity |
| `regression_sdf_computation` | `compute_sdf()`, `sample_sdf()`, gradient | sdf |
| `regression_interpolation_kernels` | B-spline kernel (MUST sum to 1.0) | interp |
| `regression_full_step` | Entire `update()` pipeline | ALL |
| `regression_sediment_settling` | deposition, entrainment | sediment |
| `regression_stability_100_steps` | No NaN/Inf for 100 frames | ALL |

### Pre-Commit Checklist (Each Agent)

```bash
# 1. REGRESSION tests (most important!)
cargo test -p sim --test refactor_regression

# 2. Build check
cargo build -p sim

# 3. Full test suite
cargo test -p sim

# 4. Lint check
cargo clippy -p sim -- -D warnings

# 5. Format check (new files only)
cargo fmt -p sim -- --check

# 6. Rebase on main
git fetch origin main
git rebase origin/main
```

### If Regression Tests Fail

**STOP immediately.** A failing regression test means behavior changed.

1. **Identify which test failed** - maps to which module
2. **Check your changes** - did you accidentally modify logic?
3. **Common causes:**
   - Missing `use super::*;` in new module
   - Wrong visibility on a method
   - Import path changed
   - Accidentally changed a constant
4. **Fix the issue** - don't update golden values unless intentional
5. **Re-run tests** - confirm fix worked

### Regenerating Golden Values (ONLY if intentional)

If you've made an **intentional** behavior change:

```bash
# Regenerate golden values
cargo test -p sim --test capture_golden_values capture_all_golden_values -- --ignored --nocapture

# Copy printed values into refactor_regression.rs
# Commit the updated golden values with explanation
```

### Smoke Test (After Each Merge)

```bash
# Run visual simulation to verify behavior unchanged
cargo run --bin game --release
# Observe for 30 seconds, check for obvious breakage
```

### Full Validation (Phase 3)

```bash
# Complete test suite
cargo test -p sim --all-features

# Release build
cargo build --release

# Run all examples
cargo run --example settling_columns --release
```

---

## 4. Communication Protocol

### Status File

Agents update `docs/plans/flip-refactor/STATUS.md`:

```markdown
# Refactor Status

Last updated: 2025-12-29 15:30 UTC

## Phase 0: Scaffolding
| Task | Agent | Status | PR | Notes |
|------|-------|--------|-----|-------|
| grid/mod.rs scaffold | Lead | MERGED | #1 | |
| flip/mod.rs scaffold | Lead | MERGED | #2 | |

## Phase 1: Grid Split
| Module | Agent | Status | PR | Depends On | Notes |
|--------|-------|--------|-----|------------|-------|
| interp | G3 | IN_PROGRESS | - | scaffold | |
| cell_types | G1 | WAITING | - | scaffold | |
| sdf | G1 | WAITING | - | cell_types | |
| velocity | G2 | WAITING | - | interp | |
| ...

## Phase 2: FLIP Split
| Module | Agent | Status | PR | Depends On | Notes |
|--------|-------|--------|-----|------------|-------|
| spawning | F1 | WAITING | - | grid merged | |
| ...
```

### Signal Types

- `WAITING` - Blocked on dependency
- `IN_PROGRESS` - Agent actively working
- `READY_FOR_REVIEW` - PR created
- `CHANGES_REQUESTED` - Needs fixes
- `APPROVED` - Ready to merge
- `MERGED` - Complete
- `BLOCKED_BY:<reason>` - Unexpected blocker
- `FAILED` - Tests failing, needs help

### Escalation Path

1. **Test Failure**: Agent stops, updates status to `FAILED`, describes error
2. **Merge Conflict**: Agent updates status to `BLOCKED_BY:conflict`, Lead intervenes
3. **Dependency Missing**: Agent updates status to `BLOCKED_BY:<other-module>`, waits

---

## 5. Rollback Procedures

### Single Agent Rollback

If an agent's PR breaks things after merge:

```bash
# Identify the merge commit
git log --oneline | head -5

# Revert the merge
git revert <merge-commit-sha> --mainline 1
git push origin main
```

### Full Rollback

If refactoring is fundamentally broken:

```bash
# Return to pre-refactor state
git checkout backup/pre-module-split-2025-12-29
git branch -D main
git checkout -b main
git push origin main --force
```

### Checkpoints

Create tags after each major milestone:

```bash
git tag refactor-phase0-complete -m "Scaffolding done"
git tag refactor-phase1-complete -m "Grid split done"
git tag refactor-phase2-complete -m "FLIP split done"
git push origin --tags
```

---

## 6. Agent-Specific Instructions

### G1: grid/cell_types + grid/sdf

1. Create `grid/cell_types.rs`
2. Move `CellType` enum, `DepositedCell` struct, `MultigridLevel` struct
3. Move cell index methods
4. Create `grid/sdf.rs`
5. Move SDF computation methods
6. Add `mod cell_types; mod sdf;` to grid/mod.rs
7. Add `pub use cell_types::*; pub use sdf::*;` for re-exports
8. Run tests

### G2: grid/velocity + grid/vorticity

1. Create `grid/velocity.rs`
2. Move velocity sampling methods
3. Move gravity/viscosity methods
4. Create `grid/vorticity.rs`
5. Move vorticity methods
6. Add module declarations
7. Run tests

### G3: grid/interp + grid/pressure + grid/extrapolation

1. **Start with interp** (other modules depend on it)
2. Create `grid/interp.rs`
3. Move `quadratic_bspline_1d`, `quadratic_bspline`, `apic_d_inverse`
4. Create `grid/pressure.rs`
5. Move pressure solver methods
6. Create `grid/extrapolation.rs`
7. Move extrapolation methods
8. Run tests

### F1: flip/spawning + flip/pile

1. Create `flip/spawning.rs`
2. Move spawn methods
3. Create `flip/pile.rs`
4. Move pile heightfield methods
5. Add module declarations
6. Run tests

### F2: flip/diagnostics

1. Create `flip/diagnostics.rs`
2. Move profiled update methods
3. Move energy/enstrophy methods
4. Move test initialization methods
5. Run tests

### F3: flip/advection

1. Create `flip/advection.rs`
2. Move advection methods
3. Move spatial hash methods
4. Move neighbor count methods
5. Run tests

### F4: flip/transfer + flip/pressure (HIGH RISK)

**flip/transfer.rs:**
1. Create `flip/transfer.rs`
2. Move `particles_to_grid`
3. Move `store_old_velocities`
4. Move `grid_to_particles`
5. **Verify kernel consistency** - all three must use same B-spline kernel

**flip/pressure.rs:**
6. Create `flip/pressure.rs`
7. Move `apply_pressure_gradient_two_way`
8. Move `apply_porosity_drag`
9. Ensure access to `sand_volume_u/v`, `water_volume_u/v` buffers

**Verification:**
10. Run transfer-specific tests
11. Run regression_flip_cycle test
12. Run full test suite

### F5: flip/sediment (HIGH RISK)

1. Create `flip/sediment.rs`
2. Move sediment force methods
3. Move DEM settling methods
4. Move deposition/entrainment methods
5. Run sediment-specific tests
6. Run full test suite

### Lead: Scaffolding + Integration

**Phase 0:**
1. Create backup branch
2. Move `grid.rs` → `grid/mod.rs`
3. Move `flip.rs` → `flip/mod.rs`
4. Create empty submodule files
5. Verify compilation
6. Merge scaffolding PRs

**Phase 3:**
1. After all agent PRs merged, delete moved methods from mod.rs files
2. Clean up imports
3. Update lib.rs exports
4. Run full validation
5. Create completion tag

---

## 7. Field Visibility Strategy

When methods are split across files, some fields need visibility changes:

### Grid Fields

```rust
pub struct Grid {
    // Public API
    pub width: usize,
    pub height: usize,
    pub cell_size: f32,

    // Accessible to submodules
    pub(super) solid: Vec<bool>,
    pub(super) cell_type: Vec<CellType>,
    pub(super) u: Vec<f32>,
    pub(super) v: Vec<f32>,
    pub(super) pressure: Vec<f32>,
    pub(super) divergence: Vec<f32>,
    pub(super) vorticity: Vec<f32>,
    pub(super) sdf: Vec<f32>,
    // ... etc
}
```

### FlipSimulation Fields

```rust
pub struct FlipSimulation {
    // === Public API ===
    pub grid: Grid,
    pub particles: Particles,
    pub pile_height: Vec<f32>,

    // === Public Configuration Flags ===
    pub use_ferguson_church: bool,
    pub use_hindered_settling: bool,
    pub use_variable_diameter: bool,
    pub diameter_variation: f32,
    pub use_viscosity: bool,
    pub viscosity: f32,
    pub sand_pic_ratio: f32,

    // === P2G Transfer Buffers (transfer.rs) ===
    pub(super) u_sum: Vec<f32>,
    pub(super) u_weight: Vec<f32>,
    pub(super) v_sum: Vec<f32>,
    pub(super) v_weight: Vec<f32>,

    // === Two-Way Coupling Volume Fractions (pressure.rs) ===
    pub(super) sand_volume_u: Vec<f32>,
    pub(super) water_volume_u: Vec<f32>,
    pub(super) sand_volume_v: Vec<f32>,
    pub(super) water_volume_v: Vec<f32>,

    // === Spatial Hash (advection.rs) ===
    pub(super) cell_head: Vec<i32>,
    pub(super) particle_next: Vec<i32>,
    pub(super) impulse_buffer: Vec<Vec2>,
    pub(super) near_force_buffer: Vec<Vec2>,

    // === Neighbor Tracking (sediment.rs) ===
    pub(super) neighbor_counts: Vec<u16>,
    pub(super) water_neighbor_counts: Vec<u16>,

    // === Deposition Mass (sediment.rs) ===
    pub(super) deposited_mass_mud: Vec<f32>,
    pub(super) deposited_mass_sand: Vec<f32>,
    pub(super) deposited_mass_magnetite: Vec<f32>,
    pub(super) deposited_mass_gold: Vec<f32>,

    // === Internal State ===
    frame: u32,  // Private - only used internally
}
```

**Field-to-Module Mapping:**

| Field | Used By Module |
|-------|----------------|
| `u_sum`, `u_weight`, `v_sum`, `v_weight` | transfer.rs |
| `sand_volume_*`, `water_volume_*` | pressure.rs |
| `cell_head`, `particle_next`, `impulse_buffer`, `near_force_buffer` | advection.rs |
| `neighbor_counts`, `water_neighbor_counts` | sediment.rs |
| `deposited_mass_*` | sediment.rs |
| `pile_height` | pile.rs |

---

## 8. Test Organization

### Inline Tests Per Module

Each new module file includes its own tests:

```rust
// grid/velocity.rs

impl Grid {
    pub fn sample_velocity(&self, pos: Vec2) -> Vec2 { ... }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_velocity_interpolation() {
        let grid = Grid::new(10, 10, 1.0);
        // ...
    }
}
```

### Shared Test Utilities

Create `crates/sim/src/test_utils.rs` for common helpers:

```rust
// test_utils.rs
pub fn create_test_grid(width: usize, height: usize) -> Grid { ... }
pub fn create_test_simulation() -> FlipSimulation { ... }
```

---

## 9. Timeline Expectations

**Note:** No time estimates - focus on completion criteria, not deadlines.

### Phase 0: Scaffolding
- **Completion Criteria:**
  - `grid/mod.rs` and `flip/mod.rs` exist
  - Empty submodule files created
  - `cargo build -p sim` passes
  - `cargo test -p sim` passes

### Phase 1: Grid Split
- **Completion Criteria:**
  - All grid submodule files populated
  - All grid methods have home in submodule
  - No duplicated code
  - All tests pass

### Phase 2: FLIP Split
- **Completion Criteria:**
  - All flip submodule files populated
  - All flip methods have home in submodule
  - No duplicated code
  - All tests pass

### Phase 3: Integration
- **Completion Criteria:**
  - Original grid.rs and flip.rs contain only struct definitions and mod declarations
  - `cargo test -p sim --all-features` passes
  - `cargo build --release` passes
  - Visual simulation runs correctly
