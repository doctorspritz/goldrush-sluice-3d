---
status: pending
priority: p2
issue_id: "005"
tags: [code-review, architecture, dfsph, refactoring]
dependencies: ["002", "003", "004"]
---

# Architecture: Merge DFSPH Crate into sim

## Problem Statement

DFSPH exists as a separate crate (`crates/dfsph`) but heavily depends on `sim` crate types while **duplicating significant infrastructure** (spatial hashing, kernels, collision). This violates DRY and complicates maintenance.

## Findings

**Current Structure:**
```
crates/
├── sim/          # FLIP/APIC, PBF, particles, physics, grid
├── dfsph/        # DFSPH - depends on sim but duplicates code
└── game/         # Rendering - depends on both
```

**Duplicated Code:**

| Component | sim | dfsph | Duplicate LOC |
|-----------|-----|-------|---------------|
| Spatial Hash | `flip.rs:1331` | `simulation.rs:431` | ~50 lines |
| SPH Kernels | `pbf.rs:340+` | `simulation.rs:452+` | ~20 lines |
| Collision Grid | `Grid::is_solid()` | `grid_solid[idx]` | ~30 lines |
| Total | | | **~100 lines** |

**Dependency Issues:**
- `game` depends on both `sim` and `dfsph`
- `dfsph` depends on `sim` but reimplements core infrastructure
- Version mismatch: dfsph uses glam 0.27, game uses glam 0.24

## Proposed Solutions

### Option A: Merge into sim as dfsph.rs (Recommended)
- **Pros:** Eliminates duplication, single source of truth, consistent API
- **Cons:** sim crate grows, initial merge effort
- **Effort:** 4-6 hours
- **Risk:** Medium

**Proposed Structure:**
```
crates/sim/src/
├── lib.rs          # Add: pub mod dfsph; pub use dfsph::DfsphSimulation;
├── dfsph.rs        # Moved from crates/dfsph/src/simulation.rs
├── spatial_hash.rs # NEW: Shared spatial acceleration
├── kernels.rs      # NEW: Shared SPH kernels
└── ... existing files
```

### Option B: Create shared sim-core crate
- **Pros:** Clean separation of concerns
- **Cons:** More crates to maintain, more complex dependencies
- **Effort:** 8-12 hours
- **Risk:** Medium

### Option C: Keep separate but extract shared code
- **Pros:** Minimal structural change
- **Cons:** Still two crates, just smaller duplication
- **Effort:** 3-4 hours
- **Risk:** Low

## Recommended Action

**Option A** - Merge into sim crate. This project already has FLIP, PBF in sim; DFSPH belongs there too.

## Technical Details

**Migration Steps:**
1. Move `crates/dfsph/src/simulation.rs` → `crates/sim/src/dfsph.rs`
2. Update imports to use `sim::Grid`, `sim::physics`
3. Extract spatial hash to shared module
4. Extract kernels to shared module (poly6, spiky)
5. Delete `crates/dfsph`
6. Update `game/Cargo.toml` to remove dfsph dependency
7. Update `Cargo.toml` workspace members

## Acceptance Criteria

- [ ] DFSPH code lives in `sim/dfsph.rs`
- [ ] `DfsphSimulation` exported from `sim` crate
- [ ] No code duplication for spatial hash or kernels
- [ ] `game` only depends on `sim`
- [ ] All tests pass
- [ ] Cargo clippy passes

## Work Log

| Date | Action | Notes |
|------|--------|-------|
| 2025-12-22 | Created | Architecture review finding |

## Resources

- Architecture strategist analysis
- Pattern recognition specialist report
