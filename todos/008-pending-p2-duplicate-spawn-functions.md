---
status: resolved
priority: p2
issue_id: "008"
tags: [code-review, simplification, dfsph]
dependencies: []
---

# Simplification: Consolidate 5 Duplicate Spawn Functions

## Problem Statement

DFSPH has 5 nearly identical spawn functions (`spawn_water`, `spawn_mud`, `spawn_sand`, `spawn_magnetite`, `spawn_gold`) that only differ in material type. This violates DRY and wastes 36 lines.

## Findings

**Location:** `crates/dfsph/src/simulation.rs:75-110`

**Current Code (Pattern repeated 5 times):**
```rust
pub fn spawn_water(&mut self, x: f32, y: f32, vx: f32, vy: f32, count: usize) {
    for _ in 0..count {
        let jx = x + (rand::random::<f32>() - 0.5) * 2.0;
        let jy = y + (rand::random::<f32>() - 0.5) * 2.0;
        self.spawn_particle_internal(Vec2::new(jx, jy), Vec2::new(vx, vy), ParticleMaterial::Water);
    }
}
// spawn_mud, spawn_sand, spawn_magnetite, spawn_gold are IDENTICAL except material
```

**Note:** `spawn_magnetite` and `spawn_gold` don't even apply jitter, showing inconsistency.

## Proposed Solutions

### Option A: Single generic spawn function (Recommended)
- **Pros:** DRY, flexible, matches `sim::Particles` API
- **Cons:** Slightly more verbose call site
- **Effort:** 30 minutes
- **Risk:** Low

```rust
pub fn spawn_particles(
    &mut self,
    x: f32, y: f32,
    vx: f32, vy: f32,
    count: usize,
    material: ParticleMaterial,
    jitter: f32,  // 0.0 for no jitter, 2.0 for default
) {
    for _ in 0..count {
        let pos = Vec2::new(
            x + (rand::random::<f32>() - 0.5) * jitter,
            y + (rand::random::<f32>() - 0.5) * jitter,
        );
        self.spawn_particle_internal(pos, Vec2::new(vx, vy), material);
    }
}
```

### Option B: Make spawn_particle_internal public with jitter param
- **Pros:** Even simpler API
- **Cons:** Exposes internal function
- **Effort:** 15 minutes
- **Risk:** Low

### Option C: Keep convenience methods, delegate to generic
- **Pros:** Backward compatible
- **Cons:** Still 5 methods, just shorter
- **Effort:** 20 minutes
- **Risk:** Low

## Recommended Action

**Option A** - Replace all 5 functions with single `spawn_particles()`.

## Technical Details

**Affected File:** `crates/dfsph/src/simulation.rs`

**Lines to Remove:** 75-110 (35 lines)
**Lines to Add:** ~12 lines for new function

**Net Reduction:** ~23 lines

**Call Site Update Required:**
```rust
// Before:
sim.spawn_water(20.0, 20.0, 50.0, 0.0, 1);

// After:
sim.spawn_particles(20.0, 20.0, 50.0, 0.0, 1, ParticleMaterial::Water, 2.0);
```

## Acceptance Criteria

- [x] Single `spawn_particles()` function exists
- [x] All 5 material-specific functions removed
- [x] `game/main.rs` updated to use new API
- [x] Tests updated/added
- [x] No functionality regression

## Work Log

| Date | Action | Notes |
|------|--------|-------|
| 2025-12-22 | Created | Code simplicity review |
| 2025-12-22 | Resolved | Replaced 5 duplicate spawn functions with single generic `spawn_particles()`. Updated API in game/main.rs. Compilation verified. |

## Resources

- Code simplicity reviewer analysis
- DRY principle
