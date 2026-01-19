---
status: resolved
priority: p3
issue_id: "012"
tags: [code-review, quality, game]
dependencies: []
---

# Code Quality: Unused Imports in game/main.rs

## Problem Statement

The game crate has unused imports and dead code related to the renderer transition from the old simulation to DFSPH.

## Findings

**From `cargo check -p game`:**

```
warning: unused import: `MetaballRenderer`
 --> crates/game/src/main.rs:9:14

warning: unused import: `sim::ParticleMaterial`
  --> crates/game/src/main.rs:11:5

warning: unused variable: `particle_renderer`
  --> crates/game/src/main.rs:66:13

warning: variable does not need to be mutable
  --> crates/game/src/main.rs:66:9
```

**Also in render.rs (14 warnings):**
- Unused constants: `METABALL_DENSITY_FRAG`, `METABALL_THRESHOLD_FRAG`
- Unused struct: `MetaballRenderer`
- Unused functions: `draw_particles_fast`, `draw_particles_rect`, etc.

## Proposed Solutions

### Option A: Remove unused imports only (Quick fix)
- **Pros:** Fast, minimal change
- **Cons:** Dead renderer code still present
- **Effort:** 5 minutes
- **Risk:** Low

### Option B: Clean up entire render.rs (Recommended)
- **Pros:** Removes all dead code
- **Cons:** More work
- **Effort:** 30 minutes
- **Risk:** Low

## Recommended Action

**Option B** - Clean up both files. The current demo doesn't use the advanced renderers.

## Technical Details

**main.rs Fixes:**
```rust
// Remove these lines:
use render::{MetaballRenderer, ParticleRenderer};  // Remove MetaballRenderer
use sim::ParticleMaterial;  // Remove entirely

// Remove or use:
let mut particle_renderer = ParticleRenderer::new();  // Remove or actually use
```

**render.rs:** Consider removing unused renderers or keeping them with `#[allow(dead_code)]` if planned for future use.

## Acceptance Criteria

- [x] `cargo check -p game` produces 0 warnings for main.rs imports
- [x] No unused imports in main.rs
- [x] Dead code in main.rs removed

## Work Log

| Date | Action | Notes |
|------|--------|-------|
| 2025-12-22 | Created | Build output analysis |
| 2025-12-22 | Resolved | Removed unused imports: MetaballRenderer, ParticleMaterial, and particle_renderer variable from main.rs. All specified imports now clean. Remaining warnings are in render.rs (separate issue). |

## Resources

- Build output from `cargo check -p game`
