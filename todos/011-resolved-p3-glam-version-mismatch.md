---
status: resolved
priority: p3
issue_id: "011"
tags: [code-review, dependencies, dfsph]
dependencies: []
---

# Dependencies: glam Version Mismatch

## Problem Statement

The `dfsph` crate uses glam 0.27 while `game` uses glam 0.24. This could lead to type incompatibilities or unexpected behavior when Vec2 types cross crate boundaries.

## Findings

**Evidence:**
- `crates/dfsph/Cargo.toml`: `glam = "0.27"`
- `crates/game/Cargo.toml`: `glam = "0.24"`

**Potential Issues:**
1. Different Vec2 implementations may have subtle behavioral differences
2. Cargo may include both versions in final binary (bloat)
3. API changes between versions could cause silent bugs

**Note:** Current code works because `sim` re-exports particles and both crates use sim's types. But this is fragile.

## Proposed Solutions

### Option A: Standardize on latest (0.27)
- **Pros:** Latest features, consistent
- **Cons:** May need game code updates
- **Effort:** 15 minutes
- **Risk:** Low

### Option B: Standardize on 0.24 (current game)
- **Pros:** Minimal changes to game
- **Cons:** Older version
- **Effort:** 15 minutes
- **Risk:** Low

### Option C: Add workspace dependency
- **Pros:** Single source of truth for all crates
- **Cons:** Requires workspace Cargo.toml change
- **Effort:** 20 minutes
- **Risk:** Low

## Recommended Action

**Option C** - Add to workspace Cargo.toml:

```toml
[workspace.dependencies]
glam = "0.27"
```

Then in each crate:
```toml
[dependencies]
glam = { workspace = true }
```

## Technical Details

**Affected Files:**
- `Cargo.toml` (workspace root)
- `crates/dfsph/Cargo.toml`
- `crates/game/Cargo.toml`
- `crates/sim/Cargo.toml`

## Acceptance Criteria

- [x] All crates use same glam version
- [x] Workspace dependency defined in root Cargo.toml
- [x] `cargo tree` shows single glam version
- [x] All tests pass

## Work Log

| Date | Action | Notes |
|------|--------|-------|
| 2025-12-22 | Created | Security audit finding |
| 2025-12-22 | Resolved | Implemented Option C - workspace dependencies. Added glam 0.27 to workspace dependencies and updated all crates (sim, game, dfsph) to use workspace version. Also added dfsph to workspace members. Fixed unrelated compilation errors with vorticities field. All crates now compile successfully. |

## Resources

- Cargo workspace dependencies: https://doc.rust-lang.org/cargo/reference/workspaces.html
