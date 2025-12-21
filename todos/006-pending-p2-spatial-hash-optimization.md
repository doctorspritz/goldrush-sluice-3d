---
status: pending
priority: p2
issue_id: "006"
tags: [code-review, performance, dfsph]
dependencies: ["002"]
---

# Performance: Replace Linked-List Spatial Hash with Dense Array

## Problem Statement

The current spatial hash uses **linked-list traversal** which causes cache misses and poor memory locality. The worktree contains an optimized `neighbor.rs` implementation that should be used instead.

## Findings

**Current Implementation:** `crates/dfsph/src/simulation.rs:431-448`

```rust
// Linked-list based (cache-unfriendly)
let mut j = heads[cell];
while j != -1 {
    let j_idx = j as usize;
    let pos_j = positions[j_idx];  // Random access!
    j = next[j_idx];  // Cache miss!
}
```

**Optimized Version Exists:** `worktrees/dfsph-rebuild/crates/dfsph/src/neighbor.rs`

Uses prefix-sum approach for dense, cache-friendly storage:
- All particles in cell stored contiguously
- Sequential memory access pattern
- 3-pass algorithm: count → prefix sum → scatter

**Performance Impact:**
- Current: ~1-2ms at 20k particles
- Optimized: ~0.3-0.5ms
- **Expected Gain:** 1.5ms per frame (at 60fps = 90ms/second saved)

## Proposed Solutions

### Option A: Use neighbor.rs from worktree (Recommended)
- **Pros:** Already implemented and tested
- **Cons:** Need to integrate
- **Effort:** 2-3 hours
- **Risk:** Low

### Option B: Rewrite current hash with prefix-sum
- **Pros:** Stays in single file
- **Cons:** Reimplementing existing solution
- **Effort:** 3-4 hours
- **Risk:** Medium

## Recommended Action

**Option A** - Integrate `neighbor.rs` SpatialHash from worktree.

## Technical Details

**Integration Steps:**
1. Copy `worktrees/dfsph-rebuild/crates/dfsph/src/neighbor.rs` to main repo
2. Add `pub mod neighbor;` to `lib.rs`
3. Replace `grid_heads/grid_next` with `SpatialHash` instance
4. Update `build_spatial_hash()` to use `SpatialHash::rebuild()`
5. Update neighbor iteration loops to use `SpatialHash::query_neighbors()`
6. Benchmark before/after

## Acceptance Criteria

- [ ] `SpatialHash` struct used for neighbor queries
- [ ] No linked-list traversal in hot paths
- [ ] Benchmark shows 15-20% neighbor search speedup
- [ ] All tests pass

## Work Log

| Date | Action | Notes |
|------|--------|-------|
| 2025-12-22 | Created | Performance analysis finding |

## Resources

- Performance oracle analysis
- `worktrees/dfsph-rebuild/crates/dfsph/src/neighbor.rs`
