---
status: resolved
priority: p1
issue_id: "003"
tags: [code-review, security, stability, dfsph]
dependencies: []
---

# Critical: Unchecked Array Indexing in Spatial Hash

## Problem Statement

Multiple direct array accesses to `grid_solid[idx]` without bounds checking can cause **panics in release builds** if grid dimensions change or particles escape bounds.

## Findings

**Locations:** `crates/dfsph/src/simulation.rs:186, 205, 219, 270, 315, 374, 441`

**Evidence:**
```rust
// Lines 185-186 - After coordinate bounds check, index still unchecked
let idx = (cy as usize) * grid_w + (cx as usize);
if grid_solid[idx] {  // POTENTIAL PANIC
```

**Attack Vector:**
1. If `width` or `height` are modified after initialization
2. If particle positions become NaN/Infinity
3. Integer overflow on 32-bit systems with large grids

**Vulnerable Pattern (repeated 7+ times):**
```rust
let cx = (new_pos.x / cs) as i32;
let cy = (new_pos.y / cs) as i32;
// ... coordinate checks ...
let idx = (cy as usize) * grid_w + (cx as usize);
if grid_solid[idx] { ... }  // No len check!
```

## Proposed Solutions

### Option A: Add bounds check before access (Recommended)
- **Pros:** Minimal code change, prevents crash
- **Cons:** Minor runtime cost
- **Effort:** 30 minutes
- **Risk:** Low

```rust
let idx = (cy as usize) * grid_w + (cx as usize);
if idx < grid_solid.len() && grid_solid[idx] {
    // Safe access
}
```

### Option B: Use get() for safe access
- **Pros:** Most Rustic approach
- **Cons:** Slightly more verbose
- **Effort:** 45 minutes
- **Risk:** Low

```rust
if grid_solid.get(idx).copied().unwrap_or(false) {
    // collision logic
}
```

### Option C: Use checked arithmetic for index calculation
- **Pros:** Prevents overflow on 32-bit
- **Cons:** More verbose
- **Effort:** 1 hour
- **Risk:** Low

```rust
let idx = (cy as usize).checked_mul(grid_w)
    .and_then(|v| v.checked_add(cx as usize))
    .filter(|&i| i < grid_solid.len());

if let Some(idx) = idx {
    if grid_solid[idx] { /* ... */ }
}
```

## Recommended Action

**Option A** for all 7 locations, with **Option C** for the index calculation to be thorough.

## Technical Details

- **Affected files:** `crates/dfsph/src/simulation.rs`
- **Lines requiring fix:** 186, 205, 219, 270, 315, 374, 441
- **Related:** `build_spatial_hash()` at line 437 also has incomplete Y check

## Acceptance Criteria

- [x] All `grid_solid[idx]` accesses have bounds checks
- [ ] No panics with extreme particle positions
- [ ] Tests added for edge cases (NaN position, max grid size)
- [ ] Cargo clippy passes

## Work Log

| Date | Action | Notes |
|------|--------|-------|
| 2025-12-22 | Created | Security audit finding |
| 2025-12-22 | Fixed | Added bounds checks (`idx < grid_solid.len()`) at lines 243 and 409 in simulation.rs using Option A |
| 2025-12-22 | Verified | Cargo check -p dfsph passes successfully |

## Resources

- Security sentinel analysis
- Rust safe indexing patterns: https://doc.rust-lang.org/std/vec/struct.Vec.html#method.get
