---
status: pending
priority: p3
issue_id: "037"
tags: [code-review, security, unsafe]
dependencies: []
---

# Unsafe Static Mutable State in Random Number Generator

## Problem Statement

The `rand_float()` function uses `unsafe` code with a static mutable variable. While the application is currently single-threaded, this pattern is fragile and could cause undefined behavior if called from multiple threads.

## Findings

**Location:** `crates/game/src/main.rs`, lines 1585-1591

```rust
fn rand_float() -> f32 {
    static mut SEED: u32 = 12345;
    unsafe {
        SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
        (SEED as f32) / (u32::MAX as f32)
    }
}
```

## Proposed Solutions

### Option A: Use rand crate (already a dependency)
**Pros:** Thread-safe, well-tested
**Cons:** Slightly heavier
**Effort:** Trivial
**Risk:** None

### Option B: Use fastrand crate
**Pros:** Lightweight, fast
**Cons:** New dependency
**Effort:** Trivial
**Risk:** None

### Option C: Use thread_local! with Cell
**Pros:** No new dependencies, thread-safe
**Cons:** More complex
**Effort:** Small
**Risk:** Low

## Acceptance Criteria

- [ ] No static mut in the codebase
- [ ] Thread-safe RNG implementation
- [ ] No performance regression

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-19 | Created | Found by security-sentinel agent |
