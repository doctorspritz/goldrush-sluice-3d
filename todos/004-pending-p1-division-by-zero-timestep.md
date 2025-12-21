---
status: resolved
priority: p1
issue_id: "004"
tags: [code-review, security, stability, dfsph]
dependencies: []
---

# Critical: Division by Zero - Timestep Validation Missing

## Problem Statement

The velocity update divides by `dt` without validation. If `dt` is zero, negative, or NaN, this causes **division by zero or NaN propagation** corrupting the simulation state.

## Findings

**Location:** `crates/dfsph/src/simulation.rs:421`

**Evidence:**
```rust
p.velocity = (p.position - *old_pos) / dt;  // NO dt VALIDATION
```

**Attack Vectors:**
1. User calls `update(0.0)` - division by zero → Infinity
2. User calls `update(-1.0)` - backward physics → chaos
3. User calls `update(f32::NAN)` - NaN propagation → all particles corrupted

**Impact:**
- Infinity/NaN velocities propagate through simulation
- Particle positions become corrupted
- Simulation enters unrecoverable invalid state
- Game crash or visual glitches

## Proposed Solutions

### Option A: Early return for invalid dt (Recommended)
- **Pros:** Simple, no allocation, fast path for valid input
- **Cons:** Silently skips frame (may mask bugs)
- **Effort:** 5 minutes
- **Risk:** Low

```rust
pub fn update(&mut self, dt: f32) {
    if dt <= 0.0 || !dt.is_finite() {
        return; // Skip invalid timesteps
    }
    // ... rest of update
}
```

### Option B: Debug assert with early return
- **Pros:** Catches bugs in development
- **Cons:** Only active in debug builds
- **Effort:** 5 minutes
- **Risk:** Low

```rust
pub fn update(&mut self, dt: f32) {
    debug_assert!(dt > 0.0 && dt.is_finite(), "Invalid timestep: {}", dt);
    if dt <= 0.0 || !dt.is_finite() {
        return;
    }
    // ...
}
```

### Option C: Clamp dt to valid range
- **Pros:** Always produces output, handles edge cases
- **Cons:** May mask frame timing issues
- **Effort:** 10 minutes
- **Risk:** Low

```rust
const MIN_DT: f32 = 1.0 / 240.0;  // 240fps max
const MAX_DT: f32 = 1.0 / 15.0;   // 15fps min
let dt = dt.clamp(MIN_DT, MAX_DT);
```

## Recommended Action

**Option B** - Debug assert for development + early return for robustness.

## Technical Details

- **Affected files:** `crates/dfsph/src/simulation.rs`
- **Location:** Top of `update()` function (line 140)
- **Related:** Also validate `dt` in kernel division at line 463

## Acceptance Criteria

- [ ] `update(0.0)` does not crash
- [ ] `update(-1.0)` does not corrupt state
- [ ] `update(f32::NAN)` does not crash
- [ ] Debug builds assert on invalid dt
- [ ] Unit tests added for edge cases

## Work Log

| Date | Action | Notes |
|------|--------|-------|
| 2025-12-22 | Created | Security audit finding |
| 2025-12-22 | Resolved | Implemented Option B: Added debug_assert and early return validation at start of update() function. Code compiles successfully. |

## Resources

- Security sentinel analysis
- Related: M-2 (kernel division edge cases)
