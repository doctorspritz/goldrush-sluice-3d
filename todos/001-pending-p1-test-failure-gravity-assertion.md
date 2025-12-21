---
status: resolved
priority: p1
issue_id: "001"
tags: [code-review, testing, dfsph]
dependencies: []
---

# Test Failure: Gravity Application Assertion Too Strict

## Problem Statement

The DFSPH test suite has a **failing test** that blocks the build:

```
test test_gravity_application ... FAILED
Velocity Y should be approx 4, got 3.959894
```

The test expects exact velocity match but the simulation includes damping (0.99 factor), causing a small deviation.

## Findings

**Location:** `crates/dfsph/tests/simulation_tests.rs:22`

**Evidence:**
```rust
// Test expects exact: GRAVITY * dt = 250 * 0.016 = 4.0
let expected_vy = physics::GRAVITY * dt;
assert!((updated_p.velocity.y - expected_vy).abs() < 1e-4, ...);

// But simulation.rs:168 applies damping:
p.velocity *= 0.99;

// Actual: 4.0 * 0.99 = 3.96 (matches error message!)
```

The 1e-4 tolerance is far too strict for a simulation with damping applied.

## Proposed Solutions

### Option A: Widen test tolerance (Recommended)
- **Pros:** Quick fix, acknowledges damping is intentional
- **Cons:** Less precise testing
- **Effort:** 5 minutes
- **Risk:** Low

```rust
// Allow 2% tolerance for damping effects
assert!((updated_p.velocity.y - expected_vy).abs() < expected_vy * 0.02, ...);
```

### Option B: Account for damping in expected value
- **Pros:** Precise test that documents damping behavior
- **Cons:** Test is coupled to implementation detail
- **Effort:** 5 minutes
- **Risk:** Low

```rust
let expected_vy = physics::GRAVITY * dt * 0.99; // Include damping
```

### Option C: Remove damping from prediction step
- **Pros:** Cleaner physics separation
- **Cons:** May affect simulation stability
- **Effort:** 30 minutes
- **Risk:** Medium

## Recommended Action

**Option B** - Update the test to account for the damping factor, making the test document the actual behavior.

## Technical Details

- **Affected files:** `crates/dfsph/tests/simulation_tests.rs`
- **Root cause:** Test written before damping was added
- **Related:** Damping at line 168 in `simulation.rs`

## Acceptance Criteria

- [x] `cargo test -p dfsph` passes
- [x] Test documents expected behavior including damping
- [x] No regressions in floor collision test

## Work Log

| Date | Action | Notes |
|------|--------|-------|
| 2025-12-22 | Created | Found during code review |
| 2025-12-22 | Resolved | Updated test to account for damping factor (0.999) in expected velocity calculation. Test now passes with proper documentation of the damping behavior. |

## Resources

- Related code: `crates/dfsph/src/simulation.rs:168`
- Test file: `crates/dfsph/tests/simulation_tests.rs`
