---
status: pending
priority: p2
issue_id: "029"
tags: [code-review, performance, physics]
dependencies: []
---

# Pressure Solver Over-Iteration (120 iterations)

## Problem Statement

The pressure solver runs **120 iterations** of Red-Black SOR, creating **240 compute dispatches**. Most FLIP simulations achieve visual quality with 40-80 iterations.

## Findings

**Location:** `crates/game/src/main.rs`, line 35
```rust
const PRESSURE_ITERS: u32 = 120;
```

### Analysis:
- Grid size: 162x52x40
- Optimal omega for N=162: `2/(1+sin(π/162)) ≈ 1.96`
- Current omega=1.85 is conservative
- Convergence typically achieved in 40-80 iterations for visual quality

## Proposed Solutions

### Option A: Reduce to 60-80 iterations (Recommended)
**Pros:** 40-50% reduction in pressure solve time
**Cons:** May affect extreme scenarios
**Effort:** Trivial
**Risk:** Low

### Option B: Implement multi-grid preconditioner
**Pros:** Faster convergence
**Cons:** Complex implementation
**Effort:** Large

## Acceptance Criteria

- [ ] Pressure iterations reduced to 60-80
- [ ] Visual quality maintained
- [ ] Frame time improved

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-19 | Created | Found by performance-oracle agent |
