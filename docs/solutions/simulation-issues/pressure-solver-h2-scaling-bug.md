---
title: "Pressure Solver Missing h² Scaling Factor"
category: simulation-issues
component: sim
severity: critical
symptoms:
  - Divergence only reduced by 25% instead of 90%+
  - Water compacting instead of spreading
  - Vertical ejections
  - Velocity explosion over time
tags: [flip, grid, pressure, poisson]
date: 2025-12-20
---

# Pressure Solver Missing h² Scaling Factor

## Symptom

The pressure solver was only reducing divergence by ~25% per frame instead of the expected 90%+. This caused:
- Water appearing viscous/compacting
- Velocity explosion (max_v reaching 2860)
- Vertical particle ejections

## Root Cause

The Poisson equation for pressure is:

```
∇²p = div
```

Discretized on a MAC grid:

```
(p_L + p_R + p_B + p_T - 4*p) / h² = div
```

Solving for p (Gauss-Seidel update):

```
p = (p_L + p_R + p_B + p_T - h²*div) / 4
```

The code was missing the **h²** factor:

```rust
// WRONG - missing h² scaling
self.pressure[idx] = (p_left + p_right + p_bottom + p_top - div) * 0.25;
```

With `cell_size = 2.0`, this made pressure corrections **4x too weak**.

## Solution

Add the h² scaling factor to the pressure update:

```rust
// CORRECT - include h² scaling
let h_sq = self.cell_size * self.cell_size;
self.pressure[idx] = (p_left + p_right + p_bottom + p_top - h_sq * div) * 0.25;
```

**File:** `crates/sim/src/grid.rs` lines 363-367

## Results

| Metric | Before | After |
|--------|--------|-------|
| Divergence reduction | 25% | ~100% (div→0) |
| Max velocity | 2860 (exploding) | 166 (stable) |
| Stress tests | 2/13 passing | 13/13 passing |

## Prevention

When implementing numerical methods:
1. **Derive from first principles** - Don't copy code without understanding the math
2. **Check dimensional analysis** - Ensure factors of h, h², dt are correct
3. **Unit tests with known solutions** - Static water column should have zero divergence after pressure solve
4. **Vary cell_size in tests** - Bugs often appear when h ≠ 1.0

## Related

- The divergence computation (`compute_divergence()`) correctly uses `1/h` scaling
- The pressure gradient application (`apply_pressure_gradient()`) correctly uses `1/h` scaling
- Only the Gauss-Seidel update was missing the `h²` factor
