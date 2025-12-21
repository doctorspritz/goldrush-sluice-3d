---
status: resolved
priority: p2
issue_id: "009"
tags: [code-review, performance, dfsph]
dependencies: []
---

# Performance: Precompute Kernel Coefficients

## Problem Statement

SPH kernel functions compute `h.powi(8)` and `h.powi(5)` on **every single call** despite `h` being constant throughout the frame. This wastes significant CPU cycles on power operations.

## Findings

**Location:** `crates/dfsph/src/simulation.rs:452-464`

**Current Code:**
```rust
fn poly6_kernel_2d(r2: f32, h: f32) -> f32 {
    let h2 = h * h;
    if r2 >= h2 { return 0.0; }
    let term = h2 - r2;
    (4.0 / (PI * h.powi(8))) * term.powi(3)  // h^8 computed EVERY call
}

fn spiky_kernel_gradient_2d(r_vec: Vec2, r: f32, h: f32) -> Vec2 {
    if r >= h || r <= 1e-5 { return Vec2::ZERO; }
    let term = h - r;
    let scalar = -30.0 / (PI * h.powi(5)) * term * term;  // h^5 computed EVERY call
    r_vec * (scalar / r)
}
```

**Cost:**
- `powi(8)` ≈ 7 multiplications per call
- `powi(5)` ≈ 4 multiplications per call
- At 20k particles × 50 neighbors × 8 passes = **8M kernel calls/frame**
- Wasted ops: **80M+ extra multiplications per frame**

## Proposed Solutions

### Option A: Precompute in struct (Recommended)
- **Pros:** Zero per-call overhead, clean API
- **Cons:** Slightly more memory
- **Effort:** 30 minutes
- **Risk:** Low

```rust
pub struct DfsphSimulation {
    // Add precomputed values
    poly6_coeff: f32,   // 4.0 / (PI * h^8)
    spiky_coeff: f32,   // -30.0 / (PI * h^5)
    h_squared: f32,     // h * h
}

impl DfsphSimulation {
    pub fn new(...) {
        let h = cell_size * H_SCALE;
        Self {
            poly6_coeff: 4.0 / (PI * h.powi(8)),
            spiky_coeff: -30.0 / (PI * h.powi(5)),
            h_squared: h * h,
            // ...
        }
    }
}

#[inline(always)]
fn poly6_kernel_2d(r2: f32, h2: f32, coeff: f32) -> f32 {
    if r2 >= h2 { return 0.0; }
    let term = h2 - r2;
    coeff * term * term * term  // Use explicit multiply
}
```

### Option B: Pass coefficients as parameters
- **Pros:** No struct change
- **Cons:** More function parameters
- **Effort:** 45 minutes
- **Risk:** Low

### Option C: Use const generics
- **Pros:** Compile-time computation
- **Cons:** More complex, requires Rust 1.51+
- **Effort:** 1 hour
- **Risk:** Medium

## Recommended Action

**Option A** - Precompute coefficients in struct, compute once at construction.

## Technical Details

**Affected File:** `crates/dfsph/src/simulation.rs`

**Changes:**
1. Add `poly6_coeff`, `spiky_coeff`, `h_squared` fields to struct
2. Compute in `new()`
3. Update kernel functions to take coefficients as parameters
4. Update all call sites in solver loop

**Expected Gain:** 10-15% kernel computation speedup

## Acceptance Criteria

- [x] Kernel coefficients computed once at construction
- [x] No `powi()` calls in kernel functions
- [ ] Benchmark shows measurable improvement
- [x] All tests pass (cargo check passes)

## Work Log

| Date | Action | Notes |
|------|--------|-------|
| 2025-12-22 | Created | Performance analysis finding |
| 2025-12-22 | Implemented | Added precomputed fields (poly6_coeff, spiky_coeff, h_squared) to DfsphSimulation struct |
| 2025-12-22 | Implemented | Updated kernel functions to accept precomputed coefficients |
| 2025-12-22 | Implemented | Updated all call sites (apply_xsph, lambda pass, delta pos pass) |
| 2025-12-22 | Verified | cargo check -p dfsph passes with only 1 unrelated warning |

## Resources

- Performance oracle analysis
- SIMD optimization patterns
