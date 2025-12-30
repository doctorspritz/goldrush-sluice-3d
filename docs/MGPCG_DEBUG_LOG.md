# MGPCG Debug Log - What's Been Tried

## Current State
- Branch: `feature/gpu-integration`
- Worktree: `/Users/simonheikkila/Documents/antigravity-dev/goldrush-gpu-integration`
- Status: PCG converges slowly, div_out still too high (5.7 after 20 iterations with input of 124)

## Symptoms
1. Initial version: NaN explosion after a few seconds
2. After fixes: No NaN, but poor convergence (div_out stays high)

---

## Fixes Applied

### 1. Dot Finalize Shader (Previous Session)
**File:** `crates/game/src/gpu/shaders/pcg_ops.wgsl`

**Problem:** Only loading 256 of 1024 partial sums in finalization.

**Fix:** Added strided loop to handle all partial sums:
```wgsl
for (var i = tid; i < num_partials; i += 256u) {
    local_sum += buffer_d[i];
}
```

### 2. Restriction Operator - Sum to Average
**File:** `crates/game/src/gpu/shaders/mg_restrict.wgsl`

**Problem:** Restriction was summing 2x2 fine grid values instead of averaging.

**Fix:** Changed from `sum` to `sum / f32(count)`

**Rationale:** Multigrid restriction should preserve amplitude to match prolongation.

### 3. Sign Conventions for CG
**File:** `crates/game/src/gpu/shaders/pcg_ops.wgsl`

**Problem:** rz and pAp were negative (should be positive for CG).

**Root cause:** Discrete Laplacian `L = sum(neighbors - center)` is negative semi-definite. CG requires positive definite operator.

**Fixes applied:**
1. `compute_pcg_residual`: Changed from `r = div - Laplacian(p)` to `r = Laplacian(p) - div`
2. `apply_laplacian`: Now returns `-Laplacian(p)` instead of `Laplacian(p)`
3. `copy_buffer`: Modified to support scaling via `params.alpha`

**File:** `crates/game/src/gpu/mgpcg.rs`
4. In `apply_preconditioner`: Set `alpha = -1.0` when copying r to divergence (negate input to V-cycle)
5. Set `alpha = 1.0` when copying pressure to z

### 4. Limited V-cycle to 2 Levels
**File:** `crates/game/src/gpu/mgpcg.rs`

**Change:** `let max_level = 1.min(self.num_levels - 1);`

**Rationale:** Testing if multi-level V-cycle has accumulated errors.

### 5. Increased PCG Iterations
**File:** `crates/game/src/main.rs`

**Change:** From 10 to 20 iterations.

---

## Current Test Results
```
GPU: div_in=0.14 -> div_out=0.00   (first frame - good)
GPU: div_in=123.97 -> div_out=5.70 (later frames - still too high)
```

- rz and pAp are now positive (correct)
- No NaN or Inf values
- But divergence reduction is poor (~20x in 20 iterations, should be much better)

---

## What Has NOT Been Checked

1. **Bind group buffer assignments** - Are the right buffers bound to the right slots?
2. **V-cycle recursion logic** - Is the recursive V-cycle correct?
3. **Smoothing shader** - Is mg_smooth.wgsl solving the same equation as pcg_ops.wgsl?
4. **Prolongation operator** - Does it match the restriction operator's scaling?
5. **Coarse solve** - Is the coarsest level solving correctly?
6. **Cell type handling** - Consistent across all shaders?
7. **Initial pressure state** - Is x being cleared properly before PCG starts?

---

## Files Modified

1. `crates/game/src/gpu/shaders/pcg_ops.wgsl` - Sign fixes, copy scaling
2. `crates/game/src/gpu/shaders/mg_restrict.wgsl` - Sum to average
3. `crates/game/src/gpu/mgpcg.rs` - Alpha params, level limiting, debug logging
4. `crates/game/src/main.rs` - PCG iteration count

---

## Recommendation for Fresh Start

Consider comparing against a known-working reference implementation:
- Bridson's "Fluid Simulation for Computer Graphics" has pseudocode
- Robert Bridson's online FLIP solver examples
- The CPU pressure solver in this same codebase (`crates/sim/src/grid/pressure.rs`)

The GPU solver should produce identical results to the CPU solver for the same input.
