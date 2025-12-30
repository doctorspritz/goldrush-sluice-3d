# MGPCG Debug Log - What's Been Tried

## Current State
- Branch: `feature/gpu-integration`
- Worktree: `/Users/simonheikkila/Documents/antigravity-dev/goldrush-gpu-integration`
- Status: **ARCHIVED** - Switched to SOR solver due to sync overhead (2024-12-30)

## Resolution (2024-12-30)

**MGPCG has been archived in favor of the simpler Checkerboard SOR solver.**

### Why MGPCG was abandoned:

1. **Sync overhead**: PCG requires 2 dot product readbacks per iteration (blocking GPU→CPU).
   With 20 iterations, that's 40+ blocking operations per frame.

2. **Slower than CPU**: MGPCG took **121.83ms** for pressure solve (78% of total sim time).
   The simpler SOR solver takes **~5ms** - a **22x speedup**.

3. **Poor convergence**: Even after fixing operator consistency, MGPCG achieved only ~44x
   divergence reduction (vs expected 100-1000x).

### Current solution:
```rust
let use_mgpcg = false; // ARCHIVED: Use SOR solver instead
```

The `GpuPressureSolver` uses Checkerboard SOR with no dot products needed.

### Performance comparison:
| Solver | Pressure Time | Convergence |
|--------|---------------|-------------|
| MGPCG (20 iter) | 121.83ms | ~44x |
| SOR | ~5ms | ~27x |

SOR is good enough for visual simulation, and **22x faster**.

---

## Historical Debug Log

### Symptoms (before archival)
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

### 6. CRITICAL: Standardized Laplacian Stencil (2024-12-30)
**Files:**
- `crates/game/src/gpu/shaders/mg_smooth.wgsl`
- `crates/game/src/gpu/shaders/pcg_ops.wgsl`
- `crates/game/src/gpu/shaders/mg_residual.wgsl`

**Problem:** The smoother used **variable neighbor count** (`/ neighbor_count`) while PCG used **fixed stencil**. This meant the V-cycle preconditioner was solving a DIFFERENT equation than the PCG outer loop.

**Root Cause Analysis:**
- `mg_smooth.wgsl`: Was dividing by `neighbor_count` (1-4 depending on solid neighbors)
- `pcg_ops.wgsl`: Was using implicit fixed stencil (sum of `p_neighbor - p_center`)
- `mg_residual.wgsl`: Was tracking `neighbor_count` but never using it (dead code)

**Fix:** All three shaders now use consistent fixed 4-neighbor stencil with Neumann BC mirroring:
```wgsl
// For solid neighbors or boundaries, mirror pressure: p_neighbor = p_center
// Then: Laplacian = (p_L + p_R + p_D + p_U - 4*p_center)
// GS update: p = (p_L + p_R + p_D + p_U - div) * 0.25
```

### 7. PCG Residual Sign - KEPT ORIGINAL (2024-12-30)
**File:** `crates/game/src/gpu/shaders/pcg_ops.wgsl`

**Investigation:** Initially changed residual from `Laplacian(p) - div` to `div + Laplacian(p)`.
This caused simulation explosion because it changed the RHS of the equation.

**Analysis:** The original formula is CORRECT:
- The system is `Laplacian(p) = div`, rewritten as `(-Laplacian)p = -div` for SPD
- So `A = -Laplacian` and `b = -div` (not `+div`)
- Correct residual: `r = b - Ax = (-div) - (-Laplacian(p)) = Laplacian(p) - div` ✓

**Result:** Kept original formula. Only the Laplacian stencil was changed (to use Neumann mirroring).

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

## What Has Been Checked & Fixed (2024-12-30)

1. ✅ **Smoothing shader** - mg_smooth.wgsl now uses same fixed 4-neighbor stencil as pcg_ops.wgsl
2. ✅ **Residual sign** - pcg_ops.wgsl now computes r = b - Ax correctly
3. ✅ **Laplacian consistency** - All three shaders use identical Laplacian with Neumann BC mirroring

## What Still Needs Verification

1. **Bind group buffer assignments** - Are the right buffers bound to the right slots?
2. **V-cycle recursion logic** - Is the recursive V-cycle correct?
3. **Prolongation operator** - Does it match the restriction operator's scaling?
4. **Coarse solve** - Is the coarsest level solving correctly?
5. **Cell type handling** - Consistent across all shaders?
6. **Initial pressure state** - Is x being cleared properly before PCG starts?
7. **h² scaling** - Is divergence properly scaled between CPU and GPU?

---

## Files Modified

1. `crates/game/src/gpu/shaders/pcg_ops.wgsl` - Sign fixes, copy scaling, fixed Laplacian stencil
2. `crates/game/src/gpu/shaders/mg_restrict.wgsl` - Sum to average
3. `crates/game/src/gpu/shaders/mg_smooth.wgsl` - Fixed 4-neighbor stencil with Neumann BC mirroring
4. `crates/game/src/gpu/shaders/mg_residual.wgsl` - Fixed Laplacian stencil, removed dead code
5. `crates/game/src/gpu/mgpcg.rs` - Alpha params, level limiting, debug logging
6. `crates/game/src/main.rs` - PCG iteration count

---

## Recommendation for Fresh Start

Consider comparing against a known-working reference implementation:
- Bridson's "Fluid Simulation for Computer Graphics" has pseudocode
- Robert Bridson's online FLIP solver examples
- The CPU pressure solver in this same codebase (`crates/sim/src/grid/pressure.rs`)

The GPU solver should produce identical results to the CPU solver for the same input.
