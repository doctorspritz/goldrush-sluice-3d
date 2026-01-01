# FLIP Velocity Damping - Systemic Issues Investigation

**Date:** 2025-12-27
**Status:** UNRESOLVED - Root cause not yet found
**Category:** logic-errors, physics, investigation
**Severity:** CRITICAL - Water unusable

---

## Problem Statement

Water in FLIP simulation flows like honey/treacle. Velocity dies from 30 units at inlet to ~3 units within 5 seconds. Should accelerate downhill but instead slows to a crawl.

## Damping Sources Identified

### 1. Anisotropic FLIP Ratio (FIXED)

**Location:** `flip.rs:553-582`

**What it did:**
```rust
const FLIP_Y_SURFACE: f32 = 0.70; // 30% PIC damping at surface!
let near_surface = /* check if adjacent to air */;
let flip_y = if near_surface { FLIP_Y_SURFACE } else { FLIP_Y_BULK };
```

**Impact:** 30% of vertical velocity replaced by grid-averaged velocity every frame for particles near free surface.

**Fix applied:** Replaced with uniform `FLIP_RATIO = 0.97`

**Result:** Improved from 6.4% to 9.4% velocity retention. Not sufficient.

---

### 2. Surface Vertical Damping (FIXED)

**Location:** `grid.rs:788` - `damp_surface_vertical()`

**What it did:**
```rust
// Multiplied v.y by depth factor (0 at surface, 1 at depth 3+)
self.v[v_idx] *= t;  // t = 0 at surface!
```

**Impact:** Zeroed out all vertical velocity in top 3 cells of water column.

**Fix applied:** Removed call from update loop

**Result:** Minimal improvement (~8.3% velocity retention)

---

### 3. Near-Pressure SPH (REVERTED)

**Location:** `flip.rs:191` - `apply_near_pressure(dt)`

**What it was:** Clavet-style double-density relaxation for particle separation.

**Problem:** Inappropriate for FLIP. FLIP particles communicate only through grid, not directly.

**Fix attempted:** Removed call

**Result:** WORSE - velocity dropped to 3%, particles jammed at x=50

**Conclusion:** Near-pressure was band-aiding a deeper problem. Removing it exposes particle overlap issues.

---

### 4. Unknown Fundamental Issue (NOT FOUND)

Even with fixes 1 & 2 applied, velocity still dies to 8-9% of inlet.

**Test results:**
```
t=  0s: avg_vx=29 (inlet=30)
t=  5s: avg_vx=4.9
t= 10s: avg_vx=4.3
t= 25s: avg_vx=2.5
```

**Candidates not yet investigated:**
- P2G transfer (weight normalization, kernel function)
- G2P transfer (FLIP delta calculation, sampling positions)
- Pressure solver (over-correction, boundary conditions)
- Grid boundary handling
- Staggered grid sampling positions

---

## Red Herrings

### APIC Re-enable
**Hypothesis:** APIC affine term was disabled, causing angular momentum loss.
**Result:** No improvement when re-enabled.
**Lesson:** APIC was disabled for a reason (didn't help before either).

### Parameter Tweaks
Previous failed attempts:
- Pressure iterations 15 → 40: No effect
- Timestep 1/60 → 1/120: No effect
- Vorticity 0.05 → 0.10: No effect
- Surface skip 3 → 1: No effect

---

## TDD Test Created

**File:** `crates/sim/tests/momentum_test.rs`

**Test:** `test_steady_state_flow_velocity`
- Creates 30-second sluice flow
- Measures inlet vs average velocity
- Asserts velocity ratio > 30%
- Currently fails at 8-9%

---

## Key Learnings

1. **Multiple overlapping damping mechanisms** were added over time to "fix" surface noise symptoms
2. **Each damping was hiding the effects of others** - removing one doesn't help much
3. **The root cause is NOT in the damping code** - it's somewhere in the core FLIP algorithm
4. **TDD is essential** - the test reveals the true magnitude of the problem (90%+ velocity loss)
5. **Parameter tweaking is futile** - need to compare against reference FLIP implementation line-by-line

---

## Next Steps

1. **Compare P2G/G2P against reference** - Bridson notes, SPlisHSPlasH, or Houdini
2. **Check FLIP delta calculation** - Is `new_grid - old_grid` sampled correctly?
3. **Verify weight normalization** - Do B-spline weights sum to 1.0?
4. **Debug with single particle** - Track one particle through entire cycle
5. **Check pressure solver** - Is it over-correcting?

---

## Files Modified (Current State)

| File | Change | Status |
|------|--------|--------|
| `flip.rs:553-558` | Uniform FLIP ratio 0.97 | Applied |
| `flip.rs:165-168` | Removed damp_surface_vertical call | Applied |
| `flip.rs:189` | Removed near_pressure call | REVERTED (made it worse) |

---

## Related Documents

- `docs/compound/failure-claude-impulsive-fixes-2025-12-27.md` - Previous failed attempts
- `docs/compound/failure-apic-reenable-2025-12-27.md` - APIC attempt
- `docs/analysis/anisotropic-flip-damping-analysis.md` - Detailed analysis
- `todos/005-pending-p1-remove-near-pressure.md` - Near-pressure analysis
- `plans/fix-flip-velocity-damping.md` - Original (over-engineered) plan
