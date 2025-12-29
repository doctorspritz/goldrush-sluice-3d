# Failure: Claude's Impulsive Surface-Level Fixes

**Date:** 2025-12-27
**Status:** UNRESOLVED - Water still flows like honey
**Category:** logic-errors, physics, process-failure

## Problem Statement

Water in FLIP simulation behaves like honey/treacle instead of flowing naturally. User asked for comprehensive diagnosis. Claude made surface-level parameter tweaks without finding the actual root cause.

## ALL Attempted Fixes (ALL FAILED)

### Round 1: Parameter Tweaks
- `flip.rs:154`: Pressure iterations 15 → 40
- `main.rs:389`: Timestep 1/60 → 1/120
- `flip.rs:149`: Vorticity confinement 0.05 → 0.10
- `grid.rs:932`: Surface skip depth 3 → 1

**Result: Water STILL flows like honey.**

### Round 2: Pure FLIP (No PIC Smoothing)
- `flip.rs:522`: FLIP_RATIO 0.99 → 1.0

Hypothesis: PIC blending was smoothing out velocity.

**Result: Still ~2% momentum loss per frame. FAILED.**

### Round 3: Disabled APIC Affine Velocity
- `flip.rs:308-316, 351-359`: Commented out APIC C matrix contribution in P2G

Hypothesis: APIC affine term was damping rotational motion.

**Result: Still losing momentum. FAILED.**

## Diagnostic Results

Added momentum tracking through the entire pipeline:

```
MOMENTUM: particles=10471 → P2G_grid=8092 → +gravity=582580 → +pressure=224370 → G2P_particles=10280 → advect=10225
```

### What This Shows:
1. **Particles start with**: 10471 total velocity magnitude
2. **After P2G**: Grid has 8092 (23% lost to grid transfer)
3. **After gravity**: 582580 (gravity adds massive velocity)
4. **After pressure**: 224370 (pressure removes 62% - this is correct, enforcing incompressibility)
5. **After G2P**: Particles have 10280 (FLIP recovers most of the loss)
6. **After advection**: 10225 (slight loss)

### The Problem:
- **Net loss per frame: ~2%** (10471 → 10225)
- **Compounded over 1 second (60 frames)**: 0.98^60 = 30% retained = **70% velocity lost**
- **This is why water flows like honey**

### What's NOT the cause:
- ❌ PIC smoothing (pure FLIP still loses 2%)
- ❌ APIC affine velocity (disabled it, still loses momentum)
- ❌ Vorticity confinement (increased it, no effect)
- ❌ Pressure iterations (increased from 15 to 40, no effect)
- ❌ Surface skip depth (changed from 3 to 1, no effect)

## What Claude Did Wrong

### 1. Made Changes Without Understanding
Made 6 different changes hoping something would work. None did.

### 2. Didn't Compare With Reference Implementation
The FLIP algorithm has specific formulas. I should have compared each step with academic papers or known-working implementations.

### 3. Focused on Symptoms Not Cause
Kept trying to "boost" velocity instead of finding WHERE it's being lost.

## Areas Still Not Investigated

1. **P2G Weight Normalization**
   - Are quadratic B-spline weights summing to 1.0?
   - Is the mass weighting correct?

2. **G2P FLIP Formula**
   - Is `new_grid_vel - old_grid_vel` sampled at correct positions?
   - Is particle velocity updated correctly?

3. **Staggered Grid Sampling**
   - Are velocities being sampled from correct staggered positions?
   - Is bilinear interpolation handling grid boundaries correctly?

4. **Pressure Gradient Application**
   - Is the gradient being applied to the correct velocity components?
   - Is the sign correct?

5. **Grid Cell Classification**
   - Are "fluid" vs "air" cells being classified correctly?
   - Could boundary conditions be causing velocity loss?

## The Core Mystery

The ~2% per-frame loss happens somewhere in the P2G → pressure → G2P cycle. But:
- P2G shows 23% loss to grid
- G2P recovers most of it via FLIP
- Net is ~2% loss

**This 2% is coming from somewhere in the FLIP update formula or the grid velocity sampling.**

## Lessons Learned

1. **Don't guess at parameters** - they didn't help
2. **Diagnostics first** - should have done this immediately
3. **Compare with reference** - need to check against known-working FLIP code
4. **The bug is likely mathematical** - a sign error, wrong position, or incorrect formula

## Next Steps (For Fresh Start)

1. Read a reference FLIP implementation (e.g., Bridson's notes, SPlisHSPlasH)
2. Compare G2P formula line-by-line
3. Check P2G weight computation
4. Verify staggered grid sampling positions
5. Look for off-by-one or sign errors
