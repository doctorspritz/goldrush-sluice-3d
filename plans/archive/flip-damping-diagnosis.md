# FLIP Velocity Damping Diagnosis

## Problem

Water loses 92% momentum in 1 second (expected loss: <30%)

## Root Cause #1: KERNEL MISMATCH ✓ FIXED

**Original Test Result (isolated FLIP cycle, NO forces):**
```
Per-cycle loss: 4.25%
After 10 cycles: 64.8% retention
```

**After Fix:**
```
Per-cycle loss: 0.11%
After 10 cycles: 98.9% retention
```

**Cause:** `store_old_velocities()` used bilinear (4-point), but `grid_to_particles()` used quadratic B-spline (9-point). Fixed by making both use quadratic B-spline.

**Fix location:** `flip.rs:390-464` - rewrote `store_old_velocities()` to use quadratic B-spline sampling.

---

## Remaining Issues (NOT YET FIXED)

### Full Momentum Test Results (after kernel fix):

1. **test_kinetic_energy_conservation**: ✓ PASSES
   - Retention: 59.5% (threshold: >50%)
   - Improved from ~20% to 59.5%

2. **test_surface_particles_not_overdamped**: ✗ FAILS
   - Retention: 67.6% (needs >80%)
   - Lost 32.4% in 0.5s (expected <20%)

3. **test_steady_state_flow_velocity**: ✗ FAILS
   - Velocity ratio: 0.03 (needs >0.30)
   - Water at 3% of inlet velocity after 30s

### Observations

The kernel fix improved isolated FLIP cycles dramatically but full simulation still has issues.

**What's happening:**
- Water accumulates (particle count grows to 3600)
- max_x only reaches 60 in 30 seconds (should be much further)
- avg_vx drops from 30 to ~1 very quickly

**Potential remaining causes (NOT YET INVESTIGATED):**
1. Pressure solver removing too much momentum (62% reduction per step)
2. Test geometry issues (sloped channel setup may be flawed)
3. Boundary interactions
4. Weight partition bug (~95% instead of 100%)

---

## Status

- [x] Kernel mismatch diagnosed
- [x] Kernel mismatch fix implemented and verified
- [ ] Full momentum tests still failing
- [ ] Need new hypothesis for remaining issues
