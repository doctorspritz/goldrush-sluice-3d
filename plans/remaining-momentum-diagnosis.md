# Remaining Momentum Loss Diagnosis - REVISED

## Current State

**FLIP cycle (no forces):** ✓ PASSES - 0.11% loss per cycle
**Kinetic energy test:** ✓ PASSES - 59.5% retention
**Surface particle test:** ✗ FAILS - 67.6% retention (needs >80%)
**Steady-state flow test:** ✗ FAILS - 3% velocity ratio (needs >30%)

The core FLIP algorithm is working. Something in the forces/boundary/pressure is causing momentum loss.

---

## Review Feedback Summary

Three reviewers identified critical issues with the original plan:

### 1. DON'T CREATE NEW METHOD
- **Existing `update_with_diagnostics()` already measures momentum at each phase**
- No need to duplicate the update pipeline
- Just READ the existing diagnostic output

### 2. STEADY-STATE TEST GEOMETRY IS WRONG
The test creates water that flows **UPHILL**:
```rust
GRAVITY = +350  // Positive = downward in screen coords (+y is down)
floor_y = HEIGHT - 10 - (i * 0.1)  // Floor surface RISES as x increases

// Result: Water spawned at (x=10, y=64) with vx=30 flows toward
// higher floor_y values (lower screen position = higher elevation)
// This is UPHILL against gravity!
```

### 3. MOST LIKELY ROOT CAUSES (from Performance Oracle)
- **70% probability**: Boundary velocity leak - `apply_pressure_gradient()` modifies wall velocities
- **25% probability**: Free-surface BC wrong - air cells should use p=0, not ∂p/∂n=0
- **CFL = 0.5 is too high** - should be <0.25 for accuracy

---

## Revised Test Strategy (SIMPLER)

### Test 0: Verify Test Geometry First
**Before any diagnosis, check if the failing tests have correct geometry.**

```rust
// In steady_state test, add:
println!("Gravity: {} (positive = down)", GRAVITY);
println!("Inlet: x={}, y={}", inlet_x, inlet_y);
println!("Floor at inlet: y={}", floor_y_at_inlet);
println!("Floor at outlet: y={}", floor_y_at_outlet);
println!("Flow direction: {} (should be with gravity for downhill)", ...);
```

**Fix if needed:**
```rust
// Change line 56 to make floor descend WITH gravity:
let floor_y = (10.0 + (i as f32 * slope)) as usize;  // Low at inlet, high at outlet
```

### Test 1: Run Existing Diagnostics
Use `update_with_diagnostics()` which already exists. No new code needed.

**What to look for:**
- Which phase shows largest momentum drop?
- Does gravity ADD momentum (should be positive delta)?
- Does pressure REMOVE more than expected?

### Test 2: Boundary Velocity Leak Detection (NEW - 15 min)
Add simple check:

```rust
pub fn check_boundary_velocity_leaks(&self) -> f32 {
    let mut leak = 0.0;
    for j in 0..self.height {
        leak += self.u[self.u_index(0, j)].abs();
        leak += self.u[self.u_index(self.width, j)].abs();
    }
    for i in 0..self.width {
        leak += self.v[self.v_index(i, 0)].abs();
        leak += self.v[self.v_index(i, self.height)].abs();
    }
    leak
}
```

**Call sequence:**
1. After `enforce_boundary_conditions()` → should be 0
2. After `apply_pressure_gradient()` → if NOT 0, **BUG FOUND**

### Test 3: Pressure Residual Check (NEW - 5 min)
After `solve_pressure(15)`, measure actual divergence:

```rust
let residual = grid.compute_total_divergence();
println!("Pressure residual: {}", residual);
// If > 0.01 per cell → solver not converging
```

### Test 4: CFL Sensitivity (5 min)
Re-run steady-state test with `dt = 1/120` instead of `1/60`:
- If much better → CFL was causing numerical diffusion
- If same → problem is elsewhere

---

## Priority Order

1. **Test 0: Geometry check** (5 min) - May explain entire steady-state failure
2. **Test 2: Boundary leak** (15 min) - 70% probability root cause
3. **Test 1: Read diagnostics** (5 min) - See where momentum actually goes
4. **Test 3: Pressure residual** (5 min) - Verify solver converges
5. **Test 4: CFL sensitivity** (5 min) - Rule out timestep effects

**Total time: ~35 minutes** (not 100+ LOC of new code)

---

## Acceptance Criteria

- Identify which system(s) cause >5% momentum loss per second
- Propose fix for each identified issue
- **All fixes require review before implementation**

## Status

- [ ] Test 0: Verify test geometry
- [ ] Test 2: Check boundary velocity leaks
- [ ] Test 1: Run existing diagnostics
- [ ] Test 3: Check pressure residuals
- [ ] Test 4: CFL sensitivity test
- [ ] Document findings
- [ ] Propose fixes for review
