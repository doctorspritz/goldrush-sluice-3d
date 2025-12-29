# Two-Way Coupling Phase 2 - Progress Documentation

**Date:** 2025-12-28
**Status:** Blocked - sand gets negative horizontal velocity
**Category:** logic-errors

## Goal

Implement two-way coupling where sand particles affect water flow through mixture density, and water carries sand through the sluice.

## What We Know

### Physics Model (Intended)
1. **P2G (Particle to Grid):**
   - Water contributes velocity to grid (APIC with affine terms)
   - Sand tracks volume only (does NOT contribute velocity - this prevents killing water velocity)

2. **Pressure Solve:**
   - Mixture density: `ρ_mix = ρ_water * (1 - φ) + ρ_sand * φ`
   - Higher sand fraction → higher ρ_mix → slower pressure acceleration
   - Capped at 1.5x to prevent complete blockage

3. **G2P (Grid to Particle):**
   - Water: FLIP ratio 0.97 with APIC C matrix
   - Sand: FLIP ratio 0.95 (no APIC) + settling

4. **Settling:**
   - Simple gravity-based: `velocity.y += GRAVITY * SETTLING_FACTOR * dt`
   - SETTLING_FACTOR = 0.62 (based on density difference)

### Key Discovery: Sand P2G Velocity Contribution Kills Water

When sand contributed its (slow, settling) velocity to P2G, the grid velocity became a weighted average of water and sand velocities. This "killed" water velocity because:
- Water has fast horizontal velocity
- Sand has slow velocity due to settling
- Averaged together → water slows down dramatically

**Solution:** Sand should NOT contribute velocity to P2G. Two-way coupling happens through mixture density in pressure gradient only.

## What We Tried

### Attempt 1: Sand Contributes to P2G (FAILED)
- Sand contributed linear velocity (no APIC)
- Result: Water velocity killed, everything slowed down

### Attempt 2: Sand Passive in P2G + Mixture Density (PARTIAL)
- Sand tracks volume but doesn't contribute velocity
- Mixture density affects pressure gradient
- Result: Water flows better, but sand settles too quickly

### Attempt 3: Reduce Sand Diameter (PARTIAL)
- Changed from 2.0 → 0.8 → 0.3 (finer sand)
- Result: Slower settling, but still too fast

### Attempt 4: Increase Water Spawn Rate (PARTIAL)
- Changed from 4 → 40 particles/frame
- Result: Stronger flow, better transport

### Attempt 5: Lower Bedload Thresholds (PARTIAL)
- Made sand easier to re-entrain
- Result: Some improvement but legacy system causing issues

### Attempt 6: Disable Legacy Sediment System (CURRENT)
- Commented out: `apply_sediment_forces()`, `update_particle_states()`, `compute_pile_heightfield()`, `enforce_pile_constraints()`
- Added simple settling in G2P: `velocity.y += GRAVITY * 0.62 * dt`
- Result: Sand gets NEGATIVE horizontal velocity when it should flow downhill

## Current Problem

**Sand gets negative horizontal velocity when flowing downhill.**

This suggests:
1. The FLIP delta might be inverted somewhere
2. The pressure gradient might be pushing sand backwards
3. The settling might be interacting badly with grid velocity

### Relevant Code Locations

**G2P for sand** (`flip.rs:761-773`):
```rust
if particle.is_sediment() {
    const SAND_FLIP_RATIO: f32 = 0.95;
    let grid_delta = v_grid - particle.old_grid_velocity;
    particle.velocity = particle.velocity + grid_delta * SAND_FLIP_RATIO;
    particle.old_grid_velocity = v_grid;

    // Simple settling
    const SETTLING_FACTOR: f32 = 0.62;
    particle.velocity.y += GRAVITY * SETTLING_FACTOR * dt;
    return;
}
```

**Pressure gradient with mixture density** (`flip.rs:565-576`):
```rust
let rho_mix = if total_vol > 0.0 {
    let sand_frac = sand_vol / total_vol;
    let raw_rho = WATER_DENSITY * (1.0 - sand_frac) + SAND_DENSITY * sand_frac;
    raw_rho.min(1.5)  // Cap at 1.5x water density
} else {
    WATER_DENSITY
};
let grad = (self.grid.pressure[idx_right] - self.grid.pressure[idx_left]) * scale;
self.grid.u[u_idx] -= grad / rho_mix;
```

## Files Modified

- `crates/sim/src/flip.rs` - P2G, G2P, pressure gradient
- `crates/sim/src/particle.rs` - Sand diameter (0.3)
- `crates/game/src/main.rs` - Spawn rate (40)
- `crates/sim/tests/two_way_coupling_test.rs` - Updated tests

## Next Steps to Investigate

1. **Check `old_grid_velocity` storage:**
   - Is it being stored at the right time (before or after forces)?
   - Is the sampling position correct for sand?

2. **Check pressure gradient direction:**
   - Is the gradient being applied in the correct direction?
   - Does sand volume affect the gradient direction?

3. **Simplify further:**
   - Try pure PIC for sand (ratio = 0.0) to see if it follows grid correctly
   - Print diagnostics: grid velocity at sand position, FLIP delta

4. **Consider alternative approaches:**
   - Sand could sample water velocity directly (not grid)
   - Drift-flux model where sand velocity = water velocity + slip

## Lessons Learned

1. **Sand contributing to P2G velocity kills water momentum** - Don't do this
2. **Legacy sediment code is complex** - Better to disable and rebuild simply
3. **FLIP ratio affects settling behavior** - Higher ratio = follows flow better but harder to settle
4. **Mixture density needs capping** - Otherwise sand accumulations block water completely
