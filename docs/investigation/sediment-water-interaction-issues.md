# Sediment-Water Interaction Issues

**Date:** 2025-12-28
**Status:** RESOLVED - Phase 1 fix already implemented

**Resolution:** Only water particles mark cells as Fluid (lines 261-273 in flip.rs).
Sediment is purely Lagrangian - carried by water via drag forces.
Divergence stable at 5-11 even with sediment spawning.

## Summary

The FLIP water simulation works correctly in isolation. When sediment particles are added, the water exhibits compression/stacking behavior at riffle boundaries. The pressure solver cannot maintain incompressibility.

## Observed Symptoms

1. **Water-only**: Works fine, divergence stays low (0.3-15), flow is natural
2. **With sediment**: Divergence explodes (106 → 359 within seconds), water compresses instead of flowing over riffles
3. **Visual behavior**: "Particle count goes up but volume doesn't increase" - classic compression symptom

## Diagnostic Data

### Water-only run (first 60 seconds)
```
t= 43s: 12207 p, fps= 35, sim=23.9ms, div= 0.3
t= 45s: 12816 p, fps= 32, sim=25.6ms, div=63.5
t= 57s: 16419 p, fps= 28, sim=31.0ms, div= 1.4
```
Divergence stays manageable even at 16k+ particles.

### With sediment (after reset at t=60s)
```
t= 62s:  1790 p, fps= 57, sim=15.9ms, div=106.4
t= 66s:  3590 p, fps= 55, sim=16.3ms, div=161.2
t= 67s:  4190 p, fps= 50, sim=16.3ms, div=268.5
t= 70s:  5990 p, fps= 54, sim=16.8ms, div=343.5
t= 76s:  8990 p, fps= 49, sim=17.5ms, div=359.2
```
Divergence grows rapidly with fewer particles than water-only case.

## Current Sediment Implementation

### P2G Transfer (flip.rs:297-300)
```rust
for particle in self.particles.iter() {
    // Sediment is passive (one-way coupling) - doesn't affect grid velocities
    let weight_scale = if particle.is_sediment() { 0.0 } else { 1.0 };
```
Sediment does NOT contribute to grid velocity.

### Cell Classification (flip.rs:237-238)
```rust
/// ALL particles mark cells as fluid (for pressure boundary conditions).
/// Sediment doesn't contribute to P2G velocity, but it occupies space.
```
Sediment DOES mark cells as fluid.

### G2P Transfer (flip.rs:527-530)
```rust
if particle.is_sediment() {
    particle.old_grid_velocity = v_grid;
    return;  // Early return - no APIC for sediment
}
```
Sediment samples grid velocity but doesn't participate in APIC.

## Hypothesized Root Causes

### 1. Cell Classification Mismatch
Sediment marks cells as Fluid but contributes zero velocity to P2G. This creates cells where:
- Pressure solver sees "fluid" and expects velocity
- Actual grid velocity is only from water particles
- Creates phantom divergence sources

### 2. Volume Displacement Not Modeled
When sediment occupies space:
- Water should be displaced
- But sediment doesn't push water away (no two-way coupling)
- Results in water and sediment occupying same volume

### 3. Drag Force Imbalance
Sediment experiences drag from water velocity but:
- Water doesn't feel equal-and-opposite drag from sediment
- Violates Newton's third law
- May cause momentum to disappear from system

## Pressure Solver Observations

The multigrid solver (2 V-cycles) cannot converge when:
- Divergence sources appear faster than solver can remove them
- Cell classification doesn't match actual fluid content
- Residual divergence accumulates frame over frame

## What Works

1. Water-only FLIP simulation
2. Pressure solver convergence without sediment
3. Flow over riffles without sediment
4. Multigrid V-cycle implementation

## What Needs Rework

1. **Sediment-fluid coupling model** - Current one-way coupling breaks incompressibility
2. **Cell classification** - Sediment shouldn't mark cells as Fluid if it doesn't contribute velocity
3. **Volume displacement** - Need proper mechanism for sediment to displace water
4. **Two-way coupling** - Water should feel reaction force from sediment drag

## Potential Solutions to Explore

### Option A: Don't mark sediment cells as Fluid
```rust
// Only water marks cells as fluid
if particle.is_water() {
    self.grid.mark_fluid(i, j);
}
```
Risk: May break sediment collision detection.

### Option B: Two-way coupling
Add reaction force to water when sediment experiences drag:
```rust
// In apply_sediment_forces
water_force -= sediment_drag;  // Equal and opposite
```
Risk: Complex, may need restructuring.

### Option C: Separate sediment simulation
Run sediment as purely Lagrangian particles without grid interaction:
- Use DEM (Discrete Element Method) for sediment
- Only interact with water through local SPH-like forces
Risk: Performance, complexity.

### Option D: Volume fraction approach
Track sediment volume fraction per cell:
- Reduce effective fluid volume based on sediment concentration
- Adjust pressure solve accordingly
Risk: Significant code changes.

## Files Involved

- `crates/sim/src/flip.rs` - Main simulation, P2G/G2P, sediment forces
- `crates/sim/src/grid.rs` - Pressure solver, cell classification
- `crates/sim/src/particle.rs` - Particle types, is_sediment()
- `crates/game/src/main.rs` - Spawn rates, diagnostics

## Next Steps

1. Decide on sediment coupling approach
2. Prototype simplest fix first (Option A)
3. Test with visual inspection and divergence monitoring
4. If needed, implement more sophisticated coupling

## Related Files

- `todos/` - May have related pending work items
- `docs/solutions/` - Check for related documented solutions
