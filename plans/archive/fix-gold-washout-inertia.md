# Fix: Gold Particles Wash Out Too Easily

## Problem Statement

Gold particles (density 19.3) are being carried by water flow almost as easily as sand (density 2.65), when they should settle ~3-4x faster and resist horizontal flow much more strongly.

### Root Cause Analysis

**Issue 1: G2P Transfer Ignores Particle Mass**

In `flip.rs:198-216`, the Grid-to-Particle transfer gives ALL particles the same velocity from the grid:

```rust
fn grid_to_particles(&mut self) {
    const ALPHA: f32 = 0.15;
    // ...
    let v_grid = grid.sample_velocity(particle.position);
    let v_pic = v_grid;
    let v_flip = particle.velocity + (v_grid - particle.old_grid_velocity);
    particle.velocity = v_pic * ALPHA + v_flip * (1.0 - ALPHA);  // SAME FOR ALL!
}
```

This is wrong because heavy particles should have more **inertia** - they resist velocity changes from the fluid.

**Issue 2: Settling Only Affects Vertical Velocity**

Our `apply_settling()` only modifies `velocity.y`, but heavy particles should also resist horizontal acceleration. Gold in a stream doesn't just sink faster - it also gets carried sideways slower.

**Issue 3: Wrong Settling Formula for Particle Size**

We use Stokes law: `v_terminal ∝ (ρ - 1)` which gives gold 11x faster settling than sand.

But Stokes only applies to very fine particles in viscous regime. For visible particles (transitional/turbulent regime), the [Rubey formula](https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/erosion-and-sediment-transport-under-construction/fall-velocity-and-settling) is more appropriate:

```
v_fall = F₁ × √((s-1) × g × d)
```

Where s = specific gravity. This gives:
- Gold: √(19.3-1) = √18.3 ≈ 4.28
- Sand: √(2.65-1) = √1.65 ≈ 1.28
- **Ratio: ~3.3x** (not 11x)

## Physics Background

### Inertia and Drag

When a particle moves through fluid, it experiences drag force:
```
F_drag = -k × (v_particle - v_fluid)
```

The acceleration is:
```
a = F_drag / m = -k/m × Δv
```

For heavy particles, `m` is larger, so acceleration is smaller. They resist velocity changes.

### Density Segregation

Per [research on granular segregation](https://www.annualreviews.org/content/journals/10.1146/annurev-fluid-122316-045201):
- Heavy particles experience **granular buoyancy** - they sink through lighter particles
- The segregation velocity depends on density ratio and local shear
- In flowing water, heavy particles "drop out" of the flow behind obstacles (riffles)

## Proposed Solution

### Fix 1: Inertia-Weighted G2P Transfer

Modify `grid_to_particles()` to account for particle mass:

```rust
fn grid_to_particles(&mut self) {
    const ALPHA: f32 = 0.15;
    const INERTIA_SCALE: f32 = 0.3; // How strongly inertia affects coupling

    let grid = &self.grid;
    self.particles.list.par_iter_mut().for_each(|particle| {
        let v_grid = grid.sample_velocity(particle.position);

        // Calculate how much particle should adopt grid velocity
        // Heavy particles (high density) adopt less, resist more
        let density_ratio = particle.density() / 1.0; // ratio to water
        let inertia_factor = 1.0 / (1.0 + INERTIA_SCALE * (density_ratio - 1.0));
        // Gold (19.3): factor = 1/(1 + 0.3*18.3) = 0.15 (adopts 15% of change)
        // Sand (2.65): factor = 1/(1 + 0.3*1.65) = 0.67 (adopts 67% of change)
        // Water (1.0): factor = 1.0 (adopts 100% - moves with fluid)

        // PIC component (direct grid velocity, scaled by inertia)
        let v_pic = particle.velocity + (v_grid - particle.velocity) * inertia_factor;

        // FLIP component (velocity change, scaled by inertia)
        let delta_v = v_grid - particle.old_grid_velocity;
        let v_flip = particle.velocity + delta_v * inertia_factor;

        // Blend
        particle.velocity = v_pic * ALPHA + v_flip * (1.0 - ALPHA);
    });
}
```

### Fix 2: Use Rubey Settling (Square Root Formula)

Update `apply_settling()` to use transitional regime formula:

```rust
fn apply_settling(&mut self, dt: f32) {
    const WATER_DENSITY: f32 = 1.0;
    // Rubey-style coefficient for transitional regime
    // v_fall = RUBEY_COEFF * sqrt(density - 1)
    const RUBEY_COEFF: f32 = 40.0; // Tuned for our scale
    const RELAXATION: f32 = 5.0;

    // ... (cell type check as before) ...

    // Rubey formula: v ∝ √(s-1) for transitional regime
    let density_diff = (density - WATER_DENSITY).max(0.0);
    let v_terminal = RUBEY_COEFF * density_diff.sqrt();

    // Gold: 40 * √18.3 = 171 px/s
    // Sand: 40 * √1.65 = 51 px/s
    // Ratio: 3.3x (physically correct)

    // Relax toward terminal velocity
    let relative_vy = particle.velocity.y - fluid_vy;
    particle.velocity.y = fluid_vy + relative_vy + RELAXATION * (v_terminal - relative_vy) * dt;
}
```

### Fix 3: Horizontal Drag for Heavy Particles

Add horizontal velocity damping based on density - heavy particles resist horizontal flow:

```rust
// In apply_settling, also damp horizontal velocity for heavy particles
let density_ratio = density / WATER_DENSITY;
if density_ratio > 1.5 {
    // Heavy particles experience drag that slows horizontal motion relative to fluid
    let fluid_vx = sample_grid_u(particle.position); // horizontal fluid velocity
    let relative_vx = particle.velocity.x - fluid_vx;

    // Drag pulls particle velocity toward fluid velocity, but slowly for heavy particles
    let drag_factor = 1.0 / density_ratio; // Gold: 0.05, Sand: 0.38
    particle.velocity.x = fluid_vx + relative_vx * (1.0 - drag_factor * RELAXATION * dt);
}
```

## Alternative Approaches Considered

### 1. Two-Phase FLIP
Run separate pressure solves for water and sediment phases. Too expensive for real-time.

### 2. DEM (Discrete Element Method)
Full particle-particle collision physics. Very accurate but O(n²) complexity.

### 3. Drift-Flux Model
Statistical approach treating sediment as a continuous phase with slip velocity. Good for high concentrations, overkill for sparse gold particles.

**Recommendation:** Fix 1 + Fix 2 + Fix 3 combined. Simple, physically motivated, maintains real-time performance.

## Acceptance Criteria

- [ ] Gold particles visibly settle faster than sand behind riffles
- [ ] Gold accumulates at riffle bases while sand washes over
- [ ] Magnetite (density 5.2) shows intermediate behavior
- [ ] Water particles still flow freely (no inertia effect)
- [ ] FPS remains above 30 with current particle counts

## Files to Modify

| File | Changes |
|------|---------|
| `crates/sim/src/flip.rs:198-216` | Add inertia factor to G2P |
| `crates/sim/src/flip.rs:226-277` | Switch to Rubey formula, add horizontal drag |

## Success Metrics

Visual test: After 30 seconds of flow:
- Gold particles (yellow) should be concentrated behind first 2-3 riffles
- Sand particles (tan) should be spread across all riffles
- Clear color gradient from gold-heavy (upstream riffles) to sand-heavy (downstream)

## References

- [Fall Velocity and Settling - HEC HMS](https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/erosion-and-sediment-transport-under-construction/fall-velocity-and-settling) - Rubey formula
- [Particle Segregation in Dense Granular Flows](https://www.annualreviews.org/content/journals/10.1146/annurev-fluid-122316-045201) - Density segregation mechanisms
- [Granular Segregation Velocity Models](https://arxiv.org/html/2410.08350) - Force balance approach
- Multi-phase FLIP research compiled in `MULTI_PHASE_FLIP_RESEARCH.md`
\n\n---\n## ARCHIVED: 2026-01-09\n\n**Status:** Problem persists but approach evolved\n\nWe've moved from the 2D FLIP G2P inertia approach described here to a 3D DEM sediment process. Currently exploring DEM-grid scaling to support gravel and smaller particle sizes.\n\nThis doc describes the original G2P-based thinking which is now superseded by DEM-based sediment physics.
