# Hybrid Fluid Architecture: PBF + PIC/FLIP

## Overview

Based on the hydrodynamic analysis, neither PBF nor FLIP alone solves the mining game requirements. We need a **hybrid architecture** that uses each solver where it excels.

| Phase | Solver | Why |
|-------|--------|-----|
| Dirt piles (dry) | PBF | Granular stacking, angle of repose |
| Sluice water | FLIP | Vorticity preservation, riffle eddies |
| Sediment (gold/sand) | Drift-Flux | One-way coupled to FLIP, slip velocity |
| Tailings pile | PBF or Heightfield | Post-sluice stacking |

---

## The Core Problem

### Sluice Box Physics
The sluice separates materials by **density stratification**:
- Water (SG 1.0) creates turbulent vortices behind riffles
- Light sand (SG 2.65) gets ejected by vortex energy
- Heavy gold (SG 19.3) "slips" through vortex and settles

**Critical insight**: The vortex behind each riffle is the "trap" - it must be energetic enough to eject sand but allow gold to penetrate.

### Why PBF Fails for Sluice
1. **Vorticity damping** - Position projection acts as low-pass filter
2. **Mass ratio instability** - Gold (19:1) causes tunneling/clumping
3. **Laminar appearance** - Water reattaches smoothly instead of recirculating

### Why FLIP Fails for Dirt
1. **No granular stacking** - Particles form flat pools, not piles
2. **No friction model** - Can't simulate angle of repose

---

## Phase 1: Optimize Current FLIP for Sluice

### 1.1 Pre-allocate P2G Buffers
**File**: `crates/sim/src/flip.rs:121-124`

```rust
// Current: allocates 4 vectors per frame (~320KB)
let mut u_sum = vec![0.0f32; self.grid.u.len()];

// Fix: Add as struct fields
pub struct FlipSimulation {
    // ... existing ...
    u_sum: Vec<f32>,
    u_weight: Vec<f32>,
    v_sum: Vec<f32>,
    v_weight: Vec<f32>,
}
```

**Impact**: 5-10% FPS improvement

### 1.2 Add Rayon Parallelism to FLIP
**Priority operations**:
1. Grid-to-Particles (G2P) - trivially parallel
2. Advection - independent per particle
3. Particle separation - use atomics for impulse accumulation

```rust
use rayon::prelude::*;

// G2P transfer
self.particles.list.par_iter_mut().for_each(|particle| {
    let v_grid = self.grid.sample_velocity(particle.position);
    // FLIP blend...
});
```

**Impact**: 2-3x speedup on 4+ cores

### 1.3 Implement Drift-Flux Sediment Transport
**Key equation** (Rubey settling velocity):
```
w_s = F * sqrt((SG - 1) * g * d)
```

Where:
- SG = specific gravity (gold=19.3, sand=2.65)
- d = particle diameter
- F = shape factor (flaky gold settles slower)

**Implementation**:
```rust
fn advect_sediment(&mut self, dt: f32) {
    for particle in self.sediment.iter_mut() {
        // Get fluid velocity at particle position
        let u_fluid = self.grid.sample_velocity(particle.position);

        // Slip velocity (settling)
        let sg_ratio = 1.0 - (WATER_DENSITY / particle.density);
        let settling = sg_ratio * GRAVITY * dt;

        // Drag (opposes settling when flow is strong)
        let drag = particle.drag_coeff * (u_fluid - particle.velocity).length();

        // Update: fluid advection + slip velocity
        particle.velocity = u_fluid + Vec2::new(0.0, settling - drag);
        particle.position += particle.velocity * dt;
    }
}
```

**Result**: Gold naturally "falls out" of vortices while sand is ejected.

---

## Phase 2: Add PBF for Granular Phase

### 2.1 Keep PBF as Separate Solver
Don't merge with FLIP. Keep `pbf.rs` for:
- Dry dirt in trucks/buckets
- Tailings piles after sluice
- Pre-wash hopper

### 2.2 Implement Phase Transition
When PBF dirt enters sluice water zone:

```rust
fn transition_dirt_to_slurry(&mut self, pbf_particles: &mut Vec<PbfParticle>) {
    let sluice_bounds = self.sluice_water_zone();

    pbf_particles.retain(|p| {
        if sluice_bounds.contains(p.position) {
            // Delete PBF particle, spawn FLIP equivalents
            self.flip.spawn_water(p.position, p.velocity, 3); // muddy water
            self.sediment.spawn(p.position, p.velocity, p.material_type);
            false // remove from PBF
        } else {
            true // keep in PBF
        }
    });
}
```

### 2.3 Tailings Transition (Exit Sluice)
When sediment exits sluice, convert back to PBF or heightfield:

```rust
fn transition_tailings(&mut self, sediment: &mut Vec<SedimentParticle>) {
    let exit_zone = self.sluice_exit_zone();

    sediment.retain(|p| {
        if exit_zone.contains(p.position) {
            // Spawn PBF granular particle
            self.pbf.spawn_granular(p.position, p.velocity, p.material_type);
            false
        } else {
            true
        }
    });
}
```

---

## Phase 3: Sediment Classification System

### 3.1 Material Types
```rust
#[derive(Clone, Copy)]
pub enum SedimentType {
    QuartzSand,   // SG 2.65, easily ejected
    Magnetite,    // SG 5.2, "black sand" indicator
    Gold,         // SG 19.3, the prize
}

impl SedimentType {
    pub fn specific_gravity(&self) -> f32 {
        match self {
            Self::QuartzSand => 2.65,
            Self::Magnetite => 5.2,
            Self::Gold => 19.3,
        }
    }

    pub fn settling_velocity(&self, diameter: f32) -> f32 {
        // Rubey equation
        let sg = self.specific_gravity();
        let f = self.shape_factor(); // flaky gold = 0.7, spherical = 1.0
        f * ((sg - 1.0) * GRAVITY * diameter).sqrt()
    }
}
```

### 3.2 Riffle Trap Detection
When sediment velocity drops below threshold near riffle, mark as "trapped":

```rust
fn classify_trapped_sediment(&mut self) {
    for particle in self.sediment.iter_mut() {
        if self.is_in_riffle_zone(particle.position) {
            let speed = particle.velocity.length();
            let threshold = particle.material.settling_velocity(particle.diameter);

            if speed < threshold {
                particle.state = SedimentState::Trapped;
                // Remove from active simulation, add to riffle accumulator
                self.riffle_contents.add(particle.material, particle.position);
            }
        }
    }
}
```

---

## Phase 4: Performance Optimizations

### 4.1 Narrow-Band FLIP
Only simulate FLIP in the sluice water zone, not entire world:

```rust
pub struct NarrowBandFlip {
    bounds: Rect,        // Sluice box bounds
    grid: Grid,          // Only covers sluice
    particles: Vec<Particle>,
}
```

### 4.2 Sediment LOD
- **Near camera**: Full physics, individual particles
- **Far/dense**: Aggregate into "clumps" with combined mass

### 4.3 GPU Compute (Future)
For 100k+ particles, port to compute shaders:

| Kernel | Description |
|--------|-------------|
| Advect | Update positions, apply slip velocity |
| Binning | Sort particles into grid cells |
| P2G | Rasterize velocities to grid |
| Pressure | Jacobi/Multigrid solve |
| G2P | Sample velocities back to particles |
| Classify | Check riffle traps |

---

## Implementation Order

### Immediate (This Week)
1. [ ] Pre-allocate FLIP P2G buffers (`flip.rs:121-124`)
2. [ ] Add rayon to FLIP G2P transfer
3. [ ] Add `SedimentParticle` struct with material type + settling

### Short Term (Next 2 Weeks)
4. [ ] Implement drift-flux advection for sediment
5. [ ] Add riffle trap detection
6. [ ] Test gold settling through water vortices

### Medium Term (Month)
7. [ ] Phase transition: PBF dirt → FLIP slurry
8. [ ] Tailings pile formation
9. [ ] "Clean out" mechanic (harvest trapped gold)

### Long Term (Future)
10. [ ] GPU compute port
11. [ ] Narrow-band optimization
12. [ ] Multi-sluice support

---

## Success Criteria

1. **Visual**: Water creates visible eddies behind riffles
2. **Physical**: Gold settles faster than sand (observable)
3. **Gameplay**: Sluice angle affects recovery rate
4. **Performance**: 60 FPS with 10k particles

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `crates/sim/src/flip.rs` | Add P2G buffers, rayon parallelism |
| `crates/sim/src/sediment.rs` | NEW - Drift-flux particles |
| `crates/sim/src/sluice.rs` | Modify - Riffle trap detection |
| `crates/sim/src/phase_transition.rs` | NEW - PBF ↔ FLIP conversion |
| `crates/sim/src/lib.rs` | Export new modules |

---

## References

- Rubey (1933) - Settling velocity equation
- Bridson (2015) - Fluid Simulation for Computer Graphics
- Müller/Macklin (2014) - Position Based Fluids
- Stomakhin (2013) - FLIP for sand/snow
\n\n---\n## ARCHIVED: 2026-01-09\n\n**Superseded by:** Pure 3D FLIP/DEM approach\n\nThe hybrid PBF+FLIP architecture proposed here was superseded by a unified 3D FLIP + DEM for sediment approach. We no longer use PBF for granular materials.
