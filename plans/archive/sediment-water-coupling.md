<!-- STATUS: Work In Progress -->

# Sediment-Water Two-Way Coupling Implementation Plan

**Status:** In Progress (Phase 1 Complete)
**Date:** 2025-12-28

## Problem Statement

Current one-way coupling (sediment feels water, water doesn't feel sediment) breaks incompressibility. Sediment marks cells as Fluid but contributes zero velocity to P2G, creating phantom divergence.

## Solution Overview

Three-phase particle model:
1. **Suspended** - Two-way coupled FLIP (sediment contributes to grid)
2. **Bedload** - DEM on bed surface (friction, particle collisions)
3. **Deposited** - Static heightfield (becomes new floor)

## Implementation Phases

### Phase 1: Fix Immediate Divergence Issue [COMPLETE]
**Goal:** Unblock water simulation while building proper coupling

**Changes:**
- Don't mark sediment cells as Fluid for pressure solve
- Sediment becomes purely Lagrangian (advected by sampled velocity)
- Water simulation works correctly again

**Files:** `crates/sim/src/flip.rs`

**Risk:** Low - minimal change, easily reversible

**Status:** Already implemented (lines 261-273). Verified working - divergence stays 5-11 with sediment.

---

### Phase 2: Two-Way Coupling for Suspended Sediment
**Goal:** Sediment properly participates in FLIP simulation

#### 2a: Volume Fraction Tracking
Track sediment concentration per cell during P2G:

```rust
struct CellData {
    velocity: Vec2,
    weight: f32,
    sediment_volume: f32,  // NEW: accumulated sediment volume
}
```

During P2G:
- Water contributes velocity with weight 1.0
- Sediment contributes velocity with weight based on its volume
- Track total sediment volume fraction per cell

#### 2b: Mixture Density for Pressure Solve
Adjust pressure solve to account for mixture density:

```rust
// Effective density increases with sediment concentration
let alpha = sediment_volume_fraction.clamp(0.0, 0.5);
let mixture_density = WATER_DENSITY * (1.0 - alpha) + SEDIMENT_DENSITY * alpha;

// Pressure gradient scaled by density
let pressure_accel = -pressure_gradient / mixture_density;
```

#### 2c: Settling Velocity in G2P
Apply settling velocity during G2P for sediment:

```rust
if particle.is_sediment() {
    let settling = particle.material.settling_velocity(particle.diameter);
    let hindered = settling * hindered_settling_factor(local_concentration);
    particle.velocity.y -= hindered * dt;
}
```

**Files:**
- `crates/sim/src/flip.rs` (P2G, G2P, pressure solve)
- `crates/sim/src/grid.rs` (cell data structure)

---

### Phase 3: Bedload State Transition
**Goal:** Particles on bed switch to friction-based movement

#### 3a: Bed Contact Detection
Detect when particle touches the SDF floor:

```rust
fn is_on_bed(&self, particle: &Particle, sdf: &Sdf) -> bool {
    let distance = sdf.sample(particle.position);
    distance < BED_CONTACT_THRESHOLD  // e.g., 0.5 * cell_size
}
```

#### 3b: Shields Criterion for State Transition
Use existing `shields_critical()` to determine if particle should suspend or stay bedload:

```rust
// Calculate local shear stress from velocity gradient
let du_dy = (grid.velocity(i, j+1).x - grid.velocity(i, j).x) / cell_size;
let tau = WATER_DENSITY * KINEMATIC_VISCOSITY * du_dy.abs();

// Shields parameter
let tau_star = tau / ((particle.density() - WATER_DENSITY) * GRAVITY * particle.diameter);

if particle.state == ParticleState::Suspended && is_on_bed {
    if tau_star < particle.material.shields_critical() {
        particle.state = ParticleState::Bedload;
    }
} else if particle.state == ParticleState::Bedload {
    if tau_star > particle.material.shields_critical() * 1.5 {  // Hysteresis
        particle.state = ParticleState::Suspended;
    }
}
```

#### 3c: Bedload Friction Model
For bedload particles, apply Coulomb friction:

```rust
if particle.state == ParticleState::Bedload {
    // Normal force from gravity (submerged weight)
    let submerged_weight = (particle.density() - WATER_DENSITY) * GRAVITY * volume;

    // Friction opposes motion
    let friction_mag = particle.material.dynamic_friction() * submerged_weight;
    let friction_dir = -particle.velocity.normalize_or_zero();

    particle.velocity += friction_dir * friction_mag * dt / mass;

    // Clamp to bed surface
    particle.position.y = particle.position.y.max(bed_height + radius);
}
```

**Files:**
- `crates/sim/src/flip.rs` (state transitions, friction)
- `crates/sim/src/particle.rs` (already has ParticleState enum)

---

### Phase 4: DEM Collisions for Bedload
**Goal:** Bedload particles can form piles, collide with each other

#### 4a: Spatial Hash for Collision Detection
```rust
struct SpatialHash {
    cell_size: f32,
    cells: HashMap<(i32, i32), Vec<usize>>,  // particle indices
}

impl SpatialHash {
    fn query_neighbors(&self, pos: Vec2, radius: f32) -> impl Iterator<Item = usize>;
}
```

#### 4b: Spring-Damper Contact Model
```rust
fn dem_collision(p1: &mut Particle, p2: &mut Particle) {
    let delta = p2.position - p1.position;
    let dist = delta.length();
    let overlap = (p1.radius() + p2.radius()) - dist;

    if overlap > 0.0 {
        let normal = delta / dist;
        let relative_vel = p2.velocity - p1.velocity;

        // Spring force (repulsion)
        let spring_force = SPRING_CONSTANT * overlap;

        // Damping force
        let damping_force = DAMPING * relative_vel.dot(normal);

        let force = (spring_force - damping_force) * normal;

        // Apply equal and opposite (assuming equal mass)
        p1.velocity -= force * dt / mass;
        p2.velocity += force * dt / mass;
    }
}
```

**Files:**
- `crates/sim/src/spatial_hash.rs` (new file)
- `crates/sim/src/flip.rs` (integrate DEM step)

---

### Phase 5: Deposition System
**Goal:** Stationary bedload becomes permanent bed

#### 5a: Deposition Trigger
```rust
// Track time at low velocity
if particle.state == ParticleState::Bedload && particle.velocity.length() < DEPOSIT_VEL_THRESHOLD {
    particle.jam_time += dt;

    if particle.jam_time > DEPOSIT_TIME_THRESHOLD {
        deposit_particle(particle);
        // Remove from active simulation
    }
} else {
    particle.jam_time = 0.0;
}
```

#### 5b: Heightfield Storage
```rust
struct DepositedBed {
    // Height of deposited material per column
    heights: Vec<f32>,  // One per grid column
    cell_size: f32,
}

impl DepositedBed {
    fn deposit(&mut self, x: f32, volume: f32) {
        let col = (x / self.cell_size) as usize;
        self.heights[col] += volume / self.cell_size;  // Height increase
    }

    fn height_at(&self, x: f32) -> f32 {
        let col = (x / self.cell_size) as usize;
        self.heights.get(col).copied().unwrap_or(0.0)
    }
}
```

#### 5c: SDF Integration
Modify SDF queries to include deposited bed:

```rust
fn effective_floor(&self, x: f32) -> f32 {
    let base_floor = self.sdf.floor_at(x);
    let deposit = self.deposited_bed.height_at(x);
    base_floor + deposit
}
```

**Files:**
- `crates/sim/src/deposit.rs` (new file)
- `crates/sim/src/flip.rs` (deposition logic)
- `crates/sim/src/grid.rs` or `sdf.rs` (floor queries)

---

## Testing Strategy

### Phase 1 Tests
- [ ] Water-only simulation maintains low divergence
- [ ] Sediment advects with water velocity

### Phase 2 Tests
- [ ] Volume fraction calculated correctly
- [ ] Higher sediment concentration slows flow
- [ ] Settling velocity applied correctly
- [ ] Gold settles faster than sand

### Phase 3 Tests
- [ ] Particles transition to bedload on floor contact
- [ ] High shear re-suspends particles
- [ ] Friction slows bedload particles
- [ ] Bedload stays on bed surface

### Phase 4 Tests
- [ ] Particles don't overlap
- [ ] Pile formation works (angle of repose)
- [ ] Performance acceptable at 10k particles

### Phase 5 Tests
- [ ] Stationary particles deposit
- [ ] Deposit raises effective floor
- [ ] Water flows over deposited bed

---

## Performance Considerations

- **Spatial hash**: O(n) build, O(1) query - critical for DEM at 10k particles
- **Volume fraction**: Already computing P2G weights, minimal overhead
- **Deposit heightfield**: One float per column, negligible memory

---

## Open Questions

1. Should bedload particles still sample grid velocity for drag, or use local fluid velocity estimate?
2. What's the right spring constant for DEM? (Tune for stability vs realism)
3. Do we need sub-stepping for DEM stability?

---

## References

- [Hybrid Grains (SIGGRAPH Asia 2018)](https://dl.acm.org/doi/10.1145/3272127.3275095)
- [SPH-DEM Coupling](https://www.sciencedirect.com/science/article/abs/pii/S0301932213001882)
- [Shields Parameter](https://en.wikipedia.org/wiki/Shields_parameter)
- Ferguson & Church 2004 (settling velocity)
