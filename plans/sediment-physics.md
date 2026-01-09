<!-- STATUS: Work In Progress -->

# Sediment Physics - Unified Plan

> **Consolidated from:** sediment-entrainment.md, sediment-water-coupling.md, multi-sediment-types.md
> 
> **Date:** 2026-01-09

---

## Design Decision: Clumps vs FLIP Particles (2026-01-09)

> **Decision:** Use `clump.rs` rigid clumps for sediment in 3D detail zones.

### Implications for Sediment Physics
- **Entrainment:** Clumps are picked up when flow exceeds threshold (Shields-like but for clump drag)
- **Coupling:** Clumps feel water drag, water doesn't need to simulate individual grains
- **Material types:** Different clump densities/shapes for gold vs sand vs magnetite

### Key Insight
For clump-based sediment, the physics concepts (Shields parameter, bedload vs suspended, stratification) still apply but at the **clump level** rather than individual particle level.

---

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


---

# Sediment Entrainment Plan

## Goal

Allow deposited sediment to be re-entrained (picked back up) when flow velocity exceeds a critical threshold. This completes the sediment transport cycle: suspension → settling → deposition → entrainment.

## Physics Background

### Shields Parameter

The Shields parameter determines when sediment begins to move:

```
τ* = τ_b / ((ρ_s - ρ_w) * g * d)

Where:
- τ_b = bed shear stress
- ρ_s = sediment density (2650 kg/m³ for sand)
- ρ_w = water density (1000 kg/m³)
- g = gravity
- d = particle diameter
```

Critical Shields number τ*_c ≈ 0.03-0.06 for sand.

### Bed Shear Stress

For channel flow:
```
τ_b = ρ_w * u*²

Where u* = friction velocity ≈ κ * u / ln(z/z0)
```

Simplified for game: use velocity magnitude at cell as proxy for shear.

## Implementation Plan

### Step 1: Track Deposited Cell Ages

Add `deposit_time: Vec<u32>` to grid - frames since deposition. Fresh deposits are more easily entrained.

### Step 2: Sample Flow Velocity at Deposits

For each deposited cell, sample grid velocity just above it:
```rust
let vel_above = grid.sample_velocity(Vec2::new(x + 0.5, y - 0.5));
let speed = vel_above.length();
```

### Step 3: Compute Entrainment Threshold

```rust
const CRITICAL_VELOCITY: f32 = 15.0;  // cells/frame - tune empirically
const ENTRAINMENT_RATE: f32 = 0.1;    // probability per frame when exceeded
```

### Step 4: Spawn Entrained Particles

When velocity exceeds threshold:
1. Probabilistically remove deposited cell
2. Spawn N sand particles at that location
3. Give particles initial velocity matching flow + small random component

### Step 5: Update SDF

After removing deposited cells, recompute SDF in affected region.

## Code Location

New function in `flip.rs`:
```rust
fn entrain_deposited_sediment(&mut self, dt: f32) {
    // For each deposited cell...
    // Check velocity above
    // Probabilistic entrainment
    // Spawn particles, clear cell
}
```

Call after `deposit_settled_sediment()` in update loop.

## Tuning Parameters

| Parameter | Initial Value | Effect |
|-----------|---------------|--------|
| CRITICAL_VELOCITY | 15.0 | Higher = more stable deposits |
| ENTRAINMENT_RATE | 0.1 | Higher = faster erosion |
| PARTICLES_PER_CELL | 4 | Matches deposition count |
| AGE_FACTOR | 0.01 | Older deposits harder to move |

## Testing

1. Build up deposits at low flow
2. Increase inlet velocity (→ key)
3. Observe deposits eroding from upstream
4. Verify particles re-enter suspension

## Edge Cases

- Don't entrain original terrain (only `is_deposited()` cells)
- Rate-limit entrainment to avoid sudden SDF changes
- Consider neighbor stability (isolated cells easier to entrain)
\n\n---\n\n# Implementation Plan (Merged)\n\n_The following was merged from sediment-entrainment-implementation.md_\n
# Sediment Entrainment Implementation Plan

**Type**: feat: Re-entrainment of deposited sediment
**Priority**: P1
**Date**: 2025-12-28

## Overview

Allow deposited sediment cells to be re-entrained (picked back up) when flow velocity exceeds a critical threshold. This completes the sediment transport cycle: **suspension → settling → deposition → entrainment**.

## Problem Statement

Currently, deposited sediment is permanent. Once particles settle and convert to solid terrain via `deposit_settled_sediment()`, they can never return to the flow. This is unrealistic - real sediment gets eroded and re-deposited as flow conditions change.

## Proposed Solution

Add `entrain_deposited_sediment()` function that:
1. Samples velocity above each deposited cell
2. Compares to critical threshold
3. Probabilistically removes cells exceeding threshold
4. Spawns sand particles at eroded locations
5. Updates SDF after terrain changes

---

## Technical Approach

### Key Decisions (Based on Research)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Velocity Sampling** | Top face center `(i+0.5)*h, (j+0.5)*h - 0.5*h` | Sample just above cell, use `sample_velocity()` |
| **Threshold Method** | Simplified velocity threshold | Shields conversion is complex; velocity proxy is sufficient for games |
| **Threshold Value** | 15.0 cells/frame (~1.5 m/s at typical scale) | Empirically tunable, matches existing plan |
| **Probability Function** | Linear with excess: `min(0.3, BASE_RATE * (v/v_c - 1.0))` | Smooth erosion, capped to prevent sudden mass loss |
| **Execution Order** | Step 8g: AFTER `deposit_settled_sediment()` | Prevents same-frame oscillation via frame marking |
| **Particles per Cell** | 4 (symmetric with `MASS_PER_CELL`) | Mass conservation |
| **Initial Velocity** | Grid velocity at cell center | Physical continuity |
| **Oscillation Prevention** | Skip cells deposited this frame | Simple, effective |

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Simulation Loop                         │
├─────────────────────────────────────────────────────────────┤
│ 8e. apply_dem_settling(dt)                                  │
│ 8f. deposit_settled_sediment(dt)  ← marks cells_deposited   │
│ 8g. entrain_deposited_sediment(dt) ← NEW: checks velocity,  │
│     │                                 removes cells,         │
│     │                                 spawns particles       │
│     └─> grid.clear_deposited(i, j)                          │
│     └─> spawn_sand(...)                                      │
│     └─> grid.compute_sdf()  (if any cells changed)          │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Grid Support (grid.rs)

Add helper function to clear deposited status:

```rust
// crates/sim/src/grid.rs:416 (after is_deposited)
/// Clear deposited status and solid flag for a cell
/// Used during entrainment when velocity exceeds threshold
pub fn clear_deposited(&mut self, i: usize, j: usize) {
    if i < self.width && j < self.height {
        let idx = self.cell_index(i, j);
        self.solid[idx] = false;
        self.deposited[idx] = false;
    }
}
```

**Files**: `crates/sim/src/grid.rs:416`
**Effort**: Small (5 lines)

---

### Phase 2: Core Entrainment Function (flip.rs)

Add new function after `deposit_settled_sediment()`:

```rust
// crates/sim/src/flip.rs:1544 (after deposit_settled_sediment)

/// Step 8g: Entrain deposited sediment when flow velocity exceeds threshold
///
/// Checks each deposited cell for high velocity flow above it.
/// If velocity exceeds critical threshold, probabilistically removes
/// the cell and spawns sand particles to re-enter the flow.
fn entrain_deposited_sediment(&mut self, dt: f32) {
    // === Thresholds ===
    const CRITICAL_VELOCITY: f32 = 15.0;  // cells/frame
    const BASE_ENTRAINMENT_RATE: f32 = 0.1;  // probability scaling
    const MAX_PROBABILITY: f32 = 0.3;  // cap per frame
    const PARTICLES_PER_CELL: usize = 4;  // matches MASS_PER_CELL

    let cell_size = self.grid.cell_size;
    let width = self.grid.width;
    let height = self.grid.height;
    let v_scale = cell_size / dt;  // Convert to cells/frame

    let mut cells_to_clear: Vec<(usize, usize)> = Vec::new();
    let mut particles_to_spawn: Vec<(Vec2, Vec2)> = Vec::new();  // (position, velocity)

    let mut rng = rand::rng();

    // Iterate deposited cells
    for j in 1..height-1 {  // Skip boundary rows
        for i in 1..width-1 {  // Skip boundary columns
            if !self.grid.is_deposited(i, j) {
                continue;
            }

            // Sample velocity just above the cell
            let sample_pos = Vec2::new(
                (i as f32 + 0.5) * cell_size,
                (j as f32 - 0.5) * cell_size,  // Half cell above
            );
            let vel_above = self.grid.sample_velocity(sample_pos);
            let speed = vel_above.length() / v_scale;  // cells/frame

            if speed <= CRITICAL_VELOCITY {
                continue;  // Not fast enough
            }

            // Compute entrainment probability
            let excess_ratio = speed / CRITICAL_VELOCITY - 1.0;
            let probability = (BASE_ENTRAINMENT_RATE * excess_ratio).min(MAX_PROBABILITY);

            // Stochastic check
            if rng.random::<f32>() >= probability {
                continue;  // Not entrained this frame
            }

            // Check support: don't entrain if it would leave floating cells above
            let has_deposit_above = j > 0 && self.grid.is_deposited(i, j - 1);
            if has_deposit_above {
                // Check if there's lateral support for the cell above
                let left_support = i > 0 && self.grid.is_solid(i - 1, j - 1);
                let right_support = i < width - 1 && self.grid.is_solid(i + 1, j - 1);
                if !left_support && !right_support {
                    continue;  // Would create floating deposit
                }
            }

            // Mark for removal
            cells_to_clear.push((i, j));

            // Queue particle spawns
            let cell_center = Vec2::new(
                (i as f32 + 0.5) * cell_size,
                (j as f32 + 0.5) * cell_size,
            );

            for _ in 0..PARTICLES_PER_CELL {
                // Jitter position within cell
                let jitter = Vec2::new(
                    (rng.random::<f32>() - 0.5) * cell_size * 0.6,
                    (rng.random::<f32>() - 0.5) * cell_size * 0.6,
                );
                let pos = cell_center + jitter;

                // Initial velocity from grid + slight upward lift
                let vel = vel_above * 0.8 + Vec2::new(0.0, -2.0 * v_scale);

                particles_to_spawn.push((pos, vel));
            }
        }
    }

    // Clear cells
    for (i, j) in &cells_to_clear {
        self.grid.clear_deposited(*i, *j);
        // Reset accumulated mass for this cell
        let idx = *j * width + *i;
        self.deposited_mass[idx] = 0.0;
    }

    // Spawn particles
    for (pos, vel) in particles_to_spawn {
        self.particles.spawn_sand(pos.x, pos.y, vel.x, vel.y);
    }

    // Update SDF if terrain changed
    if !cells_to_clear.is_empty() {
        self.grid.compute_sdf();
        self.grid.compute_bed_heights();

        // Debug output
        if self.frame % 60 == 0 {
            eprintln!(
                "[Entrainment] Eroded {} cells, spawned {} particles",
                cells_to_clear.len(),
                cells_to_clear.len() * PARTICLES_PER_CELL
            );
        }
    }
}
```

**Files**: `crates/sim/src/flip.rs:1544`
**Effort**: Medium (~80 lines)

---

### Phase 3: Integration into Update Loop

Add call to entrainment after deposition:

```rust
// crates/sim/src/flip.rs:232 (in update())

// 8f. Deposition: stable piles become solid terrain
self.deposit_settled_sediment(dt);

// 8g. Entrainment: high flow erodes deposited cells
self.entrain_deposited_sediment(dt);
```

**Files**: `crates/sim/src/flip.rs:232`
**Effort**: Trivial (2 lines)

---

## Acceptance Criteria

### Functional Requirements

- [ ] Deposited cells with velocity > 15 cells/frame above them can be entrained
- [ ] Entrainment is probabilistic (not all cells above threshold entrain immediately)
- [ ] Entrained cells spawn 4 sand particles with flow-matching velocity
- [ ] SDF is updated after entrainment to reflect terrain change
- [ ] No floating deposits (support check prevents mid-pile erosion)
- [ ] No same-frame deposit-entrain cycles

### Non-Functional Requirements

- [ ] Performance: < 1ms for entrainment step (128x128 grid)
- [ ] No new memory allocations per frame (reuse vectors)
- [ ] No visual flicker or oscillation

### Quality Gates

- [ ] `cargo test -p sim` passes
- [ ] `cargo clippy` passes
- [ ] Visual test: deposits erode when inlet velocity increased (→ key)

---

## Testing Plan

### Test 1: Entrainment Threshold Test

```rust
// crates/sim/tests/entrainment_test.rs

#[test]
fn test_entrainment_threshold() {
    let mut sim = FlipSimulation::new(32, 32, 1.0);

    // Create deposited cell
    sim.grid.set_deposited(16, 16);

    // Set velocity above cell below threshold
    sim.grid.u[/*index*/] = 10.0;  // Below 15.0 threshold
    sim.entrain_deposited_sediment(1.0/60.0);

    assert!(sim.grid.is_deposited(16, 16), "Should not entrain below threshold");

    // Set velocity above threshold
    sim.grid.u[/*index*/] = 20.0;  // Above threshold

    // Run many frames (probabilistic)
    for _ in 0..100 {
        sim.entrain_deposited_sediment(1.0/60.0);
    }

    assert!(!sim.grid.is_deposited(16, 16), "Should entrain above threshold");
}
```

### Test 2: Particle Spawning Test

```rust
#[test]
fn test_entrainment_spawns_particles() {
    let mut sim = FlipSimulation::new(32, 32, 1.0);
    sim.grid.set_deposited(16, 16);

    // Set high velocity to guarantee entrainment
    sim.grid.u[/*index*/] = 30.0;

    let particle_count_before = sim.particles.list.len();

    // Run until entrained
    for _ in 0..100 {
        sim.entrain_deposited_sediment(1.0/60.0);
        if !sim.grid.is_deposited(16, 16) {
            break;
        }
    }

    let particle_count_after = sim.particles.list.len();
    assert_eq!(particle_count_after - particle_count_before, 4, "Should spawn 4 particles");
}
```

### Test 3: Visual Integration Test

1. Run game: `cargo run --bin game --release`
2. Wait for deposits to form at riffle base
3. Increase inlet velocity (→ key)
4. Observe deposits eroding from upstream edges
5. Particles should rejoin flow and potentially re-deposit downstream

---

## Tuning Parameters

| Parameter | Final Value | Effect |
|-----------|-------------|--------|
| `CRITICAL_VELOCITY` | 0.5 | Any flow erodes - very responsive |
| `BASE_ENTRAINMENT_RATE` | 1.0 | Fast erosion scaling |
| `MAX_PROBABILITY` | 0.95 | Nearly instant erosion per frame |
| `PARTICLES_PER_CELL` | 4 | Mass per eroded cell |

---

## Edge Cases & Mitigations

| Edge Case | Mitigation |
|-----------|------------|
| Oscillation (deposit → entrain → deposit) | Skip cells deposited same frame via order (entrain AFTER deposit) |
| Floating deposits | Support check: require lateral support before allowing mid-pile erosion |
| Mass loss | Spawn exactly PARTICLES_PER_CELL to match deposition |
| CFL violation | Clamp spawned particle velocity to max safe speed |
| Sudden terrain collapse | MAX_PROBABILITY caps erosion to ~30% per frame per cell |

---

## Files Modified

| File | Changes |
|------|---------|
| `crates/sim/src/grid.rs` | Add `clear_deposited()` function (~5 lines) |
| `crates/sim/src/flip.rs` | Add `entrain_deposited_sediment()` (~80 lines), call in update (~2 lines) |
| `crates/sim/tests/entrainment_test.rs` | New test file (~60 lines) |

---

## References

### Internal
- `flip.rs:1397-1542` - `deposit_settled_sediment()` (symmetric function)
- `flip.rs:1207-1388` - `apply_dem_settling()` (DEM contact model)
- `grid.rs:401-415` - `set_deposited()` / `is_deposited()`
- `particle.rs:124-129` - `shields_critical()` = 0.045 for sand

### External
- [Shields Parameter](https://en.wikipedia.org/wiki/Shields_parameter) - Critical threshold physics
- [HEC-RAS Sediment Transport](https://www.hec.usace.army.mil/confluence/rasdocs/d2sd/ras2dsedtr/6.6/model-description/critical-thresholds-for-transport-and-erosion) - USACE reference
- [Simple Particle-Based Hydraulic Erosion](https://nickmcd.me/2020/04/10/simple-particle-based-hydraulic-erosion/) - Game implementation reference


---

# Multi-Sediment Types Implementation Plan

## Goal
Add support for different sediment types (quartz sand, black sand/magnetite) with distinct physical properties:
- Different densities → different settling velocities
- Different Shields parameters → different entrainment thresholds
- Mixed beds with material composition tracking

## Research Summary

### Material Properties

| Property | Quartz Sand | Black Sand (Magnetite) |
|----------|-------------|------------------------|
| Density (g/cm³) | 2.65 | 5.0 |
| Typical diameter | 0.3 px | 0.2 px (smaller grains) |
| Shape factor C₂ | 1.0 | 1.1 (angular) |
| Shields θ_cr | 0.045 | 0.06 (harder to move) |
| Color | tan | dark gray/black |

### Key Physics (from [sediment transport research](https://geo.libretexts.org/Bookshelves/Oceanography/Coastal_Dynamics_(Bosboom_and_Stive)/06:_Sediment_transport)):

1. **Settling velocity** - Ferguson-Church already handles density:
   - `w ∝ (ρ_p - ρ_f) × D²` in Stokes regime
   - Heavy minerals settle faster at same size
   - Magnetite grains in nature are often smaller because they settle at same velocity as larger quartz

2. **Critical entrainment** - Shields parameter varies by material:
   - Higher density → higher critical shear stress to move
   - θ_cr for magnetite ~30% higher than quartz ([source](https://www.hec.usace.army.mil/confluence/rasdocs/d2sd/ras2dsedtr/6.6/model-description/critical-thresholds-for-transport-and-erosion))

3. **Hiding/exposure effects** (for mixed beds):
   - Fine particles sheltered by coarse ones
   - Coarse particles more exposed to flow
   - Creates "[armoring](https://www.nature.com/articles/s41467-017-01681-3)" - heavy minerals concentrate at surface

4. **[Placer formation](https://en.wikipedia.org/wiki/Placer_deposit)**:
   - Heavy minerals drop out where velocity decreases
   - Creates natural stratification: heavy at bottom
   - Gold/magnetite concentrate in same locations

---

## Implementation Plan

### Phase 1: Extend ParticleMaterial enum

**File: `crates/sim/src/particle.rs`**

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParticleMaterial {
    Water,
    Sand,       // Quartz sand (existing)
    BlackSand,  // Magnetite/heavy minerals
}

impl ParticleMaterial {
    pub fn density(&self) -> f32 {
        match self {
            Self::Water => 1.0,
            Self::Sand => 2.65,
            Self::BlackSand => 5.0,
        }
    }

    pub fn typical_diameter(&self) -> f32 {
        match self {
            Self::Water => 0.0,
            Self::Sand => 0.3,
            Self::BlackSand => 0.2,  // Smaller (hydraulic equivalence)
        }
    }

    pub fn shields_critical(&self) -> f32 {
        match self {
            Self::Water => 0.0,
            Self::Sand => 0.045,
            Self::BlackSand => 0.06,  // Harder to entrain
        }
    }

    pub fn shape_factor(&self) -> f32 {
        match self {
            Self::Water => 1.0,
            Self::Sand => 1.0,
            Self::BlackSand => 1.1,  // Slightly more angular
        }
    }

    pub fn color(&self) -> [u8; 4] {
        match self {
            Self::Water => [50, 140, 240, 100],
            Self::Sand => [194, 178, 128, 255],
            Self::BlackSand => [40, 40, 45, 255],  // Dark gray
        }
    }
}
```

**Work:**
- Add `BlackSand` variant
- Add all physical properties
- Add `Particle::black_sand()` constructor
- Update `Particles::count_by_material()` to include black sand

---

### Phase 2: Mixed-material deposited cells

**File: `crates/sim/src/grid.rs`**

Current: `deposited: Vec<bool>` - just marks if cell is sediment

New: Store material composition per cell.

```rust
/// Material composition of a deposited cell
#[derive(Clone, Copy, Debug, Default)]
pub struct DepositedCell {
    pub sand_fraction: f32,       // 0.0-1.0
    pub black_sand_fraction: f32, // 0.0-1.0
    // Fractions should sum to 1.0 when cell is deposited
}

impl DepositedCell {
    pub fn is_deposited(&self) -> bool {
        self.sand_fraction + self.black_sand_fraction > 0.0
    }

    /// Weighted average Shields parameter for entrainment
    pub fn effective_shields_critical(&self) -> f32 {
        let sand_shields = 0.045;
        let black_sand_shields = 0.06;
        let total = self.sand_fraction + self.black_sand_fraction;
        if total <= 0.0 { return sand_shields; }

        (self.sand_fraction * sand_shields + self.black_sand_fraction * black_sand_shields) / total
    }

    /// Weighted average density for visualization/physics
    pub fn effective_density(&self) -> f32 {
        let total = self.sand_fraction + self.black_sand_fraction;
        if total <= 0.0 { return 2.65; }

        (self.sand_fraction * 2.65 + self.black_sand_fraction * 5.0) / total
    }

    /// Blend color based on composition
    pub fn color(&self) -> [u8; 4] {
        let sand_color = [194u8, 178, 128, 255];
        let black_color = [40u8, 40, 45, 255];
        let total = self.sand_fraction + self.black_sand_fraction;
        if total <= 0.0 { return sand_color; }

        let t = self.black_sand_fraction / total;
        [
            lerp_u8(sand_color[0], black_color[0], t),
            lerp_u8(sand_color[1], black_color[1], t),
            lerp_u8(sand_color[2], black_color[2], t),
            255,
        ]
    }
}
```

**Grid changes:**
```rust
pub struct Grid {
    // Replace:  deposited: Vec<bool>
    // With:
    pub deposited: Vec<DepositedCell>,

    // Keep deposited_mass for accumulation, but track per-material:
    pub deposited_mass_sand: Vec<f32>,
    pub deposited_mass_black: Vec<f32>,
}
```

**Work:**
- Add `DepositedCell` struct
- Replace `Vec<bool>` with `Vec<DepositedCell>`
- Split `deposited_mass` into per-material accumulators
- Update `set_deposited()` to take material fractions
- Update `is_deposited()` to check struct
- Update `clear_deposited()` to reset struct

---

### Phase 3: Update deposition logic

**File: `crates/sim/src/flip.rs` (Step 8f)**

Current flow:
1. Count particles in each cell
2. If count >= 4, convert to solid

New flow:
1. Count particles by material in each cell
2. Track separate accumulators for sand and black_sand
3. When total mass >= threshold, convert to deposited with appropriate fractions

```rust
// In step_8f_deposition:
for particle in settled_particles {
    let (i, j) = particle_to_cell(particle);
    match particle.material {
        ParticleMaterial::Sand => grid.deposited_mass_sand[idx] += 1.0,
        ParticleMaterial::BlackSand => grid.deposited_mass_black[idx] += 1.0,
        _ => {}
    }
}

// Convert when total mass reaches threshold
let total_mass = grid.deposited_mass_sand[idx] + grid.deposited_mass_black[idx];
if total_mass >= DEPOSITION_THRESHOLD {
    let sand_frac = grid.deposited_mass_sand[idx] / total_mass;
    let black_frac = grid.deposited_mass_black[idx] / total_mass;
    grid.set_deposited_with_composition(i, j, sand_frac, black_frac);
}
```

---

### Phase 4: Update entrainment logic

**File: `crates/sim/src/flip.rs` (Step 8g)**

Current: Uses fixed `CRITICAL_VELOCITY` for all deposited cells

New: Use cell's effective Shields parameter

```rust
// In step_8g_entrainment:
let cell_shields = grid.deposited[idx].effective_shields_critical();

// Higher Shields = higher critical velocity needed
let critical_velocity = BASE_CRITICAL_VELOCITY * (cell_shields / 0.045);

// When eroding, spawn particles proportional to composition
let composition = grid.deposited[idx];
let num_sand = (PARTICLES_PER_CELL as f32 * composition.sand_fraction).round() as usize;
let num_black = (PARTICLES_PER_CELL as f32 * composition.black_sand_fraction).round() as usize;

for _ in 0..num_sand {
    spawn_particle(ParticleMaterial::Sand, ...);
}
for _ in 0..num_black {
    spawn_particle(ParticleMaterial::BlackSand, ...);
}
```

---

### Phase 5: Hiding/exposure effects (optional, for realism)

When multiple sediment types are present, finer particles are protected:

```rust
/// Hiding correction factor (Wu et al. 2000)
/// Reduces effective Shields for small particles hidden by larger ones
fn hiding_exposure_factor(particle_diameter: f32, bed_median_diameter: f32) -> f32 {
    // Smaller particles are hidden: factor > 1 (harder to move)
    // Larger particles are exposed: factor < 1 (easier to move)
    let ratio = particle_diameter / bed_median_diameter;
    ratio.powf(-0.6)  // Empirical exponent from literature
}
```

This creates armoring behavior where:
- Light sand erodes first from mixed beds
- Heavy black sand concentrates at surface
- Eventual equilibrium with black sand "pavement"

---

### Phase 6: Rendering updates

**File: `crates/game/src/render.rs`**

1. Particle rendering already uses `particle.material.color()` - just add black sand color
2. Deposited cell rendering needs to use `DepositedCell::color()` for blended appearance

---

### Phase 7: Input/spawning

Add controls to spawn black sand:
- Separate spawn function or ratio parameter
- Could have "ore vein" areas that spawn mixed material

---

## Testing Strategy

1. **Unit tests** (particle.rs):
   - Verify black sand settles faster than quartz at same size
   - Verify shields_critical is higher for black sand
   - Verify density ordering: black_sand > sand > water

2. **Integration tests**:
   - Mixed material deposition creates correct composition
   - Entrainment respects composition-weighted Shields
   - Particle spawning from entrained cell matches composition

3. **Visual tests**:
   - Black sand visibly settles faster
   - Black sand concentrates at bottom of deposits
   - Harder to wash away deposited black sand

---

## Expected Behavior

1. **Settling**: Black sand drops out of flow sooner (velocity ∝ density)
2. **Stratification**: Natural layering with heavy minerals at bottom
3. **Entrainment resistance**: Black sand beds require stronger flow to erode
4. **Placer formation**: Black sand accumulates where velocity decreases (like behind riffles, in eddies)

---

## Open Questions

1. **Mass balance**: Should 4 black sand particles = 1 cell, or weight by density?
   - Option A: Same count threshold (4 particles = 1 cell regardless of type)
   - Option B: Weight by density (fewer heavy particles needed)
   - Recommend: Option A for simplicity, Option B for realism

2. **Vertical stratification within cells**: Track layers or just fractions?
   - Simple: Just fractions (what we propose)
   - Complex: Track depth order (heavy at bottom, light at top)
   - Recommend: Start simple, add layers if needed

3. **Armoring dynamics**: Implement hiding/exposure or use simple mixing?
   - Simple: Weighted average Shields
   - Complex: Hiding/exposure with surface layer tracking
   - Recommend: Start with weighted average

---

## Implementation Order

1. [ ] Add `BlackSand` to `ParticleMaterial` with all properties
2. [ ] Add `Particle::black_sand()` constructor and spawning methods
3. [ ] Create `DepositedCell` struct with composition tracking
4. [ ] Update `Grid` to use `Vec<DepositedCell>`
5. [ ] Update deposition to track per-material mass
6. [ ] Update entrainment to use composition-weighted Shields
7. [ ] Update entrainment to spawn correct material mix
8. [ ] Add rendering for black sand particles
9. [ ] Add deposited cell color blending
10. [ ] Add spawn controls for mixed material
11. [ ] Write tests
12. [ ] Visual verification

---

## References

- [Critical shear stress and settling velocity](https://geo.libretexts.org/Bookshelves/Oceanography/Coastal_Dynamics_(Bosboom_and_Stive)/06:_Sediment_transport/6.08:_Some_aspects_of_(very)_fine_sediment_transport/6.8.2:_Critical_shear_stress_and_settling_velocity)
- [HEC-RAS Hiding and Exposure Corrections](https://www.hec.usace.army.mil/confluence/rasdocs/d2sd/ras2dsedtr/latest/model-description/hiding-and-exposure-corrections)
- [Placer deposits - Wikipedia](https://en.wikipedia.org/wiki/Placer_deposit)
- [Magnetite properties](https://sandatlas.org/magnetite/)
- [River-bed armoring as granular segregation](https://www.nature.com/articles/s41467-017-01681-3)


---

# Vorticity-Driven Suspension (Merged)

_Merged from vorticity-suspension.md_

## Design Note: Clumps vs Particles in Vortices

> For clump-based sediment, vorticity affects the **clump as a whole** rather than individual particles. Larger clumps have more inertia and resist being lifted by eddies. This naturally creates stratification where smaller clumps swirl more in high-vorticity regions.


# Plan: Vorticity-Driven Sand Suspension

## Goal
Make sand particles swirl within the water flow instead of just dragging along the bottom.

## Key Insight
Vorticity (curl of velocity field) measures local rotation. High vorticity regions have eddies that can carry particles against gravity. We already compute vorticity for confinement - reuse it for sand.

---

## Current State

### Update Loop Order (flip.rs:150-230)
1. P2G transfer
2. Store old velocities
3. Apply gravity
4. **Vorticity confinement** ← vorticity computed here
5. Pressure projection
6. Extrapolate velocities
7. **G2P transfer** ← sand gets velocity here (lines 767-826)
8. Build spatial hash
9. Advect particles

### Current Sand G2P (flip.rs:767-826)
```rust
if particle.is_sediment() {
    let v_grid = grid.sample_velocity_bspline(pos);

    if cell_type == CellType::Fluid {
        // PIC/FLIP blend to follow water
        let pic_vel = v_grid;
        let flip_vel = particle.velocity + grid_delta;
        particle.velocity = sand_pic_ratio * pic_vel + (1.0 - sand_pic_ratio) * flip_vel;
    }

    // PROBLEM: Always applies constant settling
    const SETTLING_FACTOR: f32 = 0.62;
    particle.velocity.y += GRAVITY * SETTLING_FACTOR * dt;
    return;
}
```

### Vorticity Infrastructure
- `grid.vorticity: Vec<f32>` - stored at cell centers
- `grid.compute_vorticity()` - computes ω = ∂v/∂x - ∂u/∂y
- Called inside `apply_vorticity_confinement_with_piles()` before G2P

---

## Implementation Plan

### Step 1: Add Vorticity Sampling Method

**File**: `crates/sim/src/grid.rs`

Add bilinear interpolation for vorticity at arbitrary position:

```rust
/// Sample vorticity at position using bilinear interpolation
/// Vorticity is stored at cell centers
pub fn sample_vorticity(&self, pos: Vec2) -> f32 {
    // Cell center coordinates
    let x = pos.x / self.cell_size - 0.5;
    let y = pos.y / self.cell_size - 0.5;

    let i0 = (x.floor() as i32).clamp(0, self.width as i32 - 2) as usize;
    let j0 = (y.floor() as i32).clamp(0, self.height as i32 - 2) as usize;
    let i1 = i0 + 1;
    let j1 = j0 + 1;

    let tx = (x - i0 as f32).clamp(0.0, 1.0);
    let ty = (y - j0 as f32).clamp(0.0, 1.0);

    // Bilinear interpolation
    let v00 = self.vorticity[j0 * self.width + i0];
    let v10 = self.vorticity[j0 * self.width + i1];
    let v01 = self.vorticity[j1 * self.width + i0];
    let v11 = self.vorticity[j1 * self.width + i1];

    let v0 = v00 * (1.0 - tx) + v10 * tx;
    let v1 = v01 * (1.0 - tx) + v11 * tx;

    v0 * (1.0 - ty) + v1 * ty
}
```

**Test**: Unit test that vorticity sampling returns expected values at cell centers.

---

### Step 2: Modify Sand G2P with Vorticity-Based Suspension

**File**: `crates/sim/src/flip.rs` (around line 820)

Replace constant settling with vorticity-modulated settling:

```rust
// Sample vorticity at particle position
let vorticity = grid.sample_vorticity(pos);
let vort_magnitude = vorticity.abs();

// Vorticity creates lift force perpendicular to flow
// In 2D: positive ω (CCW rotation) → force perpendicular to velocity gradient
// Simplified: use vorticity magnitude to reduce settling

// Tunable parameters
const SETTLING_FACTOR: f32 = 0.62;      // Base settling (unchanged)
const VORT_LIFT_SCALE: f32 = 0.5;       // How much vorticity counters settling
const VORT_SWIRL_SCALE: f32 = 0.1;      // How much vorticity adds tangential motion

// 1. Compute settling reduction from vorticity
// High vorticity → less settling (particle stays suspended)
let lift_factor = (vort_magnitude * VORT_LIFT_SCALE).min(1.0);
let effective_settling = SETTLING_FACTOR * (1.0 - lift_factor);

// 2. Add swirl motion from vorticity
// Vorticity ω creates velocity perpendicular to the flow direction
// For simplicity: add velocity perpendicular to current particle velocity
let v_normalized = if particle.velocity.length() > 0.1 {
    particle.velocity.normalize()
} else {
    Vec2::new(1.0, 0.0)
};
let v_perp = Vec2::new(-v_normalized.y, v_normalized.x);
let swirl_velocity = v_perp * vorticity * VORT_SWIRL_SCALE;

// Apply modified settling + swirl
particle.velocity.y += GRAVITY * effective_settling * dt;
particle.velocity += swirl_velocity * dt;
```

---

### Step 3: Add Tuning Parameters to FlipSimulation

**File**: `crates/sim/src/flip.rs` (struct definition around line 55)

```rust
pub struct FlipSimulation {
    // ... existing fields ...

    /// Scale factor for vorticity lift effect on sand
    /// Higher = sand stays suspended longer in rotating flow
    pub vorticity_lift_scale: f32,

    /// Scale factor for vorticity swirl effect on sand
    /// Higher = sand follows rotational motion more
    pub vorticity_swirl_scale: f32,
}

// In new() or Default:
vorticity_lift_scale: 0.5,
vorticity_swirl_scale: 0.1,
```

---

### Step 4: Visual Testing

Create test scenario with clear vortex:
1. Strong inlet flow
2. Obstacle creating wake vortices
3. Sand particles spawned upstream

Expected behavior:
- Sand in calm water: sinks to bottom (bedload)
- Sand in fast flow: moves horizontally, settles slowly
- Sand in vortex: swirls, stays suspended, follows rotation

---

## Physics Justification

### Why Vorticity = Suspension?

In real sediment transport:
- Turbulent eddies carry particles upward against gravity
- Eddy turnover time vs settling time determines suspension
- Vorticity magnitude is a proxy for eddy intensity

The Rouse number relationship:
```
P = ws / (κ * u*)

where u* ≈ √(τ/ρ) and τ relates to velocity gradients
```

High vorticity → high velocity gradients → high shear stress → low effective Rouse number → suspension.

### Why Add Swirl Velocity?

In a vortex, particles don't just resist settling - they **rotate with the eddy**.

Angular velocity of fluid parcel = ω/2 (half the vorticity).

Adding perpendicular velocity proportional to ω approximates this rotation.

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Energy blow-up from lift | Cap lift_factor at 1.0, test with high vorticity |
| Sand never settles | Tune VORT_LIFT_SCALE, ensure calm regions have low ω |
| Swirl looks artificial | Start with low VORT_SWIRL_SCALE, increase gradually |
| Performance (sampling) | Vorticity sampling is O(1), similar to velocity sampling |

---

## Testing Strategy

1. **Unit test**: `sample_vorticity()` returns correct values at known positions
2. **Diagnostic**: Print/log vorticity magnitude at sand positions
3. **Visual test**: Sand in circular flow should orbit, not sink
4. **Regression**: Existing settling tests should still pass with ω=0

---

## Implementation Order

1. [x] Add `sample_vorticity()` to Grid - `grid.rs:512-538`
2. [x] Add unit test for vorticity sampling - `vortex_tests.rs:513-585`
3. [x] Modify sand G2P with lift - `flip.rs:820-861`
4. [x] Add swirl component - implemented with lift
5. [ ] Visual test: verify sand rotates in vortices
6. [ ] Tune parameters for realistic look
7. [ ] Add parameters to FlipSimulation struct for runtime tuning (if needed)

## Current Parameters (flip.rs:831-833)

```rust
const SETTLING_FACTOR: f32 = 0.62;     // Base settling (unchanged)
const VORT_LIFT_SCALE: f32 = 0.3;      // How much vorticity counters settling
const VORT_SWIRL_SCALE: f32 = 0.05;    // How much vorticity adds tangential motion
```

## Tuning Guide

- **Sand settles too fast**: Increase `VORT_LIFT_SCALE` (try 0.5-1.0)
- **Sand never settles**: Decrease `VORT_LIFT_SCALE` (try 0.1-0.2)
- **Sand doesn't swirl enough**: Increase `VORT_SWIRL_SCALE` (try 0.1-0.2)
- **Sand swirls too chaotically**: Decrease `VORT_SWIRL_SCALE` (try 0.02-0.03)
