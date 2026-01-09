<!-- STATUS: Work In Progress -->

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
