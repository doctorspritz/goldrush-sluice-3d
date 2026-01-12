# Headless Physics Tests - Automated Validation

## Overview

Automated headless tests that verify physics behavior with clear pass/fail criteria.
Each test runs without user interaction, records video, outputs verdict.

## Test Infrastructure

```
cargo test -p game --test physics_validation -- --nocapture
```

Each test:
1. Sets up scenario using EditorLayout::new_connected()
2. Runs simulation for N frames (physics-based duration, not arbitrary)
3. Records to MP4 in `test_output/`
4. Checks pass/fail criteria
5. Outputs clear verdict with metrics

## Physics Rules (INVIOLABLE)

These are physics laws, not tunable parameters:

1. **Conservation of mass**: Total particle count stable (inlet = outlet over time)
2. **Conservation of momentum**: No spontaneous velocity changes
3. **Conservation of energy**: System energy decreases (friction) or stays same, never increases
4. **Gravity**: All unsupported objects accelerate at 9.81 m/s² downward
5. **Collision**: No interpenetration - SDF distance always >= particle radius
6. **Buoyancy**: Dense particles sink, light particles float (Archimedes)

## Test Definitions

### TEST 1: DEM Floor Collision
**Setup**: Drop 50 particles from 10cm above gutter floor
**Duration**: 5 seconds (300 frames at 60fps)
**Pass criteria**:
- ALL particles have y > floor_y (no penetration)
- ALL particles have velocity < 0.01 m/s after 3s (settled)
- Max bounce height decreases each bounce (energy dissipation)
**Fail criteria**:
- Any particle y < floor_y - particle_radius
- Particles still bouncing after 5s
- Bounce height increases (energy creation)

### TEST 2: DEM Wall Collision
**Setup**: 20 particles with lateral velocity toward wall
**Duration**: 3 seconds
**Pass criteria**:
- ALL particles remain within gutter width bounds
- Velocity reverses direction on wall contact
- No particle escapes through wall
**Fail criteria**:
- Any particle |z| > gutter_half_width
- Particle passes through wall without velocity change

### TEST 3: DEM Density Separation
**Setup**: 25 gold (19300 kg/m³) + 25 sand (2650 kg/m³) dropped into water
**Duration**: 10 seconds
**Pass criteria**:
- Average gold_y < average sand_y after settling
- Gold settles to bottom third of water column
- Sand remains in top two-thirds
**Fail criteria**:
- Gold floats above sand
- No vertical separation after 10s

### TEST 4: DEM Settling Time
**Setup**: 100 particles dropped in pile
**Duration**: 10 seconds
**Pass criteria**:
- Average velocity < 0.01 m/s within 5 seconds
- No particle velocity > 0.1 m/s after 8 seconds
**Fail criteria**:
- Particles still moving significantly after 5s
- Jittering (oscillating velocity) detected

### TEST 5: Fluid Flow Direction
**Setup**: Water released at top of tilted gutter (angle > 0)
**Duration**: 5 seconds
**Pass criteria**:
- Average fluid velocity.x has same sign as -sin(gutter_angle)
- Fluid reaches outlet end of gutter
**Fail criteria**:
- Fluid flows uphill
- Fluid stuck at inlet

### TEST 6: Fluid Pool Equilibrium
**Setup**: Flat pool of water (no slope)
**Duration**: 5 seconds after initial settling
**Pass criteria**:
- RMS velocity < 0.01 m/s (still water)
- No net momentum
- Water level constant (no evaporation/creation)
**Fail criteria**:
- Spontaneous flow
- Energy increasing over time

### TEST 7: Fluid Wall Containment
**Setup**: Water poured into gutter
**Duration**: 10 seconds
**Pass criteria**:
- ALL fluid particles within gutter bounds
- No leaks through floor or walls
**Fail criteria**:
- Any particle outside gutter geometry
- Particle count outside bounds > 0

### TEST 8: Sediment Settling in Still Water
**Setup**: Sediment dropped into still pool
**Duration**: 10 seconds
**Pass criteria**:
- Sediment y decreases over time (sinking)
- Terminal velocity reached (Stokes' law)
- Sediment reaches floor
**Fail criteria**:
- Sediment floats
- Sediment stuck mid-water
- Instant teleport to floor

### TEST 9: Sediment Advection
**Setup**: Sediment in flowing water
**Duration**: 10 seconds
**Pass criteria**:
- Sediment velocity correlates with local fluid velocity
- Sediment moves downstream
**Fail criteria**:
- Sediment moves against flow
- Sediment stationary in flowing water

### TEST 10: Sluice Riffle Capture
**Setup**: Gold + sand in water flowing over sluice riffles
**Duration**: 30 seconds
**Pass criteria**:
- Gold accumulates behind riffles (gold_count_behind_riffle increases)
- Sand washes over riffles (sand_count_at_outlet > sand_count_behind_riffle)
- Separation ratio > 2.0 (gold capture rate / sand capture rate)
**Fail criteria**:
- All material washes through
- All material stuck
- No separation

## Video Recording

Each test records MP4 to `test_output/{test_name}_{timestamp}.mp4`
- 1280x720 resolution
- 60 fps
- Duration matches test duration
- Overlays: test name, frame count, key metrics, pass/fail status

## Implementation

File: `crates/game/tests/physics_validation.rs`

```rust
// Pseudocode structure
#[test]
fn test_dem_floor_collision() {
    let mut harness = TestHarness::new("dem_floor_collision");
    harness.setup_connected_layout();
    harness.spawn_particles_above_floor(50, 0.10);

    harness.run_for_seconds(5.0, |frame, state| {
        // Record frame to video
        // Check invariants each frame
        for particle in state.particles() {
            assert!(particle.y >= floor_y - EPSILON, "Penetration detected");
        }
    });

    // Final checks
    let avg_vel = harness.average_particle_velocity();
    assert!(avg_vel < 0.01, "Particles not settled: avg_vel={}", avg_vel);

    harness.finalize_video();
    harness.write_report();
}
```

## Run Command

```bash
# Run all physics tests, generate videos
cargo test -p game --test physics_validation --release -- --nocapture

# View results
ls test_output/*.mp4
cat test_output/report.txt
```
