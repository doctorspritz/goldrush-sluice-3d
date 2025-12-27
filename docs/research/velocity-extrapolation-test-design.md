# Velocity Extrapolation: Test Design

**Date:** 2025-12-28
**Purpose:** Define behavior tests that model reality and prevent implementation cheating

## Anti-Cheat Principles

### What We're Preventing

Previous failures show Claude will:
1. Tweak constants (gravity, damping) to make specific test values pass
2. Add hacks that pass isolated tests but break real behavior
3. Optimize for specific test scenarios instead of general correctness

### How We Prevent It

1. **Use physical invariants, not magic numbers**
   - Conservation laws that MUST hold regardless of parameters
   - Ratios and relative comparisons, not absolute values

2. **Test multiple configurations**
   - Randomized initial conditions
   - Various grid sizes and particle counts
   - Different geometries

3. **Test the REAL simulation**
   - Full timestep, not isolated functions
   - Multiple frames, not single-shot

4. **Comparison tests, not value tests**
   - "A should equal B" not "A should equal 42.5"
   - "After should be close to Before" not "After should be 100.0"

---

## Test 1: Isolated FLIP Cycle Conservation

**What it tests:** P2G → extrapolate → store_old → G2P cycle conserves momentum when no forces applied

**Why it can't be cheated:**
- Uses ratio (after/before), not absolute value
- Tests with RANDOM particle velocities (can't tune for specific case)
- Threshold is physics-based (numerical precision), not tunable

```rust
#[test]
fn flip_cycle_conserves_momentum_with_extrapolation() {
    // Run 10 times with different random seeds
    for seed in 0..10 {
        let mut sim = create_test_sim_with_random_velocities(seed);

        let momentum_before = sim.total_particle_momentum();

        // Run isolated cycle: P2G → extrapolate → store_old → G2P
        // NO forces, NO pressure, NO boundaries modified
        sim.run_isolated_flip_cycle_with_extrapolation();

        let momentum_after = sim.total_particle_momentum();

        // Conservation ratio must be > 0.99 (1% tolerance for numerics)
        let ratio = momentum_after.length() / momentum_before.length();
        assert!(ratio > 0.99, "Seed {}: ratio {} < 0.99", seed, ratio);
        assert!(ratio < 1.01, "Seed {}: ratio {} > 1.01 (energy gain)", seed, ratio);
    }
}
```

**Anti-cheat notes:**
- Random velocities prevent tuning to specific case
- 10 seeds means can't just tweak one scenario
- Ratio-based, not value-based
- Both lower AND upper bound (no artificial energy gain either)

---

## Test 2: No Phantom Delta at Air Boundaries

**What it tests:** Particles near air cells get zero FLIP delta when grid unchanged

**Why it can't be cheated:**
- Compares delta to ZERO (physical requirement)
- Places particles at MULTIPLE positions near boundary
- Random particle positions within boundary zone

```rust
#[test]
fn no_phantom_delta_at_air_boundary() {
    for seed in 0..5 {
        let mut sim = create_sim_with_fluid_block(seed);

        // Place test particle at random position within 1 cell of air
        let test_particle = place_particle_near_air_boundary(&sim, seed);
        let original_velocity = test_particle.velocity;

        // P2G, extrapolate, store_old, (NO forces), G2P
        sim.run_isolated_flip_cycle_with_extrapolation();

        let new_velocity = sim.get_particle_velocity(test_particle.id);
        let delta = new_velocity - original_velocity;

        // Delta must be essentially zero (no phantom forces)
        // Use relative threshold: delta < 1% of original velocity
        let threshold = original_velocity.length() * 0.01;
        assert!(
            delta.length() < threshold,
            "Seed {}: phantom delta {} exceeds threshold {}",
            seed, delta.length(), threshold
        );
    }
}
```

**Anti-cheat notes:**
- Threshold is RELATIVE to particle velocity, not absolute
- Random positions prevent tuning for specific location
- Tests the ABSENCE of phantom forces (physical invariant)

---

## Test 3: Tangential Velocity Preserved at Solid Walls

**What it tests:** Particles sliding along walls keep their tangential velocity

**Why it can't be cheated:**
- Tests component decomposition (tangent vs normal)
- Normal should be zero, tangent should be preserved
- Multiple wall orientations (horizontal, vertical)

```rust
#[test]
fn tangential_velocity_preserved_at_walls() {
    // Test both horizontal and vertical walls
    for wall_orientation in [Wall::Horizontal, Wall::Vertical] {
        for seed in 0..5 {
            let mut sim = create_sim_with_wall(wall_orientation);

            // Particle moving parallel to wall
            let tangent_velocity = match wall_orientation {
                Wall::Horizontal => Vec2::new(10.0, 0.0),
                Wall::Vertical => Vec2::new(0.0, 10.0),
            };
            let test_particle = place_particle_at_wall(&sim, tangent_velocity, seed);

            // Run one full timestep
            sim.step(DT);

            let new_velocity = sim.get_particle_velocity(test_particle.id);

            // Tangent component should be preserved (> 95%)
            let tangent_after = project_tangent(new_velocity, wall_orientation);
            let ratio = tangent_after.length() / tangent_velocity.length();

            assert!(
                ratio > 0.95,
                "Wall {:?} seed {}: tangent ratio {} < 0.95",
                wall_orientation, seed, ratio
            );
        }
    }
}
```

**Anti-cheat notes:**
- Tests BOTH orientations (can't tune for just one)
- Uses ratio, not absolute value
- Physical requirement: free-slip condition

---

## Test 4: Multi-Frame Momentum Stability

**What it tests:** Momentum doesn't decay exponentially over time

**Why it can't be cheated:**
- Tests over 60 frames (1 second at 60fps)
- Compares frame 60 to frame 1
- Must maintain at least 90% after full second

```rust
#[test]
fn momentum_stable_over_time() {
    // Test with flowing water (not stationary)
    let mut sim = create_flowing_water_sim();

    // Run 10 frames to let simulation stabilize
    for _ in 0..10 {
        sim.step(DT);
    }

    let momentum_at_frame_10 = sim.total_water_momentum();

    // Run 60 more frames
    for _ in 0..60 {
        sim.step(DT);
    }

    let momentum_at_frame_70 = sim.total_water_momentum();

    // Must retain at least 90% over 60 frames (1 second)
    // This allows for ~0.17% loss per frame (numerical precision)
    // Previous bug: 2% per frame = 30% retained = FAIL
    let ratio = momentum_at_frame_70.length() / momentum_at_frame_10.length();

    assert!(
        ratio > 0.90,
        "Momentum decayed to {}% after 60 frames (expected > 90%)",
        ratio * 100.0
    );
}
```

**Anti-cheat notes:**
- 60 frames amplifies small per-frame errors
- 90% threshold catches 2% per-frame loss (30% retained = fail)
- Tests REAL simulation, not isolated function
- Can't be cheated by single-frame fixes

---

## Test 5: Extrapolation Correctness (Behavior, Not Values)

**What it tests:** Extrapolated velocities are consistent with fluid neighbors

**Why it can't be cheated:**
- Tests that extrapolated value is AVERAGE of neighbors
- Not a specific value, but a relationship

```rust
#[test]
fn extrapolated_velocity_matches_neighbor_average() {
    let mut sim = create_sim_with_known_fluid_velocities();

    // Set specific velocities in fluid cells
    // (tests use cell coordinates, not world positions)
    sim.set_cell_velocity(5, 5, Vec2::new(10.0, 0.0)); // fluid
    sim.set_cell_velocity(5, 6, Vec2::new(20.0, 0.0)); // fluid
    sim.set_cell_type(5, 7, CellType::Air);

    // Run extrapolation
    sim.extrapolate_velocities(1); // 1 layer

    // Air cell (5,7) should have average of its fluid neighbors
    let extrapolated = sim.get_cell_velocity(5, 7);
    let expected_avg = Vec2::new(15.0, 0.0); // (10 + 20) / 2

    let diff = (extrapolated - expected_avg).length();
    assert!(
        diff < 0.01,
        "Extrapolated {:?} doesn't match expected average {:?}",
        extrapolated, expected_avg
    );
}
```

**Anti-cheat notes:**
- Tests the ALGORITHM (average of neighbors), not magic values
- The "expected" value is computed from inputs, not hardcoded
- If implementation is correct, this passes regardless of constants

---

## Test 6: Regression Test - Real Game Scenario

**What it tests:** Water in sluice doesn't flow like honey

**Why it can't be cheated:**
- Uses REAL game configuration
- Tests downstream velocity > inlet velocity (gravity accelerates)
- Visual sanity check (can also be run manually)

```rust
#[test]
fn water_accelerates_downhill_in_sluice() {
    let mut sim = create_game_mirror_sim(); // Matches real game

    // Run for 2 seconds
    for _ in 0..120 {
        sim.step(DT);
    }

    // Measure average velocity at inlet vs outlet
    let inlet_velocity = sim.average_particle_velocity_in_region(inlet_region());
    let outlet_velocity = sim.average_particle_velocity_in_region(outlet_region());

    // Outlet should be faster (gravity accelerates flow)
    // At minimum, outlet should not be SLOWER than inlet
    assert!(
        outlet_velocity.length() >= inlet_velocity.length() * 0.9,
        "Outlet velocity {} is much slower than inlet {} - honey behavior!",
        outlet_velocity.length(), inlet_velocity.length()
    );
}
```

**Anti-cheat notes:**
- Tests REAL behavior user cares about
- Uses 0.9 factor to allow for some friction/turbulence
- If this fails, user will SEE the problem in real game

---

## Summary of Anti-Cheat Properties

| Test | Anti-Cheat Property |
|------|---------------------|
| 1. FLIP Conservation | Random seeds, ratio-based, both bounds |
| 2. No Phantom Delta | Relative threshold, random positions |
| 3. Tangent Preserved | Multiple orientations, ratio-based |
| 4. Multi-Frame Stability | 60-frame amplification, real sim |
| 5. Extrapolation | Tests algorithm relationship, not values |
| 6. Sluice Regression | Real game config, observable behavior |

## Implementation Notes

- All tests use `DT = 1.0/60.0` (real game timestep)
- All tests use production grid/particle parameters
- NO special "test mode" constants
- If a test fails, the IMPLEMENTATION is wrong, not the test
