# DEM Physics Tests Plan

> **Created:** 2026-01-11
> **Status:** Planning

## Overview

Physics validation tests for the Discrete Element Method (DEM) implementation in `crates/sim3d/src/clump.rs`. These tests validate gravel/sediment/rock behavior against analytical solutions and expected physical behavior.

## Test Categories

### 1. Gravity and Free Fall (Analytical)

**Test: `test_dem_freefall`**
- Drop a clump from height h with no collisions
- Expected: y(t) = h - 0.5*g*t², v(t) = -g*t
- Validates basic integration and gravity application
- Tolerance: <1% error after 1 second

**Test: `test_dem_terminal_velocity`**
- Drop clump in water (FLIP coupled)
- Expected: reaches terminal velocity where drag = weight
- v_terminal ≈ sqrt(2mg / (ρ_water * C_d * A))
- Validates drag force implementation

### 2. Collision Response

**Test: `test_dem_floor_collision`**
- Drop clump onto solid floor
- Expected: bounces to height h * e² (e = restitution)
- With e=0.2, second bounce should reach ~4% of drop height
- Validates normal collision response and energy dissipation

**Test: `test_dem_wall_collision`**
- Slide clump into vertical wall at angle
- Expected: reflects with damped velocity
- Validates SDF collision detection and gradient normal

**Test: `test_dem_clump_collision`**
- Two clumps approach head-on at equal speeds
- Expected: symmetric bounce, momentum conserved
- Validates clump-clump collision detection

**Test: `test_dem_collision_no_penetration`**
- Many clumps settling in a box
- Expected: no clump centers inside solid SDF
- Validates collision resolution prevents tunneling

### 3. Friction Behavior

**Test: `test_dem_static_friction`**
- Clump on inclined plane below critical angle
- Expected: remains stationary (static friction holds)
- Critical angle = atan(μ_static)
- With μ=0.6, should hold at 30° but slide at 35°

**Test: `test_dem_kinetic_friction`**
- Push clump on flat surface, release
- Expected: decelerates at a = μ*g
- Validates friction force = μ * N

**Test: `test_dem_wet_vs_dry_friction`**
- Same incline test with wet=true vs wet=false
- Expected: wet clump slides at lower angle (μ=0.08 vs 0.4)
- Critical wet angle ≈ 4.6° vs dry ≈ 22°

**Test: `test_dem_friction_saturation`**
- Apply large tangential force
- Expected: friction capped at μ*N (Coulomb limit)
- Clump slides rather than infinite friction

### 4. Rolling Dynamics

**Test: `test_dem_rolling_friction`**
- Roll sphere on flat surface
- Expected: angular velocity decays due to rolling friction
- ω(t) = ω₀ * exp(-μ_roll * g * t / r)

**Test: `test_dem_spin_conservation`**
- Spin clump in free space (no contacts)
- Expected: angular momentum conserved perfectly
- Validates no spurious angular damping

**Test: `test_dem_rotation_from_contact`**
- Clump hits floor off-center
- Expected: gains angular velocity from contact torque
- Validates tangential force creates torque

### 5. Settling and Separation

**Test: `test_dem_settling_time`**
- Drop clumps, measure time to reach stable pile
- Expected: velocity < threshold within reasonable time
- Validates damping leads to stable equilibrium

**Test: `test_dem_density_separation`**
- Mix of heavy (gold: 19.3) and light (gangue: 2.7) clumps in water
- Expected: heavy clumps settle to bottom, light stay higher
- Measures center-of-mass separation after settling

**Test: `test_dem_angle_of_repose`**
- Pour clumps into pile, measure final slope angle
- Expected: angle ≈ atan(μ) ± 5°
- Validates friction creates realistic granular piles

### 6. Water Coupling (FLIP-DEM)

**Test: `test_dem_buoyancy`**
- Submerged clump with density < water
- Expected: rises (net upward force)
- Submerged clump with density > water
- Expected: sinks (net downward force)

**Test: `test_dem_drag_force`**
- Clump moving through still water
- Expected: decelerates exponentially
- v(t) = v₀ * exp(-drag * t / m)

**Test: `test_dem_water_velocity_coupling`**
- Clump in flowing water (uniform flow field)
- Expected: accelerates toward water velocity
- Steady state: clump velocity ≈ water velocity

### 7. Energy and Stability

**Test: `test_dem_energy_dissipation`**
- Drop clump, measure total energy over time
- Expected: energy monotonically decreases
- No energy injection from contact model

**Test: `test_dem_no_explosion`**
- Many clumps in small space
- Expected: velocities remain bounded (<100 m/s)
- Validates stiffness/damping balance

**Test: `test_dem_determinism`**
- Run same scenario twice
- Expected: identical results (deterministic)
- Validates no random behavior in physics

### 8. Spatial Hashing

**Test: `test_dem_spatial_hash_correctness`**
- Place clumps in known positions
- Expected: all colliding pairs detected
- No false negatives in neighbor finding

**Test: `test_dem_sparse_distribution`**
- Very spread out clumps (no collisions)
- Expected: O(n) performance, no spurious collisions

## Implementation Structure

```rust
// crates/game/examples/test_dem_physics.rs

fn main() {
    let mut passed = 0;
    let mut failed = 0;

    // Gravity tests
    run_test("Freefall", test_dem_freefall, &mut passed, &mut failed);
    run_test("Terminal Velocity", test_dem_terminal_velocity, &mut passed, &mut failed);

    // Collision tests
    run_test("Floor Collision", test_dem_floor_collision, &mut passed, &mut failed);
    run_test("No Penetration", test_dem_collision_no_penetration, &mut passed, &mut failed);

    // Friction tests
    run_test("Static Friction", test_dem_static_friction, &mut passed, &mut failed);
    run_test("Wet vs Dry", test_dem_wet_vs_dry_friction, &mut passed, &mut failed);

    // ... etc

    println!("DEM Physics: {}/{} passed", passed, passed + failed);
}
```

## Test Parameters

```rust
const CLUMP_RADIUS: f32 = 0.01;      // 1cm gravel
const CLUMP_DENSITY: f32 = 2650.0;   // kg/m³ (granite)
const GOLD_DENSITY: f32 = 19300.0;   // kg/m³
const WATER_DENSITY: f32 = 1000.0;   // kg/m³
const GRAVITY: f32 = -9.81;
const DT: f32 = 1.0 / 120.0;         // 120 Hz for stability
const TOLERANCE: f32 = 0.05;         // 5% error tolerance
```

## Priority Order

1. **P0 (Must have):**
   - Freefall (validates integration)
   - Floor collision (validates basic response)
   - No penetration (validates safety)
   - Energy dissipation (validates stability)

2. **P1 (Should have):**
   - Static/kinetic friction
   - Wet vs dry friction
   - Density separation
   - Settling time

3. **P2 (Nice to have):**
   - Rolling friction
   - Angle of repose
   - Determinism
   - Spatial hash correctness

## Integration with Test Suite

Add to `scripts/run_tests.sh`:
```bash
echo "Running DEM Physics Tests..."
if cargo run --example test_dem_physics --release; then
    echo "PASS: DEM Physics Tests"
    ((PASSED++))
else
    echo "FAIL: DEM Physics Tests"
    ((FAILED++))
fi
```

## Success Criteria

- All P0 tests pass
- 80%+ of P1 tests pass
- Physical behaviors match analytical predictions within tolerance
- No instabilities or explosions under stress testing
