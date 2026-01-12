//! Physics tests for Shallow Water Equations
//!
//! These tests verify that the SWE implementation matches real water behavior:
//! 1. A flat pool should remain still (no spurious oscillations)
//! 2. Water flows downhill at physically correct velocities
//! 3. Mass is conserved
//! 4. Steady-state flow follows Manning's equation

use sim3d::{World, WorldParams};

const CELL_SIZE: f32 = 0.5;
const GRAVITY: f32 = 9.81;

/// Helper to create a simple test world
fn create_world(width: usize, depth: usize, cell_size: f32) -> World {
    World::new(width, depth, cell_size, 0.0)
}

/// Helper to set flat terrain at a given height
fn set_flat_terrain(world: &mut World, height: f32) {
    for i in 0..world.bedrock_elevation.len() {
        world.bedrock_elevation[i] = height;
        world.paydirt_thickness[i] = 0.0;
        world.gravel_thickness[i] = 0.0;
        world.overburden_thickness[i] = 0.0;
        world.terrain_sediment[i] = 0.0;
    }
}

/// Helper to add water at uniform depth
fn add_water(world: &mut World, depth: f32) {
    for x in 0..world.width {
        for z in 0..world.depth {
            let idx = world.idx(x, z);
            let ground = world.ground_height(x, z);
            world.water_surface[idx] = ground + depth;
        }
    }
}

/// Helper to compute total water volume
fn total_water_volume(world: &World) -> f32 {
    let cell_area = world.cell_size * world.cell_size;
    let mut total = 0.0;
    for x in 0..world.width {
        for z in 0..world.depth {
            let idx = world.idx(x, z);
            let ground = world.ground_height(x, z);
            let depth = (world.water_surface[idx] - ground).max(0.0);
            total += depth * cell_area;
        }
    }
    total
}

/// Helper to compute max velocity magnitude
fn max_velocity(world: &World) -> f32 {
    let mut max_v = 0.0f32;
    for i in 0..world.water_flow_x.len() {
        max_v = max_v.max(world.water_flow_x[i].abs());
    }
    for i in 0..world.water_flow_z.len() {
        max_v = max_v.max(world.water_flow_z[i].abs());
    }
    max_v
}

// =============================================================================
// TEST 1: FLAT POOL STABILITY
// A flat pool of water should remain perfectly still - no oscillations
// =============================================================================

#[test]
fn flat_pool_remains_still() {
    let mut world = create_world(32, 32, CELL_SIZE);
    // Use closed boundaries for this test - no draining at edges
    world.params.open_boundaries = false;
    set_flat_terrain(&mut world, 0.0);

    // Add 0.5m of water EVERYWHERE (including edges) - truly flat surface
    let water_height = 0.5;
    for x in 0..world.width {
        for z in 0..world.depth {
            let idx = world.idx(x, z);
            world.water_surface[idx] = water_height;
        }
    }

    // Zero all velocities
    for v in &mut world.water_flow_x {
        *v = 0.0;
    }
    for v in &mut world.water_flow_z {
        *v = 0.0;
    }

    let initial_volume = total_water_volume(&world);

    // Run simulation for 100 steps
    let dt = 0.01;
    for _ in 0..100 {
        world.update_water_flow(dt);
    }

    // Velocity should be essentially zero (< 1mm/s)
    // We check interior cells only since boundary handling may differ
    let mut max_v = 0.0f32;
    for z in 2..world.depth - 2 {
        for x in 2..world.width - 2 {
            let fx = world.flow_x_idx(x, z);
            let fz = world.flow_z_idx(x, z);
            max_v = max_v.max(world.water_flow_x[fx].abs());
            max_v = max_v.max(world.water_flow_z[fz].abs());
        }
    }

    assert!(
        max_v < 0.001,
        "Flat pool should be still, but max interior velocity = {} m/s",
        max_v
    );

    // Volume in interior should be mostly conserved
    let final_volume = total_water_volume(&world);
    let volume_change = (final_volume - initial_volume).abs();
    // Allow some loss at boundaries
    assert!(
        volume_change < initial_volume * 0.1,
        "Lost too much volume: {:.2}m³ of {:.2}m³",
        volume_change,
        initial_volume
    );
}

// =============================================================================
// TEST 2: WATER FLOWS DOWNHILL
// Water on a slope should flow in the downhill direction
// =============================================================================

#[test]
fn water_flows_downhill() {
    let mut world = create_world(32, 16, CELL_SIZE);

    // Create slope: high on left (x=0), low on right (x=max)
    // 1m drop over 16m = 6.25% slope
    for z in 0..world.depth {
        for x in 0..world.width {
            let idx = world.idx(x, z);
            let height = 1.0 - (x as f32 / world.width as f32);
            world.bedrock_elevation[idx] = height;
        }
    }

    // Add water in the middle
    for z in 4..12 {
        for x in 10..20 {
            let idx = world.idx(x, z);
            let ground = world.ground_height(x, z);
            world.water_surface[idx] = ground + 0.2;
        }
    }

    // Run simulation
    let dt = 0.01;
    for _ in 0..50 {
        world.update_water_flow(dt);
    }

    // Check that net flow is in positive X direction (downhill)
    let mut net_flow_x = 0.0;
    for i in 0..world.water_flow_x.len() {
        net_flow_x += world.water_flow_x[i];
    }

    assert!(
        net_flow_x > 0.0,
        "Water should flow downhill (+X), but net flow = {} m/s",
        net_flow_x
    );
}

// =============================================================================
// TEST 3: STEADY-STATE VELOCITY (Manning's Equation)
// For uniform flow down a slope, velocity should match Manning's equation:
// V = (1/n) * R^(2/3) * S^(1/2)
// where n ≈ 0.03 for smooth channel, R = hydraulic radius, S = slope
// =============================================================================

#[test]
fn steady_state_velocity_matches_manning() {
    let mut world = create_world(64, 16, CELL_SIZE);

    // Slope: 2% grade (0.02)
    let slope = 0.02;
    for z in 0..world.depth {
        for x in 0..world.width {
            let idx = world.idx(x, z);
            let height = 2.0 - (x as f32 * world.cell_size * slope);
            world.bedrock_elevation[idx] = height;
        }
    }

    // Wide channel flow - add water across the middle
    let water_depth = 0.3; // 30cm
    for z in 4..12 {
        for x in 5..60 {
            let idx = world.idx(x, z);
            let ground = world.ground_height(x, z);
            world.water_surface[idx] = ground + water_depth;
        }
    }

    // Run to steady state (longer simulation)
    let dt = 0.005;
    for _ in 0..500 {
        world.update_water_flow(dt);

        // Re-add water at upstream to maintain flow
        for z in 4..12 {
            for x in 5..10 {
                let idx = world.idx(x, z);
                let ground = world.ground_height(x, z);
                world.water_surface[idx] = ground + water_depth;
            }
        }
    }

    // Measure average velocity in the middle section
    let mut sum_vel = 0.0;
    let mut count = 0;
    for z in 6..10 {
        for x in 25..35 {
            let idx = world.idx(x, z);
            sum_vel += world.water_flow_x[idx].abs();
            count += 1;
        }
    }
    let avg_velocity = sum_vel / count as f32;

    // Manning's equation prediction
    // For wide channel, R ≈ depth
    // Using n = 0.03 (smooth channel)
    let n = 0.03;
    let r = water_depth; // hydraulic radius ≈ depth for wide channel
    let manning_velocity = (1.0 / n) * r.powf(2.0 / 3.0) * slope.sqrt();

    // Allow 50% tolerance (SWE is simplified, not full Manning)
    let tolerance = 0.5;
    let error = (avg_velocity - manning_velocity).abs() / manning_velocity;

    println!("Measured velocity: {:.3} m/s", avg_velocity);
    println!("Manning prediction: {:.3} m/s", manning_velocity);
    println!("Error: {:.1}%", error * 100.0);

    assert!(
        error < tolerance,
        "Velocity {:.3} m/s should be within {:.0}% of Manning prediction {:.3} m/s",
        avg_velocity,
        tolerance * 100.0,
        manning_velocity
    );
}

// =============================================================================
// TEST 4: MASS CONSERVATION
// Total water volume should be conserved (closed system)
// =============================================================================

#[test]
fn mass_conservation_closed_system() {
    let mut world = create_world(32, 32, CELL_SIZE);
    // Closed boundaries - no water drains at edges
    world.params.open_boundaries = false;
    set_flat_terrain(&mut world, 0.0);

    // Add varying water depths (not flat)
    for x in 0..world.width {
        for z in 0..world.depth {
            let idx = world.idx(x, z);
            // Gaussian mound of water
            let cx = world.width as f32 / 2.0;
            let cz = world.depth as f32 / 2.0;
            let dx = x as f32 - cx;
            let dz = z as f32 - cz;
            let r2 = dx * dx + dz * dz;
            let depth = 0.5 * (-r2 / 50.0).exp();
            world.water_surface[idx] = depth;
        }
    }

    let initial_volume = total_water_volume(&world);

    // Run simulation for many steps
    let dt = 0.01;
    for _ in 0..200 {
        world.update_water_flow(dt);
    }

    let final_volume = total_water_volume(&world);

    // Volume should be conserved within 0.1%
    let volume_error = (final_volume - initial_volume).abs() / initial_volume;
    assert!(
        volume_error < 0.001,
        "Volume error {:.3}% exceeds 0.1% tolerance. Initial: {:.3}m³, Final: {:.3}m³",
        volume_error * 100.0,
        initial_volume,
        final_volume
    );
}

// =============================================================================
// TEST 5: WAVE SPEED
// Shallow water wave speed should be c = sqrt(g * h)
// =============================================================================

#[test]
fn wave_speed_matches_theory() {
    let mut world = create_world(100, 16, CELL_SIZE);
    set_flat_terrain(&mut world, 0.0);

    // Base water depth
    let base_depth = 0.5;
    add_water(&mut world, base_depth);

    // Add a small perturbation (bump) at x=20
    for z in 4..12 {
        for x in 18..22 {
            let idx = world.idx(x, z);
            world.water_surface[idx] += 0.1; // 10cm bump
        }
    }

    // Track the position of the wave front
    let mut wave_positions = Vec::new();

    let dt = 0.005;
    for step in 0..100 {
        world.update_water_flow(dt);

        // Find rightmost cell with elevated water (wave front)
        let mut max_x = 20;
        for x in 20..world.width {
            let idx = world.idx(x, 8);
            let ground = world.ground_height(x, 8);
            let depth = world.water_surface[idx] - ground;
            if depth > base_depth + 0.01 {
                max_x = x;
            }
        }

        if step % 20 == 0 {
            wave_positions.push((step as f32 * dt, max_x as f32 * world.cell_size));
        }
    }

    // Calculate observed wave speed from positions
    if wave_positions.len() >= 2 {
        let (t1, x1) = wave_positions[0];
        let (t2, x2) = wave_positions[wave_positions.len() - 1];
        let observed_speed = (x2 - x1) / (t2 - t1);

        // Theoretical wave speed
        let theoretical_speed = (GRAVITY * base_depth).sqrt();

        println!("Observed wave speed: {:.2} m/s", observed_speed);
        println!("Theoretical (sqrt(gh)): {:.2} m/s", theoretical_speed);

        // Allow 30% tolerance (numerical diffusion will slow waves)
        let error = (observed_speed - theoretical_speed).abs() / theoretical_speed;
        assert!(
            error < 0.3,
            "Wave speed {:.2} m/s should be within 30% of theoretical {:.2} m/s",
            observed_speed,
            theoretical_speed
        );
    }
}

// =============================================================================
// TEST 6: NO ENERGY CREATION
// A system should not spontaneously gain energy (create oscillations from nothing)
// =============================================================================

#[test]
fn no_spontaneous_energy_creation() {
    let mut world = create_world(32, 32, CELL_SIZE);
    // Closed boundaries - no draining at edges
    world.params.open_boundaries = false;
    set_flat_terrain(&mut world, 0.0);

    // Perfectly flat, still water - EVERYWHERE including edges
    let water_height = 0.3;
    for x in 0..world.width {
        for z in 0..world.depth {
            let idx = world.idx(x, z);
            world.water_surface[idx] = water_height;
        }
    }

    // Zero all velocities explicitly
    for i in 0..world.water_flow_x.len() {
        world.water_flow_x[i] = 0.0;
    }
    for i in 0..world.water_flow_z.len() {
        world.water_flow_z[i] = 0.0;
    }

    // Track kinetic energy in INTERIOR cells only
    let compute_interior_ke = |w: &World| -> f32 {
        let mut ke = 0.0;
        // Only check interior to avoid boundary effects
        for z in 2..w.depth - 2 {
            for x in 2..w.width - 2 {
                let fx = w.flow_x_idx(x, z);
                let fz = w.flow_z_idx(x, z);
                ke += w.water_flow_x[fx] * w.water_flow_x[fx];
                ke += w.water_flow_z[fz] * w.water_flow_z[fz];
            }
        }
        ke
    };

    let initial_ke = compute_interior_ke(&world);

    // Run simulation
    let dt = 0.01;
    let mut max_ke = initial_ke;
    for _ in 0..100 {
        world.update_water_flow(dt);
        max_ke = max_ke.max(compute_interior_ke(&world));
    }

    // Kinetic energy should never exceed initial (which is 0) by more than tiny numerical error
    assert!(
        max_ke < 1e-6,
        "Energy created from nothing! Max interior KE = {}, should be ~0",
        max_ke
    );
}
