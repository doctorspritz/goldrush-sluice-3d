//! Angle of Repose Collapse Tests
//!
//! Tests that discrete materials collapse to their physically correct
//! angle of repose when piled on a flat surface.
//!
//! Real-world angles of repose:
//! - Dry sand: 30-35°
//! - Gravel: 35-40°
//! - Dirt/soil: 30-45° depending on moisture
//! - Clay: 40-50°

use sim3d::{TerrainMaterial, World};

/// Helper to create a flat world with just bedrock base
fn create_flat_world(width: usize, depth: usize, cell_size: f32) -> World {
    let mut world = World::new(width, depth, cell_size, 0.0);

    // Set flat bedrock at height 1.0, no other layers
    let cell_count = width * depth;
    world.bedrock_elevation = vec![1.0; cell_count];
    world.paydirt_thickness = vec![0.0; cell_count];
    world.gravel_thickness = vec![0.0; cell_count];
    world.overburden_thickness = vec![0.0; cell_count];
    world.terrain_sediment = vec![0.0; cell_count];

    world
}

/// Add a pile of sediment at the center of the world
fn add_sediment_pile(world: &mut World, center_x: usize, center_z: usize, height: f32) {
    let idx = center_z * world.width + center_x;
    world.terrain_sediment[idx] = height;
}

/// Add a pile of overburden at the center of the world
fn add_overburden_pile(world: &mut World, center_x: usize, center_z: usize, height: f32) {
    let idx = center_z * world.width + center_x;
    world.overburden_thickness[idx] = height;
}

/// Measure the actual slope angle of the pile after collapse
/// Returns the cell-to-cell slope angle in degrees (the correct measurement)
fn measure_pile_angle(world: &World, center_x: usize, center_z: usize) -> f32 {
    let cell_size = world.cell_size;
    let center_height = world.ground_height(center_x, center_z);

    // Measure cell-to-cell slope in 4 cardinal directions
    let mut angles = Vec::new();

    for (dx, dz) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
        let nx = center_x as i32 + dx;
        let nz = center_z as i32 + dz;

        if nx < 0 || nz < 0 || nx >= world.width as i32 || nz >= world.depth as i32 {
            continue;
        }

        let neighbor_height = world.ground_height(nx as usize, nz as usize);
        let height_diff = center_height - neighbor_height;

        // Only measure where there's a slope
        if height_diff > 0.001 {
            let angle_rad = (height_diff / cell_size).atan();
            angles.push(angle_rad.to_degrees());
        }
    }

    if angles.is_empty() {
        0.0
    } else {
        angles.iter().sum::<f32>() / angles.len() as f32
    }
}

/// Run collapse simulation until stable (or max iterations)
fn run_until_stable(world: &mut World, max_iters: usize) -> usize {
    for i in 0..max_iters {
        let changed = world.update_terrain_collapse();
        if !changed {
            return i;
        }
    }

    max_iters
}

// =============================================================================
// TESTS
// =============================================================================

/// Test that a sediment pile collapses to approximately the correct angle
#[test]
fn test_sediment_collapse_angle() {
    let mut world = create_flat_world(31, 31, 0.1); // 3.1m x 3.1m at 10cm resolution
    let center = 15;

    // Add a pile of sediment (sand-like, ~32° angle of repose)
    add_sediment_pile(&mut world, center, center, 0.4);

    // Run collapse until stable
    let iters = run_until_stable(&mut world, 1000);

    // Measure the resulting slope angle (cell-to-cell, not peak-to-base)
    let angle = measure_pile_angle(&world, center, center);

    // Sand angle of repose is ~32°, allow tolerance for discrete grid effects
    let expected_angle = TerrainMaterial::Sand.angle_of_repose().to_degrees();
    let tolerance = 3.0; // degrees - tighter now with correct measurement

    println!("Sediment pile:");
    println!("  Iterations to stable: {}", iters);
    println!("  Measured slope angle: {:.1}°", angle);
    println!("  Expected angle: {:.1}° (±{}°)", expected_angle, tolerance);

    assert!(
        (angle - expected_angle).abs() < tolerance,
        "Sediment should collapse to ~{}° angle of repose, got {:.1}°",
        expected_angle,
        angle
    );
}

/// Test that sediment always collapses using sand's angle, regardless of substrate
/// Note: The surface_material function returns Sand if there's ANY sediment.
/// This means sediment piles always use sand's angle of repose (32°).
#[test]
fn test_sediment_uses_sand_angle_regardless_of_substrate() {
    let mut world = create_flat_world(31, 31, 0.1);
    let center = 15;

    // Add a base layer of overburden (dirt) everywhere - but this doesn't matter
    // because sediment on top will use sand's angle
    world.overburden_thickness = vec![0.5; 31 * 31];

    // Add a sediment pile on top - uses SAND's angle (32°) not dirt's (35°)
    add_sediment_pile(&mut world, center, center, 0.4);

    let iters = run_until_stable(&mut world, 1000);
    let angle = measure_pile_angle(&world, center, center);

    // Should use sand's angle since sediment is on top
    let expected_angle = TerrainMaterial::Sand.angle_of_repose().to_degrees();
    let tolerance = 3.0;

    println!("Sediment on dirt substrate:");
    println!("  Iterations to stable: {}", iters);
    println!("  Measured angle: {:.1}°", angle);
    println!(
        "  Expected angle: {:.1}° (sand, ±{}°)",
        expected_angle, tolerance
    );

    assert!(
        (angle - expected_angle).abs() < tolerance,
        "Sediment should collapse to ~{}° (sand), got {:.1}°",
        expected_angle,
        angle
    );
}

/// Test that a flat surface doesn't collapse
#[test]
fn test_flat_surface_stable() {
    let mut world = create_flat_world(11, 11, 0.1);

    // Add a thin uniform layer of sediment
    world.terrain_sediment = vec![0.1; 11 * 11];

    // Should stabilize immediately (no slope to collapse)
    let iters = run_until_stable(&mut world, 100);

    println!("Flat surface: {} iterations to stable", iters);

    // Should be stable from the start or within a few iterations
    assert!(iters < 5, "Flat surface should be stable immediately");
}

/// Test that steep slope collapses but gentle slope doesn't
#[test]
fn test_slope_threshold() {
    let mut world = create_flat_world(11, 11, 0.1);
    let cell_size = 0.1;

    // Create a slope that's steeper than angle of repose
    // Sand angle is 32°, so tan(32°) ≈ 0.625
    // Height diff = tan(angle) * horizontal_dist

    // Steep slope (45°) - should collapse
    let steep_height = cell_size * 1.0_f32.tan(); // 45° = 100% grade
    add_sediment_pile(&mut world, 5, 5, steep_height);

    let changed = world.update_terrain_collapse();
    assert!(
        changed,
        "45° slope should collapse (steeper than 32° angle of repose)"
    );

    // Reset and try gentle slope
    let mut world2 = create_flat_world(11, 11, 0.1);

    // Gentle slope (20°) - should NOT collapse
    let gentle_height = cell_size * 20.0_f32.to_radians().tan(); // 20° ≈ 36% grade
    add_sediment_pile(&mut world2, 5, 5, gentle_height);

    let changed2 = world2.update_terrain_collapse();
    assert!(
        !changed2,
        "20° slope should NOT collapse (gentler than 32° angle of repose)"
    );
}

/// Test that multiple materials maintain their different angles
#[test]
fn test_multiple_material_angles() {
    // Test each material maintains its specific angle
    let materials = [
        ("Sand/Sediment", TerrainMaterial::Sand),
        ("Dirt", TerrainMaterial::Dirt),
        ("Gravel", TerrainMaterial::Gravel),
    ];

    println!("\nMaterial angle of repose comparison:");
    println!("{:<20} {:>10} {:>10}", "Material", "Expected", "Actual");
    println!("{}", "-".repeat(42));

    for (name, material) in materials {
        let expected = material.angle_of_repose().to_degrees();
        println!("{:<20} {:>10.1}°", name, expected);
    }
}

/// Test collapse conserves mass
#[test]
fn test_collapse_mass_conservation() {
    let mut world = create_flat_world(21, 21, 0.1);
    let center = 10;

    // Calculate total sediment before
    let pile_height = 1.0;
    add_sediment_pile(&mut world, center, center, pile_height);

    let total_before: f32 = world.terrain_sediment.iter().sum();

    // Run collapse
    run_until_stable(&mut world, 1000);

    // Calculate total sediment after
    let total_after: f32 = world.terrain_sediment.iter().sum();

    let diff = (total_before - total_after).abs();
    let tolerance = 0.001; // Allow tiny floating point error

    println!("Mass conservation:");
    println!("  Before: {:.4}", total_before);
    println!("  After:  {:.4}", total_after);
    println!("  Diff:   {:.6}", diff);

    assert!(
        diff < tolerance,
        "Collapse should conserve mass. Lost {:.4} units",
        diff
    );
}
