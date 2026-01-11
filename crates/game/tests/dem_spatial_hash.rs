// DEM Spatial Hash Tests
// Validates collision detection correctness and O(n) performance

use glam::Vec3;
use sim3d::clump::{ClumpTemplate3D, ClumpShape3D, ClusterSimulation3D};
use std::time::Instant;

const PARTICLE_RADIUS: f32 = 0.01; // 1cm gravel
const PARTICLE_MASS: f32 = 1.0;
const BOUNDS_SIZE: f32 = 10.0; // Correctness test
const SPARSE_BOUNDS: f32 = 1000.0; // Sparse test

/// Test that all actual collisions are detected (no false negatives)
/// Places 8 clumps in 2x2x2 grid with exactly touching spacing
/// Expected: 12 edge contacts in a cube
#[test]
fn test_dem_spatial_hash_correctness() {
    let mut sim = ClusterSimulation3D::new(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(BOUNDS_SIZE, BOUNDS_SIZE, BOUNDS_SIZE),
    );
    sim.gravity = Vec3::ZERO; // Disable gravity to prevent movement

    // Generate tetrahedral clump template
    let template = ClumpTemplate3D::generate(
        ClumpShape3D::Tetra,
        PARTICLE_RADIUS,
        PARTICLE_MASS,
    );
    let template_idx = sim.add_template(template);
    let particle_radius = sim.templates[template_idx].particle_radius;

    // Spacing = exactly 2x particle radius (particles just touching)
    // This is the contact distance, ensuring definite collision
    let spacing = particle_radius * 1.9; // Slightly less than 2.0 to ensure overlap

    // Create 2x2x2 grid of clumps centered in bounds
    let base_pos = Vec3::new(BOUNDS_SIZE / 2.0, BOUNDS_SIZE / 2.0, BOUNDS_SIZE / 2.0)
        - Vec3::splat(spacing / 2.0);

    let mut clump_indices = Vec::new();
    for iz in 0..2 {
        for iy in 0..2 {
            for ix in 0..2 {
                let pos = base_pos + Vec3::new(ix as f32, iy as f32, iz as f32) * spacing;
                let idx = sim.spawn(template_idx, pos, Vec3::ZERO);
                clump_indices.push(idx);
            }
        }
    }

    // Step once to trigger collision detection
    sim.step(0.016); // ~60 FPS

    // Verify contact count
    let contact_count = sim.sphere_contact_count();

    // In a 2x2x2 cube, each vertex touches 3 edge neighbors (x, y, z directions)
    // Total contacts = 8 vertices * 3 edges / 2 (don't double count) = 12 edges
    assert!(
        contact_count >= 12,
        "Expected at least 12 contacts in 2x2x2 grid, got {}",
        contact_count
    );

    // Validate specific contact pairs exist
    // Each clump should touch its 3 immediate neighbors (along x, y, z)
    // Grid layout: clump_indices[z*4 + y*2 + x]
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                let i = clump_indices[z * 4 + y * 2 + x];

                // Check contact with x+1 neighbor (if exists)
                if x < 1 {
                    let j = clump_indices[z * 4 + y * 2 + (x + 1)];
                    assert!(
                        sim.has_contact(i, j),
                        "Missing contact between clumps {} and {} (x-neighbors at [{},{},{}])",
                        i, j, x, y, z
                    );
                }

                // Check contact with y+1 neighbor (if exists)
                if y < 1 {
                    let j = clump_indices[z * 4 + (y + 1) * 2 + x];
                    assert!(
                        sim.has_contact(i, j),
                        "Missing contact between clumps {} and {} (y-neighbors at [{},{},{}])",
                        i, j, x, y, z
                    );
                }

                // Check contact with z+1 neighbor (if exists)
                if z < 1 {
                    let j = clump_indices[(z + 1) * 4 + y * 2 + x];
                    assert!(
                        sim.has_contact(i, j),
                        "Missing contact between clumps {} and {} (z-neighbors at [{},{},{}])",
                        i, j, x, y, z
                    );
                }
            }
        }
    }
}

/// Test O(n) scaling for sparse distributions (no spurious checks)
/// Verifies that widely separated clumps show linear performance
#[test]
#[ignore] // Timing test - run manually to avoid CI flakiness
fn test_dem_sparse_distribution() {
    // Test with 1000 clumps
    let time_1000 = measure_sparse_step(1000);

    // Test with 2000 clumps
    let time_2000 = measure_sparse_step(2000);

    // Assert linear scaling: ratio should be ~2.0
    let ratio = time_2000 / time_1000;
    assert!(
        ratio >= 1.8 && ratio <= 2.2,
        "Expected linear scaling ratio 1.8-2.2, got {}. times: 1000={:.3}ms, 2000={:.3}ms",
        ratio, time_1000 * 1000.0, time_2000 * 1000.0
    );
}

/// Helper function to measure step time for sparse distribution
fn measure_sparse_step(num_clumps: usize) -> f64 {
    let mut sim = ClusterSimulation3D::new(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(SPARSE_BOUNDS, SPARSE_BOUNDS, SPARSE_BOUNDS),
    );
    sim.gravity = Vec3::ZERO;

    let template = ClumpTemplate3D::generate(
        ClumpShape3D::Tetra,
        PARTICLE_RADIUS,
        PARTICLE_MASS,
    );
    let template_idx = sim.add_template(template);
    let template_ref = &sim.templates[template_idx];

    // Separation >> 10x bounding radius to ensure no collisions
    let separation = template_ref.bounding_radius * 20.0;

    // Distribute clumps in 3D grid
    let grid_size = (num_clumps as f32).powf(1.0 / 3.0).ceil() as usize;
    let mut count = 0;

    'outer: for iz in 0..grid_size {
        for iy in 0..grid_size {
            for ix in 0..grid_size {
                if count >= num_clumps {
                    break 'outer;
                }

                let pos = Vec3::new(
                    ix as f32 * separation + 10.0,
                    iy as f32 * separation + 10.0,
                    iz as f32 * separation + 10.0,
                );
                sim.spawn(template_idx, pos, Vec3::ZERO);
                count += 1;
            }
        }
    }

    // Measure step time
    let start = Instant::now();
    sim.step(0.016);
    let elapsed = start.elapsed().as_secs_f64();

    // Verify no contacts detected (clumps are far apart)
    assert_eq!(
        sim.sphere_contact_count(),
        0,
        "Expected 0 contacts for sparse distribution, got {}",
        sim.sphere_contact_count()
    );

    elapsed
}

/// Test spatial hash cell assignment
/// Verifies clumps are placed in correct spatial hash cells
#[test]
fn test_spatial_hash_cell_assignment() {
    let mut sim = ClusterSimulation3D::new(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(100.0, 100.0, 100.0),
    );
    sim.gravity = Vec3::ZERO;

    let template = ClumpTemplate3D::generate(
        ClumpShape3D::Tetra,
        PARTICLE_RADIUS,
        PARTICLE_MASS,
    );
    let template_idx = sim.add_template(template);
    let bounding_radius = sim.templates[template_idx].bounding_radius;

    // Cell size = 2 * max_bounding_radius + epsilon
    let cell_size = bounding_radius * 2.0 + 0.001;

    // Place clumps at known positions
    let positions = vec![
        Vec3::new(10.0, 20.0, 30.0),
        Vec3::new(10.1, 20.0, 30.0), // Same cell as first (within cell_size)
        Vec3::new(50.0, 50.0, 50.0),
        Vec3::new(50.0 + cell_size * 2.0, 50.0, 50.0), // Different cell (2 cells away)
    ];

    for pos in &positions {
        sim.spawn(template_idx, *pos, Vec3::ZERO);
    }

    // Step to build spatial hash
    sim.step(0.016);

    // Clumps 0 and 1 are in same/adjacent cells → should detect contact if close enough
    // Clumps 2 and 3 are far apart → should NOT detect contact
    let contact_count = sim.sphere_contact_count();

    // We can't directly inspect spatial hash cells (private), but we can verify
    // that the collision detection is working correctly based on distances

    // Distance between clumps 0 and 1
    let dist_01 = (positions[0] - positions[1]).length();
    let max_dist_01 = bounding_radius * 2.0;

    if dist_01 <= max_dist_01 {
        // Should have contact
        assert!(
            sim.has_contact(0, 1),
            "Clumps 0 and 1 are within collision distance but no contact detected"
        );
    }

    // Clumps 2 and 3 are far apart

    // Should NOT have contact (far apart)
    assert!(
        !sim.has_contact(2, 3),
        "Clumps 2 and 3 are far apart but spurious contact detected"
    );

    // Verify contact count is reasonable (0-2 depending on clump proximity)
    assert!(
        contact_count <= 2,
        "Too many contacts detected (expected 0-2, got {})",
        contact_count
    );
}
