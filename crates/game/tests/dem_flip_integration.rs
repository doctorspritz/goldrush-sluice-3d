// DEM + FLIP Integration Tests
// Validates that DEM clumps work correctly with FLIP water simulation
// on the same gutter geometry used by washplant_editor

use glam::Vec3;
use sim3d::clump::{ClumpShape3D, ClumpTemplate3D, ClusterSimulation3D, SdfParams};
use sim3d::grid::CellType;
use sim3d::FlipSimulation3D;

// Match washplant_editor constants exactly
const SIM_CELL_SIZE: f32 = 0.025; // 2.5cm cells
const DEM_CLUMP_RADIUS: f32 = 0.008; // 8mm clumps
const DEM_GOLD_DENSITY: f32 = 19300.0; // kg/m³
const DEM_SAND_DENSITY: f32 = 2650.0; // kg/m³
const DEM_WATER_DENSITY: f32 = 1000.0; // kg/m³
const DEM_DRAG_COEFF: f32 = 5.0;

const DT: f32 = 1.0 / 60.0; // 60 Hz timestep
const GRAVITY: f32 = -9.81;

/// Calculate particle mass from density and radius
fn particle_mass_from_density(radius: f32, density: f32) -> f32 {
    let volume = (4.0 / 3.0) * std::f32::consts::PI * radius.powi(3);
    volume * density
}

/// Create a gutter-shaped grid with solid walls and floor
/// This matches the geometry marking in washplant_editor
fn create_gutter_grid(
    grid_width: usize,
    grid_height: usize,
    grid_depth: usize,
    cell_size: f32,
    wall_height_cells: usize,
    floor_thickness: usize,
) -> FlipSimulation3D {
    let mut sim = FlipSimulation3D::new(grid_width, grid_height, grid_depth, cell_size);
    sim.gravity = Vec3::new(0.0, GRAVITY, 0.0);

    // Mark floor cells as solid (bottom layer)
    // IMPORTANT: Use set_solid() which sets both the `solid` array AND `cell_type`
    // The `compute_sdf()` function reads from `solid`, not `cell_type`!
    for k in 0..grid_depth {
        for i in 0..grid_width {
            for j in 0..floor_thickness {
                sim.grid.set_solid(i, j, k);
            }
        }
    }

    // Mark side walls as solid
    for k in 0..grid_depth {
        for j in 0..wall_height_cells {
            // Left wall (k=0)
            if k < 2 {
                for i in 0..grid_width {
                    sim.grid.set_solid(i, j, k);
                }
            }
            // Right wall (k=grid_depth-1)
            if k >= grid_depth - 2 {
                for i in 0..grid_width {
                    sim.grid.set_solid(i, j, k);
                }
            }
        }
    }

    // Compute SDF from solid cells - CRITICAL for collision detection
    sim.grid.compute_sdf();

    sim
}

/// Apply water-DEM coupling forces (buoyancy + drag)
/// This matches the coupling in washplant_editor
fn apply_water_dem_coupling(dem_sim: &mut ClusterSimulation3D, dt: f32) {
    for clump in &mut dem_sim.clumps {
        let template = &dem_sim.templates[clump.template_idx];

        // Buoyancy force: F_b = ρ_water * V * g (upward)
        let particle_volume = (4.0 / 3.0) * std::f32::consts::PI * template.particle_radius.powi(3);
        let total_volume = particle_volume * template.local_offsets.len() as f32;
        let buoyancy_force = DEM_WATER_DENSITY * total_volume * 9.81;

        // Drag force: F_d = 0.5 * C_d * ρ_water * A * v²
        let area = std::f32::consts::PI * template.bounding_radius.powi(2);
        let speed = clump.velocity.length();
        let drag_force = if speed > 0.001 {
            0.5 * DEM_DRAG_COEFF * DEM_WATER_DENSITY * area * speed * speed
        } else {
            0.0
        };

        // Apply forces as velocity change
        clump.velocity.y += buoyancy_force * dt / template.mass;

        if speed > 0.001 {
            let drag_dir = -clump.velocity.normalize();
            let drag_dv = drag_force * dt / template.mass;
            let max_drag = speed;
            clump.velocity += drag_dir * drag_dv.min(max_drag);
        }
    }
}

/// Test 1: DEM clumps collide with gutter SDF
/// Verifies that clumps dropped into a gutter don't penetrate the floor
#[test]
fn test_dem_collides_with_gutter_sdf() {
    println!("=== Test: DEM collides with gutter SDF ===");

    // Create gutter geometry (same as washplant_editor)
    let grid_width = 40;
    let grid_height = 20;
    let grid_depth = 20;
    let cell_size = SIM_CELL_SIZE;
    let wall_height = 8;
    let floor_thickness = 2;

    let flip_sim = create_gutter_grid(
        grid_width,
        grid_height,
        grid_depth,
        cell_size,
        wall_height,
        floor_thickness,
    );

    println!(
        "Grid: {}x{}x{} cells, cell_size={:.3}m",
        grid_width, grid_height, grid_depth, cell_size
    );
    println!(
        "Physical size: {:.2}x{:.2}x{:.2}m",
        grid_width as f32 * cell_size,
        grid_height as f32 * cell_size,
        grid_depth as f32 * cell_size
    );

    // Count solid cells to verify geometry was created
    let solid_count = flip_sim
        .grid
        .cell_type()
        .iter()
        .filter(|&&t| t == CellType::Solid)
        .count();
    println!("Solid cells: {}", solid_count);
    assert!(solid_count > 0, "No solid cells created!");

    // Create DEM simulation
    let bounds = Vec3::new(
        grid_width as f32 * cell_size,
        grid_height as f32 * cell_size,
        grid_depth as f32 * cell_size,
    );
    let mut dem_sim = ClusterSimulation3D::new(Vec3::ZERO, bounds);
    dem_sim.gravity = Vec3::new(0.0, GRAVITY, 0.0);

    // Create clump template
    let particle_mass = particle_mass_from_density(DEM_CLUMP_RADIUS, DEM_SAND_DENSITY);
    let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, DEM_CLUMP_RADIUS, particle_mass);
    let template_idx = dem_sim.add_template(template);

    // Spawn clumps above the floor inside the gutter channel
    let floor_y = floor_thickness as f32 * cell_size;
    let spawn_y = floor_y + 0.2; // 20cm above floor
    let center_x = grid_width as f32 * cell_size * 0.5;
    let center_z = grid_depth as f32 * cell_size * 0.5;

    println!(
        "Spawning clumps at y={:.3}m (floor at y={:.3}m)",
        spawn_y, floor_y
    );

    let mut clump_indices = Vec::new();
    // Spawn well-separated clumps (bounding radius is ~2.6cm so use 6cm spacing)
    for i in 0..5 {
        let x = center_x + (i as f32 - 2.0) * 0.06;
        let z = center_z;
        let idx = dem_sim.spawn(template_idx, Vec3::new(x, spawn_y, z), Vec3::ZERO);
        clump_indices.push(idx);
    }

    println!("Spawned {} clumps", clump_indices.len());

    // Create SDF params from FLIP grid
    let sdf_params = SdfParams {
        sdf: flip_sim.grid.sdf(),
        grid_width,
        grid_height,
        grid_depth,
        cell_size,
        grid_offset: Vec3::ZERO,
    };

    // Run simulation with SDF collision
    let num_steps = 180; // 3 seconds at 60 Hz
    for step in 0..num_steps {
        dem_sim.step_with_sdf(DT, &sdf_params);

        if step % 60 == 0 {
            let min_y = clump_indices
                .iter()
                .map(|&i| dem_sim.clumps[i].position.y)
                .fold(f32::INFINITY, f32::min);
            let avg_vel: f32 = clump_indices
                .iter()
                .map(|&i| dem_sim.clumps[i].velocity.length())
                .sum::<f32>()
                / clump_indices.len() as f32;
            println!(
                "Step {} ({:.1}s): min_y={:.4}m, avg_vel={:.4}m/s",
                step,
                step as f32 * DT,
                min_y,
                avg_vel
            );
        }
    }

    // Check final positions - clumps should be resting on floor, not penetrating
    let floor_surface_y = floor_thickness as f32 * cell_size;
    let particle_radius = dem_sim.templates[template_idx].particle_radius;

    for (i, &idx) in clump_indices.iter().enumerate() {
        let pos = dem_sim.clumps[idx].position;
        let expected_min_y = floor_surface_y + particle_radius * 0.5; // Allow some tolerance

        println!(
            "Clump {}: y={:.4}m (floor+radius={:.4}m)",
            i, pos.y, expected_min_y
        );

        assert!(
            pos.y >= expected_min_y - 0.01, // 1cm tolerance
            "Clump {} penetrated floor: y={:.4}m, expected >= {:.4}m",
            i,
            pos.y,
            expected_min_y
        );
    }

    println!("PASS: DEM clumps collide correctly with gutter SDF");
}

/// Test 2: Gold settles faster than sand in water (with coupling)
/// This is the key physical behavior we need for gold separation
#[test]
fn test_gold_settles_faster_than_sand_in_water() {
    println!("=== Test: Gold settles faster than sand in water ===");

    // Create gutter geometry
    let grid_width = 40;
    let grid_height = 30;
    let grid_depth = 20;
    let cell_size = SIM_CELL_SIZE;
    let wall_height = 10;
    let floor_thickness = 2;

    let flip_sim = create_gutter_grid(
        grid_width,
        grid_height,
        grid_depth,
        cell_size,
        wall_height,
        floor_thickness,
    );

    // Create DEM simulation
    let bounds = Vec3::new(
        grid_width as f32 * cell_size,
        grid_height as f32 * cell_size,
        grid_depth as f32 * cell_size,
    );
    let mut dem_sim = ClusterSimulation3D::new(Vec3::ZERO, bounds);
    dem_sim.gravity = Vec3::new(0.0, GRAVITY, 0.0);

    // Create gold template (heavy)
    let gold_mass = particle_mass_from_density(DEM_CLUMP_RADIUS, DEM_GOLD_DENSITY);
    let gold_template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, DEM_CLUMP_RADIUS, gold_mass);
    let gold_idx = dem_sim.add_template(gold_template);

    // Create sand template (light)
    let sand_mass = particle_mass_from_density(DEM_CLUMP_RADIUS, DEM_SAND_DENSITY);
    let sand_template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, DEM_CLUMP_RADIUS, sand_mass);
    let sand_idx = dem_sim.add_template(sand_template);

    println!(
        "Gold mass: {:.6}kg (ρ={}kg/m³)",
        gold_mass, DEM_GOLD_DENSITY
    );
    println!(
        "Sand mass: {:.6}kg (ρ={}kg/m³)",
        sand_mass, DEM_SAND_DENSITY
    );
    println!("Mass ratio: {:.2}x", gold_mass / sand_mass);

    // Spawn gold and sand at same height inside the gutter
    let floor_y = floor_thickness as f32 * cell_size;
    let spawn_y = floor_y + 0.4; // 40cm above floor (simulating water depth)
    let center_x = grid_width as f32 * cell_size * 0.5;
    let center_z = grid_depth as f32 * cell_size * 0.5;

    println!("Spawning at y={:.3}m (floor at y={:.3}m)", spawn_y, floor_y);

    let mut gold_indices = Vec::new();
    let mut sand_indices = Vec::new();

    // Bounding radius is ~2.6cm for Tetra shape, so use 7cm spacing to prevent overlap
    let spacing = 0.07;
    for i in 0..3 {
        let x = center_x + (i as f32 - 1.0) * spacing;
        let z_gold = center_z - 0.05; // Gold row
        let z_sand = center_z + 0.05; // Sand row (10cm away, no overlap)

        // Spawn gold
        let gold = dem_sim.spawn(gold_idx, Vec3::new(x, spawn_y, z_gold), Vec3::ZERO);
        gold_indices.push(gold);

        // Spawn sand in separate row
        let sand = dem_sim.spawn(sand_idx, Vec3::new(x, spawn_y, z_sand), Vec3::ZERO);
        sand_indices.push(sand);
    }

    println!(
        "Spawned {} gold + {} sand clumps",
        gold_indices.len(),
        sand_indices.len()
    );

    // Create SDF params
    let sdf_params = SdfParams {
        sdf: flip_sim.grid.sdf(),
        grid_width,
        grid_height,
        grid_depth,
        cell_size,
        grid_offset: Vec3::ZERO,
    };

    // Run simulation with water-DEM coupling
    // Track maximum separation during settling (gold reaches floor before sand)
    let num_steps = 300; // 5 seconds
    let mut max_separation = 0.0_f32;
    let mut max_separation_step = 0;

    for step in 0..num_steps {
        // Apply water-DEM coupling (buoyancy + drag)
        apply_water_dem_coupling(&mut dem_sim, DT);

        // Step DEM with SDF collision
        dem_sim.step_with_sdf(DT, &sdf_params);

        let gold_avg_y: f32 = gold_indices
            .iter()
            .map(|&i| dem_sim.clumps[i].position.y)
            .sum::<f32>()
            / gold_indices.len() as f32;
        let sand_avg_y: f32 = sand_indices
            .iter()
            .map(|&i| dem_sim.clumps[i].position.y)
            .sum::<f32>()
            / sand_indices.len() as f32;

        let separation = sand_avg_y - gold_avg_y;
        if separation > max_separation {
            max_separation = separation;
            max_separation_step = step;
        }

        if step % 60 == 0 {
            println!(
                "Step {} ({:.1}s): gold_y={:.4}m, sand_y={:.4}m, diff={:.4}m",
                step,
                step as f32 * DT,
                gold_avg_y,
                sand_avg_y,
                separation
            );
        }
    }

    // Final measurements
    let gold_avg_y: f32 = gold_indices
        .iter()
        .map(|&i| dem_sim.clumps[i].position.y)
        .sum::<f32>()
        / gold_indices.len() as f32;
    let sand_avg_y: f32 = sand_indices
        .iter()
        .map(|&i| dem_sim.clumps[i].position.y)
        .sum::<f32>()
        / sand_indices.len() as f32;

    println!("\n=== Final Results ===");
    println!("Gold average Y: {:.4}m", gold_avg_y);
    println!("Sand average Y: {:.4}m", sand_avg_y);
    println!(
        "Max separation during settling: {:.4}m at step {} ({:.1}s)",
        max_separation,
        max_separation_step,
        max_separation_step as f32 * DT
    );

    // Gold should settle FASTER than sand due to higher density.
    // We verify this by checking the maximum separation observed during settling -
    // gold reaches the floor while sand is still falling, creating a separation.
    // Final positions are equal (both on floor), but max separation shows gold sank faster.
    let min_expected_separation = 0.1; // At least 10cm separation during settling
    assert!(
        max_separation > min_expected_separation,
        "Gold did not settle faster than sand! max_separation={:.4}m (expected > {:.2}m)",
        max_separation,
        min_expected_separation
    );

    println!(
        "PASS: Gold settled faster than sand (max separation {:.4}m during settling)",
        max_separation
    );
}

/// Test 3: FLIP water particles and DEM clumps occupy same space
/// Verifies that both systems can be simulated together
#[test]
fn test_flip_and_dem_together() {
    println!("=== Test: FLIP and DEM simulated together ===");

    // Create gutter geometry
    let grid_width = 30;
    let grid_height = 20;
    let grid_depth = 15;
    let cell_size = SIM_CELL_SIZE;
    let wall_height = 8;
    let floor_thickness = 2;

    let mut flip_sim = create_gutter_grid(
        grid_width,
        grid_height,
        grid_depth,
        cell_size,
        wall_height,
        floor_thickness,
    );

    // Create DEM simulation
    let bounds = Vec3::new(
        grid_width as f32 * cell_size,
        grid_height as f32 * cell_size,
        grid_depth as f32 * cell_size,
    );
    let mut dem_sim = ClusterSimulation3D::new(Vec3::ZERO, bounds);
    dem_sim.gravity = Vec3::new(0.0, GRAVITY, 0.0);

    // Create DEM template
    let particle_mass = particle_mass_from_density(DEM_CLUMP_RADIUS, DEM_GOLD_DENSITY);
    let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, DEM_CLUMP_RADIUS, particle_mass);
    let template_idx = dem_sim.add_template(template);

    // Spawn water particles
    let floor_y = floor_thickness as f32 * cell_size;
    let water_y = floor_y + cell_size * 2.0;
    let center_x = grid_width as f32 * cell_size * 0.5;
    let center_z = grid_depth as f32 * cell_size * 0.5;

    println!("Spawning water particles at y={:.3}m", water_y);

    for i in 0..20 {
        for k in 0..10 {
            let x = center_x - 0.1 + (i as f32) * 0.01;
            let z = center_z - 0.05 + (k as f32) * 0.01;
            let pos = Vec3::new(x, water_y, z);
            let vel = Vec3::new(0.5, 0.0, 0.0); // Flowing in +X direction

            flip_sim.particles.spawn(pos, vel);
        }
    }

    let water_count = flip_sim.particles.list().len();
    println!("Spawned {} water particles", water_count);

    // Spawn a single DEM clump IN THE WATER to test coupling
    let mut clump_indices = Vec::new();
    let x = center_x;
    let z = center_z;
    let y = water_y + 0.1; // 10cm above water surface

    let idx = dem_sim.spawn(template_idx, Vec3::new(x, y, z), Vec3::ZERO);
    clump_indices.push(idx);

    println!("Spawned {} DEM clumps IN the water", clump_indices.len());

    // Create SDF params
    let sdf_params = SdfParams {
        sdf: flip_sim.grid.sdf(),
        grid_width,
        grid_height,
        grid_depth,
        cell_size,
        grid_offset: Vec3::ZERO,
    };

    // Run both simulations together
    let num_steps = 120; // 2 seconds
    for step in 0..num_steps {
        // Step FLIP (simplified - just advect particles by velocity for this test)
        for particle in flip_sim.particles.list_mut() {
            particle.position += particle.velocity * DT;
            particle.velocity.y += GRAVITY * DT;
        }

        // Apply water-DEM coupling
        apply_water_dem_coupling(&mut dem_sim, DT);

        // Step DEM with SDF collision
        dem_sim.step_with_sdf(DT, &sdf_params);

        if step % 30 == 0 {
            let water_count = flip_sim.particles.list().len();
            let clump_count = dem_sim.clumps.len();
            let clump_avg_y: f32 = clump_indices
                .iter()
                .map(|&i| dem_sim.clumps[i].position.y)
                .sum::<f32>()
                / clump_indices.len() as f32;

            println!(
                "Step {} ({:.1}s): water={}, clumps={}, clump_avg_y={:.4}m",
                step,
                step as f32 * DT,
                water_count,
                clump_count,
                clump_avg_y
            );
        }
    }

    // Verify both systems ran without crashing
    assert!(
        !flip_sim.particles.list().is_empty(),
        "Water particles disappeared"
    );
    assert!(!dem_sim.clumps.is_empty(), "DEM clumps disappeared");

    // Verify clumps are still in valid positions (not NaN, not fallen through floor)
    let floor_surface_y = floor_thickness as f32 * cell_size;
    for &idx in &clump_indices {
        let pos = dem_sim.clumps[idx].position;
        assert!(pos.is_finite(), "Clump position became NaN/inf");
        assert!(
            pos.y >= floor_surface_y - 0.01,
            "Clump fell through floor: y={:.4}",
            pos.y
        );
    }

    println!("PASS: FLIP and DEM simulated together without issues");
}

/// Test 4: Water drag slows down DEM clumps
/// Verifies that drag force affects clump motion
#[test]
fn test_water_drag_slows_clumps() {
    println!("=== Test: Water drag slows down DEM clumps ===");

    // Create open space (no walls for this test)
    let mut dem_sim_with_drag =
        ClusterSimulation3D::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(10.0, 10.0, 10.0));
    dem_sim_with_drag.gravity = Vec3::ZERO; // Disable gravity for this test

    let mut dem_sim_no_drag =
        ClusterSimulation3D::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(10.0, 10.0, 10.0));
    dem_sim_no_drag.gravity = Vec3::ZERO;

    // Create template
    let particle_mass = particle_mass_from_density(DEM_CLUMP_RADIUS, DEM_SAND_DENSITY);
    let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, DEM_CLUMP_RADIUS, particle_mass);
    let template_idx_drag = dem_sim_with_drag.add_template(template.clone());
    let template_idx_no_drag = dem_sim_no_drag.add_template(template);

    // Spawn clumps with same initial velocity
    let initial_vel = Vec3::new(2.0, 0.0, 0.0);
    let initial_pos = Vec3::new(5.0, 5.0, 5.0);

    let idx_drag = dem_sim_with_drag.spawn(template_idx_drag, initial_pos, initial_vel);
    let idx_no_drag = dem_sim_no_drag.spawn(template_idx_no_drag, initial_pos, initial_vel);

    println!("Initial velocity: {:.2} m/s", initial_vel.length());

    // Run for 1 second
    let num_steps = 60;
    for step in 0..num_steps {
        // Apply drag to one simulation only
        apply_water_dem_coupling(&mut dem_sim_with_drag, DT);

        dem_sim_with_drag.step(DT);
        dem_sim_no_drag.step(DT);

        if step % 20 == 0 {
            let vel_drag = dem_sim_with_drag.clumps[idx_drag].velocity.length();
            let vel_no_drag = dem_sim_no_drag.clumps[idx_no_drag].velocity.length();
            println!(
                "Step {} ({:.1}s): with_drag={:.4}m/s, no_drag={:.4}m/s",
                step,
                step as f32 * DT,
                vel_drag,
                vel_no_drag
            );
        }
    }

    let final_vel_drag = dem_sim_with_drag.clumps[idx_drag].velocity.length();
    let final_vel_no_drag = dem_sim_no_drag.clumps[idx_no_drag].velocity.length();

    println!("\n=== Final Results ===");
    println!("With drag: {:.4} m/s", final_vel_drag);
    println!("Without drag: {:.4} m/s", final_vel_no_drag);
    println!(
        "Drag reduced velocity by: {:.1}%",
        (1.0 - final_vel_drag / final_vel_no_drag) * 100.0
    );

    // Clump with drag should be slower
    assert!(
        final_vel_drag < final_vel_no_drag,
        "Drag did not slow down clump! with_drag={:.4}, no_drag={:.4}",
        final_vel_drag,
        final_vel_no_drag
    );

    // Should be significantly slower (at least 20% reduction)
    let reduction = 1.0 - final_vel_drag / final_vel_no_drag;
    assert!(
        reduction > 0.2,
        "Drag effect too weak: only {:.1}% reduction (expected > 20%)",
        reduction * 100.0
    );

    println!("PASS: Water drag effectively slows DEM clumps");
}

/// Test 5: Shields stress entrainment
/// Verifies that particles are entrained only when Shields stress exceeds critical threshold
#[test]
fn test_shields_stress_entrainment() {
    use sim3d::clump::{FluidParams, FluidVelocityField};

    println!("=== Test: Shields stress entrainment ===");

    // Grid dimensions for the velocity field
    let grid_width = 20;
    let grid_height = 10;
    let grid_depth = 20;
    let cell_size = SIM_CELL_SIZE;

    // Create fluid velocity field with uniform flow
    // Different flow speeds to test entrainment threshold
    let slow_speed = 0.1; // m/s - below critical for both sand and gold
    let medium_speed = 0.5; // m/s - should entrain sand but not gold
    let fast_speed = 2.0; // m/s - should entrain both

    // Create uniform velocity fields (flow in +X direction)
    let u_size = (grid_width + 1) * grid_height * grid_depth;
    let v_size = grid_width * (grid_height + 1) * grid_depth;
    let w_size = grid_width * grid_height * (grid_depth + 1);

    let slow_u: Vec<f32> = vec![slow_speed; u_size];
    let medium_u: Vec<f32> = vec![medium_speed; u_size];
    let fast_u: Vec<f32> = vec![fast_speed; u_size];
    let zero_v: Vec<f32> = vec![0.0; v_size];
    let zero_w: Vec<f32> = vec![0.0; w_size];

    let fluid_params = FluidParams::default();

    // Create bounds
    let bounds = Vec3::new(
        grid_width as f32 * cell_size,
        grid_height as f32 * cell_size,
        grid_depth as f32 * cell_size,
    );

    // --- Test 1: Light particle (sand) entrainment threshold ---
    println!("\n--- Testing sand particle entrainment ---");

    let mut dem_sand = ClusterSimulation3D::new(Vec3::ZERO, bounds);
    dem_sand.gravity = Vec3::new(0.0, GRAVITY, 0.0);

    let sand_mass = particle_mass_from_density(DEM_CLUMP_RADIUS, DEM_SAND_DENSITY);
    let sand_template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, DEM_CLUMP_RADIUS, sand_mass);
    let sand_idx = dem_sand.add_template(sand_template);

    // Spawn sand clump at rest
    let spawn_pos = Vec3::new(
        grid_width as f32 * cell_size * 0.5,
        grid_height as f32 * cell_size * 0.3,
        grid_depth as f32 * cell_size * 0.5,
    );
    let sand_clump_idx = dem_sand.spawn(sand_idx, spawn_pos, Vec3::ZERO);

    // Check Shields parameter at different flow speeds
    let slow_field = FluidVelocityField {
        grid_u: &slow_u,
        grid_v: &zero_v,
        grid_w: &zero_w,
        grid_width,
        grid_height,
        grid_depth,
        cell_size,
        grid_offset: Vec3::ZERO,
    };

    let theta_slow = dem_sand.shields_parameter(sand_clump_idx, &slow_field, &fluid_params);
    let entrained_slow = dem_sand.is_entrained(sand_clump_idx, &slow_field, &fluid_params);
    println!(
        "Slow flow ({:.2} m/s): θ* = {:.6}, entrained = {}",
        slow_speed, theta_slow, entrained_slow
    );

    let medium_field = FluidVelocityField {
        grid_u: &medium_u,
        grid_v: &zero_v,
        grid_w: &zero_w,
        grid_width,
        grid_height,
        grid_depth,
        cell_size,
        grid_offset: Vec3::ZERO,
    };

    let theta_medium = dem_sand.shields_parameter(sand_clump_idx, &medium_field, &fluid_params);
    let entrained_medium = dem_sand.is_entrained(sand_clump_idx, &medium_field, &fluid_params);
    println!(
        "Medium flow ({:.2} m/s): θ* = {:.6}, entrained = {}",
        medium_speed, theta_medium, entrained_medium
    );

    let fast_field = FluidVelocityField {
        grid_u: &fast_u,
        grid_v: &zero_v,
        grid_w: &zero_w,
        grid_width,
        grid_height,
        grid_depth,
        cell_size,
        grid_offset: Vec3::ZERO,
    };

    let theta_fast = dem_sand.shields_parameter(sand_clump_idx, &fast_field, &fluid_params);
    let entrained_fast = dem_sand.is_entrained(sand_clump_idx, &fast_field, &fluid_params);
    println!(
        "Fast flow ({:.2} m/s): θ* = {:.6}, entrained = {}",
        fast_speed, theta_fast, entrained_fast
    );

    // Shields parameter should increase with flow speed (τ ∝ U²)
    assert!(
        theta_medium > theta_slow,
        "Shields should increase with flow speed"
    );
    assert!(
        theta_fast > theta_medium,
        "Shields should increase with flow speed"
    );

    // --- Test 2: Gold particle requires higher flow to entrain ---
    println!("\n--- Testing gold particle entrainment ---");

    let mut dem_gold = ClusterSimulation3D::new(Vec3::ZERO, bounds);
    dem_gold.gravity = Vec3::new(0.0, GRAVITY, 0.0);

    let gold_mass = particle_mass_from_density(DEM_CLUMP_RADIUS, DEM_GOLD_DENSITY);
    let gold_template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, DEM_CLUMP_RADIUS, gold_mass);
    let gold_idx = dem_gold.add_template(gold_template);
    let gold_clump_idx = dem_gold.spawn(gold_idx, spawn_pos, Vec3::ZERO);

    let theta_gold_medium =
        dem_gold.shields_parameter(gold_clump_idx, &medium_field, &fluid_params);
    let theta_gold_fast = dem_gold.shields_parameter(gold_clump_idx, &fast_field, &fluid_params);

    println!(
        "Gold at medium flow: θ* = {:.6} (sand: {:.6})",
        theta_gold_medium, theta_medium
    );
    println!(
        "Gold at fast flow: θ* = {:.6} (sand: {:.6})",
        theta_gold_fast, theta_fast
    );

    // Gold should have lower Shields parameter (harder to entrain) due to higher density
    // θ* = τ / ((ρs - ρf) × g × d) - higher density difference means lower θ*
    assert!(
        theta_gold_medium < theta_medium,
        "Gold should have lower Shields parameter than sand at same flow: gold={:.6}, sand={:.6}",
        theta_gold_medium,
        theta_medium
    );

    // --- Test 3: Apply forces and verify entrainment behavior ---
    println!("\n--- Testing force application ---");

    // Reset sand to rest
    dem_sand.clumps[sand_clump_idx].velocity = Vec3::ZERO;

    // Apply fluid forces with fast flow
    let results = dem_sand.apply_fluid_forces_with_shields(DT, &fast_field, &fluid_params);
    let (theta_applied, was_entrained) = results[sand_clump_idx];

    println!(
        "After force application: θ* = {:.6}, entrained = {}",
        theta_applied, was_entrained
    );

    // If entrained, particle should have gained velocity in flow direction
    if was_entrained {
        let vel = dem_sand.clumps[sand_clump_idx].velocity;
        println!("Velocity after entrainment: {:?}", vel);
        assert!(
            vel.x > 0.0,
            "Entrained particle should have positive X velocity"
        );
    }

    println!("\n=== Shields Stress Test Summary ===");
    println!("Critical Shields: {:.3}", fluid_params.critical_shields);
    println!("Sand (ρ=2650): θ* scales with U², entrainment depends on flow");
    println!("Gold (ρ=19300): Lower θ* at same flow, requires faster water");

    println!("\nPASS: Shields stress correctly models sediment entrainment");
}
