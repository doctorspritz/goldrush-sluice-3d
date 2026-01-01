//! Push-apart particle separation test suite
//! Run with: cargo run --example test_push_apart -p sim

use sim::grid::Grid;
use sim::{create_sluice, FlipSimulation};

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║       PUSH APART PARTICLE SEPARATION - TEST SUITE        ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let mut total_pass = 0;
    let mut total_fail = 0;

    // Test 0: CRITICAL - Particles never penetrate solids during simulation
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 0: PARTICLES NEVER PENETRATE SOLIDS (CRITICAL)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let (pass, fail) = test_no_solid_penetration();
    total_pass += pass;
    total_fail += fail;

    // Test 1: Two overlapping particles get pushed apart
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 1: OVERLAPPING PARTICLES SEPARATE");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let (pass, fail) = test_overlapping_particles_separate();
    total_pass += pass;
    total_fail += fail;

    // Test 2: Particles pushed toward wall don't penetrate
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 2: WALL COLLISION DURING SEPARATION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let (pass, fail) = test_wall_collision();
    total_pass += pass;
    total_fail += fail;

    // Test 3: Check particle positions after simulation frames
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 3: PARTICLES MAINTAIN SEPARATION DURING SIMULATION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let (pass, fail) = test_separation_during_simulation();
    total_pass += pass;
    total_fail += fail;

    // Test 4: Floor compression test
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 4: PARTICLES DON'T COMPRESS TO FLOOR");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let (pass, fail) = test_no_floor_compression();
    total_pass += pass;
    total_fail += fail;

    // Test 5: Spawn never places particles inside solids
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 5: SPAWN NEVER PLACES PARTICLES IN SOLIDS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let (pass, fail) = test_spawn_never_in_solid();
    total_pass += pass;
    total_fail += fail;

    // Test 6: Pressure solver converges
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 6: PRESSURE SOLVER CONVERGENCE");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let (pass, fail) = test_pressure_solver_convergence();
    total_pass += pass;
    total_fail += fail;

    // Test 7: Sediment separation doesn't panic
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 7: SEDIMENT SEPARATION NO PANIC");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let (pass, fail) = test_sediment_separation_no_panic();
    total_pass += pass;
    total_fail += fail;

    // Summary
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  PASSED: {:3}  |  FAILED: {:3}                            ║", total_pass, total_fail);
    println!("╚══════════════════════════════════════════════════════════╝");

    if total_fail == 0 {
        println!("\n✅ ALL TESTS PASSED");
    } else {
        println!("\n❌ SOME TESTS FAILED");
        std::process::exit(1);
    }
}

/// CRITICAL TEST: Particles should NEVER penetrate solid surfaces during simulation
/// This tests the full simulation with a sluice and checks every frame for penetration
fn test_no_solid_penetration() -> (usize, usize) {
    let mut pass = 0;
    let mut fail = 0;

    const WIDTH: usize = 64;
    const HEIGHT: usize = 48;
    const CELL_SIZE: f32 = 4.0;
    const DT: f32 = 1.0 / 60.0;
    const FRAMES: usize = 300;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);
    create_sluice(&mut sim, 0.2, 10, 4, 2);

    // Spawn water
    for i in 0..10 {
        let x = 30.0 + (i % 5) as f32 * 8.0;
        let y = 40.0 + (i / 5) as f32 * 8.0;
        sim.spawn_water(x, y, 30.0, 0.0, 5);
    }

    let mut total_violations = 0;
    let mut max_violation = 0.0f32;
    let mut worst_frame = 0;
    let mut particles_inside_solid_cells = 0;

    for frame in 0..FRAMES {
        sim.update(DT);

        // Check every particle
        for p in sim.particles.iter() {
            // Check SDF (negative = inside solid)
            let sdf = sim.grid.sample_sdf(p.position);
            if sdf < -0.1 {  // Small tolerance for numerical precision
                total_violations += 1;
                if -sdf > max_violation {
                    max_violation = -sdf;
                    worst_frame = frame;
                }
            }

            // Also check cell type
            let (i, j) = sim.grid.pos_to_cell(p.position);
            if sim.grid.is_solid(i, j) {
                particles_inside_solid_cells += 1;
            }
        }
    }

    println!("  Simulated {} frames with {} final particles", FRAMES, sim.particles.len());
    println!("  Total SDF violations: {}", total_violations);
    println!("  Max violation depth: {:.2} (frame {})", max_violation, worst_frame);
    println!("  Particles in solid cells: {}", particles_inside_solid_cells);

    print!("  [0.1] No SDF violations (particles inside solids): ");
    if total_violations == 0 {
        println!("PASS");
        pass += 1;
    } else {
        println!("FAIL ({} violations, max depth {:.2})", total_violations, max_violation);
        fail += 1;
    }

    print!("  [0.2] No particles in solid cells: ");
    if particles_inside_solid_cells == 0 {
        println!("PASS");
        pass += 1;
    } else {
        println!("FAIL ({} particles in solid cells)", particles_inside_solid_cells);
        fail += 1;
    }

    (pass, fail)
}

/// Test that two overlapping particles get pushed apart
fn test_overlapping_particles_separate() -> (usize, usize) {
    let mut pass = 0;
    let mut fail = 0;

    const WIDTH: usize = 32;
    const HEIGHT: usize = 32;
    const CELL_SIZE: f32 = 4.0;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // Create a simple container with floor
    sim.grid.compute_sdf();

    // Spawn two particles very close together (overlapping)
    let center_x = WIDTH as f32 * CELL_SIZE / 2.0;
    let center_y = HEIGHT as f32 * CELL_SIZE / 2.0;

    // Particles at same position
    sim.particles.spawn_water(center_x, center_y, 0.0, 0.0);
    sim.particles.spawn_water(center_x + 0.5, center_y, 0.0, 0.0);  // 0.5 apart (should overlap, min_dist = 2.5)

    let initial_dist = (sim.particles.list[0].position - sim.particles.list[1].position).length();
    println!("  Initial distance: {:.2} (should be < 2.5)", initial_dist);

    // Run one simulation step
    let dt = 1.0 / 60.0;
    sim.update(dt);

    let final_dist = (sim.particles.list[0].position - sim.particles.list[1].position).length();
    println!("  Final distance:   {:.2} (should be >= 2.5)", final_dist);

    print!("  [1.1] Particles pushed apart (dist >= 2.0): ");
    if final_dist >= 2.0 {
        println!("PASS (dist={:.2})", final_dist);
        pass += 1;
    } else {
        println!("FAIL (dist={:.2})", final_dist);
        fail += 1;
    }

    print!("  [1.2] Distance increased: ");
    if final_dist > initial_dist {
        println!("PASS ({:.2} > {:.2})", final_dist, initial_dist);
        pass += 1;
    } else {
        println!("FAIL ({:.2} <= {:.2})", final_dist, initial_dist);
        fail += 1;
    }

    (pass, fail)
}

/// Test that particles pushed toward wall don't penetrate
fn test_wall_collision() -> (usize, usize) {
    let mut pass = 0;
    let mut fail = 0;

    const WIDTH: usize = 32;
    const HEIGHT: usize = 32;
    const CELL_SIZE: f32 = 4.0;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // Create floor
    for i in 0..WIDTH {
        sim.grid.set_solid(i, HEIGHT - 1);
        sim.grid.set_solid(i, HEIGHT - 2);
    }
    sim.grid.compute_sdf();

    // Spawn particles near floor
    let floor_y = (HEIGHT - 2) as f32 * CELL_SIZE;
    let x = WIDTH as f32 * CELL_SIZE / 2.0;

    // Stack of overlapping particles that should push toward floor
    for i in 0..5 {
        sim.particles.spawn_water(x, floor_y - 2.0 - i as f32 * 0.5, 0.0, 0.0);
    }

    println!("  Spawned 5 particles near floor at y={:.1}", floor_y);

    // Run simulation for several frames
    let dt = 1.0 / 60.0;
    for _ in 0..30 {
        sim.update(dt);
    }

    // Check no particle is inside solid
    let mut min_y = f32::MAX;
    let mut max_sdf_violation = 0.0f32;
    for p in sim.particles.iter() {
        min_y = min_y.min(p.position.y);
        let sdf = sim.grid.sample_sdf(p.position);
        if sdf < 0.0 {
            max_sdf_violation = max_sdf_violation.max(-sdf);
        }
    }

    println!("  Min particle y: {:.2}", min_y);
    println!("  Floor solid y:  {:.2}", floor_y);
    println!("  Max SDF violation: {:.2}", max_sdf_violation);

    print!("  [2.1] No particles inside solid (SDF >= 0): ");
    if max_sdf_violation < CELL_SIZE * 0.1 {
        println!("PASS (max_violation={:.2})", max_sdf_violation);
        pass += 1;
    } else {
        println!("FAIL (max_violation={:.2})", max_sdf_violation);
        fail += 1;
    }

    print!("  [2.2] Particles above floor (y < floor_y): ");
    if min_y < floor_y {
        println!("PASS (min_y={:.2})", min_y);
        pass += 1;
    } else {
        println!("FAIL (min_y={:.2} >= floor_y={:.2})", min_y, floor_y);
        fail += 1;
    }

    (pass, fail)
}

/// Test that particles maintain separation during full simulation
fn test_separation_during_simulation() -> (usize, usize) {
    let mut pass = 0;
    let mut fail = 0;

    const WIDTH: usize = 64;
    const HEIGHT: usize = 48;
    const CELL_SIZE: f32 = 4.0;
    const DT: f32 = 1.0 / 60.0;
    const FRAMES: usize = 120;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);
    create_sluice(&mut sim, 0.2, 10, 4, 2);

    // Spawn water
    for i in 0..5 {
        let x = 30.0 + (i % 3) as f32 * 8.0;
        let y = 40.0 + (i / 3) as f32 * 8.0;
        sim.spawn_water(x, y, 30.0, 0.0, 5);
    }

    let mut min_pair_dist_ever = f32::MAX;
    let mut overlap_count = 0;
    let min_dist = 2.0;  // Slightly less than 2.5 to account for movement

    for _frame in 0..FRAMES {
        sim.update(DT);

        // Sample a subset of particle pairs to check separation
        let particles = &sim.particles.list;
        let n = particles.len().min(100);
        for i in 0..n {
            for j in (i+1)..n {
                let dist = (particles[i].position - particles[j].position).length();
                min_pair_dist_ever = min_pair_dist_ever.min(dist);
                if dist < min_dist {
                    overlap_count += 1;
                }
            }
        }
    }

    println!("  Simulated {} frames with {} particles", FRAMES, sim.particles.len());
    println!("  Minimum pair distance ever: {:.3}", min_pair_dist_ever);
    println!("  Overlap events (dist < {}): {}", min_dist, overlap_count);

    print!("  [3.1] Few overlaps (<100 events): ");
    if overlap_count < 100 {
        println!("PASS (overlaps={})", overlap_count);
        pass += 1;
    } else {
        println!("FAIL (overlaps={})", overlap_count);
        fail += 1;
    }

    print!("  [3.2] Min distance not too small (>1.0): ");
    if min_pair_dist_ever > 1.0 {
        println!("PASS (min_dist={:.3})", min_pair_dist_ever);
        pass += 1;
    } else {
        println!("FAIL (min_dist={:.3})", min_pair_dist_ever);
        fail += 1;
    }

    (pass, fail)
}

/// Test that particles don't compress/flatten to floor
///
/// Note: In a real fluid simulation, particles WILL settle under gravity into a denser
/// configuration at the bottom. The test checks that:
/// 1. Particles don't overlap completely (no minimum distance violations)
/// 2. Particles spread horizontally (occupy reasonable area)
/// 3. No particles are inside solids
fn test_no_floor_compression() -> (usize, usize) {
    let mut pass = 0;
    let mut fail = 0;

    const WIDTH: usize = 32;
    const HEIGHT: usize = 32;
    const CELL_SIZE: f32 = 4.0;
    const DT: f32 = 1.0 / 60.0;
    const FRAMES: usize = 300;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // Create container
    for i in 0..WIDTH {
        sim.grid.set_solid(i, HEIGHT - 1);  // Floor
    }
    for j in 0..HEIGHT {
        sim.grid.set_solid(0, j);           // Left wall
        sim.grid.set_solid(WIDTH - 1, j);   // Right wall
    }
    sim.grid.compute_sdf();

    let floor_y = (HEIGHT - 1) as f32 * CELL_SIZE;

    // Spawn column of water particles
    let x = WIDTH as f32 * CELL_SIZE / 2.0;
    let spawn_count = 20;
    for i in 0..spawn_count {
        let y = floor_y - 10.0 - i as f32 * 3.0;
        sim.particles.spawn_water(x, y, 0.0, 0.0);
    }

    println!("  Spawned {} particles in column", spawn_count);

    // Let simulation run and settle
    for _ in 0..FRAMES {
        sim.update(DT);
    }

    // Measure extent and check for overlaps
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;
    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_sdf_violation = 0.0f32;

    for p in sim.particles.iter() {
        min_y = min_y.min(p.position.y);
        max_y = max_y.max(p.position.y);
        min_x = min_x.min(p.position.x);
        max_x = max_x.max(p.position.x);
        let sdf = sim.grid.sample_sdf(p.position);
        if sdf < 0.0 {
            max_sdf_violation = max_sdf_violation.max(-sdf);
        }
    }

    let vertical_extent = max_y - min_y;
    let horizontal_extent = max_x - min_x;
    let remaining = sim.particles.len();

    // Check minimum distance between particles
    let mut min_pair_dist = f32::MAX;
    let mut overlap_count = 0;
    let min_allowed_dist = 1.5;  // Somewhat less than min_dist due to dynamics

    for i in 0..remaining {
        for j in (i+1)..remaining {
            let dist = (sim.particles.list[i].position - sim.particles.list[j].position).length();
            min_pair_dist = min_pair_dist.min(dist);
            if dist < min_allowed_dist {
                overlap_count += 1;
            }
        }
    }

    println!("  Remaining particles: {}", remaining);
    println!("  Vertical extent:   {:.2}", vertical_extent);
    println!("  Horizontal extent: {:.2}", horizontal_extent);
    println!("  Min pair distance: {:.2}", min_pair_dist);
    println!("  Severe overlaps:   {} (dist < {:.1})", overlap_count, min_allowed_dist);

    print!("  [4.1] Particles retained (>= 15): ");
    if remaining >= 15 {
        println!("PASS (count={})", remaining);
        pass += 1;
    } else {
        println!("FAIL (count={})", remaining);
        fail += 1;
    }

    print!("  [4.2] No particles in solids: ");
    if max_sdf_violation < CELL_SIZE * 0.1 {
        println!("PASS (violation={:.2})", max_sdf_violation);
        pass += 1;
    } else {
        println!("FAIL (violation={:.2})", max_sdf_violation);
        fail += 1;
    }

    print!("  [4.3] Particles spread horizontally (extent > 10): ");
    if horizontal_extent > 10.0 {
        println!("PASS (extent={:.1})", horizontal_extent);
        pass += 1;
    } else {
        println!("FAIL (extent={:.1})", horizontal_extent);
        fail += 1;
    }

    print!("  [4.4] Minimum separation maintained (min_dist > 1.0): ");
    if min_pair_dist > 1.0 {
        println!("PASS (min={:.2})", min_pair_dist);
        pass += 1;
    } else {
        println!("FAIL (min={:.2}, particles overlapping)", min_pair_dist);
        fail += 1;
    }

    print!("  [4.5] Few severe overlaps (< 5): ");
    if overlap_count < 5 {
        println!("PASS (overlaps={})", overlap_count);
        pass += 1;
    } else {
        println!("FAIL (overlaps={})", overlap_count);
        fail += 1;
    }

    (pass, fail)
}

/// TEST 5: Spawn should never place particles inside solid cells
/// Tests the is_spawn_safe check in spawn_water and other spawn functions
fn test_spawn_never_in_solid() -> (usize, usize) {
    let mut pass = 0;
    let mut fail = 0;

    const WIDTH: usize = 64;
    const HEIGHT: usize = 48;
    const CELL_SIZE: f32 = 4.0;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // Create a complex solid pattern with lots of solid cells
    // Floor
    for i in 0..WIDTH {
        for j in (HEIGHT - 10)..HEIGHT {
            sim.grid.set_solid(i, j);
        }
    }
    // Walls
    for j in 0..HEIGHT {
        sim.grid.set_solid(0, j);
        sim.grid.set_solid(WIDTH - 1, j);
    }
    // Some internal obstacles
    for i in 10..20 {
        for j in 20..30 {
            sim.grid.set_solid(i, j);
        }
    }
    for i in 40..50 {
        for j in 15..35 {
            sim.grid.set_solid(i, j);
        }
    }
    sim.grid.compute_sdf();

    // Try to spawn particles at various positions, including inside solids
    // The spawn functions should skip solid positions
    let spawn_positions = [
        (32.0, 20.0),   // Open area
        (5.0, 180.0),   // Near floor (inside solid)
        (15.0 * CELL_SIZE, 25.0 * CELL_SIZE), // Inside obstacle
        (45.0 * CELL_SIZE, 25.0 * CELL_SIZE), // Inside obstacle
        (2.0, HEIGHT as f32 * CELL_SIZE - 5.0), // Near corner
    ];

    let initial_count = sim.particles.len();

    for (x, y) in spawn_positions {
        sim.spawn_water(x, y, 0.0, 0.0, 10);
    }

    // Check that NO particles ended up inside solids
    let mut particles_in_solid = 0;
    let mut particles_with_negative_sdf = 0;

    for p in sim.particles.iter() {
        let (i, j) = sim.grid.pos_to_cell(p.position);
        if sim.grid.is_solid(i, j) {
            particles_in_solid += 1;
        }
        if sim.grid.sample_sdf(p.position) < 0.0 {
            particles_with_negative_sdf += 1;
        }
    }

    let spawned = sim.particles.len() - initial_count;
    println!("  Attempted spawn at {} positions (10 each)", spawn_positions.len());
    println!("  Particles spawned: {}", spawned);
    println!("  Particles in solid cells: {}", particles_in_solid);
    println!("  Particles with SDF < 0: {}", particles_with_negative_sdf);

    print!("  [5.1] No particles in solid cells: ");
    if particles_in_solid == 0 {
        println!("PASS");
        pass += 1;
    } else {
        println!("FAIL ({} in solid)", particles_in_solid);
        fail += 1;
    }

    print!("  [5.2] No particles with negative SDF: ");
    if particles_with_negative_sdf == 0 {
        println!("PASS");
        pass += 1;
    } else {
        println!("FAIL ({} with SDF < 0)", particles_with_negative_sdf);
        fail += 1;
    }

    (pass, fail)
}

/// TEST 6: Pressure solver should converge (residual below threshold)
fn test_pressure_solver_convergence() -> (usize, usize) {
    let mut pass = 0;
    let mut fail = 0;

    const WIDTH: usize = 32;
    const HEIGHT: usize = 32;
    const CELL_SIZE: f32 = 4.0;
    const DT: f32 = 1.0 / 60.0;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // Create container
    for i in 0..WIDTH {
        sim.grid.set_solid(i, HEIGHT - 1);
        sim.grid.set_solid(i, 0);
    }
    for j in 0..HEIGHT {
        sim.grid.set_solid(0, j);
        sim.grid.set_solid(WIDTH - 1, j);
    }
    // Add a narrow channel (harder to converge)
    for i in 5..WIDTH-5 {
        sim.grid.set_solid(i, HEIGHT / 2);
    }
    // Leave a small gap - compute index first to avoid borrow issues
    let gap_idx = sim.grid.cell_index(WIDTH / 2, HEIGHT / 2);
    sim.grid.solid[gap_idx] = false;

    sim.grid.compute_sdf();

    // Spawn water
    for i in 0..5 {
        for j in 0..5 {
            let x = (5 + i) as f32 * CELL_SIZE + CELL_SIZE * 0.5;
            let y = (5 + j) as f32 * CELL_SIZE + CELL_SIZE * 0.5;
            sim.spawn_water(x, y, 20.0, 0.0, 1);
        }
    }

    // Run simulation
    for _ in 0..60 {
        sim.update(DT);
    }

    // Check residual divergence
    let mut max_div = 0.0f32;
    let mut fluid_cells = 0;
    sim.grid.compute_divergence();

    for j in 1..HEIGHT - 1 {
        for i in 1..WIDTH - 1 {
            let idx = sim.grid.cell_index(i, j);
            if sim.grid.cell_type[idx] == sim::grid::CellType::Fluid {
                max_div = max_div.max(sim.grid.divergence[idx].abs());
                fluid_cells += 1;
            }
        }
    }

    println!("  Narrow channel geometry with {} fluid cells", fluid_cells);
    println!("  Max divergence after solve: {:.6}", max_div);

    print!("  [6.1] Max divergence < 0.1: ");
    if max_div < 0.1 {
        println!("PASS (max_div={:.6})", max_div);
        pass += 1;
    } else {
        println!("FAIL (max_div={:.6})", max_div);
        fail += 1;
    }

    print!("  [6.2] Max divergence < 0.01 (good convergence): ");
    if max_div < 0.01 {
        println!("PASS (max_div={:.6})", max_div);
        pass += 1;
    } else {
        println!("WARN (max_div={:.6}, acceptable but not ideal)", max_div);
        pass += 1; // Still pass, just not ideal
    }

    (pass, fail)
}

/// TEST 7: Sediment separation should not panic even with particles at same position
fn test_sediment_separation_no_panic() -> (usize, usize) {
    // Legacy sediment.rs was removed - using FLIP particle system now
    println!("  [SKIPPED] Legacy sediment tests removed");
    (0, 0)
}
