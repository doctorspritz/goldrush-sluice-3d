//! Pressure solver test suite
//! Run with: cargo run --example test_pressure_fixes -p sim

use sim::grid::{CellType, Grid};
use sim::{create_sluice, FlipSimulation};

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║       FLIP PRESSURE SOLVER - TEST SUITE                  ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let mut total_pass = 0;
    let mut total_fail = 0;

    // Test 1: Static water column
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 1: STATIC WATER COLUMN");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let (pass, fail) = test_static_water_column();
    total_pass += pass;
    total_fail += fail;

    // Test 2: Dynamic simulation
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 2: DYNAMIC SIMULATION (60 frames)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let (pass, fail) = test_dynamic_simulation();
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
    }
}

fn test_static_water_column() -> (usize, usize) {
    let mut pass = 0;
    let mut fail = 0;

    let width = 8;
    let height = 8;
    let cell_size = 1.0;
    let mut grid = Grid::new(width, height, cell_size);

    // Set up container
    for i in 0..width {
        grid.set_solid(i, height - 1);
    }
    for j in 0..height {
        grid.set_solid(0, j);
        grid.set_solid(width - 1, j);
    }

    // Mark cell types
    for j in 0..height {
        for i in 0..width {
            if grid.is_solid(i, j) {
                let idx = grid.cell_index(i, j);
                grid.cell_type[idx] = CellType::Solid;
            }
        }
    }

    // Water column
    for j in 4..7 {
        let idx = grid.cell_index(4, j);
        grid.cell_type[idx] = CellType::Fluid;
    }

    let dt = 1.0 / 60.0;
    grid.apply_gravity(dt);
    grid.enforce_boundary_conditions();
    grid.compute_divergence();
    let div_before = grid.total_divergence();

    grid.solve_pressure(40);
    grid.apply_pressure_gradient(dt);
    grid.compute_divergence();
    let div_after = grid.total_divergence();

    println!("  Divergence: {:.4} → {:.4}", div_before, div_after);

    // Check: Divergence should decrease significantly
    print!("  [1.1] Divergence decreases (>50%): ");
    if div_after < div_before * 0.5 {
        println!("PASS");
        pass += 1;
    } else {
        println!("FAIL ({:.1}% reduction)", (1.0 - div_after / div_before) * 100.0);
        fail += 1;
    }

    (pass, fail)
}

fn test_dynamic_simulation() -> (usize, usize) {
    let mut pass = 0;
    let mut fail = 0;

    const WIDTH: usize = 64;
    const HEIGHT: usize = 48;
    const CELL_SIZE: f32 = 4.0;
    const DT: f32 = 1.0 / 60.0;
    const FRAMES: usize = 300;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);
    create_sluice(&mut sim, 0.2, 10, 4, 2);

    // Spawn initial water
    for i in 0..10 {
        let x = 30.0 + (i % 5) as f32 * 8.0;
        let y = 40.0 + (i / 5) as f32 * 8.0;
        sim.spawn_water(x, y, 30.0, 0.0, 10);
    }

    let mut max_velocity: f32 = 0.0;
    let mut final_divergence: f32 = 0.0;
    let mut particle_count = 0;

    for frame in 0..FRAMES {
        sim.update(DT);

        for p in sim.particles.iter() {
            max_velocity = max_velocity.max(p.velocity.length());
        }

        if frame % 50 == 0 {
            sim.grid.compute_divergence();
            let div = sim.grid.total_divergence();
            println!("  Frame {:3}: {:4} particles, max_v: {:6.1}, div: {:6.1}",
                frame, sim.particles.len(), max_velocity, div);
        }

        if frame == FRAMES - 1 {
            sim.grid.compute_divergence();
            final_divergence = sim.grid.total_divergence();
            particle_count = sim.particles.len();
        }
    }

    println!("\n  Final: {} particles, max_v: {:.1}, div: {:.1}",
        particle_count, max_velocity, final_divergence);

    // Check: Velocity bounded (no explosion)
    print!("  [2.1] Velocity bounded (<500): ");
    if max_velocity < 500.0 {
        println!("PASS (max={:.1})", max_velocity);
        pass += 1;
    } else {
        println!("FAIL (max={:.1})", max_velocity);
        fail += 1;
    }

    // Check: Particles don't disappear (stability)
    print!("  [2.2] Particles retained (>50): ");
    if particle_count > 50 {
        println!("PASS (count={})", particle_count);
        pass += 1;
    } else {
        println!("FAIL (count={})", particle_count);
        fail += 1;
    }

    // Check: Divergence is reasonable (not exploding)
    print!("  [2.3] Divergence bounded (<1000): ");
    if final_divergence < 1000.0 {
        println!("PASS (div={:.1})", final_divergence);
        pass += 1;
    } else {
        println!("FAIL (div={:.1})", final_divergence);
        fail += 1;
    }

    (pass, fail)
}
