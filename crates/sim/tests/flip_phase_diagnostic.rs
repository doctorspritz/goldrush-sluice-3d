//! FLIP Kernel Mismatch Diagnostic Test
//!
//! This test validates the hypothesis that using different interpolation kernels
//! (bilinear for store_old, quadratic for G2P) causes phantom velocity deltas.
//!
//! Run with: cargo test -p sim --release --test flip_phase_diagnostic -- --nocapture

use sim::FlipSimulation;

/// THE ONE TEST THAT MATTERS
///
/// Run isolated FLIP cycle (P2G → store_old → G2P) with NO forces.
/// If momentum is lost despite zero grid modifications, kernel mismatch is confirmed.
#[test]
fn test_isolated_flip_cycle_no_forces() {
    const WIDTH: usize = 64;
    const HEIGHT: usize = 32;
    const CELL_SIZE: f32 = 1.0;
    const DT: f32 = 1.0 / 60.0;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // Create closed channel (top and bottom walls)
    for i in 0..WIDTH {
        sim.grid.set_solid(i, 0);
        sim.grid.set_solid(i, HEIGHT - 1);
    }
    sim.grid.compute_sdf();

    // Spawn water block in CENTER of domain (away from boundaries)
    // to eliminate weight partition loss as a factor
    let initial_vx = 50.0;
    for xi in 0..20 {
        for yi in 0..10 {
            let x = 20.0 + xi as f32;  // x: 20-40 (center of 64-wide grid)
            let y = 10.0 + yi as f32;  // y: 10-20 (center of 32-high grid)
            sim.spawn_water(x, y, initial_vx, 0.0, 1);
        }
    }

    let particle_count = sim.particles.iter().filter(|p| !p.is_sediment()).count();

    println!("\n=== ISOLATED FLIP CYCLE TEST (No Forces) ===\n");
    println!("Setup:");
    println!("  Grid: {}x{}, cell_size={}", WIDTH, HEIGHT, CELL_SIZE);
    println!("  Particles: {} (water only, centered)", particle_count);
    println!("  Initial velocity: vx={}, vy=0", initial_vx);
    println!();

    // Run multiple isolated cycles to see accumulation
    println!("{:>5} | {:>12} | {:>10} | {:>10}", "Cycle", "Momentum", "Retention", "Loss/Cycle");
    println!("{:-<5}-+-{:-<12}-+-{:-<10}-+-{:-<10}", "", "", "", "");

    let mut prev_momentum = 0.0;
    for cycle in 0..10 {
        let (before, after) = sim.run_isolated_flip_cycle(DT);

        if cycle == 0 {
            prev_momentum = before;
            println!("{:5} | {:12.1} | {:9.2}% | {:>10}",
                     "init", before, 100.0, "-");
        }

        let retention = (after / prev_momentum) * 100.0;
        let loss = (1.0 - after / prev_momentum) * 100.0;
        println!("{:5} | {:12.1} | {:9.2}% | {:9.2}%",
                 cycle, after, retention, loss);
        prev_momentum = after;
    }

    // Run one more to get final measurement
    let (_, final_momentum) = sim.run_isolated_flip_cycle(DT);
    let initial_momentum = sim.particles.iter()
        .filter(|p| !p.is_sediment())
        .count() as f32 * initial_vx;  // Approximate initial

    println!();
    println!("=== RESULTS ===");
    println!();

    // Check weight partition
    let u_weight = sim.get_u_weight_sum();
    let v_weight = sim.get_v_weight_sum();
    println!("Weight partition (should be ~{}):", particle_count);
    println!("  U weight sum: {:.1} ({:.1}%)", u_weight, u_weight / particle_count as f32 * 100.0);
    println!("  V weight sum: {:.1} ({:.1}%)", v_weight, v_weight / particle_count as f32 * 100.0);
    println!();

    // Calculate total retention over 10 cycles
    let total_retention = (final_momentum / (particle_count as f32 * initial_vx)) * 100.0;
    let per_cycle_loss = (1.0 - (final_momentum / (particle_count as f32 * initial_vx)).powf(1.0/10.0)) * 100.0;

    println!("After 10 isolated cycles:");
    println!("  Total retention: {:.1}%", total_retention);
    println!("  Per-cycle loss: {:.2}%", per_cycle_loss);
    println!();

    if per_cycle_loss > 0.5 {
        println!("HYPOTHESIS CONFIRMED: Momentum lost with NO forces applied!");
        println!("The kernel mismatch (bilinear vs quadratic) is causing phantom velocity deltas.");
        println!();
        println!("Fix: Change store_old_velocities() to use quadratic B-spline sampling.");
    } else if per_cycle_loss > 0.1 {
        println!("PARTIAL CONFIRMATION: Small momentum loss detected.");
        println!("Kernel mismatch may be a factor, but other issues exist.");
    } else {
        println!("HYPOTHESIS REJECTED: No significant momentum loss without forces.");
        println!("The problem is elsewhere (boundary conditions, pressure solver, etc.)");
    }

    // Fail the test if significant loss detected (to trigger CI alerts)
    assert!(per_cycle_loss < 1.0,
        "Kernel mismatch confirmed: {:.2}% loss per cycle with NO forces applied. \
         The FLIP algorithm uses different kernels for old_velocity (bilinear) \
         and new_velocity (quadratic B-spline), causing phantom velocity deltas.",
        per_cycle_loss);
}
