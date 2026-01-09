//! Game Mirror Test - Matches real game configuration exactly
//!
//! This test replicates the exact setup from game/src/main.rs to diagnose
//! momentum loss in real game conditions, not synthetic test scenarios.
//!
//! Run with: cargo test -p sim --release --test game_mirror_test -- --nocapture

use sim::{FlipSimulation, SluiceConfig, RiffleMode, create_sluice_with_mode};

/// Mirrors exact game configuration
#[test]
fn test_game_mirror_momentum() {
    // === EXACT GAME CONFIGURATION ===
    const SIM_WIDTH: usize = 512;
    const SIM_HEIGHT: usize = 384;
    const CELL_SIZE: f32 = 1.0;
    const DT: f32 = 1.0 / 120.0;  // Game uses 120Hz, not 60Hz

    // Inlet parameters from game
    const INLET_X: f32 = 5.0;
    const INLET_Y: f32 = 86.0;
    const INLET_VX: f32 = 80.0;
    const INLET_VY: f32 = 5.0;
    const SPAWN_RATE: usize = 4;

    let mut sim = FlipSimulation::new(SIM_WIDTH, SIM_HEIGHT, CELL_SIZE);

    // Create sluice with exact game config
    let sluice_config = SluiceConfig {
        slope: 0.25,
        riffle_spacing: 60,
        riffle_height: 6,
        riffle_width: 4,
        riffle_mode: RiffleMode::ClassicBattEdge,
        slick_plate_len: 50,
    };
    create_sluice_with_mode(&mut sim, &sluice_config);

    println!("\n=== GAME MIRROR TEST ===\n");
    println!("Configuration:");
    println!("  Grid: {}x{}, cell_size={}", SIM_WIDTH, SIM_HEIGHT, CELL_SIZE);
    println!("  dt: {} ({}Hz)", DT, (1.0/DT) as i32);
    println!("  Inlet: ({}, {}) velocity=({}, {})", INLET_X, INLET_Y, INLET_VX, INLET_VY);
    println!("  Spawn rate: {} particles/frame", SPAWN_RATE);
    println!();

    // Run for 5 seconds (600 frames at 120Hz)
    const SECONDS: usize = 5;
    const FRAMES: usize = SECONDS * 120;

    println!("{:>5} | {:>7} | {:>8} | {:>8} | {:>8}",
             "Time", "Count", "Avg VX", "Max X", "Inlet%");
    println!("{:-<5}-+-{:-<7}-+-{:-<8}-+-{:-<8}-+-{:-<8}", "", "", "", "", "");

    for frame in 0..FRAMES {
        // Spawn water (same as game)
        sim.spawn_water(INLET_X, INLET_Y, INLET_VX, INLET_VY, SPAWN_RATE);

        // Remove particles at outlet (same as game)
        let outflow_x = (SIM_WIDTH as f32 - 5.0) * CELL_SIZE;
        sim.particles.list.retain(|p| p.position.x < outflow_x);

        // Update
        sim.update(DT);

        // Log every second
        if frame % 120 == 119 {
            let count = sim.particles.iter().filter(|p| !p.is_sediment()).count();
            let avg_vx: f32 = sim.particles.iter()
                .filter(|p| !p.is_sediment())
                .map(|p| p.velocity.x)
                .sum::<f32>() / count.max(1) as f32;
            let max_x = sim.particles.iter()
                .filter(|p| !p.is_sediment())
                .map(|p| p.position.x)
                .fold(0.0f32, |a, b| a.max(b));
            let inlet_pct = (avg_vx / INLET_VX) * 100.0;

            println!("{:5}s | {:7} | {:8.1} | {:8.1} | {:7.1}%",
                     (frame + 1) / 120, count, avg_vx, max_x, inlet_pct);
        }
    }

    // Final analysis
    let final_count = sim.particles.iter().filter(|p| !p.is_sediment()).count();
    let final_avg_vx: f32 = sim.particles.iter()
        .filter(|p| !p.is_sediment())
        .map(|p| p.velocity.x)
        .sum::<f32>() / final_count.max(1) as f32;

    println!();
    println!("=== FINAL RESULTS ===");
    println!("  Particle count: {}", final_count);
    println!("  Avg velocity X: {:.1} (inlet was {})", final_avg_vx, INLET_VX);
    println!("  Velocity ratio: {:.1}%", (final_avg_vx / INLET_VX) * 100.0);

    // This is informational - we want to see the actual behavior
    // before deciding what the threshold should be
}

/// Run with full diagnostics to see where momentum goes
#[test]
fn test_game_mirror_with_diagnostics() {
    const SIM_WIDTH: usize = 512;
    const SIM_HEIGHT: usize = 384;
    const CELL_SIZE: f32 = 1.0;
    const DT: f32 = 1.0 / 120.0;

    let mut sim = FlipSimulation::new(SIM_WIDTH, SIM_HEIGHT, CELL_SIZE);

    let sluice_config = SluiceConfig {
        slope: 0.25,
        riffle_spacing: 60,
        riffle_height: 6,
        riffle_width: 4,
        riffle_mode: RiffleMode::ClassicBattEdge,
        slick_plate_len: 50,
    };
    create_sluice_with_mode(&mut sim, &sluice_config);

    // Spawn initial batch of water
    for _ in 0..100 {
        sim.spawn_water(5.0, 86.0, 80.0, 5.0, 4);
    }

    println!("\n=== GAME MIRROR DIAGNOSTICS ===\n");
    println!("Running 10 frames with full phase diagnostics...\n");

    // Run a few frames with diagnostics
    for frame in 0..10 {
        println!("--- Frame {} ---", frame);
        let diag = sim.update_with_diagnostics(DT);

        // Print phase-by-phase momentum
        for i in 0..diag.len() {
            let (name, momentum) = &diag[i];
            if i > 0 {
                let (_, prev_momentum) = &diag[i-1];
                let delta = momentum - prev_momentum;
                let pct = if *prev_momentum > 0.0 { (delta / prev_momentum) * 100.0 } else { 0.0 };
                println!("  {:20}: {:10.1} (Δ {:+8.1}, {:+6.2}%)", name, momentum, delta, pct);
            } else {
                println!("  {:20}: {:10.1}", name, momentum);
            }
        }
        println!();
    }
}
