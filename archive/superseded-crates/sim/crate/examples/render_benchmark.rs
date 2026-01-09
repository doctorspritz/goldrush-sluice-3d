//! Render benchmark - tests rendering performance at scale
//!
//! Runs simulation with high particle counts to measure real-world
//! rendering performance. Use this to validate rendering optimizations.
//!
//! Run with: cargo run --example render_benchmark --release

use sim::{create_sluice_with_mode, FlipSimulation, RiffleMode, SluiceConfig};
use std::time::{Duration, Instant};

const SIM_WIDTH: usize = 512;
const SIM_HEIGHT: usize = 256;
const CELL_SIZE: f32 = 1.0;

// Test parameters - realistic scale
const TARGET_PARTICLES: usize = 200_000;
const TEST_DURATION_SECS: u64 = 120; // 2 minutes
const SPAWN_RATE: usize = 500; // Particles per frame

fn main() {
    println!("=== RENDER BENCHMARK ===");
    println!("Target: {} particles", TARGET_PARTICLES);
    println!("Duration: {} seconds", TEST_DURATION_SECS);
    println!();

    // Create simulation with sluice
    let mut sim = FlipSimulation::new(SIM_WIDTH, SIM_HEIGHT, CELL_SIZE);

    let sluice_config = SluiceConfig {
        slope: 0.12,
        riffle_spacing: 60,
        riffle_height: 6,
        riffle_width: 4,
        riffle_mode: RiffleMode::ClassicBattEdge,
        slick_plate_len: 0,
    };
    create_sluice_with_mode(&mut sim, &sluice_config);

    // Add barrier at end (like game)
    let barrier_x = SIM_WIDTH - 10;
    for j in 0..SIM_HEIGHT {
        for i in barrier_x..SIM_WIDTH {
            let idx = j * SIM_WIDTH + i;
            sim.grid.solid[idx] = true;
        }
    }

    // Count solid cells for reference
    let solid_count: usize = sim.grid.solid.iter().filter(|&&s| s).count();
    println!("Grid: {}x{} = {} cells", SIM_WIDTH, SIM_HEIGHT, SIM_WIDTH * SIM_HEIGHT);
    println!("Solid cells: {} ({:.1}%)", solid_count, 100.0 * solid_count as f64 / (SIM_WIDTH * SIM_HEIGHT) as f64);
    println!();

    // Inlet position
    let inlet_x = 5.0;
    let inlet_y = (SIM_HEIGHT / 4 - 10) as f32;
    let inlet_vx = 80.0;
    let inlet_vy = 5.0;

    // Timing
    let start = Instant::now();
    let test_duration = Duration::from_secs(TEST_DURATION_SECS);

    let mut frame_count = 0u64;
    let mut total_sim_time = Duration::ZERO;
    let mut last_report = Instant::now();
    let report_interval = Duration::from_secs(10);

    // Warmup phase - spawn particles until we hit target
    println!("Phase 1: Warmup (spawning to {} particles)...", TARGET_PARTICLES);
    while sim.particles.len() < TARGET_PARTICLES && start.elapsed() < test_duration {
        // Spawn water
        sim.spawn_water(inlet_x, inlet_y, inlet_vx, inlet_vy, SPAWN_RATE);

        // Spawn some sediment
        if frame_count % 4 == 0 {
            sim.spawn_sand(inlet_x, inlet_y, inlet_vx, inlet_vy, 10);
        }
        if frame_count % 8 == 0 {
            sim.spawn_magnetite(inlet_x, inlet_y, inlet_vx, inlet_vy, 5);
        }
        if frame_count % 20 == 0 {
            sim.spawn_gold(inlet_x, inlet_y, inlet_vx, inlet_vy, 2);
        }

        let sim_start = Instant::now();
        sim.step(1.0 / 60.0);
        total_sim_time += sim_start.elapsed();

        frame_count += 1;

        if last_report.elapsed() >= report_interval {
            println!(
                "  {} particles, {} frames, avg sim: {:.1}ms",
                sim.particles.len(),
                frame_count,
                total_sim_time.as_secs_f64() * 1000.0 / frame_count as f64
            );
            last_report = Instant::now();
        }
    }

    let warmup_frames = frame_count;
    let warmup_time = start.elapsed();
    println!();
    println!("Warmup complete: {} particles in {:.1}s ({} frames)",
        sim.particles.len(), warmup_time.as_secs_f64(), warmup_frames);
    println!();

    // Steady state phase - measure performance
    println!("Phase 2: Steady state measurement...");
    let steady_start = Instant::now();
    let mut steady_frames = 0u64;
    let mut steady_sim_time = Duration::ZERO;
    let mut min_particles = sim.particles.len();
    let mut max_particles = sim.particles.len();

    while start.elapsed() < test_duration {
        // Maintain particle count
        let current = sim.particles.len();
        if current < TARGET_PARTICLES {
            let deficit = TARGET_PARTICLES - current;
            let spawn = deficit.min(SPAWN_RATE);
            sim.spawn_water(inlet_x, inlet_y, inlet_vx, inlet_vy, spawn);
        }

        // Spawn sediment
        if frame_count % 4 == 0 {
            sim.spawn_sand(inlet_x, inlet_y, inlet_vx, inlet_vy, 5);
        }
        if frame_count % 8 == 0 {
            sim.spawn_magnetite(inlet_x, inlet_y, inlet_vx, inlet_vy, 2);
        }
        if frame_count % 20 == 0 {
            sim.spawn_gold(inlet_x, inlet_y, inlet_vx, inlet_vy, 1);
        }

        let sim_start = Instant::now();
        sim.step(1.0 / 60.0);
        steady_sim_time += sim_start.elapsed();

        frame_count += 1;
        steady_frames += 1;

        min_particles = min_particles.min(sim.particles.len());
        max_particles = max_particles.max(sim.particles.len());

        if last_report.elapsed() >= report_interval {
            let avg_ms = steady_sim_time.as_secs_f64() * 1000.0 / steady_frames as f64;
            let fps = 1000.0 / avg_ms;
            println!(
                "  {} particles, {:.1}ms/frame ({:.1} FPS)",
                sim.particles.len(),
                avg_ms,
                fps
            );
            last_report = Instant::now();
        }
    }

    let steady_duration = steady_start.elapsed();
    let total_duration = start.elapsed();

    // Final report
    println!();
    println!("=== RESULTS ===");
    println!();
    println!("Test duration: {:.1}s total ({:.1}s warmup, {:.1}s steady)",
        total_duration.as_secs_f64(),
        warmup_time.as_secs_f64(),
        steady_duration.as_secs_f64());
    println!();
    println!("Particle range: {} - {}", min_particles, max_particles);
    println!("Total frames: {} ({} warmup, {} steady)",
        frame_count, warmup_frames, steady_frames);
    println!();

    let avg_sim_ms = steady_sim_time.as_secs_f64() * 1000.0 / steady_frames as f64;
    let steady_fps = steady_frames as f64 / steady_duration.as_secs_f64();

    println!("STEADY STATE PERFORMANCE:");
    println!("  Average sim time: {:.2} ms/frame", avg_sim_ms);
    println!("  Effective FPS: {:.1}", steady_fps);
    println!("  Target FPS: 60");
    println!("  Status: {}", if steady_fps >= 60.0 { "PASS ✓" } else { "FAIL ✗" });
    println!();

    // Breakdown
    println!("Note: This measures SIMULATION time only.");
    println!("Rendering overhead is separate (measured in game binary).");
    println!();
    println!("For rendering benchmarks, run the game and observe:");
    println!("  - TERRAIN message shows RLE rectangle count");
    println!("  - PARTICLES message shows rendered count");
    println!("  - Frame timing in status bar");
}
