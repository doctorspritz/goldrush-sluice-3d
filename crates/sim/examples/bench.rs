//! Quick benchmark for profiling the FLIP simulation
//!
//! Run with: cargo run --release --example bench -p sim
//! Profile with: cargo flamegraph --example bench -p sim

use sim::{create_sluice, FlipSimulation};
use std::time::Instant;

fn main() {
    const WIDTH: usize = 128;
    const HEIGHT: usize = 128;
    const CELL_SIZE: f32 = 2.0;
    const FRAMES: usize = 300; // 5 seconds at 60 FPS
    const DT: f32 = 1.0 / 60.0;

    println!("Setting up simulation {}x{} with cell_size={}", WIDTH, HEIGHT, CELL_SIZE);

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);
    create_sluice(&mut sim, 0.3, 20, 5);

    // Pre-spawn ~5000 particles
    println!("Spawning particles...");
    for i in 0..200 {
        let x = 20.0 + (i % 50) as f32 * 2.0;
        let y = (HEIGHT as f32 * CELL_SIZE) * 0.15 + (i / 50) as f32 * 3.0;
        sim.spawn_water(x, y, 50.0, 0.0, 25);
    }
    println!("Initial particles: {}", sim.particles.len());

    // Warm up
    println!("Warming up (50 frames)...");
    for _ in 0..50 {
        sim.update(DT);
    }
    println!("After warmup: {} particles", sim.particles.len());

    // Benchmark
    println!("Running {} frames...", FRAMES);
    let start = Instant::now();

    for frame in 0..FRAMES {
        // Spawn more particles like the game does
        if frame % 2 == 0 {
            let inlet_x = 20.0;
            let inlet_y = (HEIGHT as f32 * CELL_SIZE) * 0.2;
            sim.spawn_water(inlet_x, inlet_y, 50.0, 0.0, 3);
        }

        sim.update(DT);

        if frame % 60 == 0 {
            println!("  Frame {}: {} particles", frame, sim.particles.len());
        }
    }

    let elapsed = start.elapsed();
    let avg_frame_time = elapsed.as_secs_f64() / FRAMES as f64;
    let fps = 1.0 / avg_frame_time;

    println!("\n=== Results ===");
    println!("Total time: {:.2?}", elapsed);
    println!("Avg frame time: {:.2}ms", avg_frame_time * 1000.0);
    println!("Effective FPS: {:.1}", fps);
    println!("Final particles: {}", sim.particles.len());

    if fps < 60.0 {
        println!("\n⚠️  Below 60 FPS target - optimization needed");
    } else {
        println!("\n✅ Meeting 60 FPS target");
    }
}
