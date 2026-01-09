//! Compare performance between 512x384 and 512x256 grids

use sim::{create_sluice_with_mode, FlipSimulation, SluiceConfig, RiffleMode};
use std::time::Instant;

fn run_benchmark(width: usize, height: usize, frames: usize) -> (f64, f64, usize) {
    let cell_size = 1.0;
    let mut sim = FlipSimulation::new(width, height, cell_size);

    let config = SluiceConfig {
        slope: 0.25,
        riffle_spacing: 60,
        riffle_height: 6,
        riffle_width: 4,
        riffle_mode: RiffleMode::ClassicBattEdge,
        slick_plate_len: 50,
    };
    create_sluice_with_mode(&mut sim, &config);

    let dt = 1.0 / 60.0;
    let inlet_vx = 80.0;
    let inlet_vy = 5.0;
    let spawn_rate = 4;

    let mut total_time = 0.0;
    let base_y = height / 4;

    for frame in 0..frames {
        // Spawn water at inlet
        for i in 0..spawn_rate {
            let y = (base_y - 20 + i * 5) as f32;
            sim.spawn_water(5.0, y, inlet_vx, inlet_vy, 1);
        }

        let start = Instant::now();
        sim.update(dt);
        total_time += start.elapsed().as_secs_f64();

        // Print progress every 5 seconds
        if frame % 300 == 299 {
            let seconds = (frame + 1) as f32 / 60.0;
            let avg_ms = total_time / (frame + 1) as f64 * 1000.0;
            let fps = 1000.0 / avg_ms;
            println!("  {:.0}s: {} particles, {:.1}ms/frame, {:.0} FPS",
                seconds, sim.particles.len(), avg_ms, fps);
        }
    }

    let avg_ms = total_time / frames as f64 * 1000.0;
    let fps = 1000.0 / avg_ms;
    (avg_ms, fps, sim.particles.len())
}

fn main() {
    let frames = 600; // 10 seconds

    println!("=== Testing 512x384 (current) ===");
    let (ms_384, fps_384, particles_384) = run_benchmark(512, 384, frames);

    println!("\n=== Testing 512x256 (proposed) ===");
    let (ms_256, fps_256, particles_256) = run_benchmark(512, 256, frames);

    println!("\n=== COMPARISON ===");
    println!("Grid 512x384: {:.2}ms/frame, {:.1} FPS, {} particles", ms_384, fps_384, particles_384);
    println!("Grid 512x256: {:.2}ms/frame, {:.1} FPS, {} particles", ms_256, fps_256, particles_256);
    println!("\nSpeedup: {:.1}x faster ({:.1}ms saved per frame)",
        ms_384 / ms_256, ms_384 - ms_256);
    println!("FPS gain: +{:.1} FPS", fps_256 - fps_384);
}
