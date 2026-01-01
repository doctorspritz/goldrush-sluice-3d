//! FLIP Simulation Profiler
//!
//! Matches real game configuration (512×256 grid) and tests scaling.
//!
//! Run with: cargo run --release --example profile -p sim

use sim::{create_sluice_with_mode, FlipSimulation, Particle, SluiceConfig};

fn main() {
    println!("=== FLIP Simulation Profiler ===");
    println!("=== Real Game Configuration (512×256) ===\n");

    // Show struct size for cache analysis
    println!("Particle struct size: {} bytes", std::mem::size_of::<Particle>());
    println!("Grid size: 512×256 = 131,072 cells\n");

    // Test scaling up to find breaking point
    for &particle_count in &[10_000, 25_000, 50_000, 100_000] {
        profile_at_scale(particle_count);
    }
}

fn profile_at_scale(target_particles: usize) {
    // Match real game configuration
    const WIDTH: usize = 512;
    const HEIGHT: usize = 256;
    const CELL_SIZE: f32 = 1.0;
    const WARMUP: usize = 30;
    const FRAMES: usize = 60;  // 1 second of simulation
    const DT: f32 = 1.0 / 60.0;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // Use real sluice geometry
    let config = SluiceConfig::default();
    create_sluice_with_mode(&mut sim, &config);

    // Spawn particles across the sluice (like real game inlet)
    // Real game spawns at inlet_y = HEIGHT/4 - 10 = 54
    let inlet_y = 54.0;
    let inlet_x_start = 20.0;
    let inlet_x_end = 100.0;

    let mut spawned = 0;
    let mut y = inlet_y;
    while spawned < target_particles {
        let mut x = inlet_x_start;
        while spawned < target_particles && x < inlet_x_end {
            sim.spawn_water(x, y, 50.0, 10.0, 1);  // Match game velocity
            spawned += 1;
            x += 0.8;
        }
        y += 0.8;
        if y > inlet_y + 60.0 {
            y = inlet_y;  // Wrap around to keep spawning
        }
    }

    // Warm up - let particles flow and settle
    for _ in 0..WARMUP {
        sim.update(DT);
        // Keep topping up particles that flow out
        while sim.particles.len() < target_particles * 8 / 10 {
            sim.spawn_water(30.0, inlet_y, 50.0, 10.0, 100);
        }
    }

    let actual_particles = sim.particles.len();
    println!("--- {} particles (target: {}) ---", actual_particles, target_particles);

    // Phase names for timing breakdown
    const PHASES: [&str; 7] = ["classify", "sdf", "p2g", "pressure", "g2p", "neighbor", "rest"];

    // Accumulate phase timings
    let mut phase_totals = [0.0f32; 7];
    let mut frame_times = Vec::with_capacity(FRAMES);

    for _ in 0..FRAMES {
        // Top up particles like real game
        if sim.particles.len() < target_particles * 7 / 10 {
            sim.spawn_water(30.0, inlet_y, 50.0, 10.0, 50);
        }

        let timings = sim.update_profiled(DT);
        let total: f32 = timings.iter().sum();
        frame_times.push(total as f64);

        for (i, &t) in timings.iter().enumerate() {
            phase_totals[i] += t;
        }
    }

    let final_particles = sim.particles.len();

    // Statistics
    frame_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg = frame_times.iter().sum::<f64>() / FRAMES as f64;
    let p95 = frame_times[FRAMES * 95 / 100];

    let fps = 1000.0 / avg;
    let status = if fps >= 60.0 { "✓" } else if fps >= 30.0 { "⚠" } else { "✗" };

    println!("  {} {:.1}ms ({:.0} FPS) | p95: {:.1}ms | particles: {}",
             status, avg, fps, p95, final_particles);

    // Phase breakdown
    print!("  phases:");
    for (i, name) in PHASES.iter().enumerate() {
        let avg_ms = phase_totals[i] / FRAMES as f32;
        let pct = (phase_totals[i] / phase_totals.iter().sum::<f32>()) * 100.0;
        if pct > 5.0 {
            print!(" {}={:.1}ms({:.0}%)", name, avg_ms, pct);
        }
    }
    println!("\n");
}
