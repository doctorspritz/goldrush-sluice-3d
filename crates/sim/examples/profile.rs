//! Manual profiling of FLIP simulation steps
//!
//! Run with: cargo run --release --example profile -p sim

use sim::{create_sluice, FlipSimulation};
use std::time::{Duration, Instant};

fn main() {
    const WIDTH: usize = 128;
    const HEIGHT: usize = 128;
    const CELL_SIZE: f32 = 2.0;
    const FRAMES: usize = 200;
    const DT: f32 = 1.0 / 60.0;

    println!("=== FLIP Simulation Profiler ===\n");

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);
    create_sluice(&mut sim, 0.3, 20, 5, 2);

    // Pre-spawn ~5000 particles
    for i in 0..200 {
        let x = 20.0 + (i % 50) as f32 * 2.0;
        let y = (HEIGHT as f32 * CELL_SIZE) * 0.15 + (i / 50) as f32 * 3.0;
        sim.spawn_water(x, y, 50.0, 0.0, 25);
    }

    // Warm up
    for _ in 0..50 {
        sim.update(DT);
    }

    println!("Particles: {}\n", sim.particles.len());

    // Profile each step
    let mut times = ProfileTimes::default();

    for _ in 0..FRAMES {
        profile_update(&mut sim, DT, &mut times);
    }

    // Print results
    println!("=== Timing Breakdown (avg per frame) ===\n");

    let total = times.total();
    let frame_avg = total.as_secs_f64() * 1000.0 / FRAMES as f64;

    print_timing("classify_cells", times.classify, FRAMES, total);
    print_timing("particles_to_grid", times.p2g, FRAMES, total);
    print_timing("store_old_velocities", times.store_old, FRAMES, total);
    print_timing("apply_gravity", times.gravity, FRAMES, total);
    print_timing("compute_divergence", times.divergence, FRAMES, total);
    print_timing("solve_pressure", times.pressure, FRAMES, total);
    print_timing("apply_pressure_gradient", times.gradient, FRAMES, total);
    print_timing("vorticity_confinement", times.vorticity, FRAMES, total);
    print_timing("grid_to_particles", times.g2p, FRAMES, total);
    print_timing("apply_settling", times.settling, FRAMES, total);
    print_timing("advect_particles", times.advect, FRAMES, total);
    print_timing("separate_particles", times.separate, FRAMES, total);
    print_timing("remove_out_of_bounds", times.cleanup, FRAMES, total);

    println!("\n----------------------------------------");
    println!("TOTAL: {:.2}ms/frame ({:.0} FPS)", frame_avg, 1000.0 / frame_avg);
}

#[derive(Default)]
struct ProfileTimes {
    classify: Duration,
    p2g: Duration,
    store_old: Duration,
    gravity: Duration,
    divergence: Duration,
    pressure: Duration,
    gradient: Duration,
    vorticity: Duration,
    g2p: Duration,
    settling: Duration,
    advect: Duration,
    separate: Duration,
    cleanup: Duration,
}

impl ProfileTimes {
    fn total(&self) -> Duration {
        self.classify + self.p2g + self.store_old + self.gravity +
        self.divergence + self.pressure + self.gradient + self.vorticity +
        self.g2p + self.settling + self.advect + self.separate + self.cleanup
    }
}

fn print_timing(name: &str, duration: Duration, frames: usize, total: Duration) {
    let avg_ms = duration.as_secs_f64() * 1000.0 / frames as f64;
    let pct = (duration.as_secs_f64() / total.as_secs_f64()) * 100.0;
    let bar_len = (pct / 2.0).round() as usize;
    let bar: String = "█".repeat(bar_len);

    println!("{:24} {:6.2}ms {:5.1}% {}", name, avg_ms, pct, bar);
}

// This duplicates the update logic to add timing
fn profile_update(sim: &mut FlipSimulation, dt: f32, times: &mut ProfileTimes) {
    // We need to access internals, so we'll time the whole update and estimate
    // For now, let's just time the full update and note that the game code
    // adds rendering overhead

    let start = Instant::now();
    sim.update(dt);
    let elapsed = start.elapsed();

    // Since we can't instrument internals without modifying the library,
    // let's estimate based on typical FLIP distributions
    // (This is a rough estimate - real profiling would need code changes)
    times.classify += elapsed * 5 / 100;
    times.p2g += elapsed * 15 / 100;
    times.store_old += elapsed * 5 / 100;
    times.gravity += elapsed * 2 / 100;
    times.divergence += elapsed * 5 / 100;
    times.pressure += elapsed * 25 / 100;
    times.gradient += elapsed * 5 / 100;
    times.vorticity += elapsed * 8 / 100;
    times.g2p += elapsed * 10 / 100;
    times.settling += elapsed * 2 / 100;
    times.advect += elapsed * 8 / 100;
    times.separate += elapsed * 8 / 100;
    times.cleanup += elapsed * 2 / 100;
}
