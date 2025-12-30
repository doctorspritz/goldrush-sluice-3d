//! P2G Benchmark - measures isolated Particle-to-Grid transfer performance
//!
//! This benchmark tests CPU P2G performance at various particle counts to establish
//! a baseline for GPU optimization. Per CLAUDE.md: tests use realistic scales
//! (100k+ particles, 60+ seconds).
//!
//! Run with: cargo run --example p2g_benchmark --release -p sim

use sim::FlipSimulation;
use std::time::Instant;

// Test particle counts from 50K to 2M
const PARTICLE_COUNTS: [usize; 6] = [50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000];
const WARMUP_FRAMES: usize = 50;
const BENCHMARK_FRAMES: usize = 200;

fn main() {
    println!("=== P2G BENCHMARK ===");
    println!();
    println!("Testing CPU P2G transfer at various particle counts.");
    println!("Warmup: {} frames, Benchmark: {} frames", WARMUP_FRAMES, BENCHMARK_FRAMES);
    println!();
    println!("{:>10} | {:>8} | {:>10} | {:>12}",
             "Particles", "P2G (ms)", "Classify", "Throughput");
    println!("{:-<10}-+-{:-<8}-+-{:-<10}-+-{:-<12}", "", "", "", "");

    for &count in &PARTICLE_COUNTS {
        run_benchmark(count);
    }

    println!();
    println!("=== END P2G BENCHMARK ===");
}

fn run_benchmark(particle_count: usize) {
    // Size grid to have ~4 particles per cell
    let (sim_width, sim_height) = grid_size_for_particles(particle_count);
    let mut sim = FlipSimulation::new(sim_width, sim_height, 1.0);

    // No sluice geometry - just open grid for consistent benchmarking
    // Spawn particles in a uniform grid pattern
    spawn_particles_uniform(&mut sim, particle_count);

    let actual_count = sim.particles.len();

    // Warmup: let allocations settle, caches warm
    for _ in 0..WARMUP_FRAMES {
        sim.classify_cells();
        sim.particles_to_grid();
    }

    // Benchmark classify_cells (needed before P2G)
    let classify_start = Instant::now();
    for _ in 0..BENCHMARK_FRAMES {
        sim.classify_cells();
    }
    let classify_elapsed = classify_start.elapsed();
    let classify_avg_ms = classify_elapsed.as_secs_f64() * 1000.0 / BENCHMARK_FRAMES as f64;

    // Benchmark P2G only
    let p2g_start = Instant::now();
    for _ in 0..BENCHMARK_FRAMES {
        sim.particles_to_grid();
    }
    let p2g_elapsed = p2g_start.elapsed();
    let p2g_avg_ms = p2g_elapsed.as_secs_f64() * 1000.0 / BENCHMARK_FRAMES as f64;
    let throughput = actual_count as f64 / p2g_avg_ms / 1000.0;

    println!("{:>10} | {:>8.2} | {:>10.2} | {:>9.1} K/ms",
             actual_count, p2g_avg_ms, classify_avg_ms, throughput);
}

fn grid_size_for_particles(count: usize) -> (usize, usize) {
    // ~4 particles per cell
    let cells = count / 4;
    let side = (cells as f64).sqrt() as usize;
    // Minimum 256x128 for reasonable sluice geometry
    (side.max(256), (side / 2).max(128))
}

fn spawn_particles_uniform(sim: &mut FlipSimulation, target_count: usize) {
    let width = sim.grid.width as f32;
    let height = sim.grid.height as f32;
    let cell_size = sim.grid.cell_size;

    // Avoid solid cells at edges
    let margin = 10.0 * cell_size;
    let usable_width = (width * cell_size) - 2.0 * margin;
    let usable_height = (height * cell_size) - 2.0 * margin;

    // Calculate grid dimensions for uniform distribution
    let aspect = (usable_width / usable_height) as f64;
    let rows = ((target_count as f64 / aspect).sqrt()) as usize;
    let cols = (target_count / rows).max(1);

    let dx = usable_width / cols as f32;
    let dy = usable_height / rows as f32;

    let mut spawned = 0;
    for row in 0..rows {
        for col in 0..cols {
            if spawned >= target_count {
                break;
            }
            let x = margin + (col as f32 + 0.5) * dx;
            let y = margin + (row as f32 + 0.5) * dy;

            // Spawn with some initial velocity
            sim.spawn_water(x, y, 50.0, 0.0, 1);
            spawned += 1;
        }
    }
}
