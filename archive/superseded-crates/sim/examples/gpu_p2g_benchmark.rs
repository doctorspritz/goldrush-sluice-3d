//! GPU P2G Benchmark - measures GPU Particle-to-Grid transfer performance
//!
//! Compares GPU P2G against CPU baseline to validate correctness and measure speedup.
//! Uses fixed-point atomic scatter approach (SCALE=10^6).
//!
//! Run with: cargo run --example gpu_p2g_benchmark --release -p game

use std::time::Instant;

use glam::Vec2;
use pollster::FutureExt;
use sim::FlipSimulation;

use game::gpu::p2g::GpuP2gSolver;

// Test particle counts
const PARTICLE_COUNTS: [usize; 5] = [50_000, 100_000, 200_000, 500_000, 1_000_000];
const WARMUP_ITERS: usize = 5;
const BENCHMARK_ITERS: usize = 20;

// Set to true for detailed correctness debugging
const DEBUG_CORRECTNESS: bool = true;

fn main() {
    env_logger::init();

    println!("=== GPU P2G BENCHMARK ===");
    println!();
    println!("Testing GPU P2G transfer with fixed-point atomic scatter.");
    println!("Comparing against CPU implementation for correctness and speedup.");
    println!();

    // Create headless GPU context
    let (device, queue) = create_headless_gpu();
    println!();

    println!("{:>10} | {:>8} | {:>8} | {:>10} | {:>8} | {:>8}",
             "Particles", "CPU (ms)", "GPU (ms)", "Speedup", "Max Δ", "Avg Δ");
    println!("{:-<10}-+-{:-<8}-+-{:-<8}-+-{:-<10}-+-{:-<8}-+-{:-<8}", "", "", "", "", "", "");

    // First run a single correctness test with detailed output
    if DEBUG_CORRECTNESS {
        run_correctness_test(&device, &queue);
    }

    for &count in &PARTICLE_COUNTS {
        run_comparison(&device, &queue, count);
    }

    println!();
    println!("=== END GPU P2G BENCHMARK ===");
}

fn create_headless_gpu() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .block_on()
        .expect("Failed to find GPU adapter");

    println!("Using GPU: {:?}", adapter.get_info());

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Headless Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .block_on()
        .expect("Failed to create device");

    (device, queue)
}

fn run_comparison(device: &wgpu::Device, queue: &wgpu::Queue, particle_count: usize) {
    // Size grid to have ~4 particles per cell
    let (sim_width, sim_height) = grid_size_for_particles(particle_count);
    let cell_size = 1.0f32;

    // Create simulation and spawn particles
    let mut sim = FlipSimulation::new(sim_width, sim_height, cell_size);
    spawn_particles_uniform(&mut sim, particle_count);
    let actual_count = sim.particles.len();

    // DEBUG: For small test, print grid size and particle bounds
    if actual_count < 60000 && DEBUG_CORRECTNESS {
        let min_x = sim.particles.list.iter().map(|p| p.position.x).fold(f32::INFINITY, f32::min);
        let max_x = sim.particles.list.iter().map(|p| p.position.x).fold(f32::NEG_INFINITY, f32::max);
        let min_y = sim.particles.list.iter().map(|p| p.position.y).fold(f32::INFINITY, f32::min);
        let max_y = sim.particles.list.iter().map(|p| p.position.y).fold(f32::NEG_INFINITY, f32::max);
        eprintln!("  Grid: {}x{}, particles: {}", sim_width, sim_height, actual_count);
        eprintln!("  Particle bounds: x=[{:.1}, {:.1}], y=[{:.1}, {:.1}]", min_x, max_x, min_y, max_y);
        // For U grid, base_j for bottom-most particles
        let u_base_j_min = ((min_y / cell_size) - 0.5).floor() as i32;
        eprintln!("  U grid base_j for min_y: {} (touches j={}..{})",
                 u_base_j_min, u_base_j_min - 1, u_base_j_min + 1);
    }

    // Create GPU P2G solver
    let gpu_solver = GpuP2gSolver::new_headless(
        device,
        queue,
        sim_width as u32,
        sim_height as u32,
        actual_count,
    );

    // Prepare output buffers
    let u_size = (sim_width + 1) * sim_height;
    let v_size = sim_width * (sim_height + 1);
    let mut cpu_u = vec![0.0f32; u_size];
    let mut cpu_v = vec![0.0f32; v_size];
    let mut gpu_u = vec![0.0f32; u_size];
    let mut gpu_v = vec![0.0f32; v_size];

    // Warmup CPU
    for _ in 0..WARMUP_ITERS {
        sim.classify_cells();
        sim.particles_to_grid();
    }

    // Benchmark CPU
    let cpu_start = Instant::now();
    for _ in 0..BENCHMARK_ITERS {
        sim.classify_cells();
        sim.particles_to_grid();
    }
    let cpu_elapsed = cpu_start.elapsed();
    let cpu_avg_ms = cpu_elapsed.as_secs_f64() * 1000.0 / BENCHMARK_ITERS as f64;

    // Get CPU results
    cpu_u.copy_from_slice(&sim.grid.u);
    cpu_v.copy_from_slice(&sim.grid.v);

    // Warmup GPU
    for _ in 0..WARMUP_ITERS {
        let count = gpu_solver.upload_particles_headless(queue, &sim.particles.list, cell_size);
        gpu_solver.compute_headless(device, queue, count);
    }

    // Benchmark GPU
    let gpu_start = Instant::now();
    for _ in 0..BENCHMARK_ITERS {
        gpu_solver.execute_headless(
            device,
            queue,
            &sim.particles.list,
            cell_size,
            &mut gpu_u,
            &mut gpu_v,
        );
    }
    let gpu_elapsed = gpu_start.elapsed();
    let gpu_avg_ms = gpu_elapsed.as_secs_f64() * 1000.0 / BENCHMARK_ITERS as f64;

    // Calculate difference (correctness check)
    let (max_diff_u, avg_diff_u) = calculate_diff(&cpu_u, &gpu_u, sim_width, "U");
    let (max_diff_v, avg_diff_v) = calculate_diff(&cpu_v, &gpu_v, sim_width, "V");
    let max_diff = max_diff_u.max(max_diff_v);
    let avg_diff = (avg_diff_u + avg_diff_v) / 2.0;

    let speedup = cpu_avg_ms / gpu_avg_ms;

    println!("{:>10} | {:>8.2} | {:>8.2} | {:>10.1}x | {:>8.4} | {:>8.6}",
             actual_count, cpu_avg_ms, gpu_avg_ms, speedup, max_diff, avg_diff);
}

fn grid_size_for_particles(count: usize) -> (usize, usize) {
    // ~4 particles per cell
    let cells = count / 4;
    let side = (cells as f64).sqrt() as usize;
    (side.max(256), (side / 2).max(128))
}

/// Run a single small test to verify correctness
fn run_correctness_test(device: &wgpu::Device, queue: &wgpu::Queue) {
    eprintln!("\n=== CORRECTNESS TEST (SINGLE PARTICLE) ===");

    // Small grid for easier debugging
    let sim_width = 32usize;
    let sim_height = 16usize;
    let cell_size = 1.0f32;

    let mut sim = FlipSimulation::new(sim_width, sim_height, cell_size);

    // Spawn exactly ONE particle at a known location with NO randomness
    // We'll manually add to bypass the jitter
    sim.particles.list.push(sim::Particle::water(
        Vec2::new(10.5, 8.5),   // Center of cell (10, 8)
        Vec2::new(50.0, 10.0),  // Velocity
    ));

    let particle_count = sim.particles.len();
    eprintln!("  Spawned {} particle", particle_count);

    // Print first few particles
    for (i, p) in sim.particles.list.iter().take(3).enumerate() {
        eprintln!("  Particle {}: pos=({:.2}, {:.2}) vel=({:.2}, {:.2}) C=[{:.4}, {:.4}; {:.4}, {:.4}]",
                 i, p.position.x, p.position.y, p.velocity.x, p.velocity.y,
                 p.affine_velocity.x_axis.x, p.affine_velocity.x_axis.y,
                 p.affine_velocity.y_axis.x, p.affine_velocity.y_axis.y);
    }

    // Run CPU P2G
    sim.classify_cells();
    sim.particles_to_grid();

    let u_size = (sim_width + 1) * sim_height;
    let v_size = sim_width * (sim_height + 1);
    let mut cpu_u = vec![0.0f32; u_size];
    let mut cpu_v = vec![0.0f32; v_size];
    cpu_u.copy_from_slice(&sim.grid.u);
    cpu_v.copy_from_slice(&sim.grid.v);

    // For single-particle test, print ALL non-zero values
    let cpu_u_nonzero: Vec<_> = cpu_u.iter().enumerate()
        .filter(|(_, &v)| v.abs() > 0.0001)
        .collect();
    eprintln!("  CPU U non-zero count: {}", cpu_u_nonzero.len());
    for &(idx, &val) in &cpu_u_nonzero {
        let row = idx / (sim_width + 1);
        let col = idx % (sim_width + 1);
        eprintln!("    U[{},{}] idx={}: {:.6}", col, row, idx, val);
    }

    // Create GPU solver and run
    let gpu_solver = GpuP2gSolver::new_headless(device, queue, sim_width as u32, sim_height as u32, particle_count);

    let mut gpu_u = vec![0.0f32; u_size];
    let mut gpu_v = vec![0.0f32; v_size];

    // Debug: verify particles are uploaded correctly by checking if GPU sees similar counts
    let upload_count = gpu_solver.upload_particles_headless(queue, &sim.particles.list, cell_size);
    eprintln!("  Uploaded {} particles to GPU", upload_count);
    gpu_solver.compute_headless(device, queue, upload_count);
    gpu_solver.download_headless(device, queue, &mut gpu_u, &mut gpu_v);

    // For single-particle test, print ALL non-zero GPU values
    let gpu_u_nonzero: Vec<_> = gpu_u.iter().enumerate()
        .filter(|(_, &v)| v.abs() > 0.0001)
        .collect();
    eprintln!("  GPU U non-zero count: {}", gpu_u_nonzero.len());
    for &(idx, &val) in &gpu_u_nonzero {
        let row = idx / (sim_width + 1);
        let col = idx % (sim_width + 1);
        eprintln!("    U[{},{}] idx={}: {:.6}", col, row, idx, val);
    }

    // Compare at specific indices
    for &(idx, &cpu_val) in &cpu_u_nonzero {
        let gpu_val = gpu_u[idx];
        let diff = (cpu_val - gpu_val).abs();
        if diff > 0.1 {
            let row = idx / (sim_width + 1);
            let col = idx % (sim_width + 1);
            eprintln!("  DIFF at U[{},{}] idx={}: cpu={:.4} gpu={:.4} diff={:.4}",
                     col, row, idx, cpu_val, gpu_val, diff);
        }
    }

    // Overall max diff
    let max_u_diff = cpu_u.iter().zip(gpu_u.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_v_diff = cpu_v.iter().zip(gpu_v.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    eprintln!("  Max U diff: {:.4}, Max V diff: {:.4}", max_u_diff, max_v_diff);

    if max_u_diff < 0.01 && max_v_diff < 0.01 {
        eprintln!("  PASS: GPU matches CPU!");
    } else {
        eprintln!("  FAIL: Significant difference detected");
    }
    eprintln!("=== END CORRECTNESS TEST ===\n");
}

fn spawn_particles_uniform(sim: &mut FlipSimulation, target_count: usize) {
    let width = sim.grid.width as f32;
    let height = sim.grid.height as f32;
    let cell_size = sim.grid.cell_size;

    // Avoid edges
    let margin = 10.0 * cell_size;
    let usable_width = (width * cell_size) - 2.0 * margin;
    let usable_height = (height * cell_size) - 2.0 * margin;

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
            sim.spawn_water(x, y, 50.0, 10.0, 1);
            spawned += 1;
        }
    }
}

fn calculate_diff(a: &[f32], b: &[f32], width: usize, name: &str) -> (f32, f32) {
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f64;
    let mut count = 0;
    let mut max_idx = 0;

    // Count non-zero values for debugging
    let cpu_nonzero = a.iter().filter(|&&v| v.abs() > 0.001).count();
    let gpu_nonzero = b.iter().filter(|&&v| v.abs() > 0.001).count();

    for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (av - bv).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
        sum_diff += diff as f64;
        count += 1;
    }

    // Debug: print info about max difference location
    if max_diff > 1.0 && DEBUG_CORRECTNESS {
        // For U grid, stride is width+1; for V grid, stride is width
        let stride = if name == "U" { width + 1 } else { width };
        let row = max_idx / stride;
        let col = max_idx % stride;
        eprintln!("  {} max diff at ({},{}) idx {}: cpu={:.4} gpu={:.4} diff={:.4}",
                 name, col, row, max_idx, a[max_idx], b[max_idx], max_diff);
        eprintln!("    CPU non-zero: {}/{}, GPU non-zero: {}/{}",
                 cpu_nonzero, a.len(), gpu_nonzero, b.len());
    }

    let avg_diff = if count > 0 { sum_diff / count as f64 } else { 0.0 };
    (max_diff, avg_diff as f32)
}
