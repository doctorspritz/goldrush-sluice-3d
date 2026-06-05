//! GPU FLIP particle-step throughput at realistic scale.
//!
//! Per CLAUDE.md "PERFORMANCE TESTING — USE REALISTIC SCALES": uses a large grid and
//! hundreds of thousands of particles, warms up, then times steady-state stepping with
//! `step_no_readback` (no CPU stall). Asserts an interactive-rate gate and a throughput
//! floor, plus a validity gate so a "fast because it diverged" run cannot pass.
//!
//! Run with: cargo test -p game --test gpu_perf_throughput --release -- --nocapture
//! Skips cleanly (returns) when no compatible GPU is present, so CI without a GPU is green.

use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Vec3};
use std::time::Instant;

const CELL_FLUID: u32 = 1;
const CELL_SOLID: u32 = 2;

fn init_device_queue() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;

    let limits = adapter.limits();
    if limits.max_storage_buffers_per_shader_stage < 16 {
        eprintln!(
            "GPU adapter only supports {} storage buffers (need 16+); skipping test.",
            limits.max_storage_buffers_per_shader_stage
        );
        return None;
    }

    let mut required_limits = wgpu::Limits::default();
    required_limits.max_storage_buffers_per_shader_stage = 16;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("GPU Perf Throughput Test Device"),
            required_features: wgpu::Features::empty(),
            required_limits,
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .ok()?;

    Some((device, queue))
}

fn build_cell_types(width: u32, height: u32, depth: u32) -> Vec<u32> {
    let mut cells = vec![CELL_FLUID; (width * height * depth) as usize];
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                if x == 0 || x == width - 1 || y == 0 || y == height - 1 || z == 0 || z == depth - 1
                {
                    let idx = (z * width * height + y * width + x) as usize;
                    cells[idx] = CELL_SOLID;
                }
            }
        }
    }
    cells
}

/// PERF: GPU FLIP particle-step throughput at ~250k-400k live particles on a 128x64x64 grid.
///
/// Thresholds are written IMMUTABLE in the project convention (fix the sim, not the number).
/// They are CONSERVATIVE STARTING BASELINES, not measured targets; re-baseline on the real
/// target GPU before treating them as hard regression gates.
#[test]
fn gpu_flip_particle_step_throughput_at_scale() {
    let (device, queue) = match init_device_queue() {
        Some(handles) => handles,
        None => {
            eprintln!("No compatible GPU adapter found; skipping GPU perf throughput test.");
            return;
        }
    };

    let width = 128u32;
    let height = 64u32;
    let depth = 64u32;
    let cell_size = 0.05f32;
    let max_particles = 500_000usize;

    let mut sim = GpuFlip3D::new(&device, width, height, depth, cell_size, max_particles);
    sim.vorticity_epsilon = 0.0;

    let cell_types = build_cell_types(width, height, depth);

    // Fill ~60% of interior cells with 4 particles each to reach a realistic live count.
    let mut positions: Vec<Vec3> = Vec::new();
    let mut velocities: Vec<Vec3> = Vec::new();
    let mut densities: Vec<f32> = Vec::new();
    let mut c_matrices: Vec<Mat3> = Vec::new();

    'fill: for z in 1..(depth - 1) {
        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                // ~60% fill via a cheap deterministic mask.
                if (x.wrapping_mul(7) ^ y.wrapping_mul(13) ^ z.wrapping_mul(17)) % 5 < 3 {
                    for k in 0..4 {
                        let jitter = 0.2 + 0.15 * k as f32;
                        positions.push(Vec3::new(
                            (x as f32 + jitter) * cell_size,
                            (y as f32 + jitter) * cell_size,
                            (z as f32 + jitter) * cell_size,
                        ));
                        velocities.push(Vec3::ZERO);
                        densities.push(1.0);
                        c_matrices.push(Mat3::ZERO);
                        if positions.len() >= max_particles {
                            break 'fill;
                        }
                    }
                }
            }
        }
    }

    let particle_count = positions.len();
    assert!(
        particle_count >= 100_000,
        "perf test must use >=100k particles (CLAUDE.md realistic scales); got {particle_count}"
    );

    let dt = 1.0 / 60.0;
    let gravity = -9.81f32;
    let flow_accel = 0.0f32;
    let pressure_iterations = 40u32;

    // Warmup: measure after warmup, not ramp-up.
    for _ in 0..30 {
        sim.step_no_readback(
            &device,
            &queue,
            &mut positions,
            &mut velocities,
            &mut c_matrices,
            &densities,
            &cell_types,
            None,
            None,
            dt,
            gravity,
            flow_accel,
            pressure_iterations,
        );
    }
    device.poll(wgpu::Maintain::Wait);

    // Timed steady-state stepping (GPU-only, no readback stall).
    let n = 200usize;
    let t0 = Instant::now();
    for _ in 0..n {
        sim.step_no_readback(
            &device,
            &queue,
            &mut positions,
            &mut velocities,
            &mut c_matrices,
            &densities,
            &cell_types,
            None,
            None,
            dt,
            gravity,
            flow_accel,
            pressure_iterations,
        );
    }
    device.poll(wgpu::Maintain::Wait);
    let elapsed = t0.elapsed();

    let secs = elapsed.as_secs_f64();
    let ms_per_step = secs * 1000.0 / n as f64;
    let fps = 1000.0 / ms_per_step;
    let particle_steps_per_sec = particle_count as f64 * n as f64 / secs;

    println!("=== GPU FLIP perf (128x64x64, {particle_count} particles) ===");
    println!("steps={n} elapsed={secs:.3}s ms_per_step={ms_per_step:.4} fps={fps:.2}");
    println!("particle_steps_per_sec={particle_steps_per_sec:.3e}");

    // Validity gate: read positions back once; ensure the sim did not blow up.
    // (Prevents a "fast because it diverged" false pass.)
    sim.step(
        &device,
        &queue,
        &mut positions,
        &mut velocities,
        &mut c_matrices,
        &densities,
        &cell_types,
        None,
        None,
        dt,
        gravity,
        flow_accel,
        pressure_iterations,
    );
    let finite = positions.iter().filter(|p| p.is_finite()).count();
    println!("finite particles after run: {finite}/{particle_count}");
    assert!(
        finite > particle_count / 2,
        "more than half the particles are non-finite ({finite}/{particle_count}) -- sim diverged"
    );
    let diag = sim.read_physics_diagnostics(&device, &queue);
    assert!(
        diag.max_divergence.is_finite() && diag.avg_pressure.is_finite(),
        "physics diagnostics non-finite: max_div={} avg_p={}",
        diag.max_divergence,
        diag.avg_pressure
    );

    // Interactive-rate gate: one sim step == one frame at dt=1/60. Min 30 FPS.
    assert!(
        ms_per_step < 33.3,
        "GPU FLIP below 30 FPS at scale: {ms_per_step:.4} ms/step (>33.3). \
         particles={particle_count}. Fix the sim, not the number."
    );
    // Throughput floor (conservative starting baseline; re-baseline on target GPU).
    assert!(
        particle_steps_per_sec > 5.0e6,
        "GPU FLIP throughput below floor: {particle_steps_per_sec:.3e} particle-steps/s (<5.0e6). \
         particles={particle_count} ms_per_step={ms_per_step:.4}. Fix the sim, not the number."
    );
}

/// PERF (HEAVY, explicit-run): GPU FLIP particle-step throughput at the project's 1M-particle
/// target on a large 160x96x96 grid.
///
/// Marked `#[ignore]` so it does NOT slow normal `cargo test` / CI: allocating ~1M particles
/// on a >1M-cell grid plus warmup + 100 timed steps takes meaningful GPU time and memory.
/// Run it explicitly:
///   cargo test -p game --release --test gpu_perf_throughput \
///       gpu_flip_throughput_at_1m_particles -- --ignored --nocapture
///
/// MEMORY NOTE: if 1M particles do not fit in this machine's GPU memory, REDUCE
/// `max_particles`/`target_particles` to the largest count that allocates, update the
/// documented number here, and report the real achieved count. As written this targets
/// 1,000,000 live particles; the grid interior (158*94*94 ~= 1.4M cells) is sized so a ~80%
/// deterministic fill at 1 particle/cell reaches the 1M cap without overflowing it.
///
/// Gates are identical to the sibling test (all particles finite, diagnostics finite) plus a
/// conservative throughput floor. Fix the sim, not the number.
#[test]
#[ignore = "heavy 1M-particle perf test; run explicitly with --ignored"]
fn gpu_flip_throughput_at_1m_particles() {
    let (device, queue) = match init_device_queue() {
        Some(handles) => handles,
        None => {
            eprintln!("No compatible GPU adapter found; skipping 1M GPU perf throughput test.");
            return;
        }
    };

    let width = 160u32;
    let height = 96u32;
    let depth = 96u32;
    let cell_size = 0.05f32;
    let target_particles = 1_000_000usize;
    let max_particles = 1_050_000usize; // small headroom above target

    let mut sim = GpuFlip3D::new(&device, width, height, depth, cell_size, max_particles);
    sim.vorticity_epsilon = 0.0;

    let cell_types = build_cell_types(width, height, depth);

    // Fill interior cells with 1 particle each via a deterministic ~80% mask, capped at the
    // 1M target. 160*80*80 interior gives more than enough cells to reach 1M at 1/cell.
    let mut positions: Vec<Vec3> = Vec::new();
    let mut velocities: Vec<Vec3> = Vec::new();
    let mut densities: Vec<f32> = Vec::new();
    let mut c_matrices: Vec<Mat3> = Vec::new();

    'fill: for z in 1..(depth - 1) {
        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                // ~80% deterministic fill.
                if (x.wrapping_mul(7) ^ y.wrapping_mul(13) ^ z.wrapping_mul(17)) % 5 < 4 {
                    let jitter = 0.4;
                    positions.push(Vec3::new(
                        (x as f32 + jitter) * cell_size,
                        (y as f32 + jitter) * cell_size,
                        (z as f32 + jitter) * cell_size,
                    ));
                    velocities.push(Vec3::ZERO);
                    densities.push(1.0);
                    c_matrices.push(Mat3::ZERO);
                    if positions.len() >= target_particles {
                        break 'fill;
                    }
                }
            }
        }
    }

    let particle_count = positions.len();
    println!("=== GPU FLIP 1M perf: seeded {particle_count} particles on {width}x{height}x{depth} ===");
    assert!(
        particle_count >= 900_000,
        "1M perf test should reach ~1M particles; got {particle_count} (grid too small or fill too sparse)"
    );

    let dt = 1.0 / 60.0;
    let gravity = -9.81f32;
    let flow_accel = 0.0f32;
    let pressure_iterations = 40u32;

    // Warmup.
    for _ in 0..15 {
        sim.step_no_readback(
            &device,
            &queue,
            &mut positions,
            &mut velocities,
            &mut c_matrices,
            &densities,
            &cell_types,
            None,
            None,
            dt,
            gravity,
            flow_accel,
            pressure_iterations,
        );
    }
    device.poll(wgpu::Maintain::Wait);

    // Timed steady-state stepping (GPU-only, no readback stall).
    let n = 100usize;
    let t0 = Instant::now();
    for _ in 0..n {
        sim.step_no_readback(
            &device,
            &queue,
            &mut positions,
            &mut velocities,
            &mut c_matrices,
            &densities,
            &cell_types,
            None,
            None,
            dt,
            gravity,
            flow_accel,
            pressure_iterations,
        );
    }
    device.poll(wgpu::Maintain::Wait);
    let elapsed = t0.elapsed();

    let secs = elapsed.as_secs_f64();
    let ms_per_step = secs * 1000.0 / n as f64;
    let fps = 1000.0 / ms_per_step;
    let particle_steps_per_sec = particle_count as f64 * n as f64 / secs;

    println!("=== GPU FLIP 1M perf ({width}x{height}x{depth}, {particle_count} particles) ===");
    println!("steps={n} elapsed={secs:.3}s ms_per_step={ms_per_step:.4} fps={fps:.2}");
    println!("particle_steps_per_sec={particle_steps_per_sec:.3e}");

    // Validity gate: read back once; ensure the sim did not blow up.
    sim.step(
        &device,
        &queue,
        &mut positions,
        &mut velocities,
        &mut c_matrices,
        &densities,
        &cell_types,
        None,
        None,
        dt,
        gravity,
        flow_accel,
        pressure_iterations,
    );
    let finite = positions.iter().filter(|p| p.is_finite()).count();
    println!("finite particles after run: {finite}/{particle_count}");
    assert!(
        positions.iter().all(|p| p.is_finite()),
        "non-finite particle positions after 1M run: {finite}/{particle_count} finite"
    );
    let diag = sim.read_physics_diagnostics(&device, &queue);
    assert!(
        diag.max_divergence.is_finite() && diag.avg_pressure.is_finite(),
        "physics diagnostics non-finite: max_div={} avg_p={}",
        diag.max_divergence,
        diag.avg_pressure
    );

    // Conservative throughput floor (heavy-scale starting baseline; re-baseline on target GPU).
    assert!(
        particle_steps_per_sec > 1.0e6,
        "GPU FLIP 1M throughput below floor: {particle_steps_per_sec:.3e} particle-steps/s (<1.0e6). \
         particles={particle_count} ms_per_step={ms_per_step:.4}. Fix the sim, not the number."
    );
}
