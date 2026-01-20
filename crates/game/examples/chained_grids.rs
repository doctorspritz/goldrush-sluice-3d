//! Chained Grids Test - Two FLIP grids with ghost particle handoff
//!
//! Tests domain decomposition with velocity preservation:
//! - Grid A (upstream) receives inlet flow
//! - Particles exiting Grid A → enter Grid B with ghost particles
//! - Ghost particles establish velocity field, then are removed
//! - Measures velocity preservation and mass conservation
//!
//! Run with: cargo run --example chained_grids --release

use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Vec3};

// Grid dimensions
const GRID_WIDTH: usize = 16;
const GRID_HEIGHT: usize = 10;
const GRID_DEPTH: usize = 8;
const CELL_SIZE: f32 = 0.04;
const MAX_PARTICLES: usize = 50000;

// Weir height (particles must be above this to exit)
const OUTFLOW_MIN_Y: f32 = (GRID_HEIGHT as f32 * 0.5) * CELL_SIZE;

// Simulation parameters
const DT: f32 = 1.0 / 120.0;
const SUBSTEPS: u32 = 2;
const PRESSURE_ITERS: u32 = 50;
const GRAVITY: f32 = -9.8;
const FLOW_ACCEL: f32 = 0.0; // Zero for accurate conservation measurement

// Inlet parameters
const INLET_RATE: usize = 30;
const INLET_VELOCITY: f32 = 0.6;

// Test duration
const TEST_FRAMES: u32 = 600;
const WARMUP_FRAMES: u32 = 200; // Frames before measuring steady-state

// Ghost particle settings
const GHOST_COUNT: usize = 3; // Number of ghost particles per handoff

/// Metrics for transfer quality
#[derive(Default)]
struct TransferMetrics {
    // Counts
    total_injected: usize,
    total_handoffs: usize,
    total_exited: usize,
    total_oob_removed: usize,

    // Velocity tracking at exit from Grid B (measures true preservation)
    velocity_samples: usize,
    sum_speed_at_b_exit: f32,
    sum_speed_at_a_exit: f32,
}

impl TransferMetrics {
    fn record_a_exit(&mut self, velocity: Vec3) {
        self.sum_speed_at_a_exit += velocity.length();
    }

    fn record_b_exit(&mut self, velocity: Vec3) {
        self.velocity_samples += 1;
        self.sum_speed_at_b_exit += velocity.length();
    }

    fn velocity_preservation_ratio(&self) -> f32 {
        if self.velocity_samples == 0 || self.sum_speed_at_a_exit == 0.0 {
            return 0.0;
        }
        // Compare average speed at B exit vs A exit
        let avg_speed_a = self.sum_speed_at_a_exit / self.total_handoffs as f32;
        let avg_speed_b = self.sum_speed_at_b_exit / self.velocity_samples as f32;
        avg_speed_b / avg_speed_a
    }

    fn avg_speed_at_a_exit(&self) -> f32 {
        if self.total_handoffs == 0 { 0.0 }
        else { self.sum_speed_at_a_exit / self.total_handoffs as f32 }
    }

    fn avg_speed_at_b_exit(&self) -> f32 {
        if self.velocity_samples == 0 { 0.0 }
        else { self.sum_speed_at_b_exit / self.velocity_samples as f32 }
    }

    fn throughput_ratio(&self) -> f32 {
        if self.total_injected == 0 { 0.0 }
        else { self.total_exited as f32 / self.total_injected as f32 }
    }

    fn handoff_ratio(&self) -> f32 {
        if self.total_injected == 0 { 0.0 }
        else { self.total_handoffs as f32 / self.total_injected as f32 }
    }
}

/// Particle data for handoff (full state)
struct HandoffParticle {
    world_pos: Vec3,
    velocity: Vec3,
    c_matrix: Mat3,
    density: f32,
}

/// A single FLIP grid segment
struct GridSegment {
    gpu: GpuFlip3D,
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    c_matrices: Vec<Mat3>,
    densities: Vec<f32>,
    is_ghost: Vec<bool>,
    cell_types: Vec<u32>,
    world_offset: Vec3,
    input_buffer: Vec<(Vec3, Vec3, Mat3, f32, bool)>, // pos, vel, c_mat, density, is_ghost
}

impl GridSegment {
    fn new(device: &wgpu::Device, world_offset: Vec3) -> Self {
        let mut gpu = GpuFlip3D::new(
            device,
            GRID_WIDTH as u32,
            GRID_HEIGHT as u32,
            GRID_DEPTH as u32,
            CELL_SIZE,
            MAX_PARTICLES,
        );
        gpu.vorticity_epsilon = 0.0;
        gpu.open_boundaries = 2; // +X open

        // Initialize cell types with boundaries
        let mut cell_types = vec![0u32; GRID_WIDTH * GRID_HEIGHT * GRID_DEPTH];
        for z in 0..GRID_DEPTH {
            for y in 0..GRID_HEIGHT {
                for x in 0..GRID_WIDTH {
                    let idx = z * GRID_WIDTH * GRID_HEIGHT + y * GRID_WIDTH + x;
                    let is_floor = y == 0;
                    let is_ceiling = y == GRID_HEIGHT - 1;
                    let is_z_wall = z == 0 || z == GRID_DEPTH - 1;
                    let is_inlet = x == 0;
                    if is_floor || is_ceiling || is_z_wall || is_inlet {
                        cell_types[idx] = 2; // SOLID
                    }
                }
            }
        }

        Self {
            gpu,
            positions: Vec::new(),
            velocities: Vec::new(),
            c_matrices: Vec::new(),
            densities: Vec::new(),
            is_ghost: Vec::new(),
            cell_types,
            world_offset,
            input_buffer: Vec::new(),
        }
    }

    fn particle_count(&self) -> usize {
        self.positions.iter().zip(&self.is_ghost).filter(|(_, &g)| !g).count()
    }

    fn total_count(&self) -> usize {
        self.positions.len()
    }

    fn inject(&mut self, pos: Vec3, vel: Vec3) {
        self.positions.push(pos);
        self.velocities.push(vel);
        self.c_matrices.push(Mat3::ZERO);
        self.densities.push(1.0);
        self.is_ghost.push(false);
    }

    fn inject_full(&mut self, pos: Vec3, vel: Vec3, c_mat: Mat3, density: f32, is_ghost: bool) {
        self.positions.push(pos);
        self.velocities.push(vel);
        self.c_matrices.push(c_mat);
        self.densities.push(density);
        self.is_ghost.push(is_ghost);
    }

    fn process_input(&mut self) {
        let buf: Vec<_> = self.input_buffer.drain(..).collect();
        for (p, v, c, d, ghost) in buf {
            self.inject_full(p, v, c, d, ghost);
        }
    }

    fn step(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.positions.is_empty() {
            return;
        }

        // Update fluid cells
        for idx in 0..self.cell_types.len() {
            if self.cell_types[idx] != 2 {
                self.cell_types[idx] = 0;
            }
        }
        for pos in &self.positions {
            let i = (pos.x / CELL_SIZE).floor() as i32;
            let j = (pos.y / CELL_SIZE).floor() as i32;
            let k = (pos.z / CELL_SIZE).floor() as i32;
            if i >= 0 && i < GRID_WIDTH as i32 && j >= 0 && j < GRID_HEIGHT as i32 && k >= 0 && k < GRID_DEPTH as i32 {
                let idx = k as usize * GRID_WIDTH * GRID_HEIGHT + j as usize * GRID_WIDTH + i as usize;
                if self.cell_types[idx] != 2 {
                    self.cell_types[idx] = 1;
                }
            }
        }

        self.gpu.step(
            device,
            queue,
            &mut self.positions,
            &mut self.velocities,
            &mut self.c_matrices,
            &self.densities,
            &self.cell_types,
            None,
            None,
            DT,
            GRAVITY,
            FLOW_ACCEL,
            PRESSURE_ITERS,
        );

        // CPU collision
        let min_x = CELL_SIZE * 1.0;
        let floor_y = CELL_SIZE * 1.0;
        let ceiling_y = (GRID_HEIGHT as f32 - 1.5) * CELL_SIZE;
        let min_z = CELL_SIZE * 1.0;
        let max_z = (GRID_DEPTH as f32 - 1.5) * CELL_SIZE;

        for i in 0..self.positions.len() {
            if self.positions[i].y < floor_y {
                self.positions[i].y = floor_y;
                self.velocities[i].y = self.velocities[i].y.abs() * 0.1;
            }
            if self.positions[i].y > ceiling_y {
                self.positions[i].y = ceiling_y;
                self.velocities[i].y = -self.velocities[i].y.abs() * 0.1;
            }
            if self.positions[i].z < min_z {
                self.positions[i].z = min_z;
                self.velocities[i].z = self.velocities[i].z.abs() * 0.1;
            }
            if self.positions[i].z > max_z {
                self.positions[i].z = max_z;
                self.velocities[i].z = -self.velocities[i].z.abs() * 0.1;
            }
            if self.positions[i].x < min_x {
                self.positions[i].x = min_x;
                self.velocities[i].x = self.velocities[i].x.abs() * 0.1;
            }
        }
    }

    /// Extract particles exiting through +X boundary above weir (full state)
    fn extract_exit(&mut self) -> Vec<HandoffParticle> {
        let mut out = Vec::new();
        let exit_x = (GRID_WIDTH as f32 - 2.0) * CELL_SIZE;
        let mut i = 0;
        while i < self.positions.len() {
            if self.positions[i].x >= exit_x && self.positions[i].y >= OUTFLOW_MIN_Y && !self.is_ghost[i] {
                out.push(HandoffParticle {
                    world_pos: self.positions[i] + self.world_offset,
                    velocity: self.velocities[i],
                    c_matrix: self.c_matrices[i],
                    density: self.densities[i],
                });
                self.positions.swap_remove(i);
                self.velocities.swap_remove(i);
                self.c_matrices.swap_remove(i);
                self.densities.swap_remove(i);
                self.is_ghost.swap_remove(i);
            } else {
                i += 1;
            }
        }
        out
    }

    /// Remove ghost particles after P2G
    fn remove_ghosts(&mut self) {
        let mut i = 0;
        while i < self.positions.len() {
            if self.is_ghost[i] {
                self.positions.swap_remove(i);
                self.velocities.swap_remove(i);
                self.c_matrices.swap_remove(i);
                self.densities.swap_remove(i);
                self.is_ghost.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }

    fn remove_oob(&mut self) -> usize {
        let max_y = (GRID_HEIGHT as f32 - 0.5) * CELL_SIZE;
        let max_x = GRID_WIDTH as f32 * CELL_SIZE;
        let mut i = 0;
        let mut removed = 0;
        while i < self.positions.len() {
            let p = self.positions[i];
            if p.y < 0.0 || p.y > max_y || p.x > max_x || p.x < 0.0 || !p.is_finite() || !self.velocities[i].is_finite() {
                self.positions.swap_remove(i);
                self.velocities.swap_remove(i);
                self.c_matrices.swap_remove(i);
                self.densities.swap_remove(i);
                self.is_ghost.swap_remove(i);
                removed += 1;
            } else {
                i += 1;
            }
        }
        removed
    }

    /// Get average velocity of non-ghost particles
    fn avg_velocity(&self) -> Vec3 {
        let mut sum = Vec3::ZERO;
        let mut count = 0;
        for (vel, &ghost) in self.velocities.iter().zip(&self.is_ghost) {
            if !ghost {
                sum += *vel;
                count += 1;
            }
        }
        if count == 0 { Vec3::ZERO } else { sum / count as f32 }
    }
}

fn spawn_inlet(grid: &mut GridSegment, count: usize, seed: &mut u64) {
    let floor_y = CELL_SIZE * 1.5;
    let max_y = (GRID_HEIGHT as f32 - 2.0) * CELL_SIZE;
    let min_z = CELL_SIZE * 1.5;
    let max_z = (GRID_DEPTH as f32 - 1.5) * CELL_SIZE;
    for _ in 0..count {
        let x = CELL_SIZE * 1.5;
        let y = floor_y + rand_f32(seed) * (max_y - floor_y);
        let z = min_z + rand_f32(seed) * (max_z - min_z);
        grid.inject(Vec3::new(x, y, z), Vec3::new(INLET_VELOCITY, 0.0, 0.0));
    }
}

fn rand_f32(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*seed >> 33) as f32) / (u32::MAX as f32)
}

fn main() {
    println!("=== CHAINED GRIDS TEST (with ghost particles) ===");
    println!("Grid: {}x{}x{}, cell size: {}m", GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE);
    println!("Ghost particles per handoff: {}", GHOST_COUNT);
    println!();

    // Initialize GPU
    let (device, queue) = pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find GPU adapter");

        println!("GPU: {}", adapter.get_info().name);

        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffers_per_shader_stage = 16;
        limits.max_storage_buffer_binding_size = 256 * 1024 * 1024;

        adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Chained Grids Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .expect("Failed to create GPU device")
    });

    // Create two grids
    let grid_length = GRID_WIDTH as f32 * CELL_SIZE;
    let mut grid_a = GridSegment::new(&device, Vec3::ZERO);
    let mut grid_b = GridSegment::new(&device, Vec3::new(grid_length, 0.0, 0.0));

    let mut metrics = TransferMetrics::default();
    let mut seed = 12345u64;

    println!("Running {} frames ({} warmup + {} steady state)...", TEST_FRAMES, WARMUP_FRAMES, TEST_FRAMES - WARMUP_FRAMES);
    println!();

    for frame in 0..TEST_FRAMES {
        let is_steady_state = frame >= WARMUP_FRAMES;

        // Spawn inlet
        spawn_inlet(&mut grid_a, INLET_RATE, &mut seed);
        metrics.total_injected += INLET_RATE;

        // Process input buffers
        grid_a.process_input();
        grid_b.process_input();

        // Run substeps
        for _ in 0..SUBSTEPS {
            grid_a.step(&device, &queue);
            grid_b.step(&device, &queue);
        }

        // Handoff: A exits → B enters with ghost particles
        let exited_a = grid_a.extract_exit();
        for particle in exited_a {
            let local_pos = particle.world_pos - grid_b.world_offset;
            let local_pos = Vec3::new(CELL_SIZE * 1.5, local_pos.y, local_pos.z);

            // Record velocity at A exit (for velocity preservation comparison)
            if is_steady_state {
                metrics.record_a_exit(particle.velocity);
            }

            // Inject the actual particle
            grid_b.input_buffer.push((local_pos, particle.velocity, particle.c_matrix, particle.density, false));

            // Inject ghost particles ahead
            for dx in 1..=GHOST_COUNT {
                let ghost_pos = Vec3::new(
                    local_pos.x + dx as f32 * CELL_SIZE,
                    local_pos.y,
                    local_pos.z,
                );
                grid_b.input_buffer.push((ghost_pos, particle.velocity, particle.c_matrix, particle.density, true));
            }

            metrics.total_handoffs += 1;
        }

        // Remove ghost particles after P2G
        grid_b.remove_ghosts();

        // B exits → leave system (record velocity for preservation metric)
        let exited_b = grid_b.extract_exit();
        for particle in &exited_b {
            if is_steady_state {
                metrics.record_b_exit(particle.velocity);
            }
        }
        metrics.total_exited += exited_b.len();

        // Remove OOB (track for mass accounting)
        let oob_a = grid_a.remove_oob();
        let oob_b = grid_b.remove_oob();
        metrics.total_oob_removed += oob_a + oob_b;

        // Progress report
        if (frame + 1) % 100 == 0 {
            let state = if is_steady_state { "STEADY" } else { "WARMUP" };
            println!(
                "Frame {:3} [{}]: A={:5} B={:5} | handoffs={} exited={} | vx_A={:.2} vx_B={:.2}",
                frame + 1,
                state,
                grid_a.particle_count(),
                grid_b.particle_count(),
                metrics.total_handoffs,
                metrics.total_exited,
                grid_a.avg_velocity().x,
                grid_b.avg_velocity().x,
            );
        }
    }

    // Final statistics
    println!();
    println!("=== RESULTS ===");
    println!();
    println!("Particle Counts:");
    println!("  Total injected:        {}", metrics.total_injected);
    println!("  Total handoffs (A→B):  {}", metrics.total_handoffs);
    println!("  Total exited from B:   {}", metrics.total_exited);
    println!("  Remaining in A:        {}", grid_a.particle_count());
    println!("  Remaining in B:        {}", grid_b.particle_count());
    println!();

    println!("Flow Metrics:");
    println!("  Handoff ratio:         {:.1}%", metrics.handoff_ratio() * 100.0);
    println!("  Throughput ratio:      {:.1}%", metrics.throughput_ratio() * 100.0);
    println!();

    println!("Velocity Preservation (steady state, A exit → B exit):");
    println!("  Samples at B exit:     {}", metrics.velocity_samples);
    println!("  Avg speed at A exit:   {:.3} m/s", metrics.avg_speed_at_a_exit());
    println!("  Avg speed at B exit:   {:.3} m/s", metrics.avg_speed_at_b_exit());
    println!("  Speed preservation:    {:.1}%", metrics.velocity_preservation_ratio() * 100.0);
    println!();

    println!("Mass Conservation:");
    let in_system = grid_a.particle_count() + grid_b.particle_count();
    let accounted = metrics.total_exited + in_system + metrics.total_oob_removed;
    let mass_error = (metrics.total_injected as f32 - accounted as f32).abs() / metrics.total_injected as f32;
    println!("  In system:             {}", in_system);
    println!("  Exited normally:       {}", metrics.total_exited);
    println!("  Removed OOB:           {}", metrics.total_oob_removed);
    println!("  Accounted for:         {}", accounted);
    println!("  Mass error:            {:.2}%", mass_error * 100.0);
    println!();

    // Success criteria
    // Velocity: 95-105% (allows viscosity loss, catches over-acceleration)
    // Throughput: 30% (realistic without FLOW_ACCEL)
    // Mass: 5% tolerance
    println!("=== TEST CRITERIA ===");

    let vel_ratio = metrics.velocity_preservation_ratio();
    let velocity_pass = vel_ratio >= 0.95 && vel_ratio <= 1.05;
    let throughput_pass = metrics.throughput_ratio() >= 0.30;
    let mass_pass = mass_error < 0.05;

    println!("  Velocity preservation 95-105%: {} ({:.1}%)",
        if velocity_pass { "PASS" } else { "FAIL" },
        vel_ratio * 100.0);
    println!("  Throughput >= 30%:             {} ({:.1}%)",
        if throughput_pass { "PASS" } else { "FAIL" },
        metrics.throughput_ratio() * 100.0);
    println!("  Mass error < 5%:               {} ({:.2}%)",
        if mass_pass { "PASS" } else { "FAIL" },
        mass_error * 100.0);
    println!();

    if velocity_pass && throughput_pass && mass_pass {
        println!("=== ALL TESTS PASSED ===");
    } else {
        println!("=== SOME TESTS FAILED ===");
        std::process::exit(1);
    }
}
