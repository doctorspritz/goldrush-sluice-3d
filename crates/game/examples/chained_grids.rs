//! Chained Grids Test - Two small FLIP grids connected in series
//!
//! This tests the concept of domain decomposition where:
//! - Grid A (upstream) receives inlet flow
//! - Particles exiting Grid A's +X boundary → enter Grid B's -X boundary
//! - Grid B (downstream) has open exit at +X
//!
//! Run with: cargo run --example chained_grids --release

use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Vec3};

// Grid dimensions - small for testing
const GRID_WIDTH: usize = 24;  // Longer grids
const GRID_HEIGHT: usize = 12; // Taller for headroom
const GRID_DEPTH: usize = 8;
const CELL_SIZE: f32 = 0.04; // 4cm cells
const MAX_PARTICLES: usize = 30000;

// Simulation parameters
const DT: f32 = 1.0 / 120.0;
const SUBSTEPS: u32 = 2;
const PRESSURE_ITERS: u32 = 50;
const GRAVITY: f32 = -9.8;
const FLOW_ACCEL: f32 = 4.0; // Stronger downstream push

// Inlet parameters
const INLET_RATE: usize = 30; // Particles per frame (reduced)
const INLET_VELOCITY: f32 = 1.0; // m/s downstream (faster)

// Test duration
const TEST_FRAMES: u32 = 800; // Run longer to see steady-state

/// A single FLIP grid segment with its particle data
struct GridSegment {
    gpu: GpuFlip3D,
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    c_matrices: Vec<Mat3>,
    densities: Vec<f32>,
    cell_types: Vec<u32>,
    /// World-space offset (where this grid starts in the chain)
    world_offset: Vec3,
    /// Particles waiting to be injected (from upstream)
    input_buffer: Vec<(Vec3, Vec3)>, // (local_pos, velocity)
}

impl GridSegment {
    fn new(device: &wgpu::Device, world_offset: Vec3) -> Self {
        let gpu = GpuFlip3D::new(
            device,
            GRID_WIDTH as u32,
            GRID_HEIGHT as u32,
            GRID_DEPTH as u32,
            CELL_SIZE,
            MAX_PARTICLES,
        );

        Self {
            gpu,
            positions: Vec::with_capacity(MAX_PARTICLES),
            velocities: Vec::with_capacity(MAX_PARTICLES),
            c_matrices: Vec::with_capacity(MAX_PARTICLES),
            densities: Vec::with_capacity(MAX_PARTICLES),
            cell_types: vec![0; GRID_WIDTH * GRID_HEIGHT * GRID_DEPTH],
            world_offset,
            input_buffer: Vec::new(),
        }
    }

    fn particle_count(&self) -> usize {
        self.positions.len()
    }

    /// Inject a particle at local coordinates
    fn inject_particle(&mut self, local_pos: Vec3, velocity: Vec3) {
        self.positions.push(local_pos);
        self.velocities.push(velocity);
        self.c_matrices.push(Mat3::ZERO);
        self.densities.push(1.0); // Water density
    }

    /// Process input buffer (particles from upstream)
    fn process_input_buffer(&mut self) {
        // Collect to avoid borrow conflict
        let to_inject: Vec<_> = self.input_buffer.drain(..).collect();
        for (pos, vel) in to_inject {
            self.inject_particle(pos, vel);
        }
    }

    /// Extract particles that exited through +X boundary
    /// Returns: Vec of (world_pos, velocity) for handoff to downstream
    fn extract_exit_particles(&mut self) -> Vec<(Vec3, Vec3)> {
        let mut exited = Vec::new();
        let exit_x = (GRID_WIDTH as f32 - 1.0) * CELL_SIZE;

        let mut i = 0;
        while i < self.positions.len() {
            if self.positions[i].x >= exit_x {
                // Particle exited +X boundary
                let world_pos = self.positions[i] + self.world_offset;
                let vel = self.velocities[i];
                exited.push((world_pos, vel));

                // Remove from this grid (swap-remove for efficiency)
                self.positions.swap_remove(i);
                self.velocities.swap_remove(i);
                self.c_matrices.swap_remove(i);
                self.densities.swap_remove(i);
                // Don't increment i - we need to check the swapped element
            } else {
                i += 1;
            }
        }

        exited
    }

    /// Remove particles that exited through other boundaries (floor, walls, -X)
    fn remove_oob_particles(&mut self) {
        let max_y = (GRID_HEIGHT as f32 - 0.5) * CELL_SIZE;
        let min_z = CELL_SIZE * 0.5;
        let max_z = (GRID_DEPTH as f32 - 0.5) * CELL_SIZE;

        let mut i = 0;
        while i < self.positions.len() {
            let p = self.positions[i];
            let v = self.velocities[i];

            // Remove if out of bounds or invalid
            // Note: We allow x < 0 particles to stay briefly (they'll be bounced)
            // but remove truly invalid ones
            if p.x < -CELL_SIZE
                || p.y < 0.0
                || p.y > max_y
                || p.z < min_z
                || p.z > max_z
                || !p.is_finite()
                || !v.is_finite()
            {
                self.positions.swap_remove(i);
                self.velocities.swap_remove(i);
                self.c_matrices.swap_remove(i);
                self.densities.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }

    /// Build cell types from current particle positions
    fn update_cell_types(&mut self) {
        // Reset to air
        self.cell_types.fill(0);

        // Mark SLOPED floor as solid
        // Floor height decreases from inlet (x=0) to outlet (x=width)
        // This creates natural gravity-driven flow
        let slope = 0.15; // 15% grade
        let inlet_floor = 3; // Floor at inlet is 3 cells high

        for k in 0..GRID_DEPTH {
            for i in 0..GRID_WIDTH {
                // Floor height at this x position
                let floor_y = (inlet_floor as f32 - i as f32 * slope).max(0.0) as usize;
                for j in 0..=floor_y {
                    let idx = k * GRID_WIDTH * GRID_HEIGHT + j * GRID_WIDTH + i;
                    self.cell_types[idx] = 2; // Solid
                }
            }
        }

        // Mark side walls as solid (z=0 and z=depth-1)
        for j in 0..GRID_HEIGHT {
            for i in 0..GRID_WIDTH {
                // z=0 wall
                let idx = j * GRID_WIDTH + i;
                self.cell_types[idx] = 2;
                // z=depth-1 wall
                let idx = (GRID_DEPTH - 1) * GRID_WIDTH * GRID_HEIGHT + j * GRID_WIDTH + i;
                self.cell_types[idx] = 2;
            }
        }

        // Mark fluid cells based on particles
        for pos in &self.positions {
            let i = (pos.x / CELL_SIZE).floor() as i32;
            let j = (pos.y / CELL_SIZE).floor() as i32;
            let k = (pos.z / CELL_SIZE).floor() as i32;

            if i >= 0
                && i < GRID_WIDTH as i32
                && j >= 0
                && j < GRID_HEIGHT as i32
                && k >= 0
                && k < GRID_DEPTH as i32
            {
                let idx = k as usize * GRID_WIDTH * GRID_HEIGHT
                    + j as usize * GRID_WIDTH
                    + i as usize;
                if self.cell_types[idx] != 2 {
                    self.cell_types[idx] = 1; // Fluid
                }
            }
        }
    }

    /// Run one simulation step
    fn step(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.positions.is_empty() {
            return;
        }

        self.update_cell_types();

        self.gpu.step(
            device,
            queue,
            &mut self.positions,
            &mut self.velocities,
            &mut self.c_matrices,
            &self.densities,
            &self.cell_types,
            None, // No SDF - using grid boundaries
            None, // No bed height
            DT,
            GRAVITY,
            FLOW_ACCEL,
            PRESSURE_ITERS,
        );

        // Simple boundary enforcement (GPU doesn't know about our open +X)
        let min = CELL_SIZE * 0.5;
        let max_z = (GRID_DEPTH as f32 - 0.5) * CELL_SIZE;
        let slope = 0.15;
        let inlet_floor = 3.0;

        for i in 0..self.positions.len() {
            // Sloped floor bounce - floor height depends on x position
            let x_cell = self.positions[i].x / CELL_SIZE;
            let floor_y = ((inlet_floor - x_cell * slope).max(0.0) + 1.0) * CELL_SIZE;

            if self.positions[i].y < floor_y {
                self.positions[i].y = floor_y;
                self.velocities[i].y = self.velocities[i].y.abs() * 0.1;
            }

            // Side walls bounce
            if self.positions[i].z < min {
                self.positions[i].z = min;
                self.velocities[i].z = self.velocities[i].z.abs() * 0.1;
            }
            if self.positions[i].z > max_z {
                self.positions[i].z = max_z;
                self.velocities[i].z = -self.velocities[i].z.abs() * 0.1;
            }

            // Inlet wall bounce (-X)
            if self.positions[i].x < min {
                self.positions[i].x = min;
                self.velocities[i].x = self.velocities[i].x.abs() * 0.1;
            }
        }
    }
}

/// Spawn inlet particles at grid A's -X boundary
fn spawn_inlet(grid: &mut GridSegment, count: usize) {
    // Floor at inlet is ~3 cells high (from slope calculation)
    let floor_y = CELL_SIZE * 4.0; // Just above floor
    let max_y = CELL_SIZE * 7.0;   // 3 cells of water depth
    let min_z = CELL_SIZE * 2.0;
    let max_z = (GRID_DEPTH as f32 - 2.0) * CELL_SIZE;

    for _ in 0..count {
        let x = CELL_SIZE * 1.5; // Just inside inlet
        let y = floor_y + rand_f32() * (max_y - floor_y);
        let z = min_z + rand_f32() * (max_z - min_z);

        let vel = Vec3::new(INLET_VELOCITY, 0.0, 0.0);
        grid.inject_particle(Vec3::new(x, y, z), vel);
    }
}

/// Simple random float [0, 1)
fn rand_f32() -> f32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    static mut SEED: u64 = 12345;
    unsafe {
        let mut hasher = DefaultHasher::new();
        SEED = SEED.wrapping_mul(6364136223846793005).wrapping_add(1);
        SEED.hash(&mut hasher);
        (hasher.finish() & 0xFFFFFF) as f32 / 0xFFFFFF as f32
    }
}

fn main() {
    println!("=== CHAINED GRIDS TEST ===");
    println!("Two {}x{}x{} grids connected in series", GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH);
    println!("Cell size: {}m, Max particles per grid: {}", CELL_SIZE, MAX_PARTICLES);
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
    // Grid A: world origin
    // Grid B: offset by grid A's length in X
    let grid_length = GRID_WIDTH as f32 * CELL_SIZE;
    let mut grid_a = GridSegment::new(&device, Vec3::ZERO);
    let mut grid_b = GridSegment::new(&device, Vec3::new(grid_length, 0.0, 0.0));

    // Configure open boundaries
    // Grid A: open at +X (bit 1 = 2)
    grid_a.gpu.open_boundaries = 2;
    // Grid B: open at +X (bit 1 = 2)
    grid_b.gpu.open_boundaries = 2;

    println!("Grid A world offset: {:?}", grid_a.world_offset);
    println!("Grid B world offset: {:?}", grid_b.world_offset);
    println!();

    // Statistics
    let mut total_injected = 0usize;
    let mut total_handoffs = 0usize;
    let mut total_exited = 0usize;

    println!("Running {} frames...", TEST_FRAMES);
    println!();

    for frame in 0..TEST_FRAMES {
        // 1. Spawn inlet particles at Grid A
        spawn_inlet(&mut grid_a, INLET_RATE);
        total_injected += INLET_RATE;

        // 2. Process input buffers (particles from upstream)
        grid_a.process_input_buffer();
        grid_b.process_input_buffer();

        // 3. Run substeps
        for _ in 0..SUBSTEPS {
            grid_a.step(&device, &queue);
            grid_b.step(&device, &queue);
        }

        // 4. Extract particles exiting Grid A's +X → handoff to Grid B
        let exited_a = grid_a.extract_exit_particles();
        for (world_pos, vel) in exited_a {
            // Convert world position to Grid B's local coordinates
            let local_pos = world_pos - grid_b.world_offset;
            // Clamp X to just inside Grid B's inlet
            let local_pos = Vec3::new(
                CELL_SIZE * 0.5, // Just inside -X boundary
                local_pos.y,
                local_pos.z,
            );
            grid_b.input_buffer.push((local_pos, vel));
            total_handoffs += 1;
        }

        // 5. Extract particles exiting Grid B's +X → they leave the system
        let exited_b = grid_b.extract_exit_particles();
        total_exited += exited_b.len();

        // 6. Remove OOB particles (floor, walls)
        grid_a.remove_oob_particles();
        grid_b.remove_oob_particles();

        // Progress report
        if (frame + 1) % 50 == 0 {
            let avg_vx_a: f32 = if grid_a.positions.is_empty() {
                0.0
            } else {
                grid_a.velocities.iter().map(|v| v.x).sum::<f32>() / grid_a.positions.len() as f32
            };
            let avg_vx_b: f32 = if grid_b.positions.is_empty() {
                0.0
            } else {
                grid_b.velocities.iter().map(|v| v.x).sum::<f32>() / grid_b.positions.len() as f32
            };

            println!(
                "Frame {:3}: A={:5} (vx={:5.2}), B={:5} (vx={:5.2}), handoffs={}, exited={}",
                frame + 1,
                grid_a.particle_count(),
                avg_vx_a,
                grid_b.particle_count(),
                avg_vx_b,
                total_handoffs,
                total_exited
            );
        }
    }

    // Final statistics
    println!();
    println!("=== RESULTS ===");
    println!("Total injected at inlet:     {}", total_injected);
    println!("Total handoffs (A → B):      {}", total_handoffs);
    println!("Total exited from B:         {}", total_exited);
    println!("Remaining in Grid A:         {}", grid_a.particle_count());
    println!("Remaining in Grid B:         {}", grid_b.particle_count());
    println!(
        "Total remaining:             {}",
        grid_a.particle_count() + grid_b.particle_count()
    );
    println!(
        "Throughput:                  {:.1}%",
        total_exited as f32 / total_injected as f32 * 100.0
    );
    println!(
        "Handoff rate:                {:.1}%",
        total_handoffs as f32 / total_injected as f32 * 100.0
    );

    // Success criteria
    let success = total_handoffs > 0 && total_exited > 0;
    println!();
    if success {
        println!("✓ SUCCESS: Particles flowed through both grids!");
        if total_handoffs > total_injected / 2 {
            println!("✓ Good handoff rate - particles are transitioning between grids");
        }
        if total_exited > total_injected / 4 {
            println!("✓ Good throughput - particles are exiting the system");
        }
    } else {
        println!("✗ FAILED: Flow didn't work as expected");
        if total_handoffs == 0 {
            println!("  - No particles handed off from Grid A to Grid B");
        }
        if total_exited == 0 {
            println!("  - No particles exited Grid B");
        }
    }
}
