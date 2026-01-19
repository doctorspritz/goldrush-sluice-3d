//! GPU-accelerated 3D FLIP/APIC simulation.
//!
//! This module provides a complete GPU-based 3D fluid simulation that combines:
//! - P2G (Particle-to-Grid) with atomic scatter
//! - Pressure solve (Red-Black Gauss-Seidel)
//! - G2P (Grid-to-Particle) with FLIP/PIC blend
//!
//! The simulation maintains particle data on CPU but does all heavy computation on GPU.

use super::g2p_3d::{GpuG2p3D, SedimentParams3D};
use super::p2g_3d::GpuP2g3D;
use super::p2g_cell_centric_3d::GpuP2gCellCentric3D;
use super::particle_sort::GpuParticleSort;
use super::pressure_3d::GpuPressure3D;

use bytemuck::{Pod, Zeroable};
use std::sync::{mpsc, Arc};
use wgpu::util::DeviceExt;

pub mod diagnostics;
pub mod initialization;
pub mod params;
pub mod pipeline_builder;
pub mod readback;

pub use diagnostics::PhysicsDiagnostics;
pub use params::{
    GravelObstacle, GRAVEL_OBSTACLE_MAX,
    HYDROSTATIC_FLIP_RATIO, HYDROSTATIC_SLIP_FACTOR, HYDROSTATIC_OPEN_BOUNDARIES,
};
use params::{
    GravelObstacleParams3D, GravityParams3D, FlowParams3D, VorticityParams3D,
    VortConfineParams3D, SedimentFractionParams3D, SedimentPressureParams3D,
    PorosityDragParams3D, BcParams3D, DensityErrorParams3D, DensityPositionGridParams3D,
    DensityCorrectionParams3D, SedimentCellTypeParams3D, VelocityExtrapParams3D,
    SdfCollisionParams3D,
};
use readback::{ReadbackSlot, ReadbackMode};

/// GPU-accelerated 3D FLIP simulation
pub struct GpuFlip3D {
    // Grid dimensions
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub cell_size: f32,
    /// Bitmask for open boundaries (particles can exit without clamping):
    /// Bit 0 (1): -X open, Bit 1 (2): +X open
    /// Bit 2 (4): -Y open, Bit 3 (8): +Y open
    /// Bit 4 (16): -Z open, Bit 5 (32): +Z open
    pub open_boundaries: u32,
    /// Vorticity confinement strength (default 0.05, range 0.0-0.25)
    pub vorticity_epsilon: f32,
    /// FLIP/PIC ratio (0.0 = pure PIC, 1.0 = pure FLIP). Default 0.99 for water.
    /// Lower values add more damping, useful for settling simulations.
    pub flip_ratio: f32,
    /// Enable density projection (volume conservation). Disable for basic tests.
    pub density_projection_enabled: bool,
    /// Slip factor for tangential velocities at solid boundaries:
    /// 1.0 = free-slip (allow tangential flow, default for dynamic flow)
    /// 0.0 = no-slip (zero tangential velocity, good for hydrostatic equilibrium)
    pub slip_factor: f32,
    /// Target water particles per cell for density projection.
    /// Must match the particle seeding density (e.g., 1.0 for 1 particle/cell, 8.0 for 2x2x2 seeding).
    pub water_rest_density: f32,
    /// Target sediment particles per cell for porosity fraction.
    pub sediment_rest_particles: f32,
    /// Speed below which friction kicks in (m/s).
    pub sediment_friction_threshold: f32,
    /// How much to damp velocity when slow (0-1 per frame).
    pub sediment_friction_strength: f32,
    /// Downward settling speed for sediment particles.
    pub sediment_settling_velocity: f32,
    /// Vorticity lift applied to sediment when flow swirls.
    pub sediment_vorticity_lift: f32,
    /// Minimum vorticity magnitude to apply lift.
    pub sediment_vorticity_threshold: f32,
    /// Rate at which particle velocity approaches water velocity (1/s).
    /// Higher = more entrainment. Typical: 5.0-20.0. Scaled by 1/density.
    pub sediment_drag_coefficient: f32,
    /// Density threshold for gold particles (treat as gold above this).
    pub gold_density_threshold: f32,
    /// Drag multiplier applied to gold (fine gold entrains more).
    pub gold_drag_multiplier: f32,
    /// Settling velocity used for gold-specific lift (m/s).
    pub gold_settling_velocity: f32,
    /// Upward bias for flaky gold near the surface (m/s^2).
    pub gold_flake_lift: f32,
    /// Porosity-based drag applied to grid velocities.
    pub sediment_porosity_drag: f32,

    // Shared particle buffers (for readback scheduling)
    pub positions_buffer: Arc<wgpu::Buffer>,
    pub velocities_buffer: Arc<wgpu::Buffer>,
    pub(crate) c_col0_buffer: Arc<wgpu::Buffer>,
    pub(crate) c_col1_buffer: Arc<wgpu::Buffer>,
    pub(crate) c_col2_buffer: Arc<wgpu::Buffer>,
    pub densities_buffer: Arc<wgpu::Buffer>,

    // Sub-solvers
    p2g: GpuP2g3D,
    water_p2g: GpuP2g3D,
    g2p: GpuG2p3D,
    pressure: GpuPressure3D,

    // Gravity shader
    gravity_pipeline: wgpu::ComputePipeline,
    gravity_bind_group: wgpu::BindGroup,
    gravity_params_buffer: wgpu::Buffer,

    // Flow acceleration shader (for sluice downstream flow)
    flow_pipeline: wgpu::ComputePipeline,
    flow_bind_group: wgpu::BindGroup,
    flow_params_buffer: wgpu::Buffer,

    // Vorticity confinement
    vorticity_x_buffer: wgpu::Buffer,
    vorticity_y_buffer: wgpu::Buffer,
    vorticity_z_buffer: wgpu::Buffer,
    vorticity_mag_buffer: wgpu::Buffer,
    vorticity_compute_pipeline: wgpu::ComputePipeline,
    vorticity_compute_bind_group: wgpu::BindGroup,
    vorticity_confine_u_pipeline: wgpu::ComputePipeline,
    vorticity_confine_v_pipeline: wgpu::ComputePipeline,
    vorticity_confine_w_pipeline: wgpu::ComputePipeline,
    vorticity_confine_bind_group: wgpu::BindGroup,
    vorticity_params_buffer: wgpu::Buffer,
    vort_confine_params_buffer: wgpu::Buffer,

    // Sediment fraction (cell-centered)
    sediment_fraction_buffer: wgpu::Buffer,
    sediment_fraction_pipeline: wgpu::ComputePipeline,
    sediment_fraction_bind_group: wgpu::BindGroup,
    sediment_fraction_params_buffer: wgpu::Buffer,

    // Sediment pressure (Drucker-Prager)
    sediment_pressure_buffer: wgpu::Buffer,
    sediment_pressure_pipeline: wgpu::ComputePipeline,
    sediment_pressure_bind_group: wgpu::BindGroup,
    sediment_pressure_params_buffer: wgpu::Buffer,

    // Porosity drag (sediment damping on grid)
    porosity_drag_u_pipeline: wgpu::ComputePipeline,
    porosity_drag_v_pipeline: wgpu::ComputePipeline,
    porosity_drag_w_pipeline: wgpu::ComputePipeline,
    porosity_drag_bind_group: wgpu::BindGroup,
    porosity_drag_params_buffer: wgpu::Buffer,

    // Boundary condition enforcement shaders
    bc_u_pipeline: wgpu::ComputePipeline,
    bc_v_pipeline: wgpu::ComputePipeline,
    bc_w_pipeline: wgpu::ComputePipeline,
    bc_bind_group: wgpu::BindGroup,
    bc_params_buffer: wgpu::Buffer,

    // Grid velocity backup for FLIP delta
    grid_u_old_buffer: wgpu::Buffer,
    grid_v_old_buffer: wgpu::Buffer,
    grid_w_old_buffer: wgpu::Buffer,

    // Density projection (Implicit Density Projection for volume conservation)
    // Phase 1: Compute density error
    density_error_pipeline: wgpu::ComputePipeline,
    density_error_bind_group: wgpu::BindGroup,
    density_error_params_buffer: wgpu::Buffer,
    // Phase 2: Compute position changes on grid (blub approach)
    density_position_grid_pipeline: wgpu::ComputePipeline,
    density_position_grid_bind_group: wgpu::BindGroup,
    density_position_grid_params_buffer: wgpu::Buffer,
    position_delta_x_buffer: wgpu::Buffer, // Grid-based delta X
    position_delta_y_buffer: wgpu::Buffer, // Grid-based delta Y
    position_delta_z_buffer: wgpu::Buffer, // Grid-based delta Z
    // Phase 3: Particles sample from grid with trilinear interpolation
    density_correct_pipeline: wgpu::ComputePipeline,
    density_correct_bind_group: wgpu::BindGroup,
    density_correct_params_buffer: wgpu::Buffer,
    // Sediment density projection (granular packing)
    sediment_cell_type_pipeline: wgpu::ComputePipeline,
    sediment_cell_type_bind_group: wgpu::BindGroup,
    sediment_cell_type_params_buffer: wgpu::Buffer,
    // Fluid cell expansion (standard FLIP "7 points per particle")
    fluid_cell_expand_pipeline: wgpu::ComputePipeline,
    fluid_cell_expand_bind_group: wgpu::BindGroup,
    // Velocity extrapolation (standard FLIP - extend velocities into AIR cells)
    velocity_extrap_params_buffer: wgpu::Buffer,
    valid_u_buffer: wgpu::Buffer,
    valid_v_buffer: wgpu::Buffer,
    valid_w_buffer: wgpu::Buffer,
    velocity_extrap_init_pipeline: wgpu::ComputePipeline,
    velocity_extrap_u_pipeline: wgpu::ComputePipeline,
    velocity_extrap_v_pipeline: wgpu::ComputePipeline,
    velocity_extrap_w_pipeline: wgpu::ComputePipeline,
    velocity_extrap_finalize_pipeline: wgpu::ComputePipeline,
    velocity_extrap_bind_group: wgpu::BindGroup,
    sediment_density_error_pipeline: wgpu::ComputePipeline,
    sediment_density_error_bind_group: wgpu::BindGroup,
    sediment_density_correct_pipeline: wgpu::ComputePipeline,

    // SDF collision (advection + solid collision)
    sdf_collision_pipeline: wgpu::ComputePipeline,
    sdf_collision_bind_group: wgpu::BindGroup,
    sdf_collision_params_buffer: wgpu::Buffer,
    sdf_buffer: wgpu::Buffer,
    bed_height_buffer: Arc<wgpu::Buffer>,
    sdf_uploaded: bool,

    // Gravel obstacles (dynamic solids)
    gravel_obstacle_pipeline: wgpu::ComputePipeline,
    gravel_obstacle_bind_group: wgpu::BindGroup,
    gravel_obstacle_params_buffer: wgpu::Buffer,
    gravel_obstacle_buffer: wgpu::Buffer,
    gravel_obstacle_count: u32,
    gravel_porosity_pipeline: wgpu::ComputePipeline,
    gravel_porosity_bind_group: wgpu::BindGroup,

    // Particle sorting for cache coherence
    sorter: GpuParticleSort,
    sorted_p2g: GpuP2g3D,
    /// Cell-centric P2G (zero atomics, requires sorted particles)
    cell_centric_p2g: GpuP2gCellCentric3D,
    /// Enable particle sorting before P2G (for benchmarking)
    pub use_sorted_p2g: bool,
    /// Use cell-centric P2G (requires use_sorted_p2g = true)
    pub use_cell_centric_p2g: bool,

    // Maximum particles supported
    max_particles: usize,

    // Double-buffered async readback slots
    readback_slots: Vec<ReadbackSlot>,
    readback_cursor: usize,
}

impl GpuFlip3D {
    /// Create a new GPU 3D FLIP simulation
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        depth: u32,
        cell_size: f32,
        max_particles: usize,
    ) -> Self {
        initialization::init_complete_pipeline(device, width, height, depth, cell_size, max_particles)
    }


    /// Upload SDF data to GPU.
    pub fn upload_sdf(&mut self, queue: &wgpu::Queue, sdf: &[f32]) {
        if self.sdf_uploaded {
            return;
        }

        let expected_sdf_len = (self.width * self.height * self.depth) as usize;
        assert_eq!(
            sdf.len(),
            expected_sdf_len,
            "SDF size mismatch: got {}, expected {}",
            sdf.len(),
            expected_sdf_len
        );

        queue.write_buffer(&self.sdf_buffer, 0, bytemuck::cast_slice(sdf));
        self.sdf_uploaded = true;
    }

    /// Force upload SDF data to GPU (for dynamic obstacles).
    pub fn upload_sdf_force(&mut self, queue: &wgpu::Queue, sdf: &[f32]) {
        let expected_sdf_len = (self.width * self.height * self.depth) as usize;
        assert_eq!(
            sdf.len(),
            expected_sdf_len,
            "SDF size mismatch: got {}, expected {}",
            sdf.len(),
            expected_sdf_len
        );
        queue.write_buffer(&self.sdf_buffer, 0, bytemuck::cast_slice(sdf));
        self.sdf_uploaded = true;
    }

    pub fn upload_gravel_obstacles(&mut self, queue: &wgpu::Queue, obstacles: &[GravelObstacle]) {
        let count = obstacles.len().min(GRAVEL_OBSTACLE_MAX as usize);
        if count == 0 {
            self.gravel_obstacle_count = 0;
            return;
        }
        queue.write_buffer(
            &self.gravel_obstacle_buffer,
            0,
            bytemuck::cast_slice(&obstacles[..count]),
        );
        self.gravel_obstacle_count = count as u32;
    }

    fn apply_gravel_obstacles(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.gravel_obstacle_count == 0 {
            return;
        }

        let params = GravelObstacleParams3D {
            width: self.width,
            height: self.height,
            depth: self.depth,
            obstacle_count: self.gravel_obstacle_count,
            cell_size: self.cell_size,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        queue.write_buffer(
            &self.gravel_obstacle_params_buffer,
            0,
            bytemuck::bytes_of(&params),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Gravel Obstacle Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Gravel Obstacle Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.gravel_obstacle_pipeline);
            pass.set_bind_group(0, &self.gravel_obstacle_bind_group, &[]);
            let workgroups_x = self.width.div_ceil(8);
            let workgroups_y = self.height.div_ceil(8);
            let workgroups_z = self.depth.div_ceil(4);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }

    /// Run one simulation step (sync readback).
    ///
    /// This performs the full FLIP pipeline:
    /// 1. P2G: Transfer particle data to grid
    /// 2. Enforce boundary conditions (before storing old velocities!)
    /// 3. Save grid velocity (for FLIP delta)
    /// 4. Apply gravity (vertical)
    /// 5. Apply flow acceleration (horizontal, for sluice flow)
    /// 6. Vorticity confinement (adds rotational energy)
    /// 7. Pressure solve (includes divergence, iterations, gradient)
    /// 8. G2P: Transfer grid data back to particles
    /// 9. Optional: GPU advection + SDF collision (when `sdf` is provided)
    ///
    /// # Arguments
    /// * `flow_accel` - Downstream flow acceleration (m/s²). Set to 0.0 for closed box sims.
    ///                  For a sluice, use ~2-5 m/s² to drive water downstream.
    /// Pure GPU step (no CPU readback/upload).
    /// Assumes all data (positions, velocities, etc.) is already in GPU buffers.
    pub fn encode_step(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        particle_count: u32,
        dt: f32,
    ) {
        if particle_count == 0 {
            return;
        }

        // 1. P2G
        self.p2g.prepare(queue, particle_count, self.cell_size);
        self.water_p2g
            .prepare(queue, particle_count, self.cell_size);
        self.p2g.encode(encoder, particle_count);
        self.water_p2g.encode(encoder, particle_count);

        // 2. Pressure (Simplified for now - might need more steps for stability)
        self.pressure.encode(encoder, 40); // 40 iterations

        // 3. G2P
        let sediment_params = SedimentParams3D::default();
        self.g2p
            .upload_params(queue, particle_count, self.cell_size, dt, self.flip_ratio, sediment_params);
        self.g2p.encode(encoder, particle_count);

        // 4. SDF Collision
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("SDF Collision Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.sdf_collision_pipeline);
        pass.set_bind_group(0, &self.sdf_collision_bind_group, &[]);
        let workgroups = particle_count.div_ceil(256);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    /// Configure the solver for hydrostatic equilibrium testing.
    ///
    /// Sets validated parameters that achieve true hydrostatic equilibrium
    /// (water at rest with max velocity < 0.01 m/s). Use this configuration
    /// when testing static water scenarios.
    ///
    /// Parameters set:
    /// - `flip_ratio = 0.0` (Pure PIC for maximum damping)
    /// - `slip_factor = 0.0` (No-slip at solid boundaries)
    /// - `open_boundaries = 8` (+Y open for free surface)
    /// - `vorticity_epsilon = 0.0` (Disable vorticity confinement)
    /// - `density_projection_enabled = false` (Disable - causes particle clumping)
    pub fn configure_for_hydrostatic_equilibrium(&mut self) {
        self.flip_ratio = HYDROSTATIC_FLIP_RATIO;
        self.slip_factor = HYDROSTATIC_SLIP_FACTOR;
        self.open_boundaries = HYDROSTATIC_OPEN_BOUNDARIES;
        self.vorticity_epsilon = 0.0;
        self.density_projection_enabled = false;
    }

    pub fn step(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &mut [glam::Vec3],
        velocities: &mut [glam::Vec3],
        c_matrices: &mut [glam::Mat3],
        densities: &[f32],
        cell_types: &[u32],
        sdf: Option<&[f32]>,
        bed_height: Option<&[f32]>,
        dt: f32,
        gravity: f32,
        flow_accel: f32,
        pressure_iterations: u32,
    ) {
        let _ = self.step_internal(
            device,
            queue,
            positions,
            velocities,
            c_matrices,
            densities,
            cell_types,
            sdf,
            bed_height,
            dt,
            gravity,
            flow_accel,
            pressure_iterations,
            ReadbackMode::Sync,
        );
    }

    /// Run one simulation step without any readback (upload + GPU passes only).
    pub fn step_no_readback(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &mut [glam::Vec3],
        velocities: &mut [glam::Vec3],
        c_matrices: &mut [glam::Mat3],
        densities: &[f32],
        cell_types: &[u32],
        sdf: Option<&[f32]>,
        bed_height: Option<&[f32]>,
        dt: f32,
        gravity: f32,
        flow_accel: f32,
        pressure_iterations: u32,
    ) {
        let _ = self.step_internal(
            device,
            queue,
            positions,
            velocities,
            c_matrices,
            densities,
            cell_types,
            sdf,
            bed_height,
            dt,
            gravity,
            flow_accel,
            pressure_iterations,
            ReadbackMode::None,
        );
    }

    /// Run one simulation step and schedule an async readback.
    ///
    /// Call `try_readback` on subsequent frames to pull results without stalling.
    /// Returns false if all readback slots are still in flight.
    pub fn step_async(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &mut [glam::Vec3],
        velocities: &mut [glam::Vec3],
        c_matrices: &mut [glam::Mat3],
        densities: &[f32],
        cell_types: &[u32],
        sdf: Option<&[f32]>,
        bed_height: Option<&[f32]>,
        dt: f32,
        gravity: f32,
        flow_accel: f32,
        pressure_iterations: u32,
    ) -> bool {
        self.step_internal(
            device,
            queue,
            positions,
            velocities,
            c_matrices,
            densities,
            cell_types,
            sdf,
            bed_height,
            dt,
            gravity,
            flow_accel,
            pressure_iterations,
            ReadbackMode::Async,
        )
    }

    /// Try to fetch the latest async readback results without stalling.
    ///
    /// Returns Some(count) when a slot completes, otherwise None.
    pub fn try_readback(
        &mut self,
        device: &wgpu::Device,
        positions: &mut [glam::Vec3],
        velocities: &mut [glam::Vec3],
        c_matrices: &mut [glam::Mat3],
    ) -> Option<usize> {
        device.poll(wgpu::Maintain::Poll);
        for slot in &mut self.readback_slots {
            if let Some(count) = slot.try_read(positions, velocities, c_matrices) {
                return Some(count);
            }
        }
        None
    }

    /// Schedule a readback of the current GPU particle buffers without uploading.
    pub fn request_readback(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particle_count: usize,
    ) -> bool {
        let count = particle_count.min(self.max_particles);
        self.schedule_readback(device, queue, count)
    }

    pub fn positions_buffer(&self) -> Arc<wgpu::Buffer> {
        self.positions_buffer.clone()
    }

    pub fn velocities_buffer(&self) -> Arc<wgpu::Buffer> {
        self.velocities_buffer.clone()
    }

    pub fn densities_buffer(&self) -> Arc<wgpu::Buffer> {
        self.densities_buffer.clone()
    }

    pub fn sdf_buffer(&self) -> &wgpu::Buffer {
        &self.sdf_buffer
    }

    pub fn bed_height_buffer(&self) -> Arc<wgpu::Buffer> {
        self.bed_height_buffer.clone()
    }

    fn upload_bc_and_cell_types(&self, queue: &wgpu::Queue, cell_types: &[u32]) {
        // Upload cell types FIRST (needed for BC enforcement)
        self.pressure
            .upload_cell_types(queue, cell_types, self.cell_size, self.open_boundaries);

        // Upload BC params
        let bc_params = BcParams3D {
            width: self.width,
            height: self.height,
            depth: self.depth,
            open_boundaries: self.open_boundaries,
            slip_factor: self.slip_factor,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        queue.write_buffer(&self.bc_params_buffer, 0, bytemuck::bytes_of(&bc_params));
    }

    fn schedule_readback(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particle_count: usize,
    ) -> bool {
        if self.readback_slots.is_empty() {
            return false;
        }

        let slots = self.readback_slots.len();
        for _ in 0..slots {
            let index = self.readback_cursor % slots;
            self.readback_cursor = (self.readback_cursor + 1) % slots;
            let slot = &mut self.readback_slots[index];
            if slot.schedule(
                device,
                queue,
                &self.positions_buffer,
                &self.velocities_buffer,
                &self.c_col0_buffer,
                &self.c_col1_buffer,
                &self.c_col2_buffer,
                particle_count,
            ) {
                return true;
            }
        }

        false
    }

    pub fn step_in_place(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particle_count: u32,
        cell_types: &[u32],
        sdf: Option<&[f32]>,
        bed_height: Option<&[f32]>,
        dt: f32,
        gravity: f32,
        flow_accel: f32,
        pressure_iterations: u32,
    ) {
        if particle_count == 0 {
            return;
        }

        self.upload_bc_and_cell_types(queue, cell_types);
        self.apply_gravel_obstacles(device, queue);
        self.p2g.prepare(queue, particle_count, self.cell_size);

        let _ = self.run_gpu_passes(
            device,
            queue,
            particle_count,
            sdf,
            bed_height,
            dt,
            gravity,
            flow_accel,
            pressure_iterations,
        );
    }

    fn run_gpu_passes(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particle_count: u32,
        sdf: Option<&[f32]>,
        bed_height: Option<&[f32]>,
        dt: f32,
        gravity: f32,
        flow_accel: f32,
        pressure_iterations: u32,
    ) -> u32 {
        let count = particle_count;

        self.water_p2g.prepare(queue, count, self.cell_size);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FLIP 3D Step Encoder"),
        });

        // Run P2G scatter and divide (optionally with particle sorting)
        let (u_size, v_size, w_size) = self.p2g.grid_sizes();
        let cell_count = (self.width * self.height * self.depth) as usize;
        if self.use_sorted_p2g {
            // Sort particles by cell index for cache coherence
            self.sorter.prepare(queue, count, self.cell_size);
            self.sorter.encode(&mut encoder, queue, count);

            if self.use_cell_centric_p2g {
                // Use cell-centric P2G (zero atomics, one thread per grid node)
                self.cell_centric_p2g.prepare(queue, count, self.cell_size);
                self.cell_centric_p2g.encode(&mut encoder, count);
                // Copy results from cell_centric_p2g to p2g buffers (rest of pipeline uses p2g buffers)
                encoder.copy_buffer_to_buffer(
                    &self.cell_centric_p2g.grid_u_buffer,
                    0,
                    &self.p2g.grid_u_buffer,
                    0,
                    (u_size * 4) as u64,
                );
                encoder.copy_buffer_to_buffer(
                    &self.cell_centric_p2g.grid_v_buffer,
                    0,
                    &self.p2g.grid_v_buffer,
                    0,
                    (v_size * 4) as u64,
                );
                encoder.copy_buffer_to_buffer(
                    &self.cell_centric_p2g.grid_w_buffer,
                    0,
                    &self.p2g.grid_w_buffer,
                    0,
                    (w_size * 4) as u64,
                );
                encoder.copy_buffer_to_buffer(
                    &self.cell_centric_p2g.particle_count_buffer,
                    0,
                    &self.p2g.particle_count_buffer,
                    0,
                    (cell_count * 4) as u64,
                );
                encoder.copy_buffer_to_buffer(
                    &self.cell_centric_p2g.sediment_count_buffer,
                    0,
                    &self.p2g.sediment_count_buffer,
                    0,
                    (cell_count * 4) as u64,
                );
            } else {
                // Use sorted P2G that reads from sorted particle buffers
                self.sorted_p2g.prepare(queue, count, self.cell_size);
                self.sorted_p2g.encode(&mut encoder, count);
                // Copy results from sorted_p2g to p2g buffers (rest of pipeline uses p2g buffers)
                encoder.copy_buffer_to_buffer(
                    &self.sorted_p2g.grid_u_buffer,
                    0,
                    &self.p2g.grid_u_buffer,
                    0,
                    (u_size * 4) as u64,
                );
                encoder.copy_buffer_to_buffer(
                    &self.sorted_p2g.grid_v_buffer,
                    0,
                    &self.p2g.grid_v_buffer,
                    0,
                    (v_size * 4) as u64,
                );
                encoder.copy_buffer_to_buffer(
                    &self.sorted_p2g.grid_w_buffer,
                    0,
                    &self.p2g.grid_w_buffer,
                    0,
                    (w_size * 4) as u64,
                );
                encoder.copy_buffer_to_buffer(
                    &self.sorted_p2g.particle_count_buffer,
                    0,
                    &self.p2g.particle_count_buffer,
                    0,
                    (cell_count * 4) as u64,
                );
                encoder.copy_buffer_to_buffer(
                    &self.sorted_p2g.sediment_count_buffer,
                    0,
                    &self.p2g.sediment_count_buffer,
                    0,
                    (cell_count * 4) as u64,
                );
            }
        } else {
            self.p2g.encode(&mut encoder, count);
        }
        self.water_p2g.encode(&mut encoder, count);

        queue.submit(std::iter::once(encoder.finish()));

        let sediment_fraction_params = SedimentFractionParams3D {
            width: self.width,
            height: self.height,
            depth: self.depth,
            rest_particles: self.sediment_rest_particles,
        };
        queue.write_buffer(
            &self.sediment_fraction_params_buffer,
            0,
            bytemuck::bytes_of(&sediment_fraction_params),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Sediment Fraction 3D Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Sediment Fraction 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sediment_fraction_pipeline);
            pass.set_bind_group(0, &self.sediment_fraction_bind_group, &[]);
            let workgroups_x = self.width.div_ceil(8);
            let workgroups_y = self.height.div_ceil(8);
            let workgroups_z = self.depth.div_ceil(4);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }
        if self.gravel_obstacle_count > 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Gravel Porosity 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.gravel_porosity_pipeline);
            pass.set_bind_group(0, &self.gravel_porosity_bind_group, &[]);
            let workgroups_x = self.width.div_ceil(8);
            let workgroups_y = self.height.div_ceil(8);
            let workgroups_z = self.depth.div_ceil(4);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        queue.submit(std::iter::once(encoder.finish()));

        let sediment_pressure_params = SedimentPressureParams3D {
            width: self.width,
            height: self.height,
            depth: self.depth,
            _pad0: 0,
            cell_size: self.cell_size,
            particle_mass: 1.0,
            gravity: gravity.abs(),
            buoyancy_factor: 0.62,
        };
        queue.write_buffer(
            &self.sediment_pressure_params_buffer,
            0,
            bytemuck::bytes_of(&sediment_pressure_params),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Sediment Pressure 3D Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Sediment Pressure 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sediment_pressure_pipeline);
            pass.set_bind_group(0, &self.sediment_pressure_bind_group, &[]);
            let workgroups_x = self.width.div_ceil(8);
            let workgroups_z = self.depth.div_ceil(8);
            pass.dispatch_workgroups(workgroups_x, 1, workgroups_z);
        }

        queue.submit(std::iter::once(encoder.finish()));

        // 2. Enforce boundary conditions BEFORE storing old velocities
        // This is critical for correct FLIP delta computation!
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FLIP 3D BC Encoder"),
        });

        // Enforce BC on U
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BC U 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bc_u_pipeline);
            pass.set_bind_group(0, &self.bc_bind_group, &[]);
            let workgroups_x = (self.width + 1).div_ceil(8);
            let workgroups_y = self.height.div_ceil(8);
            let workgroups_z = self.depth.div_ceil(4);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        // Enforce BC on V
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BC V 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bc_v_pipeline);
            pass.set_bind_group(0, &self.bc_bind_group, &[]);
            let workgroups_x = self.width.div_ceil(8);
            let workgroups_y = (self.height + 1).div_ceil(8);
            let workgroups_z = self.depth.div_ceil(4);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        // Enforce BC on W
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BC W 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bc_w_pipeline);
            pass.set_bind_group(0, &self.bc_bind_group, &[]);
            let workgroups_x = self.width.div_ceil(8);
            let workgroups_y = self.height.div_ceil(8);
            let workgroups_z = (self.depth + 1).div_ceil(4);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        queue.submit(std::iter::once(encoder.finish()));

        // 3. Save grid velocity for FLIP delta (now with proper BCs!)
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FLIP 3D Grid Copy Encoder"),
        });
        encoder.copy_buffer_to_buffer(
            &self.p2g.grid_u_buffer,
            0,
            &self.grid_u_old_buffer,
            0,
            (u_size * 4) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &self.p2g.grid_v_buffer,
            0,
            &self.grid_v_old_buffer,
            0,
            (v_size * 4) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &self.p2g.grid_w_buffer,
            0,
            &self.grid_w_old_buffer,
            0,
            (w_size * 4) as u64,
        );
        queue.submit(std::iter::once(encoder.finish()));

        // 3.5 Cell type classification: mark cells as FLUID/AIR based on particle presence.
        // This MUST happen BEFORE gravity so gravity is applied to fluid cells!
        {
            let sediment_cell_params = SedimentCellTypeParams3D {
                width: self.width,
                height: self.height,
                depth: self.depth,
                _pad0: 0,
            };
            queue.write_buffer(
                &self.sediment_cell_type_params_buffer,
                0,
                bytemuck::bytes_of(&sediment_cell_params),
            );

            // Multiple iterations to propagate jamming upward (for sediment piles)
            let cell_type_iterations = 5;
            for _ in 0..cell_type_iterations {
                let mut ct_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Cell Type 3D Encoder"),
                });
                {
                    let mut pass = ct_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Cell Type 3D Pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.sediment_cell_type_pipeline);
                    pass.set_bind_group(0, &self.sediment_cell_type_bind_group, &[]);
                    let workgroups_x = (self.width + 7) / 8;
                    let workgroups_y = (self.height + 7) / 8;
                    let workgroups_z = (self.depth + 3) / 4;
                    pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
                }
                queue.submit(std::iter::once(ct_encoder.finish()));
            }

            // Fluid cell expansion: mark cells as FLUID if 2+ face-neighbors have particles
            // Conservative approach to prevent over-expansion
            {
                let mut expand_encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Fluid Cell Expand 3D Encoder"),
                    });
                {
                    let mut pass = expand_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Fluid Cell Expand 3D Pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.fluid_cell_expand_pipeline);
                    pass.set_bind_group(0, &self.fluid_cell_expand_bind_group, &[]);
                    let workgroups_x = (self.width + 7) / 8;
                    let workgroups_y = (self.height + 7) / 8;
                    let workgroups_z = (self.depth + 3) / 4;
                    pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
                }
                queue.submit(std::iter::once(expand_encoder.finish()));
            }
        }

        // 4. Apply gravity

        let gravity_params = GravityParams3D {
            width: self.width,
            height: self.height,
            depth: self.depth,
            gravity_dt: gravity * dt,
            cell_size: self.cell_size,
            open_boundaries: self.open_boundaries,
            _pad0: 0,
            _pad1: 0,
        };
        queue.write_buffer(
            &self.gravity_params_buffer,
            0,
            bytemuck::bytes_of(&gravity_params),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FLIP 3D Gravity Encoder"),
        });

        // Apply gravity
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Gravity 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.gravity_pipeline);
            pass.set_bind_group(0, &self.gravity_bind_group, &[]);
            let workgroups_x = self.width.div_ceil(8);
            let workgroups_y = (self.height + 1).div_ceil(8);
            let workgroups_z = self.depth.div_ceil(4);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        // 5. Apply flow acceleration (for sluice downstream flow)
        // This MUST happen before pressure solve so the solver can account for the flow!
        if flow_accel.abs() > 0.0001 {
            let flow_params = FlowParams3D {
                width: self.width,
                height: self.height,
                depth: self.depth,
                flow_accel_dt: flow_accel * dt,
            };
            queue.write_buffer(
                &self.flow_params_buffer,
                0,
                bytemuck::bytes_of(&flow_params),
            );

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Flow 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.flow_pipeline);
            pass.set_bind_group(0, &self.flow_bind_group, &[]);
            // U grid: (width+1) x height x depth
            let workgroups_x = (self.width + 1).div_ceil(8);
            let workgroups_y = self.height.div_ceil(8);
            let workgroups_z = self.depth.div_ceil(4);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        // 6. Vorticity confinement (adds rotational energy)
        if self.vorticity_epsilon > 0.0 {
            let vort_params = VorticityParams3D {
                width: self.width,
                height: self.height,
                depth: self.depth,
                cell_size: self.cell_size,
            };
            queue.write_buffer(
                &self.vorticity_params_buffer,
                0,
                bytemuck::bytes_of(&vort_params),
            );

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Vorticity 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.vorticity_compute_pipeline);
                pass.set_bind_group(0, &self.vorticity_compute_bind_group, &[]);
                let workgroups_x = self.width.div_ceil(8);
                let workgroups_y = self.height.div_ceil(8);
                let workgroups_z = self.depth.div_ceil(4);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }

            let confine_params = VortConfineParams3D {
                width: self.width,
                height: self.height,
                depth: self.depth,
                epsilon_h_dt: self.vorticity_epsilon * self.cell_size * dt,
            };
            queue.write_buffer(
                &self.vort_confine_params_buffer,
                0,
                bytemuck::bytes_of(&confine_params),
            );

            // Apply confinement to U faces
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Vorticity Confine U 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.vorticity_confine_u_pipeline);
                pass.set_bind_group(0, &self.vorticity_confine_bind_group, &[]);
                let workgroups_x = (self.width + 1).div_ceil(8);
                let workgroups_y = self.height.div_ceil(8);
                let workgroups_z = self.depth.div_ceil(4);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }

            // Apply confinement to V faces
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Vorticity Confine V 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.vorticity_confine_v_pipeline);
                pass.set_bind_group(0, &self.vorticity_confine_bind_group, &[]);
                let workgroups_x = self.width.div_ceil(8);
                let workgroups_y = (self.height + 1).div_ceil(8);
                let workgroups_z = self.depth.div_ceil(4);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }

            // Apply confinement to W faces
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Vorticity Confine W 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.vorticity_confine_w_pipeline);
                pass.set_bind_group(0, &self.vorticity_confine_bind_group, &[]);
                let workgroups_x = self.width.div_ceil(8);
                let workgroups_y = self.height.div_ceil(8);
                let workgroups_z = (self.depth + 1).div_ceil(4);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }
        }

        // Submit vorticity passes before pressure solve
        queue.submit(std::iter::once(encoder.finish()));

        // 7. Pressure solve (divergence → iterations → gradient)
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FLIP 3D Pressure Encoder"),
        });
        self.pressure.encode(&mut encoder, pressure_iterations);

        queue.submit(std::iter::once(encoder.finish()));

        // NOTE: Post-pressure BC REMOVED - it was introducing divergence into the
        // divergence-free velocity field, causing fluid expansion over time.
        // BC is only applied BEFORE the pressure solve (in step 6).
        // The pressure solve naturally enforces no-penetration at walls.

        if self.sediment_porosity_drag > 0.0 {
            let drag_params = PorosityDragParams3D {
                width: self.width,
                height: self.height,
                depth: self.depth,
                drag_dt: self.sediment_porosity_drag * dt,
            };
            queue.write_buffer(
                &self.porosity_drag_params_buffer,
                0,
                bytemuck::bytes_of(&drag_params),
            );

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Porosity Drag 3D Encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Porosity Drag U 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.porosity_drag_u_pipeline);
                pass.set_bind_group(0, &self.porosity_drag_bind_group, &[]);
                let workgroups_x = (self.width + 1).div_ceil(8);
                let workgroups_y = self.height.div_ceil(8);
                let workgroups_z = self.depth.div_ceil(4);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Porosity Drag V 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.porosity_drag_v_pipeline);
                pass.set_bind_group(0, &self.porosity_drag_bind_group, &[]);
                let workgroups_x = self.width.div_ceil(8);
                let workgroups_y = (self.height + 1).div_ceil(8);
                let workgroups_z = self.depth.div_ceil(4);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Porosity Drag W 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.porosity_drag_w_pipeline);
                pass.set_bind_group(0, &self.porosity_drag_bind_group, &[]);
                let workgroups_x = self.width.div_ceil(8);
                let workgroups_y = self.height.div_ceil(8);
                let workgroups_z = (self.depth + 1).div_ceil(4);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }

            queue.submit(std::iter::once(encoder.finish()));
        }

        // 7.8 Velocity extrapolation: extend velocities into AIR cells near surface
        // This is critical for FLIP stability - particles near the surface need to sample
        // from valid velocity fields, not undefined AIR cell values.
        // Run 4 passes to extrapolate 4 cells deep into AIR.
        {
            let extrap_params = VelocityExtrapParams3D {
                width: self.width,
                height: self.height,
                depth: self.depth,
                extrap_pass: 0,
            };
            queue.write_buffer(
                &self.velocity_extrap_params_buffer,
                0,
                bytemuck::bytes_of(&extrap_params),
            );

            // Calculate workgroups for U, V, W grids (they have different sizes)
            let workgroups_u = (
                (self.width + 1 + 7) / 8,
                (self.height + 7) / 8,
                (self.depth + 3) / 4,
            );
            let workgroups_v = (
                (self.width + 7) / 8,
                (self.height + 1 + 7) / 8,
                (self.depth + 3) / 4,
            );
            let workgroups_w = (
                (self.width + 7) / 8,
                (self.height + 7) / 8,
                (self.depth + 1 + 3) / 4,
            );
            // Use the maximum for init/finalize passes that touch all grids
            let workgroups_max = (
                workgroups_u.0.max(workgroups_v.0).max(workgroups_w.0),
                workgroups_u.1.max(workgroups_v.1).max(workgroups_w.1),
                workgroups_u.2.max(workgroups_v.2).max(workgroups_w.2),
            );

            // Initialize valid flags based on FLUID cells
            {
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Velocity Extrap Init Encoder"),
                    });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Velocity Extrap Init Pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.velocity_extrap_init_pipeline);
                    pass.set_bind_group(0, &self.velocity_extrap_bind_group, &[]);
                    pass.dispatch_workgroups(workgroups_max.0, workgroups_max.1, workgroups_max.2);
                }
                queue.submit(std::iter::once(encoder.finish()));
            }

            // Run 4 extrapolation passes (deeper extrapolation for better surface coverage)
            for _pass in 0..4 {
                // Extrapolate U
                {
                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Velocity Extrap U Encoder"),
                        });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Velocity Extrap U Pass"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.velocity_extrap_u_pipeline);
                        pass.set_bind_group(0, &self.velocity_extrap_bind_group, &[]);
                        pass.dispatch_workgroups(workgroups_u.0, workgroups_u.1, workgroups_u.2);
                    }
                    queue.submit(std::iter::once(encoder.finish()));
                }

                // Extrapolate V
                {
                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Velocity Extrap V Encoder"),
                        });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Velocity Extrap V Pass"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.velocity_extrap_v_pipeline);
                        pass.set_bind_group(0, &self.velocity_extrap_bind_group, &[]);
                        pass.dispatch_workgroups(workgroups_v.0, workgroups_v.1, workgroups_v.2);
                    }
                    queue.submit(std::iter::once(encoder.finish()));
                }

                // Extrapolate W
                {
                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Velocity Extrap W Encoder"),
                        });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Velocity Extrap W Pass"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.velocity_extrap_w_pipeline);
                        pass.set_bind_group(0, &self.velocity_extrap_bind_group, &[]);
                        pass.dispatch_workgroups(workgroups_w.0, workgroups_w.1, workgroups_w.2);
                    }
                    queue.submit(std::iter::once(encoder.finish()));
                }

                // Finalize: mark newly extrapolated faces as valid
                {
                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Velocity Extrap Finalize Encoder"),
                        });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Velocity Extrap Finalize Pass"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.velocity_extrap_finalize_pipeline);
                        pass.set_bind_group(0, &self.velocity_extrap_bind_group, &[]);
                        pass.dispatch_workgroups(workgroups_max.0, workgroups_max.1, workgroups_max.2);
                    }
                    queue.submit(std::iter::once(encoder.finish()));
                }
            }
        }

        // 8. Run G2P using grid buffers already on GPU
        let sediment_params = SedimentParams3D {
            settling_velocity: self.sediment_settling_velocity,
            friction_threshold: self.sediment_friction_threshold,
            friction_strength: self.sediment_friction_strength,
            vorticity_lift: self.sediment_vorticity_lift,
            vorticity_threshold: self.sediment_vorticity_threshold,
            drag_coefficient: self.sediment_drag_coefficient,
            gold_density_threshold: self.gold_density_threshold,
            gold_drag_multiplier: self.gold_drag_multiplier,
            gold_settling_velocity: self.gold_settling_velocity,
            gold_flake_lift: self.gold_flake_lift,
            _pad: [0.0; 2],
        };
        let g2p_count = self
            .g2p
            .upload_params(queue, count, self.cell_size, dt, self.flip_ratio, sediment_params);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FLIP 3D G2P Encoder"),
        });
        self.g2p.encode(&mut encoder, g2p_count);
        queue.submit(std::iter::once(encoder.finish()));

        // ========== Density Projection (Implicit Density Projection) ==========
        // Push particles from crowded regions to empty regions
        // This causes water level to "rise" when particles accumulate behind riffles
        // DISABLE for basic tests - only needed for dense water simulations
        if self.density_projection_enabled {
            // 1. Compute density error from particle counts (populated during P2G)
            let density_error_params = DensityErrorParams3D {
                width: self.width,
                height: self.height,
                depth: self.depth,
                rest_density: self.water_rest_density,
                dt,
                _pad1: 0,
                _pad2: 0,
                _pad3: 0,
            };
            queue.write_buffer(
                &self.density_error_params_buffer,
                0,
                bytemuck::bytes_of(&density_error_params),
            );

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("FLIP 3D Density Projection Encoder"),
            });

            // Dispatch density error shader - writes to divergence_buffer
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Density Error 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.density_error_pipeline);
                pass.set_bind_group(0, &self.density_error_bind_group, &[]);
                let workgroups_x = self.width.div_ceil(8);
                let workgroups_y = self.height.div_ceil(8);
                let workgroups_z = self.depth.div_ceil(4);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }

            queue.submit(std::iter::once(encoder.finish()));

            // 2. Clear pressure and run density pressure iterations
            self.pressure.clear_pressure(queue);

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("FLIP 3D Density Pressure Encoder"),
            });

            // Run pressure solver iterations with density error as RHS
            // Uses same Jacobi solver, just different input
            let density_iterations = 40; // More iterations for volume conservation
            self.pressure
                .encode_iterations_only(&mut encoder, density_iterations);

            queue.submit(std::iter::once(encoder.finish()));

            // 3. Compute position deltas on grid (blub approach)
            // Update grid shader params with dt
            let grid_params = DensityPositionGridParams3D {
                width: self.width,
                height: self.height,
                depth: self.depth,
                dt,
            };
            queue.write_buffer(
                &self.density_position_grid_params_buffer,
                0,
                bytemuck::bytes_of(&grid_params),
            );

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("FLIP 3D Position Grid Encoder"),
            });

            // Dispatch grid position delta shader
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Density Position Grid 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.density_position_grid_pipeline);
                pass.set_bind_group(0, &self.density_position_grid_bind_group, &[]);
                let workgroups_x = self.width.div_ceil(8);
                let workgroups_y = self.height.div_ceil(8);
                let workgroups_z = self.depth.div_ceil(4);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }

            queue.submit(std::iter::once(encoder.finish()));

            // 4. Apply position correction to particles (trilinear sampling from grid)
            let density_correct_params = DensityCorrectionParams3D {
                width: self.width,
                height: self.height,
                depth: self.depth,
                particle_count,
                cell_size: self.cell_size,
                dt,
                _pad1: 0,
                _pad2: 0,
            };
            queue.write_buffer(
                &self.density_correct_params_buffer,
                0,
                bytemuck::bytes_of(&density_correct_params),
            );

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("FLIP 3D Position Correction Encoder"),
            });

            // Dispatch particle position correction shader
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Density Correct 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.density_correct_pipeline);
                pass.set_bind_group(0, &self.density_correct_bind_group, &[]);
                let workgroups = particle_count.div_ceil(256);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            queue.submit(std::iter::once(encoder.finish()));
        }

        // ========== Sediment Density Projection (Granular Packing) ==========
        // NOTE: Cell type classification has been moved BEFORE pressure solve (step 6.5)
        // to ensure cells are properly marked as FLUID before the solver runs.

        if self.sediment_rest_particles > 0.0 {
            // DISABLED: Using voxel-based jamming instead of density projection
            /*
            let sediment_density_error_params = DensityErrorParams3D {
                width: self.width,
                height: self.height,
                depth: self.depth,
                rest_density: self.sediment_rest_particles,
                dt,
                _pad1: 0,
                _pad2: 0,
                _pad3: 0,
            };
            queue.write_buffer(
                &self.density_error_params_buffer,
                0,
                bytemuck::bytes_of(&sediment_density_error_params),
            );

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Sediment Density Error 3D Encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Sediment Density Error 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.sediment_density_error_pipeline);
                pass.set_bind_group(0, &self.sediment_density_error_bind_group, &[]);
                let workgroups_x = (self.width + 7) / 8;
                let workgroups_y = (self.height + 7) / 8;
                let workgroups_z = (self.depth + 3) / 4;
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }
            queue.submit(std::iter::once(encoder.finish()));

            self.pressure.clear_pressure(queue);

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Sediment Density Pressure 3D Encoder"),
            });
            let sediment_iterations = 15;
            self.pressure.encode_iterations_only(&mut encoder, sediment_iterations);
            queue.submit(std::iter::once(encoder.finish()));

            let grid_params = DensityPositionGridParams3D {
                width: self.width,
                height: self.height,
                depth: self.depth,
                dt,
            };
            queue.write_buffer(
                &self.density_position_grid_params_buffer,
                0,
                bytemuck::bytes_of(&grid_params),
            );

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Sediment Position Grid 3D Encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Sediment Position Grid 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.density_position_grid_pipeline);
                pass.set_bind_group(0, &self.density_position_grid_bind_group, &[]);
                let workgroups_x = (self.width + 7) / 8;
                let workgroups_y = (self.height + 7) / 8;
                let workgroups_z = (self.depth + 3) / 4;
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }
            queue.submit(std::iter::once(encoder.finish()));

            let density_correct_params = DensityCorrectionParams3D {
                width: self.width,
                height: self.height,
                depth: self.depth,
                particle_count: particle_count,
                cell_size: self.cell_size,
                dt,
                _pad1: 0,
                _pad2: 0,
            };
            queue.write_buffer(
                &self.density_correct_params_buffer,
                0,
                bytemuck::bytes_of(&density_correct_params),
            );

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Sediment Density Correct 3D Encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Sediment Density Correct 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.sediment_density_correct_pipeline);
                pass.set_bind_group(0, &self.density_correct_bind_group, &[]);
                let workgroups = (particle_count as u32 + 255) / 256;
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            queue.submit(std::iter::once(encoder.finish()));
            */
        }

        if let Some(bed_height) = bed_height {
            let expected_len = (self.width * self.depth) as usize;
            assert_eq!(
                bed_height.len(),
                expected_len,
                "Bed height size mismatch: got {}, expected {}",
                bed_height.len(),
                expected_len
            );
            queue.write_buffer(&self.bed_height_buffer, 0, bytemuck::cast_slice(bed_height));
        }

        if let Some(sdf) = sdf {
            if !self.sdf_uploaded {
                self.upload_sdf(queue, sdf);
            }
        }

        let sdf_params = SdfCollisionParams3D {
            width: self.width,
            height: self.height,
            depth: self.depth,
            particle_count,
            cell_size: self.cell_size,
            dt,
            open_boundaries: self.open_boundaries,
            _pad0: 0,
        };
        queue.write_buffer(
            &self.sdf_collision_params_buffer,
            0,
            bytemuck::bytes_of(&sdf_params),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FLIP 3D SDF Collision Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SDF Collision 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sdf_collision_pipeline);
            pass.set_bind_group(0, &self.sdf_collision_bind_group, &[]);
            let workgroups = g2p_count.div_ceil(256);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        g2p_count
    }

    fn step_internal(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &mut [glam::Vec3],
        velocities: &mut [glam::Vec3],
        c_matrices: &mut [glam::Mat3],
        densities: &[f32],
        cell_types: &[u32],
        sdf: Option<&[f32]>,
        bed_height: Option<&[f32]>,
        dt: f32,
        gravity: f32,
        flow_accel: f32,
        pressure_iterations: u32,
        readback: ReadbackMode,
    ) -> bool {
        let particle_count = positions.len().min(self.max_particles);
        if particle_count == 0 {
            return false;
        }

        self.upload_bc_and_cell_types(queue, cell_types);
        self.apply_gravel_obstacles(device, queue);

        // 1. Upload particles and run P2G
        let count = self.p2g.upload_particles(
            queue,
            positions,
            velocities,
            densities,
            c_matrices,
            self.cell_size,
        );

        let g2p_count = self.run_gpu_passes(
            device,
            queue,
            count,
            sdf,
            bed_height,
            dt,
            gravity,
            flow_accel,
            pressure_iterations,
        );

        match readback {
            ReadbackMode::None => true,
            ReadbackMode::Sync => {
                // Download results (velocities + C matrix + positions).
                self.g2p
                    .download(device, queue, g2p_count, velocities, c_matrices);
                let mut gpu_positions = vec![[0.0f32; 4]; particle_count];
                Self::read_buffer_vec4(
                    device,
                    queue,
                    &self.g2p.positions_buffer,
                    &mut gpu_positions,
                    particle_count,
                );
                for (i, pos) in gpu_positions.iter().enumerate() {
                    positions[i] = glam::Vec3::new(pos[0], pos[1], pos[2]);
                }
                true
            }
            ReadbackMode::Async => self.schedule_readback(device, queue, particle_count),
        }
    }

    fn read_buffer_vec4(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        output: &mut [[f32; 4]],
        count: usize,
    ) {
        let byte_size = count * std::mem::size_of::<[f32; 4]>();

        // Create a staging buffer
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Temp Vec4 Staging"),
            size: byte_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Read Vec4 Buffer Encoder"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, byte_size as u64);
        queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result); // Ignore send error if receiver dropped
        });
        device.poll(wgpu::Maintain::Wait);
        if let Err(e) = super::await_buffer_map(rx) {
            log::error!("GPU readback failed: {}", e);
            return;
        }

        {
            let data = buffer_slice.get_mapped_range();
            let slice: &[[f32; 4]] = bytemuck::cast_slice(&data);
            output[..count].copy_from_slice(&slice[..count]);
        }
        staging.unmap();
    }

}
