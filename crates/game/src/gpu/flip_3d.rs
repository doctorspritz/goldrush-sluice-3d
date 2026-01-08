//! GPU-accelerated 3D FLIP/APIC simulation.
//!
//! This module provides a complete GPU-based 3D fluid simulation that combines:
//! - P2G (Particle-to-Grid) with atomic scatter
//! - Pressure solve (Red-Black Gauss-Seidel)
//! - G2P (Grid-to-Particle) with FLIP/PIC blend
//!
//! The simulation maintains particle data on CPU but does all heavy computation on GPU.

use super::g2p_3d::{DruckerPragerParams, GpuG2p3D, SedimentParams3D};
use super::p2g_3d::GpuP2g3D;
use super::pressure_3d::GpuPressure3D;

use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use std::sync::{mpsc, Arc};

/// Gravity application parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GravityParams3D {
    width: u32,
    height: u32,
    depth: u32,
    gravity_dt: f32,
    cell_size: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Flow acceleration parameters (for sluice downstream flow)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FlowParams3D {
    width: u32,
    height: u32,
    depth: u32,
    flow_accel_dt: f32,  // flow_accel * dt
}

/// Vorticity computation parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct VorticityParams3D {
    width: u32,
    height: u32,
    depth: u32,
    cell_size: f32,
}

/// Vorticity confinement parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct VortConfineParams3D {
    width: u32,
    height: u32,
    depth: u32,
    epsilon_h_dt: f32,  // epsilon * h * dt
}

/// Sediment fraction parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SedimentFractionParams3D {
    width: u32,
    height: u32,
    depth: u32,
    rest_particles: f32,
}

/// Sediment pressure parameters (for Drucker-Prager)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SedimentPressureParams3D {
    width: u32,
    height: u32,
    depth: u32,
    _pad0: u32,
    cell_size: f32,
    particle_mass: f32,
    gravity: f32,
    buoyancy_factor: f32,
}

/// Porosity drag parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PorosityDragParams3D {
    width: u32,
    height: u32,
    depth: u32,
    drag_dt: f32,
}

/// Boundary condition parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BcParams3D {
    width: u32,
    height: u32,
    depth: u32,
    _pad: u32,
}

/// Density error computation parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct DensityErrorParams3D {
    width: u32,
    height: u32,
    depth: u32,
    rest_density: f32,  // Target particles per cell (~8 for typical FLIP)
    dt: f32,            // Timestep for scaling
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

/// Density position grid parameters (first pass - grid-based position changes)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct DensityPositionGridParams3D {
    width: u32,
    height: u32,
    depth: u32,
    dt: f32,
}

/// Density position correction parameters (blub grid-based trilinear sampling)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct DensityCorrectionParams3D {
    width: u32,
    height: u32,
    depth: u32,
    particle_count: u32,
    cell_size: f32,
    dt: f32,
    _pad1: u32,
    _pad2: u32,
}

/// Sediment cell type builder parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SedimentCellTypeParams3D {
    width: u32,
    height: u32,
    depth: u32,
    _pad0: u32,
}

/// SDF collision parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SdfCollisionParams3D {
    width: u32,
    height: u32,
    depth: u32,
    particle_count: u32,
    cell_size: f32,
    dt: f32,
    _pad0: u32,
    _pad1: u32,
}

#[derive(Copy, Clone)]
enum ReadbackMode {
    Sync,
    Async,
}

struct ReadbackSlot {
    positions_staging: wgpu::Buffer,
    velocities_staging: wgpu::Buffer,
    c_col0_staging: wgpu::Buffer,
    c_col1_staging: wgpu::Buffer,
    c_col2_staging: wgpu::Buffer,
    capacity: usize,
    count: usize,
    pending: bool,
    positions_rx: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    velocities_rx: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    c_col0_rx: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    c_col1_rx: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    c_col2_rx: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
}

impl ReadbackSlot {
    fn new(device: &wgpu::Device, max_particles: usize) -> Self {
        let buffer_size = (max_particles * std::mem::size_of::<[f32; 4]>()) as u64;
        Self {
            positions_staging: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Readback Positions Staging"),
                size: buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            velocities_staging: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Readback Velocities Staging"),
                size: buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            c_col0_staging: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Readback C Col0 Staging"),
                size: buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            c_col1_staging: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Readback C Col1 Staging"),
                size: buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            c_col2_staging: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Readback C Col2 Staging"),
                size: buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            capacity: max_particles,
            count: 0,
            pending: false,
            positions_rx: None,
            velocities_rx: None,
            c_col0_rx: None,
            c_col1_rx: None,
            c_col2_rx: None,
        }
    }

    fn schedule(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &wgpu::Buffer,
        velocities: &wgpu::Buffer,
        c_col0: &wgpu::Buffer,
        c_col1: &wgpu::Buffer,
        c_col2: &wgpu::Buffer,
        count: usize,
    ) -> bool {
        if self.pending {
            return false;
        }

        let count = count.min(self.capacity);
        if count == 0 {
            return false;
        }

        let byte_size = (count * std::mem::size_of::<[f32; 4]>()) as u64;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Readback Copy Encoder"),
        });
        encoder.copy_buffer_to_buffer(positions, 0, &self.positions_staging, 0, byte_size);
        encoder.copy_buffer_to_buffer(velocities, 0, &self.velocities_staging, 0, byte_size);
        encoder.copy_buffer_to_buffer(c_col0, 0, &self.c_col0_staging, 0, byte_size);
        encoder.copy_buffer_to_buffer(c_col1, 0, &self.c_col1_staging, 0, byte_size);
        encoder.copy_buffer_to_buffer(c_col2, 0, &self.c_col2_staging, 0, byte_size);
        queue.submit(std::iter::once(encoder.finish()));

        self.count = count;
        self.pending = true;

        let (tx, rx) = mpsc::channel();
        self.positions_staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        self.positions_rx = Some(rx);

        let (tx, rx) = mpsc::channel();
        self.velocities_staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        self.velocities_rx = Some(rx);

        let (tx, rx) = mpsc::channel();
        self.c_col0_staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        self.c_col0_rx = Some(rx);

        let (tx, rx) = mpsc::channel();
        self.c_col1_staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        self.c_col1_rx = Some(rx);

        let (tx, rx) = mpsc::channel();
        self.c_col2_staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        self.c_col2_rx = Some(rx);

        true
    }

    fn try_read(
        &mut self,
        positions_out: &mut [glam::Vec3],
        velocities_out: &mut [glam::Vec3],
        c_matrices_out: &mut [glam::Mat3],
    ) -> Option<usize> {
        if !self.pending {
            return None;
        }

        let mut failed = false;
        let mut all_ready = true;
        for rx in [
            &mut self.positions_rx,
            &mut self.velocities_rx,
            &mut self.c_col0_rx,
            &mut self.c_col1_rx,
            &mut self.c_col2_rx,
        ] {
            if let Some(receiver) = rx {
                match receiver.try_recv() {
                    Ok(Ok(())) => {
                        *rx = None;
                    }
                    Ok(Err(_)) => {
                        failed = true;
                        *rx = None;
                    }
                    Err(mpsc::TryRecvError::Empty) => {
                        all_ready = false;
                    }
                    Err(mpsc::TryRecvError::Disconnected) => {
                        failed = true;
                        *rx = None;
                    }
                }
            }
        }

        if failed {
            self.pending = false;
            self.positions_staging.unmap();
            self.velocities_staging.unmap();
            self.c_col0_staging.unmap();
            self.c_col1_staging.unmap();
            self.c_col2_staging.unmap();
            return None;
        }

        if !all_ready {
            return None;
        }

        let count = self
            .count
            .min(positions_out.len())
            .min(velocities_out.len())
            .min(c_matrices_out.len());

        {
            let data = self.positions_staging.slice(..).get_mapped_range();
            let slice: &[[f32; 4]] = bytemuck::cast_slice(&data);
            for i in 0..count {
                positions_out[i] = glam::Vec3::new(slice[i][0], slice[i][1], slice[i][2]);
            }
        }
        self.positions_staging.unmap();

        {
            let data = self.velocities_staging.slice(..).get_mapped_range();
            let slice: &[[f32; 4]] = bytemuck::cast_slice(&data);
            for i in 0..count {
                velocities_out[i] = glam::Vec3::new(slice[i][0], slice[i][1], slice[i][2]);
            }
        }
        self.velocities_staging.unmap();

        {
            let data = self.c_col0_staging.slice(..).get_mapped_range();
            let slice: &[[f32; 4]] = bytemuck::cast_slice(&data);
            for i in 0..count {
                c_matrices_out[i].x_axis = glam::Vec3::new(slice[i][0], slice[i][1], slice[i][2]);
            }
        }
        self.c_col0_staging.unmap();

        {
            let data = self.c_col1_staging.slice(..).get_mapped_range();
            let slice: &[[f32; 4]] = bytemuck::cast_slice(&data);
            for i in 0..count {
                c_matrices_out[i].y_axis = glam::Vec3::new(slice[i][0], slice[i][1], slice[i][2]);
            }
        }
        self.c_col1_staging.unmap();

        {
            let data = self.c_col2_staging.slice(..).get_mapped_range();
            let slice: &[[f32; 4]] = bytemuck::cast_slice(&data);
            for i in 0..count {
                c_matrices_out[i].z_axis = glam::Vec3::new(slice[i][0], slice[i][1], slice[i][2]);
            }
        }
        self.c_col2_staging.unmap();

        self.pending = false;
        self.count = 0;
        Some(count)
    }
}

/// GPU-accelerated 3D FLIP simulation
pub struct GpuFlip3D {
    // Grid dimensions
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub cell_size: f32,
    /// Vorticity confinement strength (default 0.05, range 0.0-0.25)
    pub vorticity_epsilon: f32,
    /// Target sediment particles per cell for porosity fraction.
    pub sediment_rest_particles: f32,
    /// Drag rate applied to sediment particles in G2P.
    pub sediment_drag_rate: f32,
    /// Downward settling speed for sediment particles.
    pub sediment_settling_velocity: f32,
    /// Vorticity lift applied to sediment when flow swirls.
    pub sediment_vorticity_lift: f32,
    /// Minimum vorticity magnitude to apply lift.
    pub sediment_vorticity_threshold: f32,
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
    position_delta_x_buffer: wgpu::Buffer,  // Grid-based delta X
    position_delta_y_buffer: wgpu::Buffer,  // Grid-based delta Y
    position_delta_z_buffer: wgpu::Buffer,  // Grid-based delta Z
    // Phase 3: Particles sample from grid with trilinear interpolation
    density_correct_pipeline: wgpu::ComputePipeline,
    density_correct_bind_group: wgpu::BindGroup,
    density_correct_params_buffer: wgpu::Buffer,
    // Sediment density projection (granular packing)
    sediment_cell_type_pipeline: wgpu::ComputePipeline,
    sediment_cell_type_bind_group: wgpu::BindGroup,
    sediment_cell_type_params_buffer: wgpu::Buffer,
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
        // Shared particle buffers for P2G/G2P
        let particle_buffer_size = (max_particles * std::mem::size_of::<[f32; 4]>()) as u64;
        let positions_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FLIP 3D Positions"),
            size: particle_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let velocities_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FLIP 3D Velocities"),
            size: particle_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let c_col0_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FLIP 3D C Col0"),
            size: particle_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let c_col1_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FLIP 3D C Col1"),
            size: particle_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let c_col2_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FLIP 3D C Col2"),
            size: particle_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let densities_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FLIP 3D Densities"),
            size: (max_particles * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Create P2G solver (owns the grid velocity buffers)
        let p2g = GpuP2g3D::new(
            device,
            width,
            height,
            depth,
            max_particles,
            Arc::clone(&positions_buffer),
            Arc::clone(&velocities_buffer),
            Arc::clone(&densities_buffer),
            Arc::clone(&c_col0_buffer),
            Arc::clone(&c_col1_buffer),
            Arc::clone(&c_col2_buffer),
        );

        // Create grid velocity backup buffers for FLIP delta
        let u_size = ((width + 1) * height * depth) as usize;
        let v_size = (width * (height + 1) * depth) as usize;
        let w_size = (width * height * (depth + 1)) as usize;

        let grid_u_old_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grid U Old 3D"),
            size: (u_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let grid_v_old_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grid V Old 3D"),
            size: (v_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let grid_w_old_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grid W Old 3D"),
            size: (w_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cell_count = (width * height * depth) as usize;

        let sediment_pressure_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sediment Pressure Buffer"),
            size: (cell_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let vorticity_x_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vorticity X 3D"),
            size: (cell_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let vorticity_y_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vorticity Y 3D"),
            size: (cell_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let vorticity_z_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vorticity Z 3D"),
            size: (cell_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let vorticity_mag_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vorticity Magnitude 3D"),
            size: (cell_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Create pressure solver (references P2G's grid buffers)
        let pressure = GpuPressure3D::new(
            device,
            width,
            height,
            depth,
            &p2g.grid_u_buffer,
            &p2g.grid_v_buffer,
            &p2g.grid_w_buffer,
        );

        // Create G2P solver (binds to P2G and old grid buffers)
        let g2p = GpuG2p3D::new(
            device,
            width,
            height,
            depth,
            max_particles,
            Arc::clone(&positions_buffer),
            Arc::clone(&velocities_buffer),
            Arc::clone(&c_col0_buffer),
            Arc::clone(&c_col1_buffer),
            Arc::clone(&c_col2_buffer),
            Arc::clone(&densities_buffer),
            &p2g.grid_u_buffer,
            &p2g.grid_v_buffer,
            &p2g.grid_w_buffer,
            &grid_u_old_buffer,
            &grid_v_old_buffer,
            &grid_w_old_buffer,
            &vorticity_mag_buffer,
            &sediment_pressure_buffer,
            &p2g.sediment_count_buffer,
            &p2g.particle_count_buffer,
        );

        // Create gravity shader
        let gravity_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Gravity 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gravity_3d.wgsl").into()),
        });

        let gravity_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gravity Params 3D"),
            size: std::mem::size_of::<GravityParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Note: We need a cell_type buffer for gravity - borrow from pressure solver
        // For now we'll create a simple gravity pipeline that just modifies grid_v
        let gravity_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Gravity 3D Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },            ],
        });

        let bed_height_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bed Height 3D"),
            size: (width as usize * depth as usize * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        // Use the pressure solver's cell_type buffer for gravity
        let gravity_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gravity 3D Bind Group"),
            layout: &gravity_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: gravity_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pressure.cell_type_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: p2g.grid_v_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: bed_height_buffer.as_entire_binding() },
            ],
        });

        let gravity_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Gravity 3D Pipeline Layout"),
            bind_group_layouts: &[&gravity_bind_group_layout],
            push_constant_ranges: &[],
        });

        let gravity_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Gravity 3D Pipeline"),
            layout: Some(&gravity_pipeline_layout),
            module: &gravity_shader,
            entry_point: Some("apply_gravity"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create flow acceleration shader (for sluice downstream flow)
        let flow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Flow 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/flow_3d.wgsl").into()),
        });

        let flow_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Flow Params 3D"),
            size: std::mem::size_of::<FlowParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Flow shader bindings: params, cell_type, grid_u
        let flow_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Flow 3D Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let flow_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Flow 3D Bind Group"),
            layout: &flow_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: flow_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pressure.cell_type_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: p2g.grid_u_buffer.as_entire_binding() },
            ],
        });

        let flow_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Flow 3D Pipeline Layout"),
            bind_group_layouts: &[&flow_bind_group_layout],
            push_constant_ranges: &[],
        });

        let flow_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Flow 3D Pipeline"),
            layout: Some(&flow_pipeline_layout),
            module: &flow_shader,
            entry_point: Some("apply_flow"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ========== Vorticity Confinement ==========
        let vorticity_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vorticity 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/vorticity_3d.wgsl").into()),
        });

        let vorticity_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vorticity Params 3D"),
            size: std::mem::size_of::<VorticityParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let vorticity_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Vorticity 3D Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let vorticity_compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Vorticity 3D Bind Group"),
            layout: &vorticity_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: vorticity_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: p2g.grid_u_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: p2g.grid_v_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: p2g.grid_w_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pressure.cell_type_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: vorticity_x_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: vorticity_y_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: vorticity_z_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: vorticity_mag_buffer.as_entire_binding() },
            ],
        });

        let vorticity_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Vorticity 3D Pipeline Layout"),
            bind_group_layouts: &[&vorticity_bind_group_layout],
            push_constant_ranges: &[],
        });

        let vorticity_compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Vorticity 3D Pipeline"),
            layout: Some(&vorticity_pipeline_layout),
            module: &vorticity_shader,
            entry_point: Some("compute_vorticity"),
            compilation_options: Default::default(),
            cache: None,
        });

        let vorticity_confine_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vorticity Confine 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/vorticity_confine_3d.wgsl").into()),
        });

        let vort_confine_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vorticity Confine Params 3D"),
            size: std::mem::size_of::<VortConfineParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let vorticity_confine_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Vorticity Confine 3D Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let vorticity_confine_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Vorticity Confine 3D Bind Group"),
            layout: &vorticity_confine_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: vort_confine_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: vorticity_x_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: vorticity_y_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: vorticity_z_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: vorticity_mag_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: pressure.cell_type_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: p2g.grid_u_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: p2g.grid_v_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: p2g.grid_w_buffer.as_entire_binding() },
            ],
        });

        let vorticity_confine_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Vorticity Confine 3D Pipeline Layout"),
            bind_group_layouts: &[&vorticity_confine_bind_group_layout],
            push_constant_ranges: &[],
        });

        let vorticity_confine_u_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Vorticity Confine U 3D Pipeline"),
            layout: Some(&vorticity_confine_pipeline_layout),
            module: &vorticity_confine_shader,
            entry_point: Some("apply_confinement_u"),
            compilation_options: Default::default(),
            cache: None,
        });

        let vorticity_confine_v_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Vorticity Confine V 3D Pipeline"),
            layout: Some(&vorticity_confine_pipeline_layout),
            module: &vorticity_confine_shader,
            entry_point: Some("apply_confinement_v"),
            compilation_options: Default::default(),
            cache: None,
        });

        let vorticity_confine_w_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Vorticity Confine W 3D Pipeline"),
            layout: Some(&vorticity_confine_pipeline_layout),
            module: &vorticity_confine_shader,
            entry_point: Some("apply_confinement_w"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ========== Sediment Fraction ==========
        let sediment_fraction_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sediment Fraction 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sediment_fraction_3d.wgsl").into()),
        });

        let sediment_fraction_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sediment Fraction Params 3D"),
            size: std::mem::size_of::<SedimentFractionParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sediment_fraction_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sediment Fraction 3D"),
            size: (cell_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let sediment_fraction_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sediment Fraction 3D Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let sediment_fraction_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sediment Fraction 3D Bind Group"),
            layout: &sediment_fraction_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sediment_fraction_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: p2g.sediment_count_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: sediment_fraction_buffer.as_entire_binding() },
            ],
        });

        let sediment_fraction_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sediment Fraction 3D Pipeline Layout"),
            bind_group_layouts: &[&sediment_fraction_bind_group_layout],
            push_constant_ranges: &[],
        });

        let sediment_fraction_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sediment Fraction 3D Pipeline"),
            layout: Some(&sediment_fraction_pipeline_layout),
            module: &sediment_fraction_shader,
            entry_point: Some("compute_sediment_fraction"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ========== Sediment Pressure ==========
        let sediment_pressure_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sediment Pressure 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sediment_pressure_3d.wgsl").into()),
        });

        let sediment_pressure_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sediment Pressure Params 3D"),
            size: std::mem::size_of::<SedimentPressureParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sediment_pressure_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sediment Pressure 3D Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let sediment_pressure_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sediment Pressure 3D Bind Group"),
            layout: &sediment_pressure_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sediment_pressure_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: p2g.sediment_count_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: sediment_pressure_buffer.as_entire_binding() },
            ],
        });

        let sediment_pressure_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sediment Pressure 3D Pipeline Layout"),
            bind_group_layouts: &[&sediment_pressure_bind_group_layout],
            push_constant_ranges: &[],
        });

        let sediment_pressure_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sediment Pressure 3D Pipeline"),
            layout: Some(&sediment_pressure_pipeline_layout),
            module: &sediment_pressure_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ========== Porosity Drag ==========
        let porosity_drag_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Porosity Drag 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/porosity_drag_3d.wgsl").into()),
        });

        let porosity_drag_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Porosity Drag Params 3D"),
            size: std::mem::size_of::<PorosityDragParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let porosity_drag_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Porosity Drag 3D Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let porosity_drag_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Porosity Drag 3D Bind Group"),
            layout: &porosity_drag_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: porosity_drag_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: sediment_fraction_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: p2g.grid_u_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: p2g.grid_v_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: p2g.grid_w_buffer.as_entire_binding() },
            ],
        });

        let porosity_drag_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Porosity Drag 3D Pipeline Layout"),
            bind_group_layouts: &[&porosity_drag_bind_group_layout],
            push_constant_ranges: &[],
        });

        let porosity_drag_u_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Porosity Drag U 3D Pipeline"),
            layout: Some(&porosity_drag_pipeline_layout),
            module: &porosity_drag_shader,
            entry_point: Some("apply_porosity_u"),
            compilation_options: Default::default(),
            cache: None,
        });

        let porosity_drag_v_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Porosity Drag V 3D Pipeline"),
            layout: Some(&porosity_drag_pipeline_layout),
            module: &porosity_drag_shader,
            entry_point: Some("apply_porosity_v"),
            compilation_options: Default::default(),
            cache: None,
        });

        let porosity_drag_w_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Porosity Drag W 3D Pipeline"),
            layout: Some(&porosity_drag_pipeline_layout),
            module: &porosity_drag_shader,
            entry_point: Some("apply_porosity_w"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create boundary condition enforcement shader
        let bc_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Enforce BC 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/enforce_bc_3d.wgsl").into()),
        });

        let bc_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BC Params 3D"),
            size: std::mem::size_of::<BcParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bc_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BC 3D Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bc_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BC 3D Bind Group"),
            layout: &bc_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: bc_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pressure.cell_type_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: p2g.grid_u_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: p2g.grid_v_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: p2g.grid_w_buffer.as_entire_binding() },
            ],
        });

        let bc_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("BC 3D Pipeline Layout"),
            bind_group_layouts: &[&bc_bind_group_layout],
            push_constant_ranges: &[],
        });

        let bc_u_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("BC U 3D Pipeline"),
            layout: Some(&bc_pipeline_layout),
            module: &bc_shader,
            entry_point: Some("enforce_bc_u"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bc_v_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("BC V 3D Pipeline"),
            layout: Some(&bc_pipeline_layout),
            module: &bc_shader,
            entry_point: Some("enforce_bc_v"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bc_w_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("BC W 3D Pipeline"),
            layout: Some(&bc_pipeline_layout),
            module: &bc_shader,
            entry_point: Some("enforce_bc_w"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ========== Density Projection (Implicit Density Projection) ==========
        // Creates pipelines for density error computation and position correction

        let cell_count = (width * height * depth) as usize;

        // Create density error shader
        let density_error_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Density Error 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/density_error_3d.wgsl").into()),
        });

        let density_error_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Density Error Params 3D"),
            size: std::mem::size_of::<DensityErrorParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Density error bindings: params, particle_count, cell_type, density_error (uses divergence_buffer)
        let density_error_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Density Error 3D Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let density_error_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Density Error 3D Bind Group"),
            layout: &density_error_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: density_error_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: p2g.particle_count_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pressure.cell_type_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: pressure.divergence_buffer.as_entire_binding() },
            ],
        });

        let density_error_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Density Error 3D Pipeline Layout"),
            bind_group_layouts: &[&density_error_bind_group_layout],
            push_constant_ranges: &[],
        });

        let density_error_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Density Error 3D Pipeline"),
            layout: Some(&density_error_pipeline_layout),
            module: &density_error_shader,
            entry_point: Some("compute_density_error"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ========== Phase 2: Density Position Grid (blub approach) ==========
        // Compute position changes on grid, then particles sample with trilinear

        // Create grid-based position delta buffers
        let position_delta_x_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Position Delta X Grid"),
            size: (cell_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let position_delta_y_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Position Delta Y Grid"),
            size: (cell_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let position_delta_z_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Position Delta Z Grid"),
            size: (cell_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Create density position grid shader
        let density_position_grid_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Density Position Grid 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/density_position_grid_3d.wgsl").into()),
        });

        let density_position_grid_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Density Position Grid Params 3D"),
            size: std::mem::size_of::<DensityPositionGridParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bindings: params, pressure, cell_type, delta_x, delta_y, delta_z
        let density_position_grid_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Density Position Grid 3D Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let density_position_grid_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Density Position Grid 3D Bind Group"),
            layout: &density_position_grid_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: density_position_grid_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pressure.pressure_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pressure.cell_type_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: position_delta_x_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: position_delta_y_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: position_delta_z_buffer.as_entire_binding() },
            ],
        });

        let density_position_grid_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Density Position Grid 3D Pipeline Layout"),
            bind_group_layouts: &[&density_position_grid_bind_group_layout],
            push_constant_ranges: &[],
        });

        let density_position_grid_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Density Position Grid 3D Pipeline"),
            layout: Some(&density_position_grid_pipeline_layout),
            module: &density_position_grid_shader,
            entry_point: Some("compute_position_grid"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ========== Phase 3: Particle Position Correction (trilinear sampling) ==========
        let density_correct_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Density Correct 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/density_correct_3d.wgsl").into()),
        });

        let density_correct_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Density Correct Params 3D"),
            size: std::mem::size_of::<DensityCorrectionParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bindings: params, delta_x, delta_y, delta_z, cell_type, positions, densities, velocities
        let density_correct_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Density Correct 3D Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let density_correct_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Density Correct 3D Bind Group"),
            layout: &density_correct_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: density_correct_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: position_delta_x_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: position_delta_y_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: position_delta_z_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pressure.cell_type_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: g2p.positions_buffer.as_ref().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: densities_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: velocities_buffer.as_ref().as_entire_binding() },
            ],
        });

        let density_correct_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Density Correct 3D Pipeline Layout"),
            bind_group_layouts: &[&density_correct_bind_group_layout],
            push_constant_ranges: &[],
        });

        let density_correct_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Density Correct 3D Pipeline"),
            layout: Some(&density_correct_pipeline_layout),
            module: &density_correct_shader,
            entry_point: Some("correct_positions"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ========== Sediment Density Projection (Granular Packing) ==========
        let sediment_cell_type_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sediment Cell Type 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sediment_cell_type_3d.wgsl").into()),
        });

        let sediment_cell_type_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sediment Cell Type Params 3D"),
            size: std::mem::size_of::<SedimentCellTypeParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sediment_cell_type_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sediment Cell Type 3D Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let sediment_cell_type_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sediment Cell Type 3D Bind Group"),
            layout: &sediment_cell_type_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sediment_cell_type_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pressure.cell_type_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: p2g.sediment_count_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: p2g.particle_count_buffer.as_entire_binding() },
            ],
        });

        let sediment_cell_type_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sediment Cell Type 3D Pipeline Layout"),
            bind_group_layouts: &[&sediment_cell_type_bind_group_layout],
            push_constant_ranges: &[],
        });

        let sediment_cell_type_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sediment Cell Type 3D Pipeline"),
            layout: Some(&sediment_cell_type_pipeline_layout),
            module: &sediment_cell_type_shader,
            entry_point: Some("build_sediment_cell_type"),
            compilation_options: Default::default(),
            cache: None,
        });

        let sediment_density_error_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sediment Density Error 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sediment_density_error_3d.wgsl").into()),
        });

        let sediment_density_error_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sediment Density Error 3D Bind Group"),
            layout: &density_error_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: density_error_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: p2g.sediment_count_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pressure.cell_type_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: pressure.divergence_buffer.as_entire_binding() },
            ],
        });

        let sediment_density_error_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sediment Density Error 3D Pipeline"),
            layout: Some(&density_error_pipeline_layout),
            module: &sediment_density_error_shader,
            entry_point: Some("compute_sediment_density_error"),
            compilation_options: Default::default(),
            cache: None,
        });

        let sediment_density_correct_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sediment Density Correct 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sediment_density_correct_3d.wgsl").into()),
        });

        let sediment_density_correct_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sediment Density Correct 3D Pipeline"),
            layout: Some(&density_correct_pipeline_layout),
            module: &sediment_density_correct_shader,
            entry_point: Some("correct_positions"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ========== SDF Collision (Advection + Solid Collision) ==========
        let sdf_collision_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Collision 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sdf_collision_3d.wgsl").into()),
        });

        let sdf_collision_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Collision Params 3D"),
            size: std::mem::size_of::<SdfCollisionParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });



        // Initialize SDF buffer with "infinity" (outside) so we default to no collision if not provided
        // We use 1000.0 * cell_size as a safe "far away" distance
        let sdf_size = (width * height * depth) as usize;
        let dummy_sdf = vec![1000.0 * cell_size; sdf_size];
        let sdf_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SDF Buffer 3D"),
            contents: bytemuck::cast_slice(&dummy_sdf),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });


        let sdf_collision_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SDF Collision 3D Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let sdf_collision_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SDF Collision 3D Bind Group"),
            layout: &sdf_collision_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sdf_collision_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: g2p.positions_buffer.as_ref().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: g2p.velocities_buffer.as_ref().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: sdf_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: bed_height_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: densities_buffer.as_entire_binding() },
            ],
        });

        let sdf_collision_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SDF Collision 3D Pipeline Layout"),
            bind_group_layouts: &[&sdf_collision_bind_group_layout],
            push_constant_ranges: &[],
        });

        let sdf_collision_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SDF Collision 3D Pipeline"),
            layout: Some(&sdf_collision_pipeline_layout),
            module: &sdf_collision_shader,
            entry_point: Some("sdf_collision"),
            compilation_options: Default::default(),
            cache: None,
        });

        let readback_slots = vec![
            ReadbackSlot::new(device, max_particles),
            ReadbackSlot::new(device, max_particles),
        ];

        Self {
            width,
            height,
            depth,
            cell_size,
            vorticity_epsilon: 0.05,
            sediment_rest_particles: 8.0,
            sediment_drag_rate: 8.0,
            sediment_settling_velocity: 0.35,
            sediment_vorticity_lift: 0.08,
            sediment_vorticity_threshold: 0.15,
            sediment_porosity_drag: 3.0,
            positions_buffer,
            velocities_buffer,
            c_col0_buffer,
            c_col1_buffer,
            c_col2_buffer,
            densities_buffer,
            p2g,
            g2p,
            pressure,
            gravity_pipeline,
            gravity_bind_group,
            gravity_params_buffer,
            flow_pipeline,
            flow_bind_group,
            flow_params_buffer,
            vorticity_x_buffer,
            vorticity_y_buffer,
            vorticity_z_buffer,
            vorticity_mag_buffer,
            vorticity_compute_pipeline,
            vorticity_compute_bind_group,
            vorticity_confine_u_pipeline,
            vorticity_confine_v_pipeline,
            vorticity_confine_w_pipeline,
            vorticity_confine_bind_group,
            vorticity_params_buffer,
            vort_confine_params_buffer,
            sediment_fraction_buffer,
            sediment_fraction_pipeline,
            sediment_fraction_bind_group,
            sediment_fraction_params_buffer,
            sediment_pressure_buffer,
            sediment_pressure_pipeline,
            sediment_pressure_bind_group,
            sediment_pressure_params_buffer,
            porosity_drag_u_pipeline,
            porosity_drag_v_pipeline,
            porosity_drag_w_pipeline,
            porosity_drag_bind_group,
            porosity_drag_params_buffer,
            bc_u_pipeline,
            bc_v_pipeline,
            bc_w_pipeline,
            bc_bind_group,
            bc_params_buffer,
            grid_u_old_buffer,
            grid_v_old_buffer,
            grid_w_old_buffer,
            density_error_pipeline,
            density_error_bind_group,
            density_error_params_buffer,
            density_position_grid_pipeline,
            density_position_grid_bind_group,
            density_position_grid_params_buffer,
            position_delta_x_buffer,
            position_delta_y_buffer,
            position_delta_z_buffer,
            density_correct_pipeline,
            density_correct_bind_group,
            density_correct_params_buffer,
            sediment_cell_type_pipeline,
            sediment_cell_type_bind_group,
            sediment_cell_type_params_buffer,
            sediment_density_error_pipeline,
            sediment_density_error_bind_group,
            sediment_density_correct_pipeline,
            sdf_collision_pipeline,
            sdf_collision_bind_group,
            sdf_collision_params_buffer,
            sdf_buffer,
            bed_height_buffer,
            sdf_uploaded: false,
            max_particles,
            readback_slots,
            readback_cursor: 0,
        }
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
        if particle_count == 0 { return; }

        // 1. P2G
        self.p2g.prepare(queue, particle_count, self.cell_size);
        self.p2g.encode(encoder, particle_count);

        // 2. Pressure (Simplified for now - might need more steps for stability)
        self.pressure.encode(encoder, 40); // 40 iterations

        // 3. G2P
        let sediment_params = SedimentParams3D {
            drag_rate: 100.0,
            settling_velocity: 1.0,
            vorticity_lift: 0.1,
            vorticity_threshold: 0.1,
            _pad: [0.0; 4],
        };
        self.g2p.upload_params(queue, particle_count, self.cell_size, dt, sediment_params);
        self.g2p.encode(encoder, particle_count);
        
        // 4. SDF Collision
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("SDF Collision Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.sdf_collision_pipeline);
        pass.set_bind_group(0, &self.sdf_collision_bind_group, &[]);
        let workgroups = (particle_count + 255) / 256;
        pass.dispatch_workgroups(workgroups, 1, 1);
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

    pub fn positions_buffer(&self) -> Arc<wgpu::Buffer> {
        self.positions_buffer.clone()
    }

    pub fn velocities_buffer(&self) -> Arc<wgpu::Buffer> {
        self.velocities_buffer.clone()
    }

    pub fn densities_buffer(&self) -> Arc<wgpu::Buffer> {
        self.densities_buffer.clone()
    }

    pub fn set_drucker_prager_params(&self, queue: &wgpu::Queue, params: DruckerPragerParams) {
        self.g2p.set_drucker_prager_params(queue, params);
    }

    pub fn sdf_buffer(&self) -> &wgpu::Buffer {
        &self.sdf_buffer
    }

    pub fn bed_height_buffer(&self) -> Arc<wgpu::Buffer> {
        self.bed_height_buffer.clone()
    }

    fn upload_bc_and_cell_types(&self, queue: &wgpu::Queue, cell_types: &[u32]) {
        // Upload cell types FIRST (needed for BC enforcement)
        self.pressure.upload_cell_types(queue, cell_types, self.cell_size);

        // Upload BC params
        let bc_params = BcParams3D {
            width: self.width,
            height: self.height,
            depth: self.depth,
            _pad: 0,
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

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FLIP 3D Step Encoder"),
        });

        // Run P2G scatter and divide
        self.p2g.encode(&mut encoder, count);

        queue.submit(std::iter::once(encoder.finish()));

        let sediment_fraction_params = SedimentFractionParams3D {
            width: self.width,
            height: self.height,
            depth: self.depth,
            rest_particles: self.sediment_rest_particles,
        };
        queue.write_buffer(&self.sediment_fraction_params_buffer, 0, bytemuck::bytes_of(&sediment_fraction_params));

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
            let workgroups_x = (self.width + 7) / 8;
            let workgroups_y = (self.height + 7) / 8;
            let workgroups_z = (self.depth + 3) / 4;
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
        queue.write_buffer(&self.sediment_pressure_params_buffer, 0, bytemuck::bytes_of(&sediment_pressure_params));

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
            let workgroups_x = (self.width + 7) / 8;
            let workgroups_z = (self.depth + 7) / 8;
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
            let workgroups_x = (self.width + 1 + 7) / 8;
            let workgroups_y = (self.height + 7) / 8;
            let workgroups_z = (self.depth + 3) / 4;
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
            let workgroups_x = (self.width + 7) / 8;
            let workgroups_y = (self.height + 1 + 7) / 8;
            let workgroups_z = (self.depth + 3) / 4;
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
            let workgroups_x = (self.width + 7) / 8;
            let workgroups_y = (self.height + 7) / 8;
            let workgroups_z = (self.depth + 1 + 3) / 4;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        queue.submit(std::iter::once(encoder.finish()));

        // 3. Save grid velocity for FLIP delta (now with proper BCs!)
        let (u_size, v_size, w_size) = self.p2g.grid_sizes();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FLIP 3D Grid Copy Encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.p2g.grid_u_buffer, 0, &self.grid_u_old_buffer, 0, (u_size * 4) as u64);
        encoder.copy_buffer_to_buffer(&self.p2g.grid_v_buffer, 0, &self.grid_v_old_buffer, 0, (v_size * 4) as u64);
        encoder.copy_buffer_to_buffer(&self.p2g.grid_w_buffer, 0, &self.grid_w_old_buffer, 0, (w_size * 4) as u64);
        queue.submit(std::iter::once(encoder.finish()));

        // 4. Apply gravity

        let gravity_params = GravityParams3D {
            width: self.width,
            height: self.height,
            depth: self.depth,
            gravity_dt: gravity * dt,
            cell_size: self.cell_size,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        queue.write_buffer(&self.gravity_params_buffer, 0, bytemuck::bytes_of(&gravity_params));

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
            let workgroups_x = (self.width + 7) / 8;
            let workgroups_y = (self.height + 1 + 7) / 8;
            let workgroups_z = (self.depth + 3) / 4;
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
            queue.write_buffer(&self.flow_params_buffer, 0, bytemuck::bytes_of(&flow_params));

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Flow 3D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.flow_pipeline);
            pass.set_bind_group(0, &self.flow_bind_group, &[]);
            // U grid: (width+1) x height x depth
            let workgroups_x = (self.width + 1 + 7) / 8;
            let workgroups_y = (self.height + 7) / 8;
            let workgroups_z = (self.depth + 3) / 4;
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
            queue.write_buffer(&self.vorticity_params_buffer, 0, bytemuck::bytes_of(&vort_params));

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Vorticity 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.vorticity_compute_pipeline);
                pass.set_bind_group(0, &self.vorticity_compute_bind_group, &[]);
                let workgroups_x = (self.width + 7) / 8;
                let workgroups_y = (self.height + 7) / 8;
                let workgroups_z = (self.depth + 3) / 4;
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }

            let confine_params = VortConfineParams3D {
                width: self.width,
                height: self.height,
                depth: self.depth,
                epsilon_h_dt: self.vorticity_epsilon * self.cell_size * dt,
            };
            queue.write_buffer(&self.vort_confine_params_buffer, 0, bytemuck::bytes_of(&confine_params));

            // Apply confinement to U faces
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Vorticity Confine U 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.vorticity_confine_u_pipeline);
                pass.set_bind_group(0, &self.vorticity_confine_bind_group, &[]);
                let workgroups_x = (self.width + 1 + 7) / 8;
                let workgroups_y = (self.height + 7) / 8;
                let workgroups_z = (self.depth + 3) / 4;
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
                let workgroups_x = (self.width + 7) / 8;
                let workgroups_y = (self.height + 1 + 7) / 8;
                let workgroups_z = (self.depth + 3) / 4;
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
                let workgroups_x = (self.width + 7) / 8;
                let workgroups_y = (self.height + 7) / 8;
                let workgroups_z = (self.depth + 1 + 3) / 4;
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }
        }

        // 7. Pressure solve (divergence → iterations → gradient)
        self.pressure.encode(&mut encoder, pressure_iterations);

        queue.submit(std::iter::once(encoder.finish()));

        if self.sediment_porosity_drag > 0.0 {
            let drag_params = PorosityDragParams3D {
                width: self.width,
                height: self.height,
                depth: self.depth,
                drag_dt: self.sediment_porosity_drag * dt,
            };
            queue.write_buffer(&self.porosity_drag_params_buffer, 0, bytemuck::bytes_of(&drag_params));

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
                let workgroups_x = (self.width + 1 + 7) / 8;
                let workgroups_y = (self.height + 7) / 8;
                let workgroups_z = (self.depth + 3) / 4;
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Porosity Drag V 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.porosity_drag_v_pipeline);
                pass.set_bind_group(0, &self.porosity_drag_bind_group, &[]);
                let workgroups_x = (self.width + 7) / 8;
                let workgroups_y = (self.height + 1 + 7) / 8;
                let workgroups_z = (self.depth + 3) / 4;
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Porosity Drag W 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.porosity_drag_w_pipeline);
                pass.set_bind_group(0, &self.porosity_drag_bind_group, &[]);
                let workgroups_x = (self.width + 7) / 8;
                let workgroups_y = (self.height + 7) / 8;
                let workgroups_z = (self.depth + 1 + 3) / 4;
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }

            queue.submit(std::iter::once(encoder.finish()));
        }

        // 8. Run G2P using grid buffers already on GPU
        let sediment_params = SedimentParams3D {
            drag_rate: self.sediment_drag_rate,
            settling_velocity: self.sediment_settling_velocity,
            vorticity_lift: self.sediment_vorticity_lift,
            vorticity_threshold: self.sediment_vorticity_threshold,
            _pad: [0.0; 4],
        };
        let g2p_count = self.g2p.upload_params(queue, count, self.cell_size, dt, sediment_params);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FLIP 3D G2P Encoder"),
        });
        self.g2p.encode(&mut encoder, g2p_count);
        queue.submit(std::iter::once(encoder.finish()));

        // ========== Density Projection (Implicit Density Projection) ==========
        // Push particles from crowded regions to empty regions
        // This causes water level to "rise" when particles accumulate behind riffles

        // 1. Compute density error from particle counts (populated during P2G)
        let density_error_params = DensityErrorParams3D {
            width: self.width,
            height: self.height,
            depth: self.depth,
            rest_density: 8.0,  // Target ~8 particles per cell
            dt,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        queue.write_buffer(&self.density_error_params_buffer, 0, bytemuck::bytes_of(&density_error_params));

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
            let workgroups_x = (self.width + 7) / 8;
            let workgroups_y = (self.height + 7) / 8;
            let workgroups_z = (self.depth + 3) / 4;
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
        let density_iterations = 40;  // More iterations for volume conservation
        self.pressure.encode_iterations_only(&mut encoder, density_iterations);

        queue.submit(std::iter::once(encoder.finish()));

        // 3. Compute position deltas on grid (blub approach)
        // Update grid shader params with dt
        let grid_params = DensityPositionGridParams3D {
            width: self.width,
            height: self.height,
            depth: self.depth,
            dt,
        };
        queue.write_buffer(&self.density_position_grid_params_buffer, 0, bytemuck::bytes_of(&grid_params));

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
            let workgroups_x = (self.width + 7) / 8;
            let workgroups_y = (self.height + 7) / 8;
            let workgroups_z = (self.depth + 3) / 4;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        queue.submit(std::iter::once(encoder.finish()));

        // 4. Apply position correction to particles (trilinear sampling from grid)
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
        queue.write_buffer(&self.density_correct_params_buffer, 0, bytemuck::bytes_of(&density_correct_params));

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
            let workgroups = (particle_count as u32 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));

        // ========== Sediment Density Projection (Granular Packing) ==========
        if self.sediment_rest_particles > 0.0 {
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

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Sediment Cell Type 3D Encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Sediment Cell Type 3D Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.sediment_cell_type_pipeline);
                pass.set_bind_group(0, &self.sediment_cell_type_bind_group, &[]);
                let workgroups_x = (self.width + 7) / 8;
                let workgroups_y = (self.height + 7) / 8;
                let workgroups_z = (self.depth + 3) / 4;
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }
            queue.submit(std::iter::once(encoder.finish()));

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
            particle_count: particle_count as u32,
            cell_size: self.cell_size,
            dt,
            _pad0: 0,
            _pad1: 0,
        };
        queue.write_buffer(&self.sdf_collision_params_buffer, 0, bytemuck::bytes_of(&sdf_params));

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
            let workgroups = (g2p_count + 255) / 256;
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
            ReadbackMode::Sync => {
                // Download results (velocities + C matrix + positions).
                self.g2p.download(device, queue, g2p_count, velocities, c_matrices);
                let mut gpu_positions = vec![[0.0f32; 4]; particle_count];
                Self::read_buffer_vec4(device, queue, &self.g2p.positions_buffer, &mut gpu_positions, particle_count);
                for (i, pos) in gpu_positions.iter().enumerate() {
                    positions[i] = glam::Vec3::new(pos[0], pos[1], pos[2]);
                }
                true
            }
            ReadbackMode::Async => self.schedule_readback(device, queue, particle_count),
        }
    }

    fn read_buffer_vec4(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer, output: &mut [[f32; 4]], count: usize) {
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
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        {
            let data = buffer_slice.get_mapped_range();
            let slice: &[[f32; 4]] = bytemuck::cast_slice(&data);
            output[..count].copy_from_slice(&slice[..count]);
        }
        staging.unmap();
    }
}
