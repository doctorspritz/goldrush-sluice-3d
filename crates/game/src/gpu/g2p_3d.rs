//! GPU-accelerated Grid-to-Particle (G2P) transfer for 3D FLIP simulation.
//!
//! This module implements G2P for APIC-FLIP using 3D compute shaders.
//! Unlike P2G (scatter), G2P is a gather operation where each particle reads
//! from its 3x3x3 grid neighborhood - no atomics needed.
//!
//! Output per particle:
//! - Updated velocity (FLIP/PIC blend)
//! - Reconstructed C matrix (3x3 APIC affine velocity)
//!
//! The FLIP delta is computed using grid_*_old (pre-force velocities)
//! stored by the caller after P2G but before forces are applied.

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

/// Parameters for G2P 3D compute shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct G2pParams3D {
    cell_size: f32,
    width: u32,
    height: u32,
    depth: u32,
    particle_count: u32,
    d_inv: f32,        // APIC D inverse = 4/dx^2
    flip_ratio: f32,   // FLIP blend ratio (0.97 for water)
    dt: f32,           // Time step for velocity clamping
    max_velocity: f32, // Safety clamp (2000.0)
    diameter_offset: u32,
    _padding: [u32; 2], // Align to 48 bytes
}

/// Sediment parameters for G2P - drag-based entrainment model.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SedimentParams3D {
    /// How fast sediment falls (m/s). Gold ~0.15, sand ~0.05, mud ~0.001
    pub settling_velocity: f32,
    /// Speed below which friction kicks in (m/s). Typical: 0.1
    pub friction_threshold: f32,
    /// Damping fraction at 60 Hz; integrated with exponential decay.
    pub friction_strength: f32,
    /// How much vorticity suspends sediment. Typical: 1.0-2.0
    pub vorticity_lift: f32,
    /// Minimum vorticity to lift. Typical: 1.0-5.0
    pub vorticity_threshold: f32,
    /// Fallback relaxation rate (1/s) when no terminal settling speed is set.
    /// With a terminal speed, the shader derives the rate from submerged gravity.
    pub drag_coefficient: f32,
    /// Density threshold for gold particles.
    pub gold_density_threshold: f32,
    /// Drag multiplier applied to gold (fine gold entrains more).
    pub gold_drag_multiplier: f32,
    /// Gold terminal settling speed in still water (m/s).
    pub gold_settling_velocity: f32,
    /// Upward bias for flaky gold near the surface (m/s^2).
    pub gold_flake_lift: f32,
    /// Gravity magnitude (m/s²) used for sediment buoyancy. Defaults to 9.81;
    /// driven from the runtime gravity in `step` so passing gravity=0 actually
    /// disables sediment sinking (previously buoyancy used a hardcoded const).
    pub gravity: f32,
    pub _pad: [f32; 1],
}

impl Default for SedimentParams3D {
    fn default() -> Self {
        Self {
            settling_velocity: 0.0,     // No prescribed terminal speed; use fallback drag
            friction_threshold: 0.0,    // No friction threshold
            friction_strength: 0.0,     // No heuristic low-speed damping
            vorticity_lift: 0.0,        // Not needed with no settling
            vorticity_threshold: 999.0, // Never triggers
            drag_coefficient: 10.0,     // Moderate drag - particles entrain in flow
            gold_density_threshold: 10.0,
            gold_drag_multiplier: 1.0,
            gold_settling_velocity: 0.0,
            gold_flake_lift: 0.0,
            gravity: 9.81, // preserve prior buoyancy behavior by default
            _pad: [0.0; 1],
        }
    }
}

/// GPU-based Grid-to-Particle transfer for 3D simulation
pub struct GpuG2p3D {
    particle_metadata_buffer: Arc<wgpu::Buffer>,
    conservative_phase_exchange: bool,
    width: u32,
    height: u32,
    depth: u32,

    // Particle buffers (positions public for density projection)
    pub positions_buffer: Arc<wgpu::Buffer>,
    pub(crate) velocities_buffer: Arc<wgpu::Buffer>,
    #[allow(dead_code)]
    densities_buffer: Arc<wgpu::Buffer>,
    // C matrix columns (output)
    c_col0_buffer: Arc<wgpu::Buffer>,
    c_col1_buffer: Arc<wgpu::Buffer>,
    c_col2_buffer: Arc<wgpu::Buffer>,

    // Staging buffers for readback
    velocities_staging: wgpu::Buffer,
    c_col0_staging: wgpu::Buffer,
    c_col1_staging: wgpu::Buffer,
    c_col2_staging: wgpu::Buffer,

    // Parameters
    params_buffer: wgpu::Buffer,
    sediment_params_buffer: wgpu::Buffer,

    // Compute pipeline
    g2p_pipeline: wgpu::ComputePipeline,

    // Bind group
    bind_group: wgpu::BindGroup,

    // Current capacity
    max_particles: usize,

    // Workgroup size
    workgroup_size: u32,
}

impl GpuG2p3D {
    /// Create a new GPU G2P 3D solver
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        depth: u32,
        max_particles: usize,
        positions_buffer: Arc<wgpu::Buffer>,
        velocities_buffer: Arc<wgpu::Buffer>,
        c_col0_buffer: Arc<wgpu::Buffer>,
        c_col1_buffer: Arc<wgpu::Buffer>,
        c_col2_buffer: Arc<wgpu::Buffer>,
        densities_buffer: Arc<wgpu::Buffer>,
        grid_u_buffer: &wgpu::Buffer,
        grid_v_buffer: &wgpu::Buffer,
        grid_w_buffer: &wgpu::Buffer,
        grid_u_old_buffer: &wgpu::Buffer,
        grid_v_old_buffer: &wgpu::Buffer,
        grid_w_old_buffer: &wgpu::Buffer,
        vorticity_mag_buffer: &wgpu::Buffer,
        water_grid_u_buffer: &wgpu::Buffer,
        water_grid_v_buffer: &wgpu::Buffer,
        water_grid_w_buffer: &wgpu::Buffer,
        water_offsets: [u64; 3],
    ) -> Self {
        Self::new_with_metadata(
            device,
            width,
            height,
            depth,
            max_particles,
            positions_buffer,
            velocities_buffer,
            c_col0_buffer,
            c_col1_buffer,
            c_col2_buffer,
            densities_buffer,
            grid_u_buffer,
            grid_v_buffer,
            grid_w_buffer,
            grid_u_old_buffer,
            grid_v_old_buffer,
            grid_w_old_buffer,
            vorticity_mag_buffer,
            water_grid_u_buffer,
            water_grid_v_buffer,
            water_grid_w_buffer,
            water_offsets,
            None,
        )
    }

    /// Shared phase metadata must contain four capacity planes (rho,d,b,c).
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_metadata(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        depth: u32,
        max_particles: usize,
        positions_buffer: Arc<wgpu::Buffer>,
        velocities_buffer: Arc<wgpu::Buffer>,
        c_col0_buffer: Arc<wgpu::Buffer>,
        c_col1_buffer: Arc<wgpu::Buffer>,
        c_col2_buffer: Arc<wgpu::Buffer>,
        densities_buffer: Arc<wgpu::Buffer>,
        grid_u_buffer: &wgpu::Buffer,
        grid_v_buffer: &wgpu::Buffer,
        grid_w_buffer: &wgpu::Buffer,
        grid_u_old_buffer: &wgpu::Buffer,
        grid_v_old_buffer: &wgpu::Buffer,
        grid_w_old_buffer: &wgpu::Buffer,
        vorticity_mag_buffer: &wgpu::Buffer,
        water_grid_u_buffer: &wgpu::Buffer,
        water_grid_v_buffer: &wgpu::Buffer,
        water_grid_w_buffer: &wgpu::Buffer,
        water_offsets: [u64; 3],
        metadata: Option<Arc<wgpu::Buffer>>,
    ) -> Self {
        let phase_buffers = metadata.is_some();
        let mut water_sizes = [
            ((width + 1) * height * depth) as u64,
            (width * (height + 1) * depth) as u64,
            (width * height * (depth + 1)) as u64,
        ]
        .map(|n| {
            if phase_buffers {
                ((n + 4).div_ceil(64) * 64 + 4 * n) * 4
            } else {
                n * 4
            }
        });
        if phase_buffers {
            // Paired voxel reflection reads immutable solid flags after the
            // response plane. Binding only that earlier prefix silently made
            // every geometry query out of bounds in the gather shader.
            let cells = u64::from(width) * u64::from(height) * u64::from(depth);
            water_sizes[0] = (super::phase_transfer_3d::PhaseTransfer3D::cell_count_offset([
                width, height, depth,
            ]) + cells)
                * 4;
            assert!(metadata.as_ref().unwrap().size() >= (max_particles as u64) * 16);
            for (axis, buffer) in [
                water_grid_u_buffer,
                water_grid_v_buffer,
                water_grid_w_buffer,
            ]
            .into_iter()
            .enumerate()
            {
                assert!(
                    buffer.size() >= water_offsets[axis] + water_sizes[axis],
                    "phase gather response or solid-marker plane is truncated"
                );
            }
        }
        let particle_metadata_buffer = metadata.unwrap_or_else(|| {
            Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Grain density and diameter metadata"),
                size: (max_particles * 8) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        });
        let phase_constants = std::collections::HashMap::from([(
            "PHASE_EXCHANGE_ALLOWED".to_string(),
            f64::from(u32::from(phase_buffers)),
        )]);
        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("G2P 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}",
                    include_str!("shaders/mac_wall_3d.wgsl"),
                    include_str!("shaders/g2p_3d.wgsl")
                )
                .into(),
            ),
        });

        // Staging buffers for readback
        let velocities_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G2P 3D Velocities Staging"),
            size: (max_particles * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let c_col0_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G2P 3D C Col0 Staging"),
            size: (max_particles * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let c_col1_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G2P 3D C Col1 Staging"),
            size: (max_particles * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let c_col2_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G2P 3D C Col2 Staging"),
            size: (max_particles * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create params buffer
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G2P 3D Params"),
            size: std::mem::size_of::<G2pParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sediment_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G2P 3D Sediment Params"),
            size: std::mem::size_of::<SedimentParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout (matches shader bindings 0-17)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("G2P 3D Bind Group Layout"),
            entries: &[
                // 0: params (uniform)
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
                // 1: positions (read)
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
                // 2: velocities (read_write)
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
                // 3-5: c_col0, c_col1, c_col2 (read_write)
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
                // 6-8: grid_u, grid_v, grid_w (read)
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 9-11: grid_u_old, grid_v_old, grid_w_old (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 12: densities (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 13: sediment params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 14: vorticity magnitude (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 15-17: water-only grid_u, grid_v, grid_w (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 15,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 16,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 17,
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

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("G2P 3D Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: positions_buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: velocities_buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: c_col0_buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: c_col1_buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: c_col2_buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: grid_u_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: grid_v_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: grid_w_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: grid_u_old_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: grid_v_old_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: grid_w_old_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: particle_metadata_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: sediment_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: vorticity_mag_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: water_grid_u_buffer,
                        offset: water_offsets[0],
                        size: std::num::NonZeroU64::new(water_sizes[0]),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: water_grid_v_buffer,
                        offset: water_offsets[1],
                        size: std::num::NonZeroU64::new(water_sizes[1]),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: water_grid_w_buffer,
                        offset: water_offsets[2],
                        size: std::num::NonZeroU64::new(water_sizes[2]),
                    }),
                },
            ],
        });

        // Create pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("G2P 3D Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let g2p_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("G2P 3D Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("g2p"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &phase_constants,
                ..Default::default()
            },
            cache: None,
        });

        Self {
            particle_metadata_buffer,
            conservative_phase_exchange: false,
            width,
            height,
            depth,
            positions_buffer,
            velocities_buffer,
            densities_buffer,
            c_col0_buffer,
            c_col1_buffer,
            c_col2_buffer,
            velocities_staging,
            c_col0_staging,
            c_col1_staging,
            c_col2_staging,
            params_buffer,
            sediment_params_buffer,
            g2p_pipeline,
            bind_group,
            max_particles,
            workgroup_size: 128,
        }
    }

    /// Metadata shares the original density binding: keep the storage-buffer
    /// requirement at 16 rather than excluding adapters with that limit.
    pub fn set_conservative_phase_exchange(&mut self, enabled: bool) {
        assert!(
            !enabled || self.particle_metadata_buffer.size() >= (self.max_particles as u64) * 16,
            "phase gather requires shared four-plane metadata"
        );
        self.conservative_phase_exchange = enabled;
    }

    pub fn upload_grain_diameters(&self, queue: &wgpu::Queue, first: usize, values: &[f32]) {
        if !values.is_empty() {
            queue.write_buffer(
                &self.particle_metadata_buffer,
                ((self.max_particles + first) * 4) as u64,
                bytemuck::cast_slice(values),
            );
        }
    }

    /// Set the boundary mask after uploading this step's shared parameters.
    pub fn set_open_boundaries(&self, queue: &wgpu::Queue, open: u32) {
        assert!(open <= 63);
        queue.write_buffer(
            &self.params_buffer,
            (std::mem::size_of::<G2pParams3D>() - 4) as u64,
            bytemuck::bytes_of(&open),
        );
    }

    /// Upload G2P params to GPU (particle data is shared with P2G).
    pub fn upload_params(
        &self,
        queue: &wgpu::Queue,
        particle_count: u32,
        cell_size: f32,
        dt: f32,
        flip_ratio: f32,
        sediment_params: SedimentParams3D,
    ) -> u32 {
        let particle_count = particle_count.min(self.max_particles as u32);

        // Upload params
        let d_inv = 4.0 / (cell_size * cell_size);
        let params = G2pParams3D {
            cell_size,
            width: self.width,
            height: self.height,
            depth: self.depth,
            particle_count,
            d_inv,
            flip_ratio,
            dt,
            max_velocity: 0.9 * cell_size / dt.max(1e-6), // Safety follows CFL, not an arbitrary 1 m/s sluice limit
            diameter_offset: self.max_particles as u32,
            _padding: [
                u32::from(self.conservative_phase_exchange),
                if self.conservative_phase_exchange {
                    63
                } else {
                    0
                },
            ],
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
        queue.write_buffer(
            &self.sediment_params_buffer,
            0,
            bytemuck::bytes_of(&sediment_params),
        );

        particle_count
    }

    /// Encode G2P compute pass into command encoder
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, particle_count: u32) {
        self.encode_profiled(encoder, particle_count, None);
    }

    pub fn encode_profiled(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        particle_count: u32,
        timer: Option<&mut super::timing::GpuStageTimer>,
    ) {
        encoder.copy_buffer_to_buffer(
            &self.densities_buffer,
            0,
            &self.particle_metadata_buffer,
            0,
            (particle_count as u64) * 4,
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("G2P 3D Pass"),
            timestamp_writes: timer.map(|timer| timer.pass("G2P 3D Pass")),
        });
        pass.set_pipeline(&self.g2p_pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        let workgroups = particle_count.div_ceil(self.workgroup_size);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }


    /// Download particle data from GPU
    pub fn download(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particle_count: u32,
        velocities: &mut [glam::Vec3],
        c_matrices: &mut [glam::Mat3],
    ) {
        let count = particle_count as usize;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("G2P 3D Download Encoder"),
        });

        // Copy to staging
        encoder.copy_buffer_to_buffer(
            &self.velocities_buffer,
            0,
            &self.velocities_staging,
            0,
            (count * 16) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &self.c_col0_buffer,
            0,
            &self.c_col0_staging,
            0,
            (count * 16) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &self.c_col1_buffer,
            0,
            &self.c_col1_staging,
            0,
            (count * 16) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &self.c_col2_buffer,
            0,
            &self.c_col2_staging,
            0,
            (count * 16) as u64,
        );

        queue.submit(std::iter::once(encoder.finish()));

        // Read velocities
        let vel_data = Self::read_staging_buffer_vec4(device, &self.velocities_staging, count);
        for (i, v) in vel_data.iter().enumerate().take(count) {
            velocities[i] = glam::Vec3::new(v[0], v[1], v[2]);
        }

        // Read C matrices
        let c0_data = Self::read_staging_buffer_vec4(device, &self.c_col0_staging, count);
        let c1_data = Self::read_staging_buffer_vec4(device, &self.c_col1_staging, count);
        let c2_data = Self::read_staging_buffer_vec4(device, &self.c_col2_staging, count);

        for i in 0..count {
            c_matrices[i] = glam::Mat3::from_cols(
                glam::Vec3::new(c0_data[i][0], c0_data[i][1], c0_data[i][2]),
                glam::Vec3::new(c1_data[i][0], c1_data[i][1], c1_data[i][2]),
                glam::Vec3::new(c2_data[i][0], c2_data[i][1], c2_data[i][2]),
            );
        }
    }

    fn read_staging_buffer_vec4(
        device: &wgpu::Device,
        buffer: &wgpu::Buffer,
        count: usize,
    ) -> Vec<[f32; 4]> {
        let buffer_slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        if let Err(e) = super::await_buffer_map(rx) {
            log::error!("GPU G2P readback failed: {}", e);
            return Vec::new();
        }

        let data = buffer_slice.get_mapped_range();
        let result: Vec<[f32; 4]> = bytemuck::cast_slice(&data)[..count].to_vec();
        drop(data);
        buffer.unmap();
        result
    }
}
