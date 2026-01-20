use super::flip_3d::GpuFlip3D;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct EmitterParams3D {
    pos: [f32; 3],
    radius: f32,
    vel: [f32; 3],
    spread: f32,
    count: u32,
    density: f32,
    time: f32,
    max_particles: u32,
    // Grid params for density checking
    grid_width: u32,
    grid_height: u32,
    grid_depth: u32,
    cell_size: f32,
    max_ppc: f32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct AbsorptionParams3D {
    particle_count: u32,
    world_width: u32,
    world_depth: u32,
    cell_size: f32,
    sediment_volume_per_particle: f32,
    water_volume_per_particle: f32,
    dt: f32,
    absorption_radius: f32,
}

/// Bridge between 3D FLIP particles and 2.5D Heightfield.
/// Handles spawning particles (Ejection) and merging them back (Absorption).
pub struct GpuBridge3D {
    emitter_pipeline: wgpu::ComputePipeline,
    absorption_pipeline: wgpu::ComputePipeline,

    // Shared GPU particle count
    pub count_buffer: wgpu::Buffer,

    // Transfer buffers (atomic<i32> fixed-point)
    pub transfer_sediment_buffer: wgpu::Buffer,
    pub transfer_water_buffer: wgpu::Buffer,

    emitter_params_buffer: wgpu::Buffer,
    absorption_params_buffer: wgpu::Buffer,

    emitter_bind_group: wgpu::BindGroup,
    absorption_bind_group: wgpu::BindGroup,

    max_particles: usize,
}

impl GpuBridge3D {
    pub fn new(
        device: &wgpu::Device,
        flip: &GpuFlip3D,
        world_width: u32,
        world_depth: u32,
    ) -> Self {
        let emitter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("3D Emitter Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/particle_emitter_3d.wgsl").into(),
            ),
        });

        let absorption_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("3D Absorption Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/particle_absorption_3d.wgsl").into(),
            ),
        });

        let count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Global Particle Count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let transfer_size = (world_width * world_depth * 4) as u64;
        let transfer_sediment_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Transfer Sediment Atomics"),
            size: transfer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let transfer_water_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Transfer Water Atomics"),
            size: transfer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let emitter_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Emitter Params 3D"),
            size: std::mem::size_of::<EmitterParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let absorption_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Absorption Params 3D"),
            size: std::mem::size_of::<AbsorptionParams3D>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // BIND GROUPS
        let emitter_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Emitter 3D Layout"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let emitter_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Emitter 3D Bind Group"),
            layout: &emitter_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: emitter_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: flip.positions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: flip.velocities_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: flip.densities_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: flip.p2g_particle_count_buffer().as_entire_binding(),
                },
            ],
        });

        let absorption_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Absorption 3D Layout"),
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
                ],
            });

        let absorption_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Absorption 3D Bind Group"),
            layout: &absorption_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: absorption_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: flip.positions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: flip.velocities_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: flip.densities_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: flip.bed_height_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: transfer_sediment_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: transfer_water_buffer.as_entire_binding(),
                },
            ],
        });

        // PIPELINES
        let emitter_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Emitter 3D Pipe Layout"),
                bind_group_layouts: &[&emitter_bind_group_layout],
                push_constant_ranges: &[],
            });

        let emitter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Emitter 3D Pipeline"),
            layout: Some(&emitter_pipeline_layout),
            module: &emitter_shader,
            entry_point: Some("spawn_particles"),
            compilation_options: Default::default(),
            cache: None,
        });

        let absorption_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Absorption 3D Pipe Layout"),
                bind_group_layouts: &[&absorption_bind_group_layout],
                push_constant_ranges: &[],
            });

        let absorption_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Absorption 3D Pipeline"),
                layout: Some(&absorption_pipeline_layout),
                module: &absorption_shader,
                entry_point: Some("absorb_particles"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            emitter_pipeline,
            absorption_pipeline,
            count_buffer,
            transfer_sediment_buffer,
            transfer_water_buffer,
            emitter_params_buffer,
            absorption_params_buffer,
            emitter_bind_group,
            absorption_bind_group,
            max_particles: flip.positions_buffer.size() as usize / 16,
        }
    }

    pub fn set_particle_count(&self, queue: &wgpu::Queue, count: u32) {
        queue.write_buffer(&self.count_buffer, 0, bytemuck::bytes_of(&count));
    }

    pub fn dispatch_emitter(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        flip: &GpuFlip3D,
        pos: [f32; 3],
        vel: [f32; 3],
        radius: f32,
        spread: f32,
        count: u32,
        density: f32,
        time: f32,
    ) {
        if count == 0 {
            return;
        }

        let params = EmitterParams3D {
            pos,
            radius,
            vel,
            spread,
            count,
            density,
            time,
            max_particles: self.max_particles as u32,
            grid_width: flip.width,
            grid_height: flip.height,
            grid_depth: flip.depth,
            cell_size: flip.cell_size,
            max_ppc: 8.0,  // Target rest density
            _pad: [0; 3],
        };
        queue.write_buffer(&self.emitter_params_buffer, 0, bytemuck::bytes_of(&params));

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Emitter 3D Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.emitter_pipeline);
        pass.set_bind_group(0, &self.emitter_bind_group, &[]);
        let groups = count.div_ceil(64);
        pass.dispatch_workgroups(groups, 1, 1);
    }

    pub fn dispatch_absorption(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        particle_count: u32,
        world_width: u32,
        world_depth: u32,
        cell_size: f32,
        dt: f32,
        absorption_radius: f32,
    ) {
        if particle_count == 0 {
            return;
        }

        let params = AbsorptionParams3D {
            particle_count,
            world_width,
            world_depth,
            cell_size,
            sediment_volume_per_particle: 0.0001, // 100cm^3?
            water_volume_per_particle: 0.0001,
            dt,
            absorption_radius,
        };
        queue.write_buffer(
            &self.absorption_params_buffer,
            0,
            bytemuck::bytes_of(&params),
        );

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Absorption 3D Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.absorption_pipeline);
        pass.set_bind_group(0, &self.absorption_bind_group, &[]);
        let groups = particle_count.div_ceil(256);
        pass.dispatch_workgroups(groups, 1, 1);
    }

    pub fn clear_transfers(&self, encoder: &mut wgpu::CommandEncoder) {
        // encoder.clear_buffer is available in wgpu 0.17+
        encoder.clear_buffer(&self.transfer_sediment_buffer, 0, None);
        encoder.clear_buffer(&self.transfer_water_buffer, 0, None);
    }
}
