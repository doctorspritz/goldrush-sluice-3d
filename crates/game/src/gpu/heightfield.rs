use std::sync::Arc;
use wgpu::util::DeviceExt;

/// GPU-accelerated Heightfield Simulation.
/// 
/// Manages the state for:
/// - Hydrodynamics (Shallow Water Equations)
/// - Erosion & Sediment Transport
/// - Multi-layer Geology (Bedrock, Paydirt, Overburden)
pub struct GpuHeightfield {
    width: u32,
    depth: u32,
    
    // Geology Buffers
    pub bedrock_buffer: wgpu::Buffer,
    pub paydirt_buffer: wgpu::Buffer,
    pub overburden_buffer: wgpu::Buffer,
    pub sediment_buffer: wgpu::Buffer, // Deposited sediment
    
    // Water State Buffers
    pub water_depth_buffer: wgpu::Buffer,
    pub water_velocity_x_buffer: wgpu::Buffer,
    pub water_velocity_z_buffer: wgpu::Buffer,
    pub suspended_sediment_buffer: wgpu::Buffer,
    
    // Derived/Intermediate Buffers
    pub water_surface_buffer: wgpu::Buffer, // Calculated as Ground + Water Depth
    pub flux_x_buffer: wgpu::Buffer,
    pub flux_z_buffer: wgpu::Buffer,
    
    // Bind Groups
    pub params_bind_group: wgpu::BindGroup,
    pub terrain_bind_group: wgpu::BindGroup,
    pub water_bind_group: wgpu::BindGroup,
    
    // Pipelines
    pub surface_pipeline: wgpu::ComputePipeline,
    pub flux_pipeline: wgpu::ComputePipeline,
    pub depth_pipeline: wgpu::ComputePipeline,
    
    pub erosion_pipeline: wgpu::ComputePipeline,
    pub sediment_transport_pipeline: wgpu::ComputePipeline,
    
    // Emitter Pipeline
    pub emitter_pipeline: wgpu::ComputePipeline,
    pub emitter_params_buffer: wgpu::Buffer,
    pub emitter_bind_group: wgpu::BindGroup,
    
    // Params Buffer (to update every frame)
    pub params_buffer: wgpu::Buffer,
}

impl GpuHeightfield {
    pub fn new(device: &wgpu::Device, width: u32, depth: u32, initial_height: f32) -> Self {
        let size = (width * depth) as usize * std::mem::size_of::<f32>();
        
        // Helper to create valid storage buffers
        let create_storage = |label: &str, init_val: f32| -> wgpu::Buffer {
            let data = vec![init_val; (width * depth) as usize];
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            })
        };

        // 1. Initialize Geology
        let bedrock = create_storage("Bedrock Buffer", initial_height * 0.5);
        let paydirt = create_storage("Paydirt Buffer", initial_height * 0.3);
        let overburden = create_storage("Overburden Buffer", initial_height * 0.2);
        let sediment = create_storage("Sediment Buffer", 0.0);
        
        // 2. Initialize Water
        let water_depth = create_storage("Water Depth Buffer", 0.0);
        let water_vel_x = create_storage("Water Vel X Buffer", 0.0);
        let water_vel_z = create_storage("Water Vel Z Buffer", 0.0);
        let suspended = create_storage("Suspended Sediment Buffer", 0.0);
        
        // 3. Initialize Intermediate
        let water_surface = create_storage("Water Surface Buffer", initial_height); // approx
        let flux_x = create_storage("Flux X Buffer", 0.0);
        let flux_z = create_storage("Flux Z Buffer", 0.0);

        // Bind Group Layouts (Placeholder - will implement in next step with proper layout)
        // For now constructing the struct fields. simpler to create layout and bindgroups here.
        
        // 4. Uniforms
        let params_size = std::mem::size_of::<[u32; 8]>(); // Alignment padding safe size
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Heightfield Params"),
            size: params_size as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Load Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Heightfield Water Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/heightfield_water.wgsl").into()),
        });

        // Group 0: Params
        let params_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Heightfield Params Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        let params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Heightfield Params Bind Group"),
            layout: &params_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() }],
        });

        // Group 1: Water State (RW)
        let water_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Water State Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None }, // depth
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None }, // vel_x
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None }, // vel_z
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None }, // surface
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None }, // flux_x
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None }, // flux_z
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None }, // suspended_sediment
            ],
        });
        
        let water_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Water Bind Group"),
            layout: &water_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: water_depth.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: water_vel_x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: water_vel_z.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: water_surface.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: flux_x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: flux_z.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: suspended.as_entire_binding() },
            ],
        });

        // Group 2: Terrain (ReadOnly)
         let terrain_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Terrain Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None }, // bedrock
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None }, // paydirt
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None }, // overburden
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None }, // sediment
            ]
        });
        
        let terrain_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Terrain Bind Group"),
            layout: &terrain_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: bedrock.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: paydirt.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: overburden.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: sediment.as_entire_binding() },
            ],
        });

        // Pipelines
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Heightfield Pipeline Layout"),
            bind_group_layouts: &[&params_layout, &water_layout, &terrain_layout],
            push_constant_ranges: &[],
        });

        let surface_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Update Surface Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("update_surface"),
            compilation_options: Default::default(),
            cache: None,
        });

        let flux_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Update Flux Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("update_flux"),
            compilation_options: Default::default(),
            cache: None,
        });

        let depth_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Update Depth Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("update_depth"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Erosion Shader & Pipelines
        let erosion_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Heightfield Erosion Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/heightfield_erosion.wgsl").into()),
        });

        let erosion_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Update Erosion Pipeline"),
            layout: Some(&pipeline_layout), // Reusing same layout
            module: &erosion_shader,
            entry_point: Some("update_erosion"),
            compilation_options: Default::default(),
            cache: None,
        });

        let sediment_transport_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sediment Transport Pipeline"),
            layout: Some(&pipeline_layout),
            module: &erosion_shader,
            entry_point: Some("update_sediment_transport"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Emitter Shader & Pipelines
        let emitter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Heightfield Emitter Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/heightfield_emitter.wgsl").into()),
        });

        // Emitter params buffer (pos_x, pos_z, radius, rate, dt, enabled, width, depth)
        let emitter_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Emitter Params Buffer"),
            size: 32, // 8 x f32/u32
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let emitter_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Emitter Bind Group Layout"),
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
            ],
        });

        let emitter_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Emitter Bind Group"),
            layout: &emitter_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: emitter_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: water_depth.as_entire_binding(),
                },
            ],
        });

        let emitter_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Emitter Pipeline Layout"),
            bind_group_layouts: &[&emitter_layout],
            push_constant_ranges: &[],
        });

        let emitter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Emitter Pipeline"),
            layout: Some(&emitter_pipeline_layout),
            module: &emitter_shader,
            entry_point: Some("add_water"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            width,
            depth,
            bedrock_buffer: bedrock,
            paydirt_buffer: paydirt,
            overburden_buffer: overburden,
            sediment_buffer: sediment,
            
            water_depth_buffer: water_depth,
            water_velocity_x_buffer: water_vel_x,
            water_velocity_z_buffer: water_vel_z,
            suspended_sediment_buffer: suspended,
            
            water_surface_buffer: water_surface,
            flux_x_buffer: flux_x,
            flux_z_buffer: flux_z,
            
            terrain_bind_group,
            water_bind_group,
            params_bind_group,
            
            surface_pipeline,
            flux_pipeline,
            depth_pipeline,
            erosion_pipeline,
            sediment_transport_pipeline,
            
            emitter_pipeline,
            emitter_params_buffer,
            emitter_bind_group,
            
            params_buffer,
        }
    }

    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, dt: f32) {
         let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Heightfield Compute Pass"),
            timestamp_writes: None,
         });
         
         let x_groups = (self.width + 15) / 16;
         let z_groups = (self.depth + 15) / 16;
         
         pass.set_bind_group(0, &self.params_bind_group, &[]);
         pass.set_bind_group(1, &self.water_bind_group, &[]);
         pass.set_bind_group(2, &self.terrain_bind_group, &[]);
         
         // 1. Update Surface
         pass.set_pipeline(&self.surface_pipeline);
         pass.dispatch_workgroups(x_groups, z_groups, 1);
         
         // 2. Erosion & Deposition (Requires Velocity, which is updated in Flux step?)
         // Wait, Flux step updates Velocity. So Erosion should be AFTER Flux?
         // Or using previous frame velocity?
         // Standard: Flux -> Depth -> Erosion (using updated depth/vel implied).
         
         // 3. Update Flux (Updates Velocity + Flux)
         pass.set_pipeline(&self.flux_pipeline);
         pass.dispatch_workgroups(x_groups, z_groups, 1);
         
         // 4. Update Depth (Volume)
         pass.set_pipeline(&self.depth_pipeline);
         pass.dispatch_workgroups(x_groups, z_groups, 1);
         
         // 5. Erosion (post-flux velocity)
         pass.set_pipeline(&self.erosion_pipeline);
         pass.dispatch_workgroups(x_groups, z_groups, 1);
         
         // 6. Sediment Transport (flux-based advection)
         pass.set_pipeline(&self.sediment_transport_pipeline);
         pass.dispatch_workgroups(x_groups, z_groups, 1);
    }
    
    pub fn update_params(&self, queue: &wgpu::Queue, dt: f32) {
        let params = [
            self.width,
            self.depth,
            0, 0, // padding
            bytemuck::cast(1.0f32), // cell_size
            bytemuck::cast(dt),
            bytemuck::cast(9.81f32), // gravity
            bytemuck::cast(0.99f32), // damping
        ];
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&params));
    }
    
    /// Update GPU emitter parameters
    pub fn update_emitter(&self, queue: &wgpu::Queue, pos_x: f32, pos_z: f32, radius: f32, rate: f32, dt: f32, enabled: bool) {
        // EmitterParams struct: pos_x, pos_z, radius, rate, dt, enabled, width, depth
        let params: [u32; 8] = [
            bytemuck::cast(pos_x),
            bytemuck::cast(pos_z),
            bytemuck::cast(radius),
            bytemuck::cast(rate),
            bytemuck::cast(dt),
            if enabled { 1 } else { 0 },
            self.width,
            self.depth,
        ];
        queue.write_buffer(&self.emitter_params_buffer, 0, bytemuck::cast_slice(&params));
    }
    
    /// Dispatch emitter compute pass - call before main dispatch
    pub fn dispatch_emitter(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Emitter Compute Pass"),
            timestamp_writes: None,
        });
        
        let x_groups = (self.width + 15) / 16;
        let z_groups = (self.depth + 15) / 16;
        
        pass.set_pipeline(&self.emitter_pipeline);
        pass.set_bind_group(0, &self.emitter_bind_group, &[]);
        pass.dispatch_workgroups(x_groups, z_groups, 1);
    }

    pub fn upload_from_world(&self, queue: &wgpu::Queue, world: &sim3d::World) {
        if world.width != self.width as usize || world.depth != self.depth as usize {
            log::error!("World size mismatch in upload");
            return;
        }
        
        // Geology
        queue.write_buffer(&self.bedrock_buffer, 0, bytemuck::cast_slice(&world.bedrock_elevation));
        queue.write_buffer(&self.paydirt_buffer, 0, bytemuck::cast_slice(&world.paydirt_thickness));
        queue.write_buffer(&self.overburden_buffer, 0, bytemuck::cast_slice(&world.overburden_thickness));
        queue.write_buffer(&self.sediment_buffer, 0, bytemuck::cast_slice(&world.terrain_sediment));
        
        // Water
        // CAUTION: World stores Absolute Surface. GPU uses Depth.
        // We must calculate depth = max(0, surface - ground_height)
        let count = (self.width * self.depth) as usize;
        let mut depth_data = vec![0.0f32; count];
        
        for i in 0..count {
             let ground = world.bedrock_elevation[i] 
                        + world.paydirt_thickness[i] 
                        + world.overburden_thickness[i] 
                        + world.terrain_sediment[i];
             depth_data[i] = (world.water_surface[i] - ground).max(0.0);
        }
        queue.write_buffer(&self.water_depth_buffer, 0, bytemuck::cast_slice(&depth_data));
        
        // Velocity
        // World uses staggered flow? flow_x[i] is at face?
        // Sim3d `water_flow_x` size is (W+1)*D.
        // Our GPU buffer is Cell-Centered W*D.
        // Rough approx: Take flux/flow and map to center? 
        // For initialization, 0 is fine.
        // queue.write_buffer(&self.water_velocity_x_buffer, 0, ...);
        
        queue.write_buffer(&self.suspended_sediment_buffer, 0, bytemuck::cast_slice(&world.suspended_sediment));
    }
    
    /// Upload only terrain buffers (for excavation) - does NOT touch water state
    pub fn upload_terrain_only(&self, queue: &wgpu::Queue, world: &sim3d::World) {
        if world.width != self.width as usize || world.depth != self.depth as usize {
            log::error!("World size mismatch in upload");
            return;
        }
        
        // Only geology - leave water/velocity untouched on GPU
        queue.write_buffer(&self.bedrock_buffer, 0, bytemuck::cast_slice(&world.bedrock_elevation));
        queue.write_buffer(&self.paydirt_buffer, 0, bytemuck::cast_slice(&world.paydirt_thickness));
        queue.write_buffer(&self.overburden_buffer, 0, bytemuck::cast_slice(&world.overburden_thickness));
        queue.write_buffer(&self.sediment_buffer, 0, bytemuck::cast_slice(&world.terrain_sediment));
    }
    
    pub async fn download_to_world(&self, device: &wgpu::Device, queue: &wgpu::Queue, world: &mut sim3d::World) {
         let size = (self.width * self.depth) as usize * std::mem::size_of::<f32>();
         
         // Helper to read buffer
         let read_buffer = |buffer: &wgpu::Buffer| -> Vec<f32> {
             let staging = device.create_buffer(&wgpu::BufferDescriptor {
                 label: Some("Staging"),
                 size: size as u64,
                 usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                 mapped_at_creation: false,
             });
             
             let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
             encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size as u64);
             queue.submit(Some(encoder.finish()));
             
             let slice = staging.slice(..);
             let (tx, rx) = std::sync::mpsc::channel();
             slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
             device.poll(wgpu::Maintain::Wait);
             rx.recv().unwrap().unwrap();
             
             let data = slice.get_mapped_range();
             let result: Vec<f32> = bytemuck::cast_slice(&*data).to_vec();
             drop(data);
             staging.unmap();
             result 
         };
         
         // This is SLOW (doing it serially 5 times). 
         // But "FPS is cooked" so this is likely faster than CPU sim.
         // Optimization: One large staging buffer and copy all regions into it?
         
         let bedrock = read_buffer(&self.bedrock_buffer);
         let paydirt = read_buffer(&self.paydirt_buffer);
         let overburden = read_buffer(&self.overburden_buffer);
         let sediment = read_buffer(&self.sediment_buffer);
         
         let water_depth = read_buffer(&self.water_depth_buffer);
         let suspended = read_buffer(&self.suspended_sediment_buffer);
         
         // Update World
         world.bedrock_elevation = bedrock;
         world.paydirt_thickness = paydirt;
         world.overburden_thickness = overburden;
         world.terrain_sediment = sediment;
         world.suspended_sediment = suspended;
         
         // Calculate Surface
         for i in 0..water_depth.len() {
              let ground = world.bedrock_elevation[i] 
                        + world.paydirt_thickness[i] 
                        + world.overburden_thickness[i] 
                        + world.terrain_sediment[i];
              world.water_surface[i] = ground + water_depth[i];
         }
    }
}
