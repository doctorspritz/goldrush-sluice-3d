//! Buffer creation utilities for GPU heightfield simulation.

use wgpu::util::DeviceExt;

/// Create a storage buffer initialized with a constant value.
pub fn create_storage_buffer(
    device: &wgpu::Device,
    label: &str,
    width: u32,
    depth: u32,
    init_val: f32,
) -> wgpu::Buffer {
    let data = vec![init_val; (width * depth) as usize];
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(&data),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    })
}

/// Create a storage buffer initialized with u32 values.
pub fn create_storage_buffer_u32(
    device: &wgpu::Device,
    label: &str,
    width: u32,
    depth: u32,
    init_val: u32,
) -> wgpu::Buffer {
    let data = vec![init_val; (width * depth) as usize];
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(&data),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    })
}

/// Holds all geology-related buffers.
pub struct GeologyBuffers {
    pub bedrock: wgpu::Buffer,
    pub paydirt: wgpu::Buffer,
    pub gravel: wgpu::Buffer,
    pub overburden: wgpu::Buffer,
    pub sediment: wgpu::Buffer,
    pub surface_material: wgpu::Buffer,
}

impl GeologyBuffers {
    pub fn new(device: &wgpu::Device, width: u32, depth: u32, initial_height: f32) -> Self {
        let bedrock = create_storage_buffer(device, "Bedrock Buffer", width, depth, initial_height * 0.5);
        let paydirt = create_storage_buffer(device, "Paydirt Buffer", width, depth, initial_height * 0.25);
        let gravel = create_storage_buffer(device, "Gravel Buffer", width, depth, initial_height * 0.05);
        let overburden = create_storage_buffer(device, "Overburden Buffer", width, depth, initial_height * 0.2);
        let sediment = create_storage_buffer(device, "Sediment Buffer", width, depth, 0.0);
        let surface_material = create_storage_buffer_u32(device, "Surface Material Buffer", width, depth, 4);

        Self {
            bedrock,
            paydirt,
            gravel,
            overburden,
            sediment,
            surface_material,
        }
    }
}

/// Holds all water state buffers.
pub struct WaterBuffers {
    pub depth: wgpu::Buffer,
    pub velocity_x: wgpu::Buffer,
    pub velocity_z: wgpu::Buffer,
    pub surface: wgpu::Buffer,
    pub flux_x: wgpu::Buffer,
    pub flux_z: wgpu::Buffer,
    pub suspended_sediment: wgpu::Buffer,
    pub suspended_sediment_next: wgpu::Buffer,
    pub suspended_overburden: wgpu::Buffer,
    pub suspended_overburden_next: wgpu::Buffer,
    pub suspended_gravel: wgpu::Buffer,
    pub suspended_gravel_next: wgpu::Buffer,
    pub suspended_paydirt: wgpu::Buffer,
    pub suspended_paydirt_next: wgpu::Buffer,
}

impl WaterBuffers {
    pub fn new(device: &wgpu::Device, width: u32, depth: u32) -> Self {
        Self {
            depth: create_storage_buffer(device, "Water Depth Buffer", width, depth, 0.0),
            velocity_x: create_storage_buffer(device, "Water Vel X Buffer", width, depth, 0.0),
            velocity_z: create_storage_buffer(device, "Water Vel Z Buffer", width, depth, 0.0),
            surface: create_storage_buffer(device, "Water Surface Buffer", width, depth, 0.0),
            flux_x: create_storage_buffer(device, "Flux X Buffer", width, depth, 0.0),
            flux_z: create_storage_buffer(device, "Flux Z Buffer", width, depth, 0.0),
            suspended_sediment: create_storage_buffer(device, "Suspended Sediment Buffer", width, depth, 0.0),
            suspended_sediment_next: create_storage_buffer(device, "Suspended Sediment Next Buffer", width, depth, 0.0),
            suspended_overburden: create_storage_buffer(device, "Suspended Overburden Buffer", width, depth, 0.0),
            suspended_overburden_next: create_storage_buffer(device, "Suspended Overburden Next Buffer", width, depth, 0.0),
            suspended_gravel: create_storage_buffer(device, "Suspended Gravel Buffer", width, depth, 0.0),
            suspended_gravel_next: create_storage_buffer(device, "Suspended Gravel Next Buffer", width, depth, 0.0),
            suspended_paydirt: create_storage_buffer(device, "Suspended Paydirt Buffer", width, depth, 0.0),
            suspended_paydirt_next: create_storage_buffer(device, "Suspended Paydirt Next Buffer", width, depth, 0.0),
        }
    }
}
