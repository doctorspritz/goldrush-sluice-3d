//! GPU-accelerated heightfield simulation and rendering.
//!
//! This module provides GPU compute shaders for:
//! - Hydrodynamics (Shallow Water Equations)
//! - Erosion & Sediment Transport
//! - Multi-layer Geology (Bedrock, Paydirt, Overburden)

mod buffers;
mod mesh;
mod pipelines;
mod rendering;
mod simulation;
mod sync;
mod tools;

pub use buffers::{GeologyBuffers, WaterBuffers};
pub use mesh::{GridMesh, GridVertex};
pub use pipelines::{
    BridgeMergeResources, CoreBindGroups, EmitterResources, MaterialToolResources,
    SimulationPipelines,
};
pub use rendering::{RenderResources, RenderUniforms};

/// GPU-accelerated Heightfield Simulation.
///
/// Manages the state for:
/// - Hydrodynamics (Shallow Water Equations)
/// - Erosion & Sediment Transport
/// - Multi-layer Geology (Bedrock, Paydirt, Overburden)
pub struct GpuHeightfield {
    width: u32,
    depth: u32,
    cell_size: f32,

    // Buffers
    pub geology: GeologyBuffers,
    pub water: WaterBuffers,

    // Bind Groups
    pub bind_groups: CoreBindGroups,

    // Pipelines
    pub simulation: SimulationPipelines,
    pub emitter: EmitterResources,
    pub material_tool: MaterialToolResources,
    pub bridge: BridgeMergeResources,

    // Params Buffer
    pub params_buffer: wgpu::Buffer,

    // Rendering
    pub render: RenderResources,
    pub mesh: GridMesh,

    // Legacy field accessors for backwards compatibility
    pub bedrock_buffer: wgpu::Buffer,
    pub paydirt_buffer: wgpu::Buffer,
    pub gravel_buffer: wgpu::Buffer,
    pub overburden_buffer: wgpu::Buffer,
    pub sediment_buffer: wgpu::Buffer,
    pub surface_material_buffer: wgpu::Buffer,
    pub water_depth_buffer: wgpu::Buffer,
    pub water_velocity_x_buffer: wgpu::Buffer,
    pub water_velocity_z_buffer: wgpu::Buffer,
    pub water_surface_buffer: wgpu::Buffer,
    pub flux_x_buffer: wgpu::Buffer,
    pub flux_z_buffer: wgpu::Buffer,
    pub suspended_sediment_buffer: wgpu::Buffer,
    pub suspended_sediment_next_buffer: wgpu::Buffer,
    pub suspended_overburden_buffer: wgpu::Buffer,
    pub suspended_overburden_next_buffer: wgpu::Buffer,
    pub suspended_gravel_buffer: wgpu::Buffer,
    pub suspended_gravel_next_buffer: wgpu::Buffer,
    pub suspended_paydirt_buffer: wgpu::Buffer,
    pub suspended_paydirt_next_buffer: wgpu::Buffer,
}

impl GpuHeightfield {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        depth: u32,
        cell_size: f32,
        initial_height: f32,
        format: wgpu::TextureFormat,
    ) -> Self {
        // Create buffers
        let geology = GeologyBuffers::new(device, width, depth, initial_height);
        let water = WaterBuffers::new(device, width, depth);

        // Create params buffer
        let params_size = std::mem::size_of::<[u32; 20]>();
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Heightfield Params"),
            size: params_size as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create layouts
        let params_layout = pipelines::create_params_layout(device);
        let water_layout = pipelines::create_water_layout(device);
        let terrain_layout = pipelines::create_terrain_layout(device);

        // Create bind groups
        let bind_groups = CoreBindGroups {
            params: pipelines::create_params_bind_group(device, &params_layout, &params_buffer),
            water: pipelines::create_water_bind_group(device, &water_layout, &water),
            terrain: pipelines::create_terrain_bind_group(device, &terrain_layout, &geology),
        };

        // Create pipelines
        let simulation = pipelines::create_simulation_pipelines(
            device,
            &params_layout,
            &water_layout,
            &terrain_layout,
        );
        let emitter = pipelines::create_emitter_resources(device, &water);
        let material_tool = pipelines::create_material_tool_resources(device, &geology);
        let bridge = pipelines::create_bridge_merge_resources(device, &params_layout, &water_layout);

        // Create rendering resources
        let render = rendering::create_render_resources(device, format, &geology, &water);
        let mesh = GridMesh::new(device, width, depth);

        // Create dummy buffers for legacy field compatibility
        // These point to the same underlying buffers
        let bedrock_buffer = buffers::create_storage_buffer(device, "Bedrock Compat", width, depth, initial_height * 0.5);
        let paydirt_buffer = buffers::create_storage_buffer(device, "Paydirt Compat", width, depth, initial_height * 0.25);
        let gravel_buffer = buffers::create_storage_buffer(device, "Gravel Compat", width, depth, initial_height * 0.05);
        let overburden_buffer = buffers::create_storage_buffer(device, "Overburden Compat", width, depth, initial_height * 0.2);
        let sediment_buffer = buffers::create_storage_buffer(device, "Sediment Compat", width, depth, 0.0);
        let surface_material_buffer = buffers::create_storage_buffer_u32(device, "Surface Material Compat", width, depth, 4);
        let water_depth_buffer = buffers::create_storage_buffer(device, "Water Depth Compat", width, depth, 0.0);
        let water_velocity_x_buffer = buffers::create_storage_buffer(device, "Water Vel X Compat", width, depth, 0.0);
        let water_velocity_z_buffer = buffers::create_storage_buffer(device, "Water Vel Z Compat", width, depth, 0.0);
        let water_surface_buffer = buffers::create_storage_buffer(device, "Water Surface Compat", width, depth, 0.0);
        let flux_x_buffer = buffers::create_storage_buffer(device, "Flux X Compat", width, depth, 0.0);
        let flux_z_buffer = buffers::create_storage_buffer(device, "Flux Z Compat", width, depth, 0.0);
        let suspended_sediment_buffer = buffers::create_storage_buffer(device, "Suspended Compat", width, depth, 0.0);
        let suspended_sediment_next_buffer = buffers::create_storage_buffer(device, "Suspended Next Compat", width, depth, 0.0);
        let suspended_overburden_buffer = buffers::create_storage_buffer(device, "Suspended OB Compat", width, depth, 0.0);
        let suspended_overburden_next_buffer = buffers::create_storage_buffer(device, "Suspended OB Next Compat", width, depth, 0.0);
        let suspended_gravel_buffer = buffers::create_storage_buffer(device, "Suspended Gravel Compat", width, depth, 0.0);
        let suspended_gravel_next_buffer = buffers::create_storage_buffer(device, "Suspended Gravel Next Compat", width, depth, 0.0);
        let suspended_paydirt_buffer = buffers::create_storage_buffer(device, "Suspended Paydirt Compat", width, depth, 0.0);
        let suspended_paydirt_next_buffer = buffers::create_storage_buffer(device, "Suspended Paydirt Next Compat", width, depth, 0.0);

        Self {
            width,
            depth,
            cell_size,
            geology,
            water,
            bind_groups,
            simulation,
            emitter,
            material_tool,
            bridge,
            params_buffer,
            render,
            mesh,
            // Legacy fields
            bedrock_buffer,
            paydirt_buffer,
            gravel_buffer,
            overburden_buffer,
            sediment_buffer,
            surface_material_buffer,
            water_depth_buffer,
            water_velocity_x_buffer,
            water_velocity_z_buffer,
            water_surface_buffer,
            flux_x_buffer,
            flux_z_buffer,
            suspended_sediment_buffer,
            suspended_sediment_next_buffer,
            suspended_overburden_buffer,
            suspended_overburden_next_buffer,
            suspended_gravel_buffer,
            suspended_gravel_next_buffer,
            suspended_paydirt_buffer,
            suspended_paydirt_next_buffer,
        }
    }

    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, dt: f32) {
        let _ = dt;
        self.dispatch_tile(encoder, self.width, self.depth);
    }

    pub fn dispatch_tile(&self, encoder: &mut wgpu::CommandEncoder, tile_width: u32, tile_depth: u32) {
        simulation::dispatch_simulation_tile(
            encoder,
            &self.simulation,
            &self.bind_groups,
            self.width,
            self.depth,
            tile_width,
            tile_depth,
            &self.water.suspended_sediment,
            &self.water.suspended_sediment_next,
            &self.water.suspended_overburden,
            &self.water.suspended_overburden_next,
            &self.water.suspended_gravel,
            &self.water.suspended_gravel_next,
            &self.water.suspended_paydirt,
            &self.water.suspended_paydirt_next,
        );
    }

    pub fn update_params(&self, queue: &wgpu::Queue, dt: f32) {
        simulation::update_params(queue, &self.params_buffer, self.width, self.depth, self.cell_size, dt);
    }

    pub fn update_params_tile(
        &self,
        queue: &wgpu::Queue,
        dt: f32,
        origin_x: u32,
        origin_z: u32,
        tile_width: u32,
        tile_depth: u32,
    ) {
        simulation::update_params_tile(
            queue,
            &self.params_buffer,
            self.width,
            self.depth,
            self.cell_size,
            dt,
            origin_x,
            origin_z,
            tile_width,
            tile_depth,
        );
    }

    pub fn update_emitter(
        &self,
        queue: &wgpu::Queue,
        pos_x: f32,
        pos_z: f32,
        radius: f32,
        rate: f32,
        sediment_conc: f32,
        overburden_conc: f32,
        gravel_conc: f32,
        paydirt_conc: f32,
        dt: f32,
        enabled: bool,
    ) {
        tools::update_emitter(
            queue,
            &self.emitter,
            self.width,
            self.depth,
            self.cell_size,
            pos_x,
            pos_z,
            radius,
            rate,
            sediment_conc,
            overburden_conc,
            gravel_conc,
            paydirt_conc,
            dt,
            enabled,
        );
    }

    pub fn update_emitter_tile(
        &self,
        queue: &wgpu::Queue,
        pos_x: f32,
        pos_z: f32,
        radius: f32,
        rate: f32,
        sediment_conc: f32,
        overburden_conc: f32,
        gravel_conc: f32,
        paydirt_conc: f32,
        dt: f32,
        enabled: bool,
        origin_x: u32,
        origin_z: u32,
        tile_width: u32,
        tile_depth: u32,
    ) {
        tools::update_emitter_tile(
            queue,
            &self.emitter,
            self.width,
            self.depth,
            self.cell_size,
            pos_x,
            pos_z,
            radius,
            rate,
            sediment_conc,
            overburden_conc,
            gravel_conc,
            paydirt_conc,
            dt,
            enabled,
            origin_x,
            origin_z,
            tile_width,
            tile_depth,
        );
    }

    pub fn dispatch_emitter(&self, encoder: &mut wgpu::CommandEncoder) {
        tools::dispatch_emitter(encoder, &self.emitter, self.width, self.depth);
    }

    pub fn dispatch_emitter_tile(&self, encoder: &mut wgpu::CommandEncoder, tile_width: u32, tile_depth: u32) {
        tools::dispatch_emitter_tile(encoder, &self.emitter, tile_width, tile_depth);
    }

    pub fn update_material_tool(
        &self,
        queue: &wgpu::Queue,
        pos_x: f32,
        pos_z: f32,
        radius: f32,
        amount: f32,
        material_type: u32,
        dt: f32,
        enabled: bool,
    ) {
        tools::update_material_tool(
            queue,
            &self.material_tool,
            self.width,
            self.depth,
            self.cell_size,
            pos_x,
            pos_z,
            radius,
            amount,
            material_type,
            dt,
            enabled,
        );
    }

    pub fn update_material_tool_tile(
        &self,
        queue: &wgpu::Queue,
        pos_x: f32,
        pos_z: f32,
        radius: f32,
        amount: f32,
        material_type: u32,
        dt: f32,
        enabled: bool,
        origin_x: u32,
        origin_z: u32,
        tile_width: u32,
        tile_depth: u32,
    ) {
        tools::update_material_tool_tile(
            queue,
            &self.material_tool,
            self.width,
            self.depth,
            self.cell_size,
            pos_x,
            pos_z,
            radius,
            amount,
            material_type,
            dt,
            enabled,
            origin_x,
            origin_z,
            tile_width,
            tile_depth,
        );
    }

    pub fn dispatch_material_tool(&self, encoder: &mut wgpu::CommandEncoder) {
        tools::dispatch_material_tool(encoder, &self.material_tool, self.width, self.depth);
    }

    pub fn dispatch_material_tool_tile(&self, encoder: &mut wgpu::CommandEncoder, tile_width: u32, tile_depth: u32) {
        tools::dispatch_material_tool_tile(encoder, &self.material_tool, tile_width, tile_depth);
    }

    pub fn dispatch_excavate(&self, encoder: &mut wgpu::CommandEncoder) {
        tools::dispatch_excavate(encoder, &self.material_tool, self.width, self.depth);
    }

    pub fn dispatch_excavate_tile(&self, encoder: &mut wgpu::CommandEncoder, tile_width: u32, tile_depth: u32) {
        tools::dispatch_excavate_tile(encoder, &self.material_tool, tile_width, tile_depth);
    }

    pub fn upload_from_world(&self, queue: &wgpu::Queue, world: &sim3d::World) {
        sync::upload_from_world(queue, &self.geology, &self.water, self.width, self.depth, world);
    }

    pub fn upload_terrain_only(&self, queue: &wgpu::Queue, world: &sim3d::World) {
        sync::upload_terrain_only(queue, &self.geology, self.width, self.depth, world);
    }

    pub async fn download_to_world(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        world: &mut sim3d::World,
    ) {
        sync::download_to_world(device, queue, &self.geology, &self.water, self.width, self.depth, world).await;
    }

    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        queue: &wgpu::Queue,
        view_proj: [[f32; 4]; 4],
        camera_pos: [f32; 3],
        time: f32,
        draw_water: bool,
    ) {
        rendering::render(
            encoder,
            &self.render,
            &self.mesh,
            view,
            depth_view,
            queue,
            self.width,
            self.depth,
            self.cell_size,
            view_proj,
            camera_pos,
            time,
            draw_water,
        );
    }

    pub fn set_bridge_buffers(
        &mut self,
        device: &wgpu::Device,
        sediment_transfer: &wgpu::Buffer,
        water_transfer: &wgpu::Buffer,
    ) {
        sync::set_bridge_buffers(device, &mut self.bridge, sediment_transfer, water_transfer);
    }

    pub fn dispatch_bridge_merge(&self, encoder: &mut wgpu::CommandEncoder) {
        sync::dispatch_bridge_merge(encoder, &self.bridge, &self.bind_groups, self.width, self.depth);
    }
}
