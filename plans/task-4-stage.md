# Task 4: WashplantStage

**File to create:** `crates/game/src/washplant/stage.rs`

## Goal
Create `WashplantStage` struct that wraps a single FLIP simulation with its GPU backend and equipment geometry. This is the core building block - each processing stage (hopper, grizzly, shaker, sluice) will be one `WashplantStage`.

## Dependencies
- `crates/game/src/washplant/config.rs` (already created - StageConfig, EquipmentType)
- `crates/game/src/washplant/metrics.rs` (already created - StageMetrics)
- `crates/game/src/gpu/flip_3d.rs` (existing - GpuFlip3D)
- `crates/sim3d/src/lib.rs` (existing - FlipSimulation3D, Grid3D)
- `crates/game/src/equipment_geometry.rs` (existing - HopperGeometryBuilder, GrateGeometryBuilder, etc.)
- `crates/game/src/sluice_geometry.rs` (existing - SluiceGeometryBuilder, SluiceConfig, SluiceVertex)

## Types to Implement

```rust
use crate::equipment_geometry::{
    GrateConfig, GrateGeometryBuilder, HopperConfig, HopperGeometryBuilder,
};
use crate::gpu::flip_3d::GpuFlip3D;
use crate::sluice_geometry::{SluiceConfig, SluiceGeometryBuilder, SluiceVertex};
use crate::washplant::config::{
    EquipmentType, GrizzlyStageConfig, HopperStageConfig, ShakerStageConfig,
    SluiceStageConfig, StageConfig,
};
use crate::washplant::metrics::StageMetrics;
use glam::Vec3;
use sim3d::{FlipSimulation3D, Grid3D};
use std::sync::Arc;

/// A single processing stage in the washplant
pub struct WashplantStage {
    /// Configuration
    pub config: StageConfig,

    /// CPU simulation state
    pub sim: FlipSimulation3D,

    /// GPU acceleration (optional - None if headless)
    pub gpu_flip: Option<GpuFlip3D>,

    /// World-space offset for rendering
    pub world_offset: Vec3,

    /// Mesh data for rendering equipment
    pub vertices: Vec<SluiceVertex>,
    pub indices: Vec<u32>,

    /// GPU buffers (created on first render)
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,

    /// Metrics
    pub metrics: StageMetrics,

    /// Particle data cache for GPU sync
    positions_cache: Vec<Vec3>,
    velocities_cache: Vec<Vec3>,
}

impl WashplantStage {
    /// Create a new stage from configuration
    pub fn new(config: StageConfig) -> Self {
        // 1. Create FLIP simulation with grid dimensions from config
        let mut sim = FlipSimulation3D::new(
            config.grid_width,
            config.grid_height,
            config.grid_depth,
            config.cell_size,
        );

        // 2. Build equipment geometry and mark solid cells
        let (vertices, indices) = Self::build_equipment_geometry(&config, &mut sim.grid);

        // 3. Compute SDF for collision
        sim.grid.compute_sdf();

        Self {
            world_offset: config.world_offset,
            metrics: StageMetrics::new(config.name),
            config,
            sim,
            gpu_flip: None,
            vertices,
            indices,
            vertex_buffer: None,
            index_buffer: None,
            positions_cache: Vec::new(),
            velocities_cache: Vec::new(),
        }
    }

    /// Build equipment geometry based on type, returns (vertices, indices)
    fn build_equipment_geometry(
        config: &StageConfig,
        grid: &mut Grid3D,
    ) -> (Vec<SluiceVertex>, Vec<u32>) {
        match &config.equipment {
            EquipmentType::Hopper(hopper_cfg) => {
                Self::build_hopper(config, hopper_cfg, grid)
            }
            EquipmentType::Grizzly(grizzly_cfg) => {
                Self::build_grizzly(config, grizzly_cfg, grid)
            }
            EquipmentType::Shaker(shaker_cfg) => {
                Self::build_shaker(config, shaker_cfg, grid)
            }
            EquipmentType::Sluice(sluice_cfg) => {
                Self::build_sluice(config, sluice_cfg, grid)
            }
        }
    }

    fn build_hopper(
        config: &StageConfig,
        hopper_cfg: &HopperStageConfig,
        grid: &mut Grid3D,
    ) -> (Vec<SluiceVertex>, Vec<u32>) {
        let hopper_config = HopperConfig {
            grid_width: config.grid_width,
            grid_height: config.grid_height,
            grid_depth: config.grid_depth,
            cell_size: config.cell_size,
            top_width: hopper_cfg.top_width,
            top_depth: hopper_cfg.top_depth,
            bottom_width: hopper_cfg.bottom_width,
            bottom_depth: hopper_cfg.bottom_depth,
            wall_thickness: hopper_cfg.wall_thickness,
        };

        let mut builder = HopperGeometryBuilder::new(hopper_config.clone());

        // Mark solid cells in grid
        for (i, j, k) in builder.solid_cells() {
            grid.set_solid(i, j, k);
        }

        // Build mesh
        builder.build_mesh(|i, j, k| hopper_config.is_solid(i, j, k));

        (builder.vertices, builder.indices)
    }

    fn build_grizzly(
        config: &StageConfig,
        grizzly_cfg: &GrizzlyStageConfig,
        grid: &mut Grid3D,
    ) -> (Vec<SluiceVertex>, Vec<u32>) {
        let grate_config = GrateConfig {
            grid_width: config.grid_width,
            grid_height: config.grid_height,
            grid_depth: config.grid_depth,
            cell_size: config.cell_size,
            bar_spacing: grizzly_cfg.bar_spacing,
            bar_thickness: grizzly_cfg.bar_thickness,
            orientation: 0, // bars parallel to X (flow direction)
        };

        let mut builder = GrateGeometryBuilder::new(grate_config.clone());

        for (i, j, k) in builder.solid_cells() {
            grid.set_solid(i, j, k);
        }

        builder.build_mesh(|i, j, k| grate_config.is_solid(i, j, k));

        (builder.vertices, builder.indices)
    }

    fn build_shaker(
        config: &StageConfig,
        shaker_cfg: &ShakerStageConfig,
        grid: &mut Grid3D,
    ) -> (Vec<SluiceVertex>, Vec<u32>) {
        // Shaker uses GrateConfig with holes
        // For now, use same as grizzly but with different spacing
        let grate_config = GrateConfig {
            grid_width: config.grid_width,
            grid_height: config.grid_height,
            grid_depth: config.grid_depth,
            cell_size: config.cell_size,
            bar_spacing: (shaker_cfg.hole_spacing / config.cell_size) as usize,
            bar_thickness: 2, // thin bars between holes
            orientation: 0,
        };

        let mut builder = GrateGeometryBuilder::new(grate_config.clone());

        for (i, j, k) in builder.solid_cells() {
            grid.set_solid(i, j, k);
        }

        builder.build_mesh(|i, j, k| grate_config.is_solid(i, j, k));

        (builder.vertices, builder.indices)
    }

    fn build_sluice(
        config: &StageConfig,
        sluice_cfg: &SluiceStageConfig,
        grid: &mut Grid3D,
    ) -> (Vec<SluiceVertex>, Vec<u32>) {
        let sluice_config = SluiceConfig {
            grid_width: config.grid_width,
            grid_height: config.grid_height,
            grid_depth: config.grid_depth,
            cell_size: config.cell_size,
            floor_height_left: sluice_cfg.floor_height_left,
            floor_height_right: sluice_cfg.floor_height_right,
            riffle_spacing: sluice_cfg.riffle_spacing,
            riffle_height: sluice_cfg.riffle_height,
            riffle_thickness: sluice_cfg.riffle_thickness,
            riffle_start_x: 10, // slick plate at start
            riffle_end_pad: 5,
            exit_width_fraction: 0.8,
            exit_height: 8,
            wall_margin: sluice_cfg.wall_margin,
        };

        let mut builder = SluiceGeometryBuilder::new(sluice_config);

        for (i, j, k) in builder.solid_cells() {
            grid.set_solid(i, j, k);
        }

        builder.build_mesh(|i, j, k| grid.is_solid(i, j, k));

        (builder.vertices, builder.indices)
    }

    /// Initialize GPU backend
    pub fn init_gpu(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        use wgpu::util::DeviceExt;

        // Create GpuFlip3D
        self.gpu_flip = Some(GpuFlip3D::new(
            device,
            queue,
            self.config.grid_width,
            self.config.grid_height,
            self.config.grid_depth,
            self.config.cell_size,
            self.config.max_particles,
        ));

        // Create vertex buffer
        if !self.vertices.is_empty() {
            self.vertex_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{} vertex buffer", self.config.name)),
                contents: bytemuck::cast_slice(&self.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }));
        }

        // Create index buffer
        if !self.indices.is_empty() {
            self.index_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{} index buffer", self.config.name)),
                contents: bytemuck::cast_slice(&self.indices),
                usage: wgpu::BufferUsages::INDEX,
            }));
        }
    }

    /// Simulation tick (CPU or GPU)
    pub fn tick(&mut self, dt: f32, queue: &wgpu::Queue) {
        let start = std::time::Instant::now();

        if let Some(ref mut gpu_flip) = self.gpu_flip {
            // Sync particles to GPU
            self.sync_to_gpu(queue);

            // Run GPU passes
            // gpu_flip.run_gpu_passes(...);
        } else {
            // CPU fallback
            self.sim.step(dt);
        }

        // Update metrics
        self.metrics.total_particles = self.sim.particles.len();
        self.metrics.last_tick_ms = start.elapsed().as_secs_f32() * 1000.0;
    }

    /// Sync particle data to GPU
    fn sync_to_gpu(&mut self, _queue: &wgpu::Queue) {
        // Cache positions/velocities for GPU upload
        self.positions_cache.clear();
        self.velocities_cache.clear();

        for p in &self.sim.particles {
            self.positions_cache.push(p.position);
            self.velocities_cache.push(p.velocity);
        }

        // TODO: Upload to GPU buffers via gpu_flip
    }

    /// Spawn a water particle
    pub fn spawn_water(&mut self, position: Vec3, velocity: Vec3) {
        if self.sim.particles.len() < self.config.max_particles {
            self.sim.spawn_particle_with_velocity(position, velocity);
            self.metrics.particles_entered += 1;
        }
    }

    /// Spawn a sediment particle
    pub fn spawn_sediment(&mut self, position: Vec3, velocity: Vec3, density: f32) {
        if self.sim.particles.len() < self.config.max_particles {
            self.sim.spawn_sediment_with_velocity(position, velocity, density);
            self.metrics.particles_entered += 1;
        }
    }

    /// Remove particle at index, returns its data
    pub fn remove_particle(&mut self, idx: usize) -> Option<(Vec3, Vec3, f32)> {
        if idx < self.sim.particles.len() {
            let p = self.sim.particles.swap_remove(idx);
            self.metrics.particles_exited += 1;
            Some((p.position, p.velocity, p.density))
        } else {
            None
        }
    }

    /// Get particle count
    pub fn particle_count(&self) -> usize {
        self.sim.particles.len()
    }

    /// Get vertex/index buffers for rendering
    pub fn render_buffers(&self) -> Option<(&wgpu::Buffer, &wgpu::Buffer, u32)> {
        match (&self.vertex_buffer, &self.index_buffer) {
            (Some(vb), Some(ib)) => Some((vb, ib, self.indices.len() as u32)),
            _ => None,
        }
    }

    /// Get grid dimensions
    pub fn grid_size(&self) -> (usize, usize, usize) {
        (self.config.grid_width, self.config.grid_height, self.config.grid_depth)
    }

    /// Get cell size
    pub fn cell_size(&self) -> f32 {
        self.config.cell_size
    }
}
```

## Update mod.rs

Add to `crates/game/src/washplant/mod.rs`:
```rust
mod stage;
pub use stage::*;
```

## Notes
- The `spawn_sediment_with_velocity` method may not exist on FlipSimulation3D - check and use appropriate method or add density to regular spawn
- GpuFlip3D::new signature may differ - check actual constructor
- Equipment builders may have slightly different APIs - adapt as needed
- Focus on getting the structure right, exact API compatibility can be fixed later

## Testing
Run `cargo check -p game` to verify compilation. Warnings about unused fields are OK.
