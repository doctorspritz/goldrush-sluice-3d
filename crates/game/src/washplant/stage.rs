//! WashplantStage - Wraps FlipSimulation3D with equipment geometry and optional GPU acceleration

use crate::equipment_geometry::{GrateGeometryBuilder, HopperGeometryBuilder};
use crate::gpu::flip_3d::GpuFlip3D;
use crate::sluice_geometry::SluiceGeometryBuilder;
use crate::washplant::config::{EquipmentConfig, StageConfig};
use glam::Vec3;
use sim3d::FlipSimulation3D;

/// Washplant stage wrapping FlipSimulation3D with equipment geometry
pub struct WashplantStage {
    sim: FlipSimulation3D,
    gpu: Option<GpuFlip3D>,

    // Equipment geometry builders
    hopper: Option<HopperGeometryBuilder>,
    grizzly: Option<GrateGeometryBuilder>,
    shaker: Option<GrateGeometryBuilder>,
    sluice: Option<SluiceGeometryBuilder>,
}

impl WashplantStage {
    /// Create new washplant stage with specified grid dimensions
    pub fn new(width: usize, height: usize, depth: usize, cell_size: f32) -> Self {
        let sim = FlipSimulation3D::new(width, height, depth, cell_size);
        Self {
            sim,
            gpu: None,
            hopper: None,
            grizzly: None,
            shaker: None,
            sluice: None,
        }
    }

    /// Build equipment geometry from stage configuration
    pub fn build_equipment_geometry(&mut self, stage_config: &StageConfig) {
        match &stage_config.equipment {
            EquipmentConfig::Hopper(hopper_config) => {
                self.build_hopper(*hopper_config);
            }
            EquipmentConfig::Grizzly(grizzly_config) => {
                self.build_grizzly(*grizzly_config);
            }
            EquipmentConfig::Shaker(shaker_config) => {
                self.build_shaker(shaker_config.clone());
            }
            EquipmentConfig::Sluice(sluice_config) => {
                self.build_sluice(*sluice_config);
            }
        }
    }

    /// Build hopper geometry and mark solid cells
    pub fn build_hopper(&mut self, config: crate::washplant::config::HopperStageConfig) {
        // Convert hopper stage config to geometry config
        // Note: config.height used for physical scaling but HopperConfig doesn't expose height field
        // Hopper geometry builder uses grid dimensions to determine geometry

        let geom_config = crate::equipment_geometry::HopperConfig {
            grid_width: self.sim.grid.width,
            grid_height: self.sim.grid.height,
            grid_depth: self.sim.grid.depth,
            cell_size: self.sim.grid.cell_size,
            top_width: (self.sim.grid.width * 3 / 4).max(4),
            top_depth: (self.sim.grid.depth * 3 / 4).max(4),
            bottom_width: (self.sim.grid.width / 4).max(2),
            bottom_depth: (self.sim.grid.depth / 4).max(2),
            wall_thickness: 1,
            ..Default::default()
        };

        let builder = HopperGeometryBuilder::new(geom_config);

        // Mark solid cells in grid
        for (i, j, k) in builder.solid_cells() {
            self.sim.grid.set_solid(i, j, k);
        }

        self.hopper = Some(builder);
    }

    /// Build grizzly (grate) geometry and mark solid cells
    pub fn build_grizzly(&mut self, config: crate::washplant::config::GrizzlyStageConfig) {
        // Convert grizzly config to grate geometry config
        // Bar spacing in cells (config.bar_spacing is in mm, cell_size is in world units)
        let bar_spacing_cells = ((config.bar_spacing / 1000.0) / self.sim.grid.cell_size).max(2.0) as usize;

        let geom_config = crate::equipment_geometry::GrateConfig {
            grid_width: self.sim.grid.width,
            grid_height: self.sim.grid.height,
            grid_depth: self.sim.grid.depth,
            cell_size: self.sim.grid.cell_size,
            bar_spacing: bar_spacing_cells,
            bar_thickness: 1,
            orientation: 0, // Parallel to X axis
            ..Default::default()
        };

        let builder = GrateGeometryBuilder::new(geom_config);

        // Mark solid cells
        for (i, j, k) in builder.solid_cells() {
            self.sim.grid.set_solid(i, j, k);
        }

        self.grizzly = Some(builder);
    }

    /// Build shaker (grate) geometry and mark solid cells
    pub fn build_shaker(&mut self, config: crate::washplant::config::ShakerStageConfig) {
        // Use top_opening for bar spacing
        let bar_spacing_cells = ((config.top_opening / 1000.0) / self.sim.grid.cell_size).max(2.0) as usize;

        let geom_config = crate::equipment_geometry::GrateConfig {
            grid_width: self.sim.grid.width,
            grid_height: self.sim.grid.height,
            grid_depth: self.sim.grid.depth,
            cell_size: self.sim.grid.cell_size,
            bar_spacing: bar_spacing_cells,
            bar_thickness: 1,
            orientation: 1, // Parallel to Z axis
            ..Default::default()
        };

        let builder = GrateGeometryBuilder::new(geom_config);

        // Mark solid cells
        for (i, j, k) in builder.solid_cells() {
            self.sim.grid.set_solid(i, j, k);
        }

        self.shaker = Some(builder);
    }

    /// Build sluice geometry and mark solid cells
    pub fn build_sluice(&mut self, config: crate::washplant::config::SluiceStageConfig) {
        // Convert sluice config to geometry config using actual config values
        // Convert riffle_height from mm to cells
        let riffle_height_cells = ((config.riffle_height / 1000.0) / self.sim.grid.cell_size).max(1.0) as usize;

        // Calculate floor slope from angle (degrees)
        let angle_rad = config.angle.to_radians();
        let length_cells = (config.length / self.sim.grid.cell_size) as usize;
        let height_drop_cells = (length_cells as f32 * angle_rad.tan()) as usize;

        let geom_config = crate::sluice_geometry::SluiceConfig {
            grid_width: self.sim.grid.width,
            grid_height: self.sim.grid.height,
            grid_depth: self.sim.grid.depth,
            cell_size: self.sim.grid.cell_size,
            floor_height_left: (self.sim.grid.height * 2 / 3).max(4),
            floor_height_right: ((self.sim.grid.height * 2 / 3).saturating_sub(height_drop_cells)).max(2),
            riffle_spacing: 12,
            riffle_height: riffle_height_cells.max(1),
            riffle_thickness: 2,
            riffle_start_x: 12,
            riffle_end_pad: 8,
            wall_margin: 4,
            exit_width_fraction: 0.67,
            exit_height: (self.sim.grid.height / 3).max(4),
            ..Default::default()
        };

        let builder = SluiceGeometryBuilder::new(geom_config);

        // Mark solid cells
        for (i, j, k) in builder.solid_cells() {
            self.sim.grid.set_solid(i, j, k);
        }

        self.sluice = Some(builder);
    }

    /// Initialize GPU acceleration
    pub fn init_gpu(&mut self, device: &wgpu::Device, max_particles: usize) {
        let gpu = GpuFlip3D::new(
            device,
            self.sim.grid.width as u32,
            self.sim.grid.height as u32,
            self.sim.grid.depth as u32,
            self.sim.grid.cell_size,
            max_particles,
        );
        self.gpu = Some(gpu);
    }

    /// Run one simulation timestep
    pub fn tick(&mut self, dt: f32) {
        self.sim.update(dt);
    }

    /// Synchronize particle data to GPU and run GPU step
    pub fn sync_to_gpu(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, dt: f32) {
        if let Some(gpu) = &mut self.gpu {
            // Extract particle data as Vec3/Mat3 for GpuFlip3D::step
            let mut positions: Vec<Vec3> = self
                .sim
                .particles
                .list
                .iter()
                .map(|p| p.position)
                .collect();

            let mut velocities: Vec<Vec3> = self
                .sim
                .particles
                .list
                .iter()
                .map(|p| p.velocity)
                .collect();

            // APIC affine velocity matrices
            let mut c_matrices: Vec<glam::Mat3> = self
                .sim
                .particles
                .list
                .iter()
                .map(|p| p.affine_velocity)
                .collect();

            // Densities
            let densities: Vec<f32> = self
                .sim
                .particles
                .list
                .iter()
                .map(|p| p.density)
                .collect();

            // Cell types from grid
            let cell_types: Vec<u32> = self
                .sim
                .grid
                .cell_type
                .iter()
                .map(|ct| match ct {
                    sim3d::CellType::Fluid => 0,
                    sim3d::CellType::Solid => 1,
                    sim3d::CellType::Air => 2,
                })
                .collect();

            // Run GPU step
            gpu.step(
                device,
                queue,
                &mut positions,
                &mut velocities,
                &mut c_matrices,
                &densities,
                &cell_types,
                None, // No SDF
                None, // No bed height
                dt,
                self.sim.gravity.y,
                0.0, // No flow acceleration
                self.sim.pressure_iterations as u32,
            );

            // Write results back to CPU particles
            for (i, particle) in self.sim.particles.list.iter_mut().enumerate() {
                if i < positions.len() {
                    particle.position = positions[i];
                    particle.velocity = velocities[i];
                    particle.affine_velocity = c_matrices[i];
                }
            }
        }
    }

    /// Spawn water particle at position
    pub fn spawn_water(&mut self, position: Vec3) {
        self.sim.spawn_particle(position);
    }

    /// Spawn sediment particle with velocity and density
    pub fn spawn_sediment(&mut self, position: Vec3, velocity: Vec3, density: f32) {
        self.sim.spawn_sediment(position, velocity, density);
    }

    /// Remove particle at index
    pub fn remove_particle(&mut self, index: usize) {
        if index < self.sim.particles.list.len() {
            self.sim.particles.list.swap_remove(index);
        }
    }

    /// Get current particle count
    pub fn particle_count(&self) -> usize {
        self.sim.particle_count()
    }

    /// Get render buffers for equipment geometry
    pub fn render_buffers(&self) -> Option<(&wgpu::Buffer, &wgpu::Buffer)> {
        // Prioritize sluice, then others
        if let Some(sluice) = &self.sluice {
            if let (Some(vb), Some(ib)) = (sluice.vertex_buffer(), sluice.index_buffer()) {
                return Some((vb, ib));
            }
        }

        if let Some(hopper) = &self.hopper {
            if let (Some(vb), Some(ib)) = (hopper.vertex_buffer(), hopper.index_buffer()) {
                return Some((vb, ib));
            }
        }

        if let Some(grizzly) = &self.grizzly {
            if let (Some(vb), Some(ib)) = (grizzly.vertex_buffer(), grizzly.index_buffer()) {
                return Some((vb, ib));
            }
        }

        if let Some(shaker) = &self.shaker {
            if let (Some(vb), Some(ib)) = (shaker.vertex_buffer(), shaker.index_buffer()) {
                return Some((vb, ib));
            }
        }

        None
    }

    /// Get grid dimensions
    pub fn grid_size(&self) -> (usize, usize, usize) {
        (
            self.sim.grid.width,
            self.sim.grid.height,
            self.sim.grid.depth,
        )
    }

    /// Get cell size
    pub fn cell_size(&self) -> f32 {
        self.sim.grid.cell_size
    }
}
