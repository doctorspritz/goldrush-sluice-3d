//! Multi-grid simulation for washplant pieces.
//!
//! This module provides a multi-grid FLIP fluid simulation that manages
//! multiple equipment pieces (gutters, sluices, shaker decks) with
//! particle transfer between them and DEM simulation for solid particles.
//!
//! # Module Structure
//!
//! - [`constants`]: Simulation constants and random number generation
//! - [`types`]: Core type definitions (PieceKind, PieceSimulation, MultiGridSim)
//! - [`solid_cells`]: Solid cell marking for equipment geometry
//! - [`pieces`]: Piece management (adding gutters, sluices, etc.)
//! - [`test_sdf`]: Test SDF geometry for isolated physics tests
//! - [`step`]: Simulation stepping and particle transfer

mod constants;
mod pieces;
mod solid_cells;
mod step;
mod test_sdf;
mod types;

pub use constants::SIM_CELL_SIZE;
pub use types::{MultiGridSim, PieceKind, PieceSimulation};

use constants::{DEM_CLUMP_RADIUS, DEM_GOLD_DENSITY, DEM_SAND_DENSITY};
use glam::Vec3;
use sim3d::clump::{ClumpShape3D, ClumpTemplate3D, ClusterSimulation3D};
use wgpu::util::DeviceExt;

impl MultiGridSim {
    /// Create a new multi-grid simulation manager.
    pub fn new() -> Self {
        // Create DEM simulation with large bounds (covers all pieces)
        let mut dem_sim =
            ClusterSimulation3D::new(Vec3::new(-10.0, -2.0, -10.0), Vec3::new(20.0, 10.0, 20.0));

        // Reduce stiffness for stability with small particles
        // Default 6000 N/m causes particles to explode on collision
        dem_sim.normal_stiffness = 100.0;
        dem_sim.tangential_stiffness = 50.0;
        dem_sim.restitution = 0.1; // Lower bounce

        // Create gold template (heavy, ~8mm clumps)
        // Gold: 19300 kg/m^3, water: 1000 kg/m^3
        // Volume of 8mm sphere ~ 2.68e-7 m^3, mass ~ 5.17g for gold
        let gold_particle_mass =
            DEM_GOLD_DENSITY * (4.0 / 3.0) * std::f32::consts::PI * DEM_CLUMP_RADIUS.powi(3);
        let gold_template = ClumpTemplate3D::generate(
            ClumpShape3D::Irregular {
                count: 5,
                seed: 42,
                style: sim3d::clump::IrregularStyle3D::Round,
            },
            DEM_CLUMP_RADIUS,
            gold_particle_mass,
        );
        let gold_template_idx = dem_sim.add_template(gold_template);

        // Create sand/gangue template (lighter)
        let sand_particle_mass =
            DEM_SAND_DENSITY * (4.0 / 3.0) * std::f32::consts::PI * DEM_CLUMP_RADIUS.powi(3);
        let sand_template = ClumpTemplate3D::generate(
            ClumpShape3D::Irregular {
                count: 4,
                seed: 123,
                style: sim3d::clump::IrregularStyle3D::Sharp,
            },
            DEM_CLUMP_RADIUS,
            sand_particle_mass,
        );
        let sand_template_idx = dem_sim.add_template(sand_template);

        Self {
            pieces: Vec::new(),
            transfers: Vec::new(),
            frame: 0,
            dem_sim,
            gpu_dem: None,
            gold_template_idx,
            sand_template_idx,
            test_sdf: None,
            test_sdf_dims: (0, 0, 0),
            test_sdf_cell_size: constants::SIM_CELL_SIZE,
            test_sdf_offset: Vec3::ZERO,
            gpu_test_sdf_buffer: None,
            test_mesh: None,
        }
    }

    /// Initialize GPU DEM simulation.
    pub fn init_gpu_dem(
        &mut self,
        device: std::sync::Arc<wgpu::Device>,
        queue: std::sync::Arc<wgpu::Queue>,
    ) {
        let mut gpu_dem =
            crate::gpu::dem_3d::GpuDem3D::new(device.clone(), queue.clone(), 50000, 10, 100000);

        // Sync templates
        for template in &self.dem_sim.templates {
            gpu_dem.add_template(template.clone());
        }

        // Sync existing clumps
        for clump in &self.dem_sim.clumps {
            gpu_dem.spawn_clump(clump.template_idx as u32, clump.position, clump.velocity);
        }

        // Create SDF buffers for existing pieces
        for piece in &mut self.pieces {
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Piece SDF Buffer"),
                contents: bytemuck::cast_slice(&piece.sim.grid.sdf()),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            piece.sdf_buffer = Some(buffer);
        }

        if let Some(test_sdf) = &self.test_sdf {
            println!(
                "Init GPU DEM: Test SDF len={}, sample[0]={}",
                test_sdf.len(),
                test_sdf[0]
            );
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Test SDF Buffer"),
                contents: bytemuck::cast_slice(test_sdf),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            self.gpu_test_sdf_buffer = Some(buffer);
        }

        // Initialize params
        gpu_dem.stiffness = self.dem_sim.normal_stiffness;
        gpu_dem.damping = 4.0; // Critical damping ~4.5 for sand mass 0.005kg

        self.gpu_dem = Some(gpu_dem);
    }
}

impl Default for MultiGridSim {
    fn default() -> Self {
        Self::new()
    }
}
