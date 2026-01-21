//! Type definitions for the multigrid simulation.

use crate::gpu::flip_3d::GpuFlip3D;
use crate::sluice_geometry::SluiceVertex;
use glam::{Mat3, Vec3};
use sim3d::clump::ClusterSimulation3D;
use sim3d::FlipSimulation3D;

/// Which type of piece this simulation belongs to.
#[derive(Clone, Copy, Debug)]
pub enum PieceKind {
    /// Index into layout.gutters.
    Gutter(usize),
    /// Index into layout.sluices.
    Sluice(usize),
    /// Index into layout.shaker_decks.
    ShakerDeck(usize),
    /// equipment_geometry test box.
    TestBox,
}

/// Per-piece simulation grid.
pub struct PieceSimulation {
    pub kind: PieceKind,

    // Grid configuration
    /// World position of grid origin.
    pub grid_offset: Vec3,
    pub grid_dims: (usize, usize, usize),
    pub cell_size: f32,

    // Simulation state
    pub sim: FlipSimulation3D,
    pub gpu_flip: Option<GpuFlip3D>,
    pub sdf_buffer: Option<wgpu::Buffer>,

    // Particle data buffers
    pub positions: Vec<Vec3>,
    pub velocities: Vec<Vec3>,
    pub affine_vels: Vec<Mat3>,
    pub densities: Vec<f32>,
}

/// Defines particle transfer between two pieces.
pub(crate) struct PieceTransfer {
    /// Index into MultiGridSim::pieces.
    pub from_piece: usize,
    pub to_piece: usize,

    /// Capture region (in from_piece's sim-space).
    pub capture_min: Vec3,
    pub capture_max: Vec3,

    /// Injection position (in to_piece's sim-space).
    #[allow(dead_code)]
    pub inject_pos: Vec3,
    #[allow(dead_code)]
    pub inject_vel: Vec3,

    /// Particles in transit (position, velocity, density).
    #[allow(dead_code)]
    pub transit_queue: Vec<(Vec3, Vec3, f32)>,
    #[allow(dead_code)]
    pub transit_time: f32,
}

/// Multi-grid simulation manager.
pub struct MultiGridSim {
    pub pieces: Vec<PieceSimulation>,
    pub(crate) transfers: Vec<PieceTransfer>,
    pub(crate) frame: u32,

    // DEM simulation (global, not per-piece)
    pub dem_sim: ClusterSimulation3D,
    pub gpu_dem: Option<crate::gpu::dem_3d::GpuDem3D>,
    pub gold_template_idx: usize,
    pub sand_template_idx: usize,
    pub gpu_test_sdf_buffer: Option<wgpu::Buffer>,

    // Test SDF for isolated physics tests (used instead of piece SDFs when set)
    pub(crate) test_sdf: Option<Vec<f32>>,
    pub(crate) test_sdf_dims: (usize, usize, usize),
    pub(crate) test_sdf_cell_size: f32,
    pub(crate) test_sdf_offset: Vec3,
    /// Renderable mesh for test geometry.
    pub test_mesh: Option<(Vec<SluiceVertex>, Vec<u32>)>,
}
