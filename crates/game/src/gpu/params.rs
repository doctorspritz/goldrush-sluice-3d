//! GPU shader parameter structs for FLIP 3D simulation.
//!
//! These are `#[repr(C)]` structs that are uploaded to GPU uniform/storage buffers
//! for compute shader parameters.

use bytemuck::{Pod, Zeroable};

/// Gravel obstacle for dynamic solid collision.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GravelObstacle {
    pub position: [f32; 3],
    pub radius: f32,
}

/// Gravel obstacle shader parameters.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct GravelObstacleParams3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub obstacle_count: u32,
    pub cell_size: f32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Gravity application parameters.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct GravityParams3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub gravity_dt: f32,
    pub cell_size: f32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Flow acceleration parameters (for sluice downstream flow).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct FlowParams3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub flow_accel_dt: f32,
}

/// Vorticity computation parameters.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct VorticityParams3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub cell_size: f32,
}

/// Vorticity confinement parameters.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct VortConfineParams3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub epsilon_h_dt: f32,
}

/// Sediment fraction parameters.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct SedimentFractionParams3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub rest_particles: f32,
}

/// Sediment pressure parameters (for Drucker-Prager).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct SedimentPressureParams3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub _pad0: u32,
    pub cell_size: f32,
    pub particle_mass: f32,
    pub gravity: f32,
    pub buoyancy_factor: f32,
}

/// Porosity drag parameters.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct PorosityDragParams3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub drag_dt: f32,
}

/// Boundary condition parameters.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct BcParams3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub _pad: u32,
}

/// Sediment cell type builder parameters.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct SedimentCellTypeParams3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub _pad0: u32,
}

/// Density error parameters.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct DensityErrorParams3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub rest_density: f32,
    pub dt: f32,
    pub surface_clamp: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}

/// Density position grid parameters.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct DensityPositionGridParams3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub dt: f32,
    pub strength: f32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}

/// Density correct parameters.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct DensityCorrectParams3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub particle_count: u32,
    pub cell_size: f32,
    pub damping: f32,
    pub _pad2: u32,
    pub _pad3: u32,
}

/// SDF collision parameters.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct SdfCollisionParams3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub particle_count: u32,
    pub cell_size: f32,
    pub dt: f32,
    /// Bitmask for open boundaries (particles can exit without clamping):
    /// Bit 0 (1): -X open, Bit 1 (2): +X open
    /// Bit 2 (4): -Y open, Bit 3 (8): +Y open
    /// Bit 4 (16): -Z open, Bit 5 (32): +Z open
    pub open_boundaries: u32,
    pub _pad0: u32,
}
