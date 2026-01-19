//! GPU parameter structs for FLIP 3D shaders.
//!
//! These `#[repr(C)]` structs are uploaded to GPU uniform buffers
//! for each compute shader pass.

use bytemuck::{Pod, Zeroable};

//==============================================================================
// HYDROSTATIC EQUILIBRIUM - VALIDATED PARAMETERS
//
// These parameters have been validated to achieve true hydrostatic equilibrium
// (water at rest with max velocity < 0.01 m/s). DO NOT CHANGE without re-running
// test_hydrostatic_equilibrium in flip_component_tests.rs.
//
// Critical settings for hydrostatic equilibrium:
//   - flip_ratio = 0.0       (Pure PIC for maximum damping)
//   - slip_factor = 0.0      (No-slip at solid boundaries)
//   - open_boundaries = 8    (+Y open for free surface, all others closed)
//   - vorticity_epsilon = 0.0 (Disable vorticity confinement)
//   - density_projection_enabled = false (Disable - causes particle clumping)
//
// Particle seeding: 8 particles per cell (2×2×2 stratified, Zhu & Bridson 2005)
//==============================================================================

/// Validated flip_ratio for hydrostatic equilibrium tests.
/// Pure PIC (0.0) provides maximum damping for equilibrium scenarios.
pub const HYDROSTATIC_FLIP_RATIO: f32 = 0.0;

/// Validated slip_factor for hydrostatic equilibrium tests.
/// No-slip (0.0) at solid boundaries is essential for proper hydrostatic equilibrium.
pub const HYDROSTATIC_SLIP_FACTOR: f32 = 0.0;

/// Validated open_boundaries for hydrostatic equilibrium tests.
/// Bit 3 (8) = +Y open for free surface at top, all other boundaries closed.
pub const HYDROSTATIC_OPEN_BOUNDARIES: u32 = 8;

pub const GRAVEL_OBSTACLE_MAX: u32 = 2048;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GravelObstacle {
    pub position: [f32; 3],
    pub radius: f32,
}

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

/// Gravity application parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct GravityParams3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub gravity_dt: f32,
    pub cell_size: f32,
    /// Bitmask for open boundaries (velocity NOT zeroed):
    /// Bit 0 (1): -X open, Bit 1 (2): +X open
    /// Bit 2 (4): -Y open, Bit 3 (8): +Y open
    /// Bit 4 (16): -Z open, Bit 5 (32): +Z open
    pub open_boundaries: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// Flow acceleration parameters (for sluice downstream flow)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct FlowParams3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub flow_accel_dt: f32, // flow_accel * dt
}

/// Vorticity computation parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct VorticityParams3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub cell_size: f32,
}

/// Vorticity confinement parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct VortConfineParams3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub epsilon_h_dt: f32, // epsilon * h * dt
}

/// Sediment fraction parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct SedimentFractionParams3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub rest_particles: f32,
}

/// Sediment pressure parameters (for Drucker-Prager)
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

/// Porosity drag parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct PorosityDragParams3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub drag_dt: f32,
}

/// Boundary condition parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct BcParams3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    /// Bitmask for open boundaries (velocity NOT zeroed):
    /// Bit 0 (1): -X open, Bit 1 (2): +X open
    /// Bit 2 (4): -Y open, Bit 3 (8): +Y open
    /// Bit 4 (16): -Z open, Bit 5 (32): +Z open
    pub open_boundaries: u32,
    /// Slip factor for tangential velocities at solid boundaries:
    /// 1.0 = free-slip (allow tangential flow, default)
    /// 0.0 = no-slip (zero tangential velocity)
    pub slip_factor: f32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Density error computation parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct DensityErrorParams3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub rest_density: f32, // Target particles per cell (~8 for typical FLIP)
    pub dt: f32,           // Timestep for scaling
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}

/// Density position grid parameters (first pass - grid-based position changes)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct DensityPositionGridParams3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub dt: f32,
}

/// Density position correction parameters (blub grid-based trilinear sampling)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct DensityCorrectionParams3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub particle_count: u32,
    pub cell_size: f32,
    pub dt: f32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Sediment cell type builder parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct SedimentCellTypeParams3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub _pad0: u32,
}

/// Velocity extrapolation parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct VelocityExtrapParams3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub extrap_pass: u32,  // Current extrapolation pass
}

/// SDF collision parameters
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
