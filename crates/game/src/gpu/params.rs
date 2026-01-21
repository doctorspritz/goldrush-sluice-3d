//! GPU shader parameter structs for FLIP 3D simulation.
//!
//! These are `#[repr(C)]` structs that are uploaded to GPU uniform/storage buffers
//! for compute shader parameters.
//!
//! # Consolidated Parameter Structs
//!
//! To reduce duplication, common patterns are consolidated:
//! - [`GridParams3D`]: Grid dimensions only (width, height, depth)
//! - [`GridScalarParams3D`]: Grid dimensions + single scalar value
//! - [`GridCellSizeParams3D`]: Grid dimensions + cell_size + extra u32 + padding
//! - [`GridCellSizeScalarParams3D`]: Grid dimensions + cell_size + scalar + padding
//!
//! Complex structs with unique field combinations remain separate.

use bytemuck::{Pod, Zeroable};

// =============================================================================
// Consolidated Base Structs
// =============================================================================

/// Base grid parameters (16 bytes).
///
/// Used for shaders that only need grid dimensions.
/// Replaces: `BcParams3D`, `SedimentCellTypeParams3D`
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(crate) struct GridParams3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub _pad: u32,
}

impl GridParams3D {
    pub fn new(width: u32, height: u32, depth: u32) -> Self {
        Self {
            width,
            height,
            depth,
            _pad: 0,
        }
    }
}

/// Grid parameters with a single scalar value (16 bytes).
///
/// Used for shaders that need grid dimensions plus one f32 parameter.
/// Shaders should access the value as `params.value`.
///
/// Replaces: `FlowParams3D`, `VorticityParams3D`, `VortConfineParams3D`,
///           `SedimentFractionParams3D`, `PorosityDragParams3D`
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(crate) struct GridScalarParams3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    /// Generic scalar value (flow_accel_dt, cell_size, epsilon_h_dt, etc.)
    pub value: f32,
}

impl GridScalarParams3D {
    pub fn new(width: u32, height: u32, depth: u32, value: f32) -> Self {
        Self {
            width,
            height,
            depth,
            value,
        }
    }
}

/// Grid parameters with cell_size and an extra u32 value (32 bytes).
///
/// Layout: [width, height, depth, extra_u32, cell_size, _pad, _pad, _pad]
/// Used for shaders needing grid + cell_size + one u32 parameter.
///
/// Replaces: `GravelObstacleParams3D`
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(crate) struct GridCellSizeU32Params3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    /// Extra u32 value (obstacle_count, etc.)
    pub extra: u32,
    pub cell_size: f32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

impl GridCellSizeU32Params3D {
    pub fn new(width: u32, height: u32, depth: u32, extra: u32, cell_size: f32) -> Self {
        Self {
            width,
            height,
            depth,
            extra,
            cell_size,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }
}

/// Grid parameters with cell_size and an extra f32 value (32 bytes).
///
/// Layout: [width, height, depth, extra_f32, cell_size, _pad, _pad, _pad]
/// Used for shaders needing grid + cell_size + one f32 parameter.
///
/// Replaces: `GravityParams3D`
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(crate) struct GridCellSizeScalarParams3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    /// Extra f32 value (gravity_dt, etc.)
    pub extra: f32,
    pub cell_size: f32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

impl GridCellSizeScalarParams3D {
    pub fn new(width: u32, height: u32, depth: u32, extra: f32, cell_size: f32) -> Self {
        Self {
            width,
            height,
            depth,
            extra,
            cell_size,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }
}

// =============================================================================
// Type Aliases for Semantic Clarity
// =============================================================================

/// Boundary condition parameters (uses GridParams3D).
pub(crate) type BcParams3D = GridParams3D;

/// Sediment cell type builder parameters (uses GridParams3D).
pub(crate) type SedimentCellTypeParams3D = GridParams3D;

/// Flow acceleration parameters (uses GridScalarParams3D, value = flow_accel_dt).
pub(crate) type FlowParams3D = GridScalarParams3D;

/// Vorticity computation parameters (uses GridScalarParams3D, value = cell_size).
pub(crate) type VorticityParams3D = GridScalarParams3D;

/// Vorticity confinement parameters (uses GridScalarParams3D, value = epsilon_h_dt).
pub(crate) type VortConfineParams3D = GridScalarParams3D;

/// Sediment fraction parameters (uses GridScalarParams3D, value = rest_particles).
pub(crate) type SedimentFractionParams3D = GridScalarParams3D;

/// Porosity drag parameters (uses GridScalarParams3D, value = drag_dt).
pub(crate) type PorosityDragParams3D = GridScalarParams3D;

/// Gravel obstacle shader parameters (uses GridCellSizeU32Params3D, extra = obstacle_count).
pub(crate) type GravelObstacleParams3D = GridCellSizeU32Params3D;

/// Gravity application parameters (uses GridCellSizeScalarParams3D, extra = gravity_dt).
pub(crate) type GravityParams3D = GridCellSizeScalarParams3D;

// =============================================================================
// Complex Structs (unique field combinations)
// =============================================================================

/// Gravel obstacle for dynamic solid collision.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GravelObstacle {
    pub position: [f32; 3],
    pub radius: f32,
}

/// Sediment pressure parameters (for Drucker-Prager).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
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

impl SedimentPressureParams3D {
    pub fn _new(
        width: u32,
        height: u32,
        depth: u32,
        cell_size: f32,
        particle_mass: f32,
        gravity: f32,
        buoyancy_factor: f32,
    ) -> Self {
        Self {
            width,
            height,
            depth,
            _pad0: 0,
            cell_size,
            particle_mass,
            gravity,
            buoyancy_factor,
        }
    }
}

/// Density error parameters.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
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

impl DensityErrorParams3D {
    pub fn _new(
        width: u32,
        height: u32,
        depth: u32,
        rest_density: f32,
        dt: f32,
        surface_clamp: bool,
    ) -> Self {
        Self {
            width,
            height,
            depth,
            rest_density,
            dt,
            surface_clamp: surface_clamp as u32,
            _pad2: 0,
            _pad3: 0,
        }
    }
}

/// Density position grid parameters.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
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

impl DensityPositionGridParams3D {
    pub fn _new(width: u32, height: u32, depth: u32, dt: f32, strength: f32) -> Self {
        Self {
            width,
            height,
            depth,
            dt,
            strength,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        }
    }
}

/// Density correct parameters.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
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

impl DensityCorrectParams3D {
    pub fn _new(
        width: u32,
        height: u32,
        depth: u32,
        particle_count: u32,
        cell_size: f32,
        damping: f32,
    ) -> Self {
        Self {
            width,
            height,
            depth,
            particle_count,
            cell_size,
            damping,
            _pad2: 0,
            _pad3: 0,
        }
    }
}

/// SDF collision parameters.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
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

impl SdfCollisionParams3D {
    pub fn _new(
        width: u32,
        height: u32,
        depth: u32,
        particle_count: u32,
        cell_size: f32,
        dt: f32,
        open_boundaries: u32,
    ) -> Self {
        Self {
            width,
            height,
            depth,
            particle_count,
            cell_size,
            dt,
            open_boundaries,
            _pad0: 0,
        }
    }
}
