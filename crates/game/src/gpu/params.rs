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

/// Boundary condition parameters (32 bytes).
///
/// Matches layout in `enforce_bc_3d.wgsl`:
/// struct Params {
///     width: u32,
///     height: u32,
///     depth: u32,
///     open_boundaries: u32,
///     slip_factor: f32,
///     _pad0: u32,
///     _pad1: u32,
///     _pad2: u32,
/// }
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(crate) struct BcParams3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub open_boundaries: u32,
    pub slip_factor: f32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

impl BcParams3D {
    pub fn new(width: u32, height: u32, depth: u32, open_boundaries: u32, slip_factor: f32) -> Self {
        Self {
            width,
            height,
            depth,
            open_boundaries,
            slip_factor,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }
}

/// Fluid cell expansion parameters (32 bytes).
///
/// Matches layout in `fluid_cell_expand_3d.wgsl`:
/// struct Params {
///     width: u32,
///     height: u32,
///     depth: u32,
///     open_boundaries: u32,
///     min_neighbors: u32,
///     _pad0: u32,
///     _pad1: u32,
///     _pad2: u32,
/// }
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(crate) struct FluidCellExpandParams3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub open_boundaries: u32,
    pub min_neighbors: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

impl FluidCellExpandParams3D {
    pub fn new(
        width: u32,
        height: u32,
        depth: u32,
        open_boundaries: u32,
        min_neighbors: u32,
    ) -> Self {
        Self {
            width,
            height,
            depth,
            open_boundaries,
            min_neighbors: min_neighbors.clamp(1, 6),
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }
}

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

/// Gravity application parameters (32 bytes).
///
/// Matches layout in `gravity_3d.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(crate) struct GravityParams3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    /// gravity_dt = gravity * dt
    pub extra: f32,
    pub cell_size: f32,
    pub open_boundaries: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

impl GravityParams3D {
    pub fn new(
        width: u32,
        height: u32,
        depth: u32,
        extra: f32,
        cell_size: f32,
        open_boundaries: u32,
    ) -> Self {
        Self {
            width,
            height,
            depth,
            extra,
            cell_size,
            open_boundaries,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::fs;
    use std::mem::{offset_of, size_of};
    use std::path::Path;

    struct WgslLayout {
        size: u32,
        offsets: HashMap<String, u32>,
    }

    fn wgsl_struct_layout(shader: &str, struct_name: &str) -> WgslLayout {
        let shader_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/shaders").join(shader);
        let source = fs::read_to_string(&shader_path)
            .unwrap_or_else(|e| panic!("Failed to read {:?}: {e}", shader_path));
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("Failed to parse {:?}: {}", shader_path, e.emit_to_string(&source)));

        let mut layouter = naga::proc::Layouter::default();
        let gctx = naga::proc::GlobalCtx {
            types: &module.types,
            constants: &module.constants,
            overrides: &module.overrides,
            global_expressions: &module.global_expressions,
        };
        layouter
            .update(gctx)
            .unwrap_or_else(|e| panic!("Failed to compute layout for {:?}: {e}", shader_path));

        let (handle, ty) = module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some(struct_name))
            .unwrap_or_else(|| panic!("Struct {struct_name} not found in {:?}", shader_path));

        let members = match &ty.inner {
            naga::TypeInner::Struct { members, .. } => members,
            _ => panic!("Type {struct_name} is not a struct in {:?}", shader_path),
        };

        let mut offsets = HashMap::new();
        for member in members {
            if let Some(name) = &member.name {
                offsets.insert(name.clone(), member.offset);
            }
        }

        WgslLayout {
            size: layouter[handle].size,
            offsets,
        }
    }

    #[test]
    fn bc_params_layout_matches_wgsl() {
        let layout = wgsl_struct_layout("enforce_bc_3d.wgsl", "Params");
        assert_eq!(layout.size as usize, size_of::<BcParams3D>());
        assert_eq!(
            *layout.offsets.get("open_boundaries").unwrap(),
            offset_of!(BcParams3D, open_boundaries) as u32
        );
        assert_eq!(
            *layout.offsets.get("slip_factor").unwrap(),
            offset_of!(BcParams3D, slip_factor) as u32
        );
    }

    #[test]
    fn gravity_params_layout_matches_wgsl() {
        let layout = wgsl_struct_layout("gravity_3d.wgsl", "Params");
        assert_eq!(layout.size as usize, size_of::<GravityParams3D>());
        assert_eq!(
            *layout.offsets.get("open_boundaries").unwrap(),
            offset_of!(GravityParams3D, open_boundaries) as u32
        );
        assert_eq!(
            *layout.offsets.get("cell_size").unwrap(),
            offset_of!(GravityParams3D, cell_size) as u32
        );
    }

    #[test]
    fn fluid_cell_expand_params_layout_matches_wgsl() {
        let layout = wgsl_struct_layout("fluid_cell_expand_3d.wgsl", "Params");
        assert_eq!(layout.size as usize, size_of::<FluidCellExpandParams3D>());
        assert_eq!(
            *layout.offsets.get("open_boundaries").unwrap(),
            offset_of!(FluidCellExpandParams3D, open_boundaries) as u32
        );
        assert_eq!(
            *layout.offsets.get("min_neighbors").unwrap(),
            offset_of!(FluidCellExpandParams3D, min_neighbors) as u32
        );
    }
}
