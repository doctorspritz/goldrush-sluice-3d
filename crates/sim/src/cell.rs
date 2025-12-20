//! Cell data structure - minimal per-pixel state.

use crate::material::Material;

/// A single cell in the simulation grid.
///
/// IMPORTANT: Update tracking is NOT stored here!
/// It's stored separately in Chunk::updated to avoid the swap bug
/// where flags would travel with cell data during swaps.
#[derive(Clone, Copy, Default)]
pub struct Cell {
    pub material: Material,
}

impl Cell {
    #[inline]
    pub const fn new(material: Material) -> Self {
        Self { material }
    }

    #[inline]
    pub const fn air() -> Self {
        Self {
            material: Material::Air,
        }
    }
}
