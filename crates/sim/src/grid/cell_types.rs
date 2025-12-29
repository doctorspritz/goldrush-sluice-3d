//! Cell type definitions and accessors.
//!
//! Defines the CellType enum and deposited cell tracking.
//!
//! NOTE: These are COPIES of methods from mod.rs for the refactor extraction.
//! The originals in mod.rs should be deleted in Phase 3 by the Lead agent.

use super::{CellType, DepositedCell, Grid};

// ============================================================================
// CELL INDEX METHODS (impl Grid)
// These use _impl suffix to avoid conflicts with the originals in mod.rs
// ============================================================================

impl Grid {
    /// Cell center index (for pressure, divergence, cell_type)
    /// Copy of method from mod.rs
    #[inline]
    pub fn cell_index_impl(&self, i: usize, j: usize) -> usize {
        j * self.width + i
    }

    /// U velocity index (staggered on left edges)
    /// Copy of method from mod.rs
    #[inline]
    pub fn u_index_impl(&self, i: usize, j: usize) -> usize {
        j * (self.width + 1) + i
    }

    /// V velocity index (staggered on bottom edges)
    /// Copy of method from mod.rs
    #[inline]
    pub fn v_index_impl(&self, i: usize, j: usize) -> usize {
        j * self.width + i
    }

    /// Check if cell coordinates are within grid bounds
    #[inline]
    pub fn is_valid_cell(&self, i: usize, j: usize) -> bool {
        i < self.width && j < self.height
    }

    // ========================================================================
    // CELL TYPE ACCESSORS
    // ========================================================================

    /// Get cell type at position
    #[inline]
    pub fn cell_type_at(&self, i: usize, j: usize) -> CellType {
        if i >= self.width || j >= self.height {
            return CellType::Solid; // Out of bounds is solid
        }
        self.cell_type[self.cell_index(i, j)]
    }

    /// Set cell type at position
    #[inline]
    pub fn set_cell_type(&mut self, i: usize, j: usize, cell_type: CellType) {
        if i < self.width && j < self.height {
            let idx = self.cell_index(i, j);
            self.cell_type[idx] = cell_type;
        }
    }

    /// Check if cell contains fluid
    /// Copy using different name to avoid conflict with is_fluid in mod.rs
    #[inline]
    pub fn is_fluid_cell(&self, i: usize, j: usize) -> bool {
        if i >= self.width || j >= self.height {
            return false;
        }
        self.cell_type[self.cell_index(i, j)] == CellType::Fluid
    }

    /// Check if cell is solid terrain
    /// Copy using different name to avoid conflict with is_solid in mod.rs
    #[inline]
    pub fn is_solid_cell(&self, i: usize, j: usize) -> bool {
        if i >= self.width || j >= self.height {
            return true; // Out of bounds is solid
        }
        self.solid[self.cell_index(i, j)]
    }

    // ========================================================================
    // DEPOSITED CELL ACCESSORS
    // ========================================================================

    /// Check if cell is deposited sediment
    /// Copy using different name to avoid conflict with is_deposited in mod.rs
    #[inline]
    pub fn is_deposited_cell(&self, i: usize, j: usize) -> bool {
        if i >= self.width || j >= self.height {
            return false;
        }
        self.deposited[self.cell_index(i, j)].is_deposited()
    }

    /// Get deposited cell reference
    /// Copy using different name to avoid conflict with get_deposited in mod.rs
    pub fn get_deposited_cell(&self, i: usize, j: usize) -> Option<&DepositedCell> {
        if i >= self.width || j >= self.height {
            return None;
        }
        let idx = self.cell_index(i, j);
        if self.deposited[idx].is_deposited() {
            Some(&self.deposited[idx])
        } else {
            None
        }
    }
}
