//! Chunk - a 64x64 region of the world.

use crate::material::Material;

pub const CHUNK_SIZE: usize = 64;
pub const CHUNK_AREA: usize = CHUNK_SIZE * CHUNK_SIZE;

/// A 64x64 chunk of the world.
///
/// Key design decision: `updated` flags are stored SEPARATELY from cells.
/// This prevents the "swap bug" where update flags would travel with cell
/// data when two cells swap positions, causing double-updates.
/// A 64x64 chunk of the world.
///
/// Refactored to Struct of Arrays (SoA) for cache efficiency and physics performance.
pub struct Chunk {
    /// Material ID for each cell (solid materials only - Rock, Soil, Gold, Mud, Air)
    pub materials: Box<[Material; CHUNK_AREA]>,

    /// Velocity X component (-128.0 to 127.0)
    pub vel_x: Box<[f32; CHUNK_AREA]>,

    /// Velocity Y component (-128.0 to 127.0)
    pub vel_y: Box<[f32; CHUNK_AREA]>,

    /// Stability factor (0.0 to 1.0)
    pub stability: Box<[f32; CHUNK_AREA]>,

    /// Update flags - tracks which positions were processed this frame.
    pub updated: Box<[bool; CHUNK_AREA]>,

    /// Scratch buffer for fluid solver (avoids allocation per frame)
    pub scratch_a: Box<[f32; CHUNK_AREA]>,
    /// Second scratch buffer for fluid solver
    pub scratch_b: Box<[f32; CHUNK_AREA]>,

    // === Virtual Pipes Water System ===
    /// Water amount per cell (0.0 = dry, 1.0+ = water present)
    pub water_mass: Box<[f32; CHUNK_AREA]>,
    /// Flow rate to right neighbor (volume/time, positive = rightward)
    pub flow_right: Box<[f32; CHUNK_AREA]>,
    /// Flow rate to bottom neighbor (volume/time, positive = downward)
    pub flow_down: Box<[f32; CHUNK_AREA]>,

    /// Set true when any cell changes - triggers re-render
    pub needs_render: bool,

    /// Set true when chunk has active (moving) particles.
    pub is_active: bool,

    /// True if chunk contains any water (optimization for water solver)
    pub has_water: bool,

    /// True if chunk contains any liquid particles (Mud)
    pub has_liquid: bool,
}

impl Default for Chunk {
    fn default() -> Self {
        Self::new()
    }
}

impl Chunk {
    /// Create a new chunk filled with air.
    pub fn new() -> Self {
        Self {
            materials: Box::new([Material::Air; CHUNK_AREA]),
            vel_x: Box::new([0.0; CHUNK_AREA]),
            vel_y: Box::new([0.0; CHUNK_AREA]),
            stability: Box::new([1.0; CHUNK_AREA]),
            updated: Box::new([false; CHUNK_AREA]),
            scratch_a: Box::new([0.0; CHUNK_AREA]),
            scratch_b: Box::new([0.0; CHUNK_AREA]),
            water_mass: Box::new([0.0; CHUNK_AREA]),
            flow_right: Box::new([0.0; CHUNK_AREA]),
            flow_down: Box::new([0.0; CHUNK_AREA]),
            needs_render: true,
            is_active: false,
            has_water: false,
            has_liquid: false,
        }
    }

    /// Create a chunk filled with a specific material.
    pub fn filled(material: Material) -> Self {
        Self {
            materials: Box::new([material; CHUNK_AREA]),
            vel_x: Box::new([0.0; CHUNK_AREA]),
            vel_y: Box::new([0.0; CHUNK_AREA]),
            stability: Box::new([1.0; CHUNK_AREA]),
            updated: Box::new([false; CHUNK_AREA]),
            scratch_a: Box::new([0.0; CHUNK_AREA]),
            scratch_b: Box::new([0.0; CHUNK_AREA]),
            water_mass: Box::new([0.0; CHUNK_AREA]),
            flow_right: Box::new([0.0; CHUNK_AREA]),
            flow_down: Box::new([0.0; CHUNK_AREA]),
            needs_render: true,
            is_active: material.is_active(),
            has_water: false,
            has_liquid: material.is_liquid(),
        }
    }

    /// Get unique index for local coordinates.
    #[inline]
    pub const fn index(x: usize, y: usize) -> usize {
        debug_assert!(x < CHUNK_SIZE && y < CHUNK_SIZE);
        y * CHUNK_SIZE + x
    }

    /// Get material at local position.
    #[inline]
    pub fn get_material(&self, x: usize, y: usize) -> Material {
        self.materials[Self::index(x, y)]
    }

    /// Set material at local position.
    /// Resets physics state for new material.
    #[inline]
    pub fn set_material(&mut self, x: usize, y: usize, material: Material) {
        let idx = Self::index(x, y);
        self.materials[idx] = material;
        self.vel_x[idx] = 0.0;
        self.vel_y[idx] = 0.0;
        self.stability[idx] = 1.0;

        self.needs_render = true;
        if material.is_active() {
            self.is_active = true;
        }
        if material.is_liquid() {
            self.has_liquid = true;
        }
    }

    /// Set full state (material + physics).
    #[inline]
    pub fn set_state(&mut self, x: usize, y: usize, material: Material, vx: f32, vy: f32) {
        let idx = Self::index(x, y);
        self.materials[idx] = material;
        self.vel_x[idx] = vx;
        self.vel_y[idx] = vy;

        self.needs_render = true;
        if material.is_active() {
            self.is_active = true;
        }
        if material.is_liquid() {
            self.has_liquid = true;
        }
    }

    /// Swap two cells within this chunk (moves ALL properties).
    /// Marks both positions as updated.
    #[inline]
    pub fn swap(&mut self, x1: usize, y1: usize, x2: usize, y2: usize) {
        let idx1 = Self::index(x1, y1);
        let idx2 = Self::index(x2, y2);

        // Swap all parallel arrays
        self.materials.swap(idx1, idx2);
        self.vel_x.swap(idx1, idx2);
        self.vel_y.swap(idx1, idx2);
        self.stability.swap(idx1, idx2);

        // Mark updated
        self.updated[idx1] = true;
        self.updated[idx2] = true;

        self.needs_render = true;
        self.is_active = true;
    }

    /// Swap materials ONLY (preserves velocity field for Navier-Stokes).
    /// Use this for liquid advection to maintain coherent vortices.
    #[inline]
    pub fn swap_material_only(&mut self, x1: usize, y1: usize, x2: usize, y2: usize) {
        let idx1 = Self::index(x1, y1);
        let idx2 = Self::index(x2, y2);

        // Swap ONLY materials - velocity stays on grid!
        self.materials.swap(idx1, idx2);

        // Mark updated
        self.updated[idx1] = true;
        self.updated[idx2] = true;

        self.needs_render = true;
        self.is_active = true;
    }

    /// Check if position was already updated this frame.
    #[inline]
    pub fn is_updated(&self, x: usize, y: usize) -> bool {
        self.updated[Self::index(x, y)]
    }

    /// Mark position as updated.
    #[inline]
    pub fn mark_updated(&mut self, x: usize, y: usize) {
        self.updated[Self::index(x, y)] = true;
    }

    /// Clear all update flags for new frame.
    pub fn clear_updated(&mut self) {
        self.updated.fill(false);
    }

    /// Check if chunk has any active materials, liquids, and water.
    pub fn recalculate_active(&mut self) {
        self.is_active = false;
        self.has_liquid = false;
        self.has_water = false;

        for (i, m) in self.materials.iter().enumerate() {
            if m.is_active() {
                self.is_active = true;
            }
            if m.is_liquid() {
                self.has_liquid = true;
            }
            if self.water_mass[i] > 0.01 {
                self.has_water = true;
                self.is_active = true; // Water movement keeps chunk active
            }
            // Early exit if all found
            if self.is_active && self.has_liquid && self.has_water {
                break;
            }
        }
    }

    /// Get water mass at local position.
    #[inline]
    pub fn get_water(&self, x: usize, y: usize) -> f32 {
        self.water_mass[Self::index(x, y)]
    }

    /// Set water mass at local position.
    #[inline]
    pub fn set_water(&mut self, x: usize, y: usize, amount: f32) {
        let idx = Self::index(x, y);
        self.water_mass[idx] = amount.max(0.0);
        if amount > 0.01 {
            self.has_water = true;
            self.is_active = true;
        }
        self.needs_render = true;
    }

    /// Add water mass at local position.
    #[inline]
    pub fn add_water(&mut self, x: usize, y: usize, amount: f32) {
        let idx = Self::index(x, y);
        self.water_mass[idx] = (self.water_mass[idx] + amount).max(0.0);
        if self.water_mass[idx] > 0.01 {
            self.has_water = true;
            self.is_active = true;
        }
        self.needs_render = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swap_preserves_update_positions() {
        let mut chunk = Chunk::new();

        // Place soil at (0, 0) and air at (0, 1)
        chunk.set_material(0, 0, Material::Soil);
        chunk.clear_updated();

        // Simulate: soil falls from (0,0) to (0,1)
        chunk.swap(0, 0, 0, 1);

        // CRITICAL: Both positions should be marked updated
        assert!(chunk.is_updated(0, 0), "Source position should be updated");
        assert!(chunk.is_updated(0, 1), "Target position should be updated");

        // Cell data should have swapped
        assert_eq!(chunk.get_material(0, 1), Material::Soil);
        assert_eq!(chunk.get_material(0, 0), Material::Air);
    }

    #[test]
    fn index_calculation() {
        assert_eq!(Chunk::index(0, 0), 0);
        assert_eq!(Chunk::index(63, 0), 63);
        assert_eq!(Chunk::index(0, 1), 64);
        assert_eq!(Chunk::index(63, 63), 4095);
    }
}
