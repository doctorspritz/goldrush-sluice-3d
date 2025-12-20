//! Material types and their physical properties.
//!
//! All materials use unified physics based on density and spread rate.
//! Heavier materials sink through lighter ones. Liquids spread horizontally.

/// All material types in the simulation.
/// Each material has unique physical behavior (falling, flowing, static).
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
#[repr(u8)]
pub enum Material {
    #[default]
    Air,
    Soil,  // Loose earth - falls and slides like sand
    PackedSoil, // Stable earth - supports weight
    Mud,   // Wet soil - slower liquid-like flow
    Rock,  // Solid - never moves
    Water, // Liquid - flows horizontally fast
    Gold,  // Heavy powder - sinks through everything except rock
}

impl Material {
    /// Density determines settling order.
    /// Higher density materials sink through lower density ones.
    /// Scale: 0 (air) to 255 (heaviest solid)
    #[inline]
    pub const fn density(self) -> u8 {
        match self {
            Material::Air => 0,
            Material::Water => 10,
            Material::Soil => 30,
            Material::PackedSoil => 35,
            Material::Mud => 40,
            Material::Rock => 200,
            Material::Gold => 250,
        }
    }

    /// Horizontal spread rate - how many cells per frame this material tries to spread.
    /// Higher = more fluid. 0 = powder (only falls, doesn't spread).
    #[inline]
    pub const fn spread_rate(self) -> u8 {
        match self {
            Material::Water => 15,     // Very fluid - spreads fast
            Material::Mud => 4,        // Viscous - spreads slowly
            Material::Soil => 0,       // Powder - doesn't spread (unless in water)
            Material::Gold => 0,       // Powder - doesn't spread
            Material::Air => 0,
            Material::Rock => 0,
            Material::PackedSoil => 0,
        }
    }

    /// Returns true if this material falls and slides (sand-like behavior).
    #[inline]
    pub const fn is_powder(self) -> bool {
        matches!(self, Material::Soil | Material::Gold)
    }

    /// Returns true if this material flows horizontally (liquid behavior).
    #[inline]
    pub const fn is_liquid(self) -> bool {
        matches!(self, Material::Water | Material::Mud)
    }

    /// Returns true if this material never moves.
    #[inline]
    pub const fn is_solid(self) -> bool {
        matches!(self, Material::Rock | Material::PackedSoil)
    }

    /// Returns true if this material can be displaced by higher-density materials.
    #[inline]
    pub const fn can_be_displaced(self) -> bool {
        matches!(self, Material::Air | Material::Water)
    }

    /// Returns true if this cell should be simulated (not air or rock).
    #[inline]
    pub const fn is_active(self) -> bool {
        !matches!(self, Material::Air | Material::Rock | Material::PackedSoil)
    }

    /// Base RGBA color for rendering.
    #[inline]
    pub const fn color(self) -> [u8; 4] {
        match self {
            Material::Air => [20, 20, 30, 255],      // Dark background
            Material::Soil => [139, 90, 43, 255],   // Brown earth
            Material::PackedSoil => [101, 67, 33, 255], // Darker, harder packed earth
            Material::Mud => [59, 41, 26, 255],     // Dark brown
            Material::Rock => [128, 128, 128, 255], // Gray
            Material::Water => [30, 100, 200, 180], // Blue, semi-transparent
            Material::Gold => [255, 215, 0, 255],   // Bright gold
        }
    }

    /// Get a slightly varied color based on position for visual interest.
    /// Uses position to create deterministic variation (no randomness).
    #[inline]
    pub fn color_varied(self, x: i32, y: i32) -> [u8; 4] {
        let base = self.color();
        if matches!(self, Material::Air) {
            return base;
        }

        // Simple hash for variation (-8 to +7 per channel)
        let hash = ((x.wrapping_mul(374761393)) ^ (y.wrapping_mul(668265263))) as u8;
        let variation = (hash & 0x0F) as i16 - 8;

        [
            (base[0] as i16 + variation).clamp(0, 255) as u8,
            (base[1] as i16 + variation).clamp(0, 255) as u8,
            (base[2] as i16 + variation).clamp(0, 255) as u8,
            base[3],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn density_ordering() {
        // Gold should sink through everything
        assert!(Material::Gold.density() > Material::Soil.density());
        assert!(Material::Gold.density() > Material::Water.density());

        // Soil should sink through water
        assert!(Material::Soil.density() > Material::Water.density());

        // Air is lightest
        assert!(Material::Air.density() < Material::Water.density());
    }

    #[test]
    fn material_categories() {
        assert!(Material::Soil.is_powder());
        assert!(Material::Gold.is_powder());
        assert!(!Material::Water.is_powder());

        assert!(Material::Water.is_liquid());
        assert!(Material::Mud.is_liquid());
        assert!(!Material::Soil.is_liquid());

        assert!(Material::Rock.is_solid());
        assert!(!Material::Soil.is_solid());
    }
}
