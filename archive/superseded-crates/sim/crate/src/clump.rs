//! Rigid particle clumps for gravel simulation
//!
//! Clumps are groups of particles that move together as a rigid body.
//! They enable realistic gravel dynamics where round clumps roll and
//! flat clumps slide/catch in sluice riffles.
//!
//! Industry-standard "multi-sphere" approach from DEM literature:
//! - PFC/LIGGGHTS/MercuryDPM all use this method
//! - Particles maintain fixed local offsets from clump center
//! - Clump has center-of-mass position, rotation angle, and velocities

use glam::Vec2;

/// Clump shape types - determines particle arrangement
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClumpShape {
    // Round shapes (square packing) - tend to roll
    Round2x2, // 4 particles
    Round3x3, // 9 particles
    Round4x4, // 16 particles
    Round5x5, // 25 particles
    // Flat shapes (planar, wider aspect ratio) - tend to slide/catch
    Flat2x2, // 4 particles
    Flat3x3, // 9 particles
    Flat4x4, // 16 particles
    Flat5x5, // 25 particles
}

impl ClumpShape {
    /// Number of particles in this shape
    pub fn particle_count(&self) -> usize {
        match self {
            Self::Round2x2 | Self::Flat2x2 => 4,
            Self::Round3x3 | Self::Flat3x3 => 9,
            Self::Round4x4 | Self::Flat4x4 => 16,
            Self::Round5x5 | Self::Flat5x5 => 25,
        }
    }

    /// Grid size (N for NxN arrangement)
    pub fn grid_size(&self) -> usize {
        match self {
            Self::Round2x2 | Self::Flat2x2 => 2,
            Self::Round3x3 | Self::Flat3x3 => 3,
            Self::Round4x4 | Self::Flat4x4 => 4,
            Self::Round5x5 | Self::Flat5x5 => 5,
        }
    }

    /// Is this a flat (planar) shape?
    pub fn is_flat(&self) -> bool {
        matches!(
            self,
            Self::Flat2x2 | Self::Flat3x3 | Self::Flat4x4 | Self::Flat5x5
        )
    }

    /// All available shape variants
    pub const ALL: [ClumpShape; 8] = [
        Self::Round2x2,
        Self::Round3x3,
        Self::Round4x4,
        Self::Round5x5,
        Self::Flat2x2,
        Self::Flat3x3,
        Self::Flat4x4,
        Self::Flat5x5,
    ];
}

/// Pre-computed clump template with particle offsets and inertial properties
#[derive(Clone, Debug)]
pub struct ClumpTemplate {
    /// Shape type
    pub shape: ClumpShape,
    /// Particle positions relative to center of mass (unrotated)
    pub local_offsets: Vec<Vec2>,
    /// Total mass (sum of particle masses)
    pub mass: f32,
    /// Moment of inertia about center of mass
    pub inertia: f32,
    /// Bounding radius for broad-phase collision
    pub bounding_radius: f32,
}

impl ClumpTemplate {
    /// Generate a clump template for the given shape
    ///
    /// # Arguments
    /// - `shape`: The clump shape to generate
    /// - `particle_spacing`: Distance between adjacent particles (typically cell_size * 0.4)
    /// - `particle_mass`: Mass of each individual particle
    pub fn generate(shape: ClumpShape, particle_spacing: f32, particle_mass: f32) -> Self {
        let n = shape.grid_size();
        let is_flat = shape.is_flat();

        // Flat shapes have 2:1 aspect ratio (stretched in X, compressed in Y)
        let x_scale = if is_flat { 1.0 } else { 1.0 };
        let y_scale = if is_flat { 0.5 } else { 1.0 };

        // Generate grid positions centered at origin
        let mut local_offsets = Vec::with_capacity(n * n);
        let half = (n as f32 - 1.0) / 2.0;

        for j in 0..n {
            for i in 0..n {
                let x = (i as f32 - half) * particle_spacing * x_scale;
                let y = (j as f32 - half) * particle_spacing * y_scale;
                local_offsets.push(Vec2::new(x, y));
            }
        }

        // Total mass
        let mass = particle_mass * local_offsets.len() as f32;

        // Moment of inertia: I = Σ m_i * |r_i|²
        let inertia: f32 = local_offsets
            .iter()
            .map(|r| particle_mass * r.length_squared())
            .sum();

        // Bounding radius: max distance from center to any particle
        let bounding_radius = local_offsets
            .iter()
            .map(|r| r.length())
            .fold(0.0_f32, f32::max);

        Self {
            shape,
            local_offsets,
            mass,
            inertia: inertia.max(0.001), // Avoid zero inertia
            bounding_radius,
        }
    }

    /// Generate all 8 clump templates
    pub fn generate_all(particle_spacing: f32, particle_mass: f32) -> Vec<ClumpTemplate> {
        ClumpShape::ALL
            .iter()
            .map(|&shape| Self::generate(shape, particle_spacing, particle_mass))
            .collect()
    }
}

/// Rigid body state for a clump of bonded particles
#[derive(Clone, Debug)]
pub struct Clump {
    /// Unique clump ID (matches clump_id in constituent particles)
    pub id: u32,
    /// Center of mass position
    pub position: Vec2,
    /// Linear velocity of center of mass
    pub velocity: Vec2,
    /// Rotation angle (radians)
    pub rotation: f32,
    /// Angular velocity (radians/second)
    pub angular_velocity: f32,
    /// Index into the clump templates array
    pub template_idx: usize,
    /// Indices of constituent particles in the Particles list
    pub particle_indices: Vec<usize>,
    /// Is the clump sleeping (stable, no updates needed)?
    pub is_sleeping: bool,
}

impl Clump {
    /// Create a new clump
    pub fn new(
        id: u32,
        position: Vec2,
        velocity: Vec2,
        rotation: f32,
        template_idx: usize,
        particle_indices: Vec<usize>,
    ) -> Self {
        Self {
            id,
            position,
            velocity,
            rotation,
            angular_velocity: 0.0,
            template_idx,
            particle_indices,
            is_sleeping: false,
        }
    }

    /// Get world-space offset for a particle at local index
    pub fn get_rotated_offset(&self, template: &ClumpTemplate, local_idx: usize) -> Vec2 {
        let local = template.local_offsets[local_idx];
        let (sin_r, cos_r) = self.rotation.sin_cos();
        Vec2::new(
            cos_r * local.x - sin_r * local.y,
            sin_r * local.x + cos_r * local.y,
        )
    }

    /// Get world-space position for a particle at local index
    pub fn get_particle_world_position(&self, template: &ClumpTemplate, local_idx: usize) -> Vec2 {
        self.position + self.get_rotated_offset(template, local_idx)
    }

    /// Get velocity for a particle at local index (includes rotational component)
    pub fn get_particle_velocity(&self, template: &ClumpTemplate, local_idx: usize) -> Vec2 {
        let offset = self.get_rotated_offset(template, local_idx);
        // Tangent vector: perpendicular to offset
        let tangent = Vec2::new(-offset.y, offset.x);
        self.velocity + tangent * self.angular_velocity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clump_template_generation() {
        let template = ClumpTemplate::generate(ClumpShape::Round2x2, 1.0, 1.0);

        assert_eq!(template.local_offsets.len(), 4);
        assert_eq!(template.mass, 4.0);
        assert!(template.inertia > 0.0);
        assert!(template.bounding_radius > 0.0);
    }

    #[test]
    fn test_clump_template_all_shapes() {
        let templates = ClumpTemplate::generate_all(1.0, 1.0);

        assert_eq!(templates.len(), 8);

        // Verify particle counts
        assert_eq!(templates[0].local_offsets.len(), 4); // Round2x2
        assert_eq!(templates[1].local_offsets.len(), 9); // Round3x3
        assert_eq!(templates[2].local_offsets.len(), 16); // Round4x4
        assert_eq!(templates[3].local_offsets.len(), 25); // Round5x5
    }

    #[test]
    fn test_flat_shape_aspect_ratio() {
        let round = ClumpTemplate::generate(ClumpShape::Round3x3, 1.0, 1.0);
        let flat = ClumpTemplate::generate(ClumpShape::Flat3x3, 1.0, 1.0);

        // Flat shapes should have smaller Y extent
        let round_y_extent: f32 = round
            .local_offsets
            .iter()
            .map(|p| p.y.abs())
            .fold(0.0, f32::max);
        let flat_y_extent: f32 = flat
            .local_offsets
            .iter()
            .map(|p| p.y.abs())
            .fold(0.0, f32::max);

        assert!(
            flat_y_extent < round_y_extent,
            "Flat shape Y extent {} should be less than round {}",
            flat_y_extent,
            round_y_extent
        );
    }

    #[test]
    fn test_clump_rotation() {
        let template = ClumpTemplate::generate(ClumpShape::Round2x2, 1.0, 1.0);
        let clump = Clump::new(1, Vec2::ZERO, Vec2::ZERO, std::f32::consts::FRAC_PI_2, 0, vec![0, 1, 2, 3]);

        // After 90 degree rotation, X should become Y and Y should become -X
        let offset0 = template.local_offsets[0];
        let rotated = clump.get_rotated_offset(&template, 0);

        // Check rotation approximately correct (allowing for float precision)
        assert!((rotated.x - (-offset0.y)).abs() < 0.001);
        assert!((rotated.y - offset0.x).abs() < 0.001);
    }

    #[test]
    fn test_clump_particle_velocity() {
        let template = ClumpTemplate::generate(ClumpShape::Round2x2, 1.0, 1.0);
        let mut clump = Clump::new(1, Vec2::ZERO, Vec2::new(10.0, 0.0), 0.0, 0, vec![0, 1, 2, 3]);

        // With only linear velocity
        let vel = clump.get_particle_velocity(&template, 0);
        assert_eq!(vel, Vec2::new(10.0, 0.0));

        // Add angular velocity
        clump.angular_velocity = 1.0;
        let vel_with_rotation = clump.get_particle_velocity(&template, 0);

        // Should have additional tangential component
        assert!(vel_with_rotation != Vec2::new(10.0, 0.0));
    }
}
