//! 3D Particle representation for FLIP/APIC simulation.

use glam::{Mat3, Vec3};

/// A single particle in the 3D FLIP simulation.
#[derive(Clone, Copy, Debug)]
pub struct Particle3D {
    /// World position
    pub position: Vec3,
    /// Current velocity
    pub velocity: Vec3,
    /// APIC affine velocity matrix (captures local velocity gradient)
    pub affine_velocity: Mat3,
    /// Grid velocity at particle position from previous step (for FLIP delta)
    pub old_grid_velocity: Vec3,
    /// Density relative to water (1.0 = water, 2.65 = sand, 19.3 = gold)
    pub density: f32,
}

impl Particle3D {
    /// Create a new water particle at the given position with initial velocity.
    pub fn new(position: Vec3, velocity: Vec3) -> Self {
        Self::with_density(position, velocity, 1.0)
    }

    /// Create a new particle with specified density.
    pub fn with_density(position: Vec3, velocity: Vec3, density: f32) -> Self {
        Self {
            position,
            velocity,
            affine_velocity: Mat3::ZERO,
            old_grid_velocity: Vec3::ZERO,
            density,
        }
    }

    /// Create a stationary water particle at the given position.
    pub fn at(position: Vec3) -> Self {
        Self::new(position, Vec3::ZERO)
    }

    /// Create a stationary sediment particle with specified density.
    pub fn sediment_at(position: Vec3, density: f32) -> Self {
        Self::with_density(position, Vec3::ZERO, density)
    }

    /// Is this a sediment particle (denser than water)?
    pub fn is_sediment(&self) -> bool {
        self.density > 1.0
    }
}

impl Default for Particle3D {
    fn default() -> Self {
        Self::new(Vec3::ZERO, Vec3::ZERO)
    }
}

/// Collection of particles.
pub struct Particles3D {
    pub list: Vec<Particle3D>,
}

impl Particles3D {
    /// Create an empty particle collection.
    pub fn new() -> Self {
        Self { list: Vec::new() }
    }

    /// Create with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            list: Vec::with_capacity(capacity),
        }
    }

    /// Add a water particle with the given position and velocity.
    pub fn spawn(&mut self, position: Vec3, velocity: Vec3) {
        self.list.push(Particle3D::new(position, velocity));
    }

    /// Add a particle with specified density.
    pub fn spawn_with_density(&mut self, position: Vec3, velocity: Vec3, density: f32) {
        self.list
            .push(Particle3D::with_density(position, velocity, density));
    }

    /// Add a stationary water particle.
    pub fn spawn_at(&mut self, position: Vec3) {
        self.list.push(Particle3D::at(position));
    }

    /// Add a stationary sediment particle with specified density.
    pub fn spawn_sediment(&mut self, position: Vec3, density: f32) {
        self.list.push(Particle3D::sediment_at(position, density));
    }

    /// Number of particles.
    pub fn len(&self) -> usize {
        self.list.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    /// Clear all particles.
    pub fn clear(&mut self) {
        self.list.clear();
    }

    /// Remove particles outside the given bounds.
    pub fn remove_out_of_bounds(&mut self, min: Vec3, max: Vec3) {
        self.list.retain(|p| {
            p.position.x >= min.x
                && p.position.x <= max.x
                && p.position.y >= min.y
                && p.position.y <= max.y
                && p.position.z >= min.z
                && p.position.z <= max.z
        });
    }
}

impl Default for Particles3D {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_particle_creation() {
        let p = Particle3D::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.1, 0.2, 0.3));
        assert_eq!(p.position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(p.velocity, Vec3::new(0.1, 0.2, 0.3));
        assert_eq!(p.affine_velocity, Mat3::ZERO);
    }

    #[test]
    fn test_particles_spawn() {
        let mut particles = Particles3D::new();
        particles.spawn(Vec3::ONE, Vec3::ZERO);
        particles.spawn_at(Vec3::new(2.0, 2.0, 2.0));
        assert_eq!(particles.len(), 2);
    }

    #[test]
    fn test_remove_out_of_bounds() {
        let mut particles = Particles3D::new();
        particles.spawn_at(Vec3::new(0.5, 0.5, 0.5)); // In bounds
        particles.spawn_at(Vec3::new(2.0, 0.5, 0.5)); // Out of bounds
        particles.spawn_at(Vec3::new(-0.1, 0.5, 0.5)); // Out of bounds

        particles.remove_out_of_bounds(Vec3::ZERO, Vec3::ONE);
        assert_eq!(particles.len(), 1);
    }
}
