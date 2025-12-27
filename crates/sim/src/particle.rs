//! Fluid particles for PIC/FLIP simulation
//!
//! Each particle has a continuous position and velocity.
//! Particles are transferred to/from a grid for pressure solving.
//!
//! WATER-ONLY VERSION
//! Sediment code archived in: archive/sediment_archive.rs

use glam::{Mat2, Vec2};

/// A water particle
#[derive(Clone, Copy, Debug)]
pub struct Particle {
    /// Continuous position in world coordinates
    pub position: Vec2,
    /// Current velocity
    pub velocity: Vec2,
    /// Affine velocity matrix for APIC transfer (captures local velocity gradients)
    /// Stores rotation and deformation for angular momentum preservation
    pub affine_velocity: Mat2,
    /// Velocity sampled from grid (for FLIP delta calculation)
    pub old_grid_velocity: Vec2,
}

impl Particle {
    /// Create a water particle
    pub fn new(position: Vec2, velocity: Vec2) -> Self {
        Self {
            position,
            velocity,
            affine_velocity: Mat2::ZERO,
            old_grid_velocity: Vec2::ZERO,
        }
    }

    /// Create a water particle (alias for new)
    pub fn water(position: Vec2, velocity: Vec2) -> Self {
        Self::new(position, velocity)
    }

    /// RGBA color for rendering
    pub fn color(&self) -> [u8; 4] {
        [50, 140, 240, 180] // Blue, semi-transparent
    }

    /// Advect micro-stepped with collision callback
    /// Ensures particle never travels more than 0.5 * cell_size per substep.
    /// The callback is executed after each substep to resolve collisions.
    pub fn advect_micro_stepped<F>(&mut self, dt: f32, cell_size: f32, mut collision_callback: F)
    where
        F: FnMut(&mut Self),
    {
        let max_disp = 0.5 * cell_size;
        let disp = self.velocity * dt;
        let steps = (disp.length() / max_disp).ceil().max(1.0) as usize;
        let sub_dt = dt / steps as f32;

        for _ in 0..steps {
            self.position += self.velocity * sub_dt;
            collision_callback(self);
        }
    }
}

/// Particle collection
pub struct Particles {
    pub list: Vec<Particle>,
}

impl Particles {
    pub fn new() -> Self {
        Self { list: Vec::new() }
    }

    pub fn spawn(&mut self, x: f32, y: f32, vx: f32, vy: f32) {
        self.list.push(Particle::new(
            Vec2::new(x, y),
            Vec2::new(vx, vy),
        ));
    }

    /// Alias for spawn (legacy compatibility)
    pub fn spawn_water(&mut self, x: f32, y: f32, vx: f32, vy: f32) {
        self.spawn(x, y, vx, vy);
    }

    /// Remove particles outside the simulation bounds
    pub fn remove_out_of_bounds(&mut self, width: f32, height: f32) {
        self.list.retain(|p| {
            p.position.x >= 0.0
                && p.position.x < width
                && p.position.y >= 0.0
                && p.position.y < height
        });
    }

    pub fn len(&self) -> usize {
        self.list.len()
    }

    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Particle> {
        self.list.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Particle> {
        self.list.iter_mut()
    }
}

impl Default for Particles {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_particle_creation() {
        let p = Particle::new(Vec2::new(10.0, 20.0), Vec2::new(1.0, 2.0));
        assert_eq!(p.position, Vec2::new(10.0, 20.0));
        assert_eq!(p.velocity, Vec2::new(1.0, 2.0));
    }

    #[test]
    fn test_particles_spawn() {
        let mut particles = Particles::new();
        particles.spawn(5.0, 10.0, 0.0, 0.0);
        assert_eq!(particles.len(), 1);
    }

    #[test]
    fn test_particles_remove_out_of_bounds() {
        let mut particles = Particles::new();
        particles.spawn(5.0, 10.0, 0.0, 0.0);  // In bounds
        particles.spawn(-5.0, 10.0, 0.0, 0.0); // Out of bounds
        particles.spawn(5.0, -10.0, 0.0, 0.0); // Out of bounds
        particles.spawn(150.0, 10.0, 0.0, 0.0); // Out of bounds (if width=100)

        particles.remove_out_of_bounds(100.0, 100.0);
        assert_eq!(particles.len(), 1);
    }

    #[test]
    fn test_micro_stepping() {
        let mut p = Particle::new(Vec2::new(0.0, 0.0), Vec2::new(100.0, 0.0));
        let cell_size = 10.0;
        let dt = 1.0;

        let mut callback_count = 0;
        p.advect_micro_stepped(dt, cell_size, |_| {
            callback_count += 1;
        });

        // With velocity 100 and max_disp 5 (0.5 * 10), should take 20 steps
        assert!(callback_count >= 20);
        // Final position should be ~100 (some floating point variance)
        assert!((p.position.x - 100.0).abs() < 0.1);
    }
}
