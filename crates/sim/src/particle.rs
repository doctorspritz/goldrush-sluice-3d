//! Fluid particles for PIC/FLIP simulation
//!
//! Each particle has a continuous position and velocity.
//! Particles are transferred to/from a grid for pressure solving.

use glam::Vec2;

/// A fluid particle - can be water or mud (different densities)
#[derive(Clone, Copy, Debug)]
pub struct Particle {
    /// Continuous position in world coordinates
    pub position: Vec2,
    /// Current velocity
    pub velocity: Vec2,
    /// Velocity sampled from grid before pressure solve (for FLIP)
    pub old_grid_velocity: Vec2,
    /// Density: water=1.0, mud=2.5
    pub density: f32,
}

impl Particle {
    /// Create a water particle
    pub fn water(position: Vec2, velocity: Vec2) -> Self {
        Self {
            position,
            velocity,
            old_grid_velocity: Vec2::ZERO,
            density: 1.0,
        }
    }

    /// Create a mud particle (denser, settles faster)
    pub fn mud(position: Vec2, velocity: Vec2) -> Self {
        Self {
            position,
            velocity,
            old_grid_velocity: Vec2::ZERO,
            density: 2.5,
        }
    }

    /// Check if this is a mud particle
    pub fn is_mud(&self) -> bool {
        self.density > 1.5
    }
}

/// Particle collection with spatial acceleration
pub struct Particles {
    pub list: Vec<Particle>,
}

impl Particles {
    pub fn new() -> Self {
        Self { list: Vec::new() }
    }

    pub fn spawn_water(&mut self, x: f32, y: f32, vx: f32, vy: f32) {
        self.list.push(Particle::water(
            Vec2::new(x, y),
            Vec2::new(vx, vy),
        ));
    }

    pub fn spawn_mud(&mut self, x: f32, y: f32, vx: f32, vy: f32) {
        self.list.push(Particle::mud(
            Vec2::new(x, y),
            Vec2::new(vx, vy),
        ));
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
