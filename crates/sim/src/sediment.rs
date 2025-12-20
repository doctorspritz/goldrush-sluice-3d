//! Sediment particles with drift-flux physics
//!
//! Sediment particles are one-way coupled to the FLIP velocity field.
//! They use the Rubey equation for settling velocity, allowing heavy
//! particles (gold) to "slip" through fluid flow and settle in traps.

use glam::Vec2;
use rayon::prelude::*;

/// Gravity constant (pixels/s²)
const GRAVITY: f32 = 980.0;

/// Water density for buoyancy calculations
const WATER_DENSITY: f32 = 1.0;

/// Sediment material types with physical properties
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SedimentType {
    /// Quartz sand - light, easily carried by flow
    QuartzSand,
    /// Magnetite (black sand) - medium density, indicator of gold
    Magnetite,
    /// Gold - very heavy, settles through vortices
    Gold,
}

impl SedimentType {
    /// Specific gravity (density relative to water)
    /// Gold is 19.3x denser than water
    pub fn specific_gravity(&self) -> f32 {
        match self {
            Self::QuartzSand => 2.65,
            Self::Magnetite => 5.2,
            Self::Gold => 19.3,
        }
    }

    /// Shape factor for Rubey equation
    /// Flaky gold settles slower than spherical particles
    pub fn shape_factor(&self) -> f32 {
        match self {
            Self::QuartzSand => 0.9,  // Roughly spherical
            Self::Magnetite => 0.85, // Angular
            Self::Gold => 0.7,       // Flaky/flat
        }
    }

    /// Drag coefficient (higher = follows fluid better)
    /// Light particles have HIGH drag (carried by flow)
    /// Heavy particles have LOW drag (sink through flow)
    pub fn drag_coefficient(&self) -> f32 {
        match self {
            Self::QuartzSand => 8.0,  // HIGH drag - follows water flow
            Self::Magnetite => 4.0,   // Medium drag
            Self::Gold => 1.5,        // LOW drag - sinks through flow
        }
    }

    /// RGBA color for rendering
    pub fn color(&self) -> [u8; 4] {
        match self {
            Self::QuartzSand => [194, 178, 128, 255], // Tan
            Self::Magnetite => [30, 30, 30, 255],     // Black
            Self::Gold => [255, 215, 0, 255],         // Gold
        }
    }
}

/// A sediment particle that follows drift-flux physics
#[derive(Clone, Copy, Debug)]
pub struct SedimentParticle {
    pub position: Vec2,
    pub velocity: Vec2,
    pub material: SedimentType,
    /// Particle diameter in simulation units
    pub diameter: f32,
    /// Current state (flowing, settling, trapped)
    pub state: SedimentState,
}

/// State of a sediment particle
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SedimentState {
    /// Actively being transported by fluid flow
    Flowing,
    /// Settling through slow-moving fluid
    Settling,
    /// Trapped in riffle mat (no longer simulated)
    Trapped,
}

impl SedimentParticle {
    /// Create a new sediment particle
    pub fn new(position: Vec2, velocity: Vec2, material: SedimentType, diameter: f32) -> Self {
        Self {
            position,
            velocity,
            material,
            diameter,
            state: SedimentState::Flowing,
        }
    }

    /// Calculate settling velocity using Rubey equation
    /// w_s = F * sqrt((SG - 1) * g * d)
    ///
    /// This is the terminal velocity a particle reaches when falling through still water.
    pub fn settling_velocity(&self) -> f32 {
        let sg = self.material.specific_gravity();
        let f = self.material.shape_factor();
        let d = self.diameter;

        // Rubey equation for coarse particles in turbulent flow
        f * ((sg - 1.0) * GRAVITY * d).sqrt()
    }

    /// Calculate buoyancy-adjusted gravity
    /// Returns the net downward force per unit mass
    pub fn effective_gravity(&self) -> f32 {
        let sg = self.material.specific_gravity();
        // Buoyancy reduces effective gravity by (1 - 1/SG)
        GRAVITY * (1.0 - WATER_DENSITY / sg)
    }
}

/// Collection of sediment particles with parallel update
pub struct Sediment {
    pub particles: Vec<SedimentParticle>,
}

impl Sediment {
    pub fn new() -> Self {
        Self {
            particles: Vec::new(),
        }
    }

    /// Spawn sediment at a position
    pub fn spawn(&mut self, position: Vec2, velocity: Vec2, material: SedimentType, diameter: f32) {
        self.particles.push(SedimentParticle::new(position, velocity, material, diameter));
    }

    /// Spawn gold particle with default diameter
    pub fn spawn_gold(&mut self, position: Vec2, velocity: Vec2) {
        self.spawn(position, velocity, SedimentType::Gold, 2.0);
    }

    /// Spawn sand particle with default diameter
    pub fn spawn_sand(&mut self, position: Vec2, velocity: Vec2) {
        self.spawn(position, velocity, SedimentType::QuartzSand, 1.5);
    }

    /// Spawn magnetite particle with default diameter
    pub fn spawn_magnetite(&mut self, position: Vec2, velocity: Vec2) {
        self.spawn(position, velocity, SedimentType::Magnetite, 1.8);
    }

    /// Update all sediment particles using drift-flux physics
    ///
    /// The velocity field sampler should return the fluid velocity at a given position.
    /// Sediment particles are advected by the fluid, but also experience:
    /// - Buoyancy-adjusted gravity (heavier particles sink faster)
    /// - Drag (opposes motion relative to fluid)
    /// - Settling velocity (terminal velocity through still water)
    pub fn update<F>(&mut self, dt: f32, sample_fluid_velocity: F)
    where
        F: Fn(Vec2) -> Vec2 + Sync,
    {
        // Use update_with_collision with no-op collision/fluid checkers
        self.update_with_collision(dt, sample_fluid_velocity, |_| false, |_| true);
    }

    /// Update with collision detection against solid terrain
    ///
    /// `is_solid` returns true if the position is inside solid geometry.
    /// `is_fluid` returns true if the position is inside fluid (for drift-flux coupling).
    pub fn update_with_collision<F, S, FL>(
        &mut self,
        dt: f32,
        sample_fluid_velocity: F,
        is_solid: S,
        is_fluid: FL,
    ) where
        F: Fn(Vec2) -> Vec2 + Sync,
        S: Fn(Vec2) -> bool + Sync,
        FL: Fn(Vec2) -> bool + Sync,
    {
        self.particles.par_iter_mut().for_each(|particle| {
            if particle.state == SedimentState::Trapped {
                return; // Trapped particles don't move
            }

            let in_fluid = is_fluid(particle.position);
            let settling = particle.settling_velocity();

            if in_fluid {
                // DRIFT-FLUX: In water, particles are advected by flow + slip
                let u_fluid = sample_fluid_velocity(particle.position);
                let drag_coeff = particle.material.drag_coefficient();

                // Target velocity: fluid velocity + downward settling slip
                let slip_velocity = Vec2::new(0.0, settling);
                let target_vel = u_fluid + slip_velocity;

                // Exponential approach to target (drag coefficient controls rate)
                // High drag = quickly match fluid, Low drag = slowly approach (keeps sinking)
                let blend = (drag_coeff * dt).min(1.0);
                particle.velocity = particle.velocity.lerp(target_vel, blend);
            } else {
                // IN AIR: Just fall with gravity (no fluid coupling)
                let effective_g = particle.effective_gravity();
                particle.velocity.y += effective_g * dt;
                // Add some air drag to prevent crazy speeds
                particle.velocity *= 0.99;
            }

            // Calculate new position
            let new_pos = particle.position + particle.velocity * dt;

            // Collision detection with solid terrain
            if is_solid(new_pos) {
                // Try sliding along surface
                let slide_x = Vec2::new(new_pos.x, particle.position.y);
                let slide_y = Vec2::new(particle.position.x, new_pos.y);

                if !is_solid(slide_x) {
                    // Can slide horizontally
                    particle.position = slide_x;
                    particle.velocity.y = 0.0;
                } else if !is_solid(slide_y) {
                    // Can slide vertically
                    particle.position = slide_y;
                    particle.velocity.x = 0.0;
                } else {
                    // Stuck - don't move, reduce velocity
                    particle.velocity *= 0.1;
                }
            } else {
                // No collision, update position normally
                particle.position = new_pos;
            }

            // Update state based on velocity
            let speed = particle.velocity.length();
            if speed < settling * 0.5 {
                particle.state = SedimentState::Settling;
            } else {
                particle.state = SedimentState::Flowing;
            }
        });
    }

    /// Separate overlapping particles (simple push-apart)
    pub fn separate_particles(&mut self, min_distance: f32) {
        let n = self.particles.len();
        if n < 2 {
            return;
        }

        // Simple O(n²) separation - fine for small particle counts
        // For larger counts, use spatial hashing
        for i in 0..n {
            for j in (i + 1)..n {
                let pi = self.particles[i].position;
                let pj = self.particles[j].position;

                let diff = pi - pj;
                let dist_sq = diff.length_squared();
                let min_dist_sq = min_distance * min_distance;

                if dist_sq < min_dist_sq && dist_sq > 0.001 {
                    let dist = dist_sq.sqrt();
                    let overlap = min_distance - dist;
                    let push = diff.normalize() * (overlap * 0.5);

                    self.particles[i].position += push;
                    self.particles[j].position -= push;
                }
            }
        }
    }

    /// Remove particles outside bounds
    pub fn remove_out_of_bounds(&mut self, width: f32, height: f32) {
        self.particles.retain(|p| {
            p.position.x >= 0.0
                && p.position.x < width
                && p.position.y >= 0.0
                && p.position.y < height
                && p.state != SedimentState::Trapped
        });
    }

    /// Count particles by material type
    pub fn count_by_type(&self) -> (usize, usize, usize) {
        let mut sand = 0;
        let mut magnetite = 0;
        let mut gold = 0;
        for p in &self.particles {
            match p.material {
                SedimentType::QuartzSand => sand += 1,
                SedimentType::Magnetite => magnetite += 1,
                SedimentType::Gold => gold += 1,
            }
        }
        (sand, magnetite, gold)
    }

    pub fn len(&self) -> usize {
        self.particles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.particles.is_empty()
    }
}

impl Default for Sediment {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_settling_velocities() {
        // Gold should settle faster than sand
        let gold = SedimentParticle::new(Vec2::ZERO, Vec2::ZERO, SedimentType::Gold, 2.0);
        let sand = SedimentParticle::new(Vec2::ZERO, Vec2::ZERO, SedimentType::QuartzSand, 2.0);

        assert!(gold.settling_velocity() > sand.settling_velocity());
    }

    #[test]
    fn test_specific_gravity_order() {
        // Gold > Magnetite > Sand
        assert!(SedimentType::Gold.specific_gravity() > SedimentType::Magnetite.specific_gravity());
        assert!(SedimentType::Magnetite.specific_gravity() > SedimentType::QuartzSand.specific_gravity());
    }

    #[test]
    fn test_effective_gravity() {
        let gold = SedimentParticle::new(Vec2::ZERO, Vec2::ZERO, SedimentType::Gold, 2.0);
        let sand = SedimentParticle::new(Vec2::ZERO, Vec2::ZERO, SedimentType::QuartzSand, 2.0);

        // Gold should experience more effective gravity (less buoyancy relative to weight)
        assert!(gold.effective_gravity() > sand.effective_gravity());
    }
}
