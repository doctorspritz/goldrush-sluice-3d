//! Fluid particles for PIC/FLIP simulation
//!
//! Each particle has a continuous position and velocity.
//! Particles are transferred to/from a grid for pressure solving.
//!
//! Supports multiple material types with different densities for
//! natural settling stratification in the sluice.

use glam::Vec2;

/// Material type for particles (affects rendering and settling)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParticleMaterial {
    Water,
    Mud,
    Sand,      // Quartz sand - light sediment
    Magnetite, // Black sand - medium sediment
    Gold,      // Heavy sediment - settles fast
}

impl ParticleMaterial {
    /// Density relative to water (specific gravity)
    pub fn density(&self) -> f32 {
        match self {
            Self::Water => 1.0,
            Self::Mud => 2.0,
            Self::Sand => 2.65,      // Quartz
            Self::Magnetite => 5.2,  // Black sand
            Self::Gold => 19.3,      // Gold!
        }
    }

    /// RGBA color for rendering
    pub fn color(&self) -> [u8; 4] {
        match self {
            Self::Water => [50, 140, 240, 255],
            Self::Mud => [139, 90, 43, 255],
            Self::Sand => [194, 178, 128, 255],    // Tan
            Self::Magnetite => [30, 30, 30, 255],  // Black
            Self::Gold => [255, 215, 0, 255],      // Gold
        }
    }

    /// Near-pressure stiffness multiplier (Clavet et al. 2005)
    /// All fluids need high stiffness for incompressibility.
    /// Sediments are stiffer - they dominate in cross-material interactions.
    pub fn near_pressure_stiffness(&self) -> f32 {
        match self {
            Self::Water => 1.0,      // Incompressible fluid
            Self::Mud => 1.2,        // Slightly stiffer
            Self::Sand => 2.0,       // Granular - harder than water
            Self::Magnetite => 3.0,  // Dense grains
            Self::Gold => 4.0,       // Hardest - dominates water
        }
    }

    /// Is this a sediment type? (anything denser than water)
    /// Only water participates in the FLIP pressure solve.
    /// Sediment is Lagrangian - carried by fluid via drag forces.
    pub fn is_sediment(&self) -> bool {
        matches!(self, Self::Mud | Self::Sand | Self::Magnetite | Self::Gold)
    }
}

/// A fluid particle - supports water, mud, and sediment types
#[derive(Clone, Copy, Debug)]
pub struct Particle {
    /// Continuous position in world coordinates
    pub position: Vec2,
    /// Current velocity
    pub velocity: Vec2,
    /// Velocity sampled from grid before pressure solve (for FLIP)
    pub old_grid_velocity: Vec2,
    /// Material type (determines density and color)
    pub material: ParticleMaterial,
    /// Near-density for Clavet pressure (computed each frame)
    pub near_density: f32,
}

impl Particle {
    /// Create a particle with specified material
    pub fn new(position: Vec2, velocity: Vec2, material: ParticleMaterial) -> Self {
        Self {
            position,
            velocity,
            old_grid_velocity: Vec2::ZERO,
            material,
            near_density: 0.0,
        }
    }

    /// Create a water particle
    pub fn water(position: Vec2, velocity: Vec2) -> Self {
        Self::new(position, velocity, ParticleMaterial::Water)
    }

    /// Create a mud particle (denser, settles faster)
    pub fn mud(position: Vec2, velocity: Vec2) -> Self {
        Self::new(position, velocity, ParticleMaterial::Mud)
    }

    /// Create a sand particle
    pub fn sand(position: Vec2, velocity: Vec2) -> Self {
        Self::new(position, velocity, ParticleMaterial::Sand)
    }

    /// Create a magnetite particle
    pub fn magnetite(position: Vec2, velocity: Vec2) -> Self {
        Self::new(position, velocity, ParticleMaterial::Magnetite)
    }

    /// Create a gold particle
    pub fn gold(position: Vec2, velocity: Vec2) -> Self {
        Self::new(position, velocity, ParticleMaterial::Gold)
    }

    /// Get density (for settling calculations)
    pub fn density(&self) -> f32 {
        self.material.density()
    }

    /// Check if this is a mud particle (legacy compatibility)
    pub fn is_mud(&self) -> bool {
        self.material == ParticleMaterial::Mud
    }

    /// Check if this is a sediment particle
    pub fn is_sediment(&self) -> bool {
        self.material.is_sediment()
    }

    /// Get RGBA color for rendering
    pub fn color(&self) -> [u8; 4] {
        self.material.color()
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

    pub fn spawn_sand(&mut self, x: f32, y: f32, vx: f32, vy: f32) {
        self.list.push(Particle::sand(
            Vec2::new(x, y),
            Vec2::new(vx, vy),
        ));
    }

    pub fn spawn_magnetite(&mut self, x: f32, y: f32, vx: f32, vy: f32) {
        self.list.push(Particle::magnetite(
            Vec2::new(x, y),
            Vec2::new(vx, vy),
        ));
    }

    pub fn spawn_gold(&mut self, x: f32, y: f32, vx: f32, vy: f32) {
        self.list.push(Particle::gold(
            Vec2::new(x, y),
            Vec2::new(vx, vy),
        ));
    }

    /// Count particles by material type
    pub fn count_by_material(&self) -> (usize, usize, usize, usize, usize) {
        let mut water = 0;
        let mut mud = 0;
        let mut sand = 0;
        let mut magnetite = 0;
        let mut gold = 0;
        for p in &self.list {
            match p.material {
                ParticleMaterial::Water => water += 1,
                ParticleMaterial::Mud => mud += 1,
                ParticleMaterial::Sand => sand += 1,
                ParticleMaterial::Magnetite => magnetite += 1,
                ParticleMaterial::Gold => gold += 1,
            }
        }
        (water, mud, sand, magnetite, gold)
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
