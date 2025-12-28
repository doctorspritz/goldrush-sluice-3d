//! Fluid particles for PIC/FLIP simulation
//!
//! Each particle has a continuous position and velocity.
//! Particles are transferred to/from a grid for pressure solving.
//!
//! Supports multiple material types with different densities for
//! natural settling stratification in the sluice.

use glam::{Mat2, Vec2};

/// Particle transport state for friction mechanics
/// Determines whether particle follows fluid (suspended) or experiences bed friction (bedload)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ParticleState {
    /// Particle is suspended in fluid - follows APIC transfer + settling velocity
    #[default]
    Suspended,
    /// Particle is on the bed - experiences friction, resists flow until Shields exceeded
    Bedload,
}

/// Material type for particles (affects rendering and settling)
/// Simplified to Water + Sand for two-way coupling development
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParticleMaterial {
    Water,
    Sand, // Quartz sand - sediment for two-way coupling
}

impl ParticleMaterial {
    /// Density relative to water (specific gravity)
    #[inline]
    pub fn density(&self) -> f32 {
        match self {
            Self::Water => 1.0,
            Self::Sand => 2.65, // Quartz
        }
    }

    /// RGBA color for rendering
    pub fn color(&self) -> [u8; 4] {
        match self {
            Self::Water => [50, 140, 240, 100], // Semi-transparent
            Self::Sand => [194, 178, 128, 255], // Tan
        }
    }

    /// Is this a sediment type? (anything denser than water)
    /// Only water participates in the FLIP pressure solve.
    /// Sediment is Lagrangian - carried by fluid via drag forces.
    pub fn is_sediment(&self) -> bool {
        matches!(self, Self::Sand)
    }

    /// Render scale multiplier (1.0 = base size)
    /// Normalized for development - all particles same size for ratio visibility
    pub fn render_scale(&self) -> f32 {
        1.0 // Normalized: all materials same size during development
        // Original values for production:
        // Water => 1.0, Mud => 0.85, Sand => 0.5, Magnetite => 0.45, Gold => 0.4
    }

    /// Edge sharpness for metaball rendering (higher = harder borders)
    /// Solids have crisp edges, water is blobby
    pub fn edge_sharpness(&self) -> f32 {
        match self {
            Self::Water => 0.03, // Soft, blobby
            Self::Sand => 0.15,  // Distinct grains
        }
    }

    /// Density contribution for metaball accumulation
    /// Lower values = less blending between particles
    pub fn density_contribution(&self) -> f32 {
        match self {
            Self::Water => 0.04,  // Blends together
            Self::Sand => 0.025, // More separate
        }
    }

    /// Typical particle diameter in simulation units (pixels)
    /// Based on realistic size ranges for each material type
    /// Smaller diameter = slower settling = stays suspended longer in flow
    #[inline]
    pub fn typical_diameter(&self) -> f32 {
        match self {
            Self::Water => 0.0, // N/A - water is the fluid
            Self::Sand => 0.3,  // Very fine sand (silt) - stays suspended in flow
        }
    }

    /// Shape factor C₂ for Ferguson-Church settling equation
    /// Higher values = more drag (flakier/less spherical particles)
    /// Reference: Ferguson & Church 2004
    pub fn shape_factor(&self) -> f32 {
        match self {
            Self::Water => 1.0, // N/A
            Self::Sand => 1.0,  // Natural rounded sand
        }
    }

    /// Static friction coefficient for bed contact
    /// Controls how easily particles slide along the sluice floor
    /// Higher values = more resistance to sliding
    pub fn static_friction(&self) -> f32 {
        match self {
            Self::Water => 0.0, // Water has no friction
            Self::Sand => 0.5,  // Rough grains
        }
    }

    /// Dynamic friction coefficient (kinetic friction)
    /// Applied when particle is sliding along the bed
    /// Typically 70-90% of static friction
    pub fn dynamic_friction(&self) -> f32 {
        self.static_friction() * 0.8
    }

    /// Critical Shields number (τ*_c) for sediment entrainment
    /// Determines threshold shear stress to mobilize bedload particles
    /// Lower values = easier to move
    /// Reference: Shields (1936)
    #[inline]
    pub fn shields_critical(&self) -> f32 {
        match self {
            Self::Water => 0.0,   // N/A - water is the fluid
            Self::Sand => 0.045, // Standard Shields value for sand
        }
    }

    /// Calculate settling velocity using Ferguson-Church universal equation
    ///
    /// Formula: w = (R * g * D²) / (C₁ * ν + √(0.75 * C₂ * R * g * D³))
    ///
    /// This naturally transitions from Stokes (viscous) to Newton (turbulent)
    /// regime based on particle size, without requiring iteration.
    ///
    /// Arguments:
    /// - diameter: particle diameter in simulation units (pixels)
    ///
    /// Returns: terminal settling velocity in simulation units (pixels/s)
    pub fn settling_velocity(&self, diameter: f32) -> f32 {
        // Water doesn't settle through water
        if *self == Self::Water || diameter <= 0.0 {
            return 0.0;
        }

        use crate::physics::{GRAVITY, KINEMATIC_VISCOSITY, WATER_DENSITY};
        const C1: f32 = 18.0; // Stokes constant

        let density = self.density();
        let c2 = self.shape_factor();

        // Relative submerged density: R = (ρ_p - ρ_f) / ρ_f
        let r = (density - WATER_DENSITY) / WATER_DENSITY;

        // Ferguson-Church equation
        let d = diameter;
        let numerator = r * GRAVITY * d * d;
        let denominator = C1 * KINEMATIC_VISCOSITY
            + (0.75 * c2 * r * GRAVITY * d * d * d).sqrt();

        numerator / denominator
    }

    /// Friction coefficient for sliding on the floor
    /// Higher values = stops faster
    pub fn friction_coefficient(&self) -> f32 {
        match self {
            Self::Water => 0.0,
            Self::Sand => 0.6,
        }
    }
}

/// A fluid particle - supports water and sand
#[derive(Clone, Copy, Debug)]
pub struct Particle {
    /// Continuous position in world coordinates
    pub position: Vec2,
    /// Current velocity
    pub velocity: Vec2,
    /// Affine velocity matrix for APIC transfer (captures local velocity gradients)
    /// Stores rotation and deformation for angular momentum preservation
    pub affine_velocity: Mat2,
    /// Velocity sampled from grid (used for sediment drag calculation)
    pub old_grid_velocity: Vec2,
    /// Material type (determines density and color)
    pub material: ParticleMaterial,
    /// Particle diameter in simulation units (for Ferguson-Church settling)
    /// If 0.0, uses material.typical_diameter() as fallback
    pub diameter: f32,
    /// Transport state (suspended vs bedload) for friction mechanics
    pub state: ParticleState,
    /// Time spent in bedload state (seconds) - used for hysteresis
    /// Particles must be bedload for MIN_JAM_TIME before unjamming
    pub jam_time: f32,
}

impl Particle {
    /// Create a particle with specified material (uses typical diameter)
    pub fn new(position: Vec2, velocity: Vec2, material: ParticleMaterial) -> Self {
        Self {
            position,
            velocity,
            affine_velocity: Mat2::ZERO,
            old_grid_velocity: Vec2::ZERO,
            material,
            diameter: material.typical_diameter(),
            state: ParticleState::Suspended,
            jam_time: 0.0,
        }
    }

    /// Create a particle with specified material and diameter
    pub fn with_diameter(position: Vec2, velocity: Vec2, material: ParticleMaterial, diameter: f32) -> Self {
        Self {
            position,
            velocity,
            affine_velocity: Mat2::ZERO,
            old_grid_velocity: Vec2::ZERO,
            material,
            diameter,
            state: ParticleState::Suspended,
            jam_time: 0.0,
        }
    }

    /// Get effective diameter (falls back to typical if not set)
    #[inline]
    pub fn effective_diameter(&self) -> f32 {
        if self.diameter > 0.0 {
            self.diameter
        } else {
            self.material.typical_diameter()
        }
    }

    /// Create a water particle
    pub fn water(position: Vec2, velocity: Vec2) -> Self {
        Self::new(position, velocity, ParticleMaterial::Water)
    }

    /// Create a sand particle
    pub fn sand(position: Vec2, velocity: Vec2) -> Self {
        Self::new(position, velocity, ParticleMaterial::Sand)
    }

    /// Get density (for settling calculations)
    pub fn density(&self) -> f32 {
        self.material.density()
    }

    /// Check if this is a sediment particle
    pub fn is_sediment(&self) -> bool {
        self.material.is_sediment()
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
        // Ensure at least 1 step, and steps is calculated based on displacement vs max_disp
        let steps = (disp.length() / max_disp).ceil().max(1.0) as usize;
        let sub_dt = dt / steps as f32;

        for _ in 0..steps {
            self.position += self.velocity * sub_dt;
            collision_callback(self);
        }
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

    pub fn spawn_sand(&mut self, x: f32, y: f32, vx: f32, vy: f32) {
        self.list.push(Particle::sand(
            Vec2::new(x, y),
            Vec2::new(vx, vy),
        ));
    }

    /// Spawn a sediment particle with diameter variation
    /// `variation` is the fractional variation (e.g., 0.3 = ±30%)
    /// `random_value` should be in [0, 1] for reproducibility
    pub fn spawn_with_variation(
        &mut self,
        x: f32,
        y: f32,
        vx: f32,
        vy: f32,
        material: ParticleMaterial,
        variation: f32,
        random_value: f32,
    ) {
        let base_diameter = material.typical_diameter();
        // random_value in [0,1] -> variation_factor in [1-variation, 1+variation]
        let variation_factor = 1.0 + variation * (2.0 * random_value - 1.0);
        let diameter = base_diameter * variation_factor;
        self.list.push(Particle::with_diameter(
            Vec2::new(x, y),
            Vec2::new(vx, vy),
            material,
            diameter,
        ));
    }

    /// Count particles by material type (water, sand)
    pub fn count_by_material(&self) -> (usize, usize) {
        let mut water = 0;
        let mut sand = 0;
        for p in &self.list {
            match p.material {
                ParticleMaterial::Water => water += 1,
                ParticleMaterial::Sand => sand += 1,
            }
        }
        (water, sand)
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

    // ==========================================================================
    // Phase 1 Tests: Particle Size and Shape Properties
    // ==========================================================================

    #[test]
    fn test_density_ordering() {
        // Sand > Water
        assert!(ParticleMaterial::Sand.density() > ParticleMaterial::Water.density());
    }

    #[test]
    fn test_typical_diameter_exists() {
        // Sand should have positive diameter
        assert!(ParticleMaterial::Sand.typical_diameter() > 0.0);
        // Water doesn't have a diameter (it's the fluid)
        assert_eq!(ParticleMaterial::Water.typical_diameter(), 0.0);
    }

    #[test]
    fn test_shape_factor_exists() {
        // Sand should have shape factor >= 1.0
        assert!(ParticleMaterial::Sand.shape_factor() >= 1.0);
    }

    // ==========================================================================
    // Phase 2 Tests: Ferguson-Church Settling Velocity
    // ==========================================================================

    #[test]
    fn test_settling_velocity_increases_with_size() {
        // Larger particles settle faster (for same material)
        let small = ParticleMaterial::Sand.settling_velocity(0.5);
        let medium = ParticleMaterial::Sand.settling_velocity(1.0);
        let large = ParticleMaterial::Sand.settling_velocity(2.0);

        assert!(large > medium, "Large ({}) > Medium ({})", large, medium);
        assert!(medium > small, "Medium ({}) > Small ({})", medium, small);
    }

    #[test]
    fn test_settling_velocity_is_positive() {
        // Sand settling velocity should be positive (downward)
        let v = ParticleMaterial::Sand.settling_velocity(1.0);
        assert!(v > 0.0, "Sand settling velocity should be positive");
    }

    #[test]
    fn test_water_has_no_settling() {
        // Water doesn't settle through water
        let v = ParticleMaterial::Water.settling_velocity(1.0);
        assert_eq!(v, 0.0, "Water should not settle");
    }

    #[test]
    fn test_fine_sand_settles_slower_than_coarse() {
        // Fine sand should settle slower than coarse
        let fine = ParticleMaterial::Sand.settling_velocity(0.1);
        let coarse = ParticleMaterial::Sand.settling_velocity(2.0);

        assert!(
            coarse > fine * 3.0,
            "Coarse sand ({}) should settle >3x faster than fine sand ({})",
            coarse,
            fine
        );
    }

    // ==========================================================================
    // Phase 3 Tests: Hindered Settling (Richardson-Zaki)
    // ==========================================================================

    #[test]
    fn test_hindered_settling_reduces_velocity() {
        // At high concentration, settling is slower
        let clear = hindered_settling_factor(0.0);
        let dilute = hindered_settling_factor(0.1);
        let concentrated = hindered_settling_factor(0.3);

        assert_eq!(clear, 1.0, "Clear water should have factor 1.0");
        assert!(dilute < 1.0, "10% concentration should reduce settling");
        assert!(concentrated < dilute, "30% should be slower than 10%");
    }

    #[test]
    fn test_hindered_settling_magnitude() {
        // Richardson-Zaki with n=4: (1-C)^4
        // At 10%: (0.9)^4 ≈ 0.656
        // At 30%: (0.7)^4 ≈ 0.240
        let at_10_pct = hindered_settling_factor(0.1);
        let at_30_pct = hindered_settling_factor(0.3);

        assert!(
            (at_10_pct - 0.656).abs() < 0.1,
            "10% hindered factor ({}) should be ~0.656",
            at_10_pct
        );
        assert!(
            (at_30_pct - 0.240).abs() < 0.1,
            "30% hindered factor ({}) should be ~0.240",
            at_30_pct
        );
    }

    // ==========================================================================
    // Phase 3 Tests: Hindered Settling Integration
    // ==========================================================================

    #[test]
    fn test_hindered_settling_applied_to_velocity() {
        // Verify that hindered settling reduces effective settling velocity
        let base_settling = ParticleMaterial::Sand.settling_velocity(1.0);

        // At 20% concentration, settling should be reduced by (1-0.2)^4 ≈ 0.41
        let concentration = 0.2;
        let hindered = base_settling * hindered_settling_factor(concentration);

        assert!(
            hindered < base_settling,
            "Hindered settling ({}) should be less than base ({})",
            hindered,
            base_settling
        );

        let expected_ratio = 0.41; // (0.8)^4
        let actual_ratio = hindered / base_settling;
        assert!(
            (actual_ratio - expected_ratio).abs() < 0.05,
            "Ratio ({:.3}) should be ~{:.3}",
            actual_ratio,
            expected_ratio
        );
    }

    #[test]
    fn test_concentration_from_neighbor_count() {
        // Test the helper function that converts neighbor count to concentration
        // Assuming ~8 particles per cell at rest density
        let rest_neighbors = 8.0;

        // Few neighbors = low concentration
        assert!(neighbor_count_to_concentration(2, rest_neighbors) < 0.15);

        // Normal neighbors = moderate concentration
        let moderate = neighbor_count_to_concentration(8, rest_neighbors);
        assert!(moderate > 0.05 && moderate < 0.3, "8 neighbors: {}", moderate);

        // Many neighbors = high concentration
        let high = neighbor_count_to_concentration(20, rest_neighbors);
        assert!(high > 0.2, "20 neighbors should give high concentration: {}", high);
    }

    // ==========================================================================
    // Diagnostic Tests (print values for verification)
    // ==========================================================================

    #[test]
    fn diagnostic_print_settling_velocities() {
        println!("\n=== Settling Velocity Comparison (pixels/s) ===");
        println!("Material       | Diameter | Settling | Density | Shape");
        println!("---------------|----------|----------|---------|------");

        for (name, mat) in [
            ("Water", ParticleMaterial::Water),
            ("Sand", ParticleMaterial::Sand),
        ] {
            let d = mat.typical_diameter();
            let v = mat.settling_velocity(d);
            println!(
                "{:14} | {:8.2} | {:8.2} | {:7.2} | {:5.2}",
                name,
                d,
                v,
                mat.density(),
                mat.shape_factor()
            );
        }

        println!("\n=== Size Effect on Sand ===");
        for size in [0.1, 0.5, 1.0, 2.0, 5.0] {
            let v = ParticleMaterial::Sand.settling_velocity(size);
            println!("Sand d={:.1}: {:.2} px/s", size, v);
        }

        println!("\n=== Hindered Settling Effect ===");
        let sand_base = ParticleMaterial::Sand.settling_velocity(1.0);
        for neighbors in [4, 8, 12, 20, 30, 50] {
            let conc = neighbor_count_to_concentration(neighbors, 8.0);
            let factor = hindered_settling_factor(conc);
            let hindered = sand_base * factor;
            println!(
                "Neighbors={:2}: conc={:.2}, factor={:.2}, sand settling={:.2} px/s",
                neighbors, conc, factor, hindered
            );
        }
    }

    // ==========================================================================
    // Phase 4 Tests: Friction Coefficients
    // ==========================================================================

    #[test]
    fn test_friction_coefficients_exist() {
        // Sand should have positive friction coefficients
        assert!(
            ParticleMaterial::Sand.static_friction() > 0.0,
            "Sand should have positive static friction"
        );
        assert!(
            ParticleMaterial::Sand.dynamic_friction() > 0.0,
            "Sand should have positive dynamic friction"
        );
    }

    #[test]
    fn test_water_has_no_friction() {
        assert_eq!(ParticleMaterial::Water.static_friction(), 0.0);
        assert_eq!(ParticleMaterial::Water.dynamic_friction(), 0.0);
    }

    #[test]
    fn test_dynamic_friction_less_than_static() {
        // Dynamic friction should be less than static for sand
        assert!(
            ParticleMaterial::Sand.dynamic_friction() < ParticleMaterial::Sand.static_friction(),
            "Sand dynamic friction should be less than static"
        );
    }

    #[test]
    fn test_friction_in_reasonable_range() {
        // Friction coefficients should be between 0 and 1
        let mu_s = ParticleMaterial::Sand.static_friction();
        let mu_d = ParticleMaterial::Sand.dynamic_friction();
        assert!(mu_s <= 1.0, "Sand static friction {} too high", mu_s);
        assert!(mu_d <= 1.0, "Sand dynamic friction {} too high", mu_d);
    }

    // ==========================================================================
    // Phase 5 Tests: Shields Critical Threshold
    // ==========================================================================

    #[test]
    fn test_shields_critical_exists() {
        // Sand should have positive Shields threshold
        assert!(
            ParticleMaterial::Sand.shields_critical() > 0.0,
            "Sand should have positive Shields critical"
        );
    }

    #[test]
    fn test_water_has_no_shields() {
        assert_eq!(ParticleMaterial::Water.shields_critical(), 0.0);
    }

    #[test]
    fn test_shields_in_physical_range() {
        // Shields criterion typically 0.03-0.06 for most particles
        let shields = ParticleMaterial::Sand.shields_critical();
        assert!(
            shields >= 0.01 && shields <= 0.1,
            "Sand Shields {} outside physical range [0.01, 0.1]",
            shields
        );
    }

    // ==========================================================================
    // Phase 6 Tests: Particle State
    // ==========================================================================

    #[test]
    fn test_particle_default_state_is_suspended() {
        let p = Particle::sand(Vec2::new(0.0, 0.0), Vec2::ZERO);
        assert_eq!(p.state, ParticleState::Suspended);
    }

    #[test]
    fn test_particle_state_can_change() {
        let mut p = Particle::sand(Vec2::new(0.0, 0.0), Vec2::ZERO);
        assert_eq!(p.state, ParticleState::Suspended);
        p.state = ParticleState::Bedload;
        assert_eq!(p.state, ParticleState::Bedload);
    }

    #[test]
    fn test_particle_with_diameter_has_suspended_state() {
        let p = Particle::with_diameter(
            Vec2::new(0.0, 0.0),
            Vec2::ZERO,
            ParticleMaterial::Sand,
            2.0,
        );
        assert_eq!(p.state, ParticleState::Suspended);
    }
}

/// Richardson-Zaki hindered settling factor
/// Returns multiplier for settling velocity based on concentration
/// concentration: volumetric fraction of solids (0.0 to ~0.6)
pub fn hindered_settling_factor(concentration: f32) -> f32 {
    const N: f32 = 4.0; // Richardson-Zaki exponent for fine particles
    let c = concentration.clamp(0.0, 0.6);
    (1.0 - c).powf(N)
}

/// Convert neighbor count to volumetric concentration estimate
/// Uses a simple model where concentration scales with neighbor density
///
/// Arguments:
/// - neighbor_count: number of particles in local neighborhood
/// - rest_neighbors: expected neighbors at "rest" (dilute) conditions
///
/// Returns: estimated volumetric concentration (0.0 to 0.6)
pub fn neighbor_count_to_concentration(neighbor_count: usize, rest_neighbors: f32) -> f32 {
    // Concentration increases with neighbor count above rest
    // At rest_neighbors, concentration is ~0.1 (dilute slurry)
    // At 2x rest, concentration approaches 0.3 (moderate)
    // At 3x+ rest, concentration approaches maximum (~0.5)
    let ratio = neighbor_count as f32 / rest_neighbors.max(1.0);

    // Smooth mapping: tanh gives nice S-curve from 0 to max
    // Scale so that ratio=1 gives ~0.1, ratio=2 gives ~0.25, ratio=3 gives ~0.4
    let concentration = 0.5 * (ratio / 2.5).tanh();

    concentration.clamp(0.0, 0.5)
}
