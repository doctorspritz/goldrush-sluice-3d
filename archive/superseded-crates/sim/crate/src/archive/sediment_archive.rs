//! ARCHIVED SEDIMENT SIMULATION CODE
//!
//! This file contains sediment transport physics that was removed to focus on
//! pure water simulation. Preserved for future re-integration.
//!
//! Archived: 2025-12-27
//!
//! Contains:
//! - ParticleMaterial::Mud, Sand, Magnetite, Gold
//! - ParticleState::Bedload
//! - Ferguson-Church settling velocity
//! - Shields criterion for entrainment
//! - Richardson-Zaki hindered settling
//! - Sediment-specific friction coefficients

// ============================================================================
// PARTICLE MATERIAL TYPES (SEDIMENTS)
// ============================================================================

/// Sediment material types with physical properties
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SedimentMaterial {
    Mud,
    Sand,      // Quartz sand - light sediment
    Magnetite, // Black sand - medium sediment
    Gold,      // Heavy sediment - settles fast
}

impl SedimentMaterial {
    /// Density relative to water (specific gravity)
    #[inline]
    pub fn density(&self) -> f32 {
        match self {
            Self::Mud => 2.0,
            Self::Sand => 2.65,      // Quartz
            Self::Magnetite => 5.2,  // Black sand
            Self::Gold => 19.3,      // Gold!
        }
    }

    /// RGBA color for rendering
    pub fn color(&self) -> [u8; 4] {
        match self {
            Self::Mud => [139, 90, 43, 255],
            Self::Sand => [194, 178, 128, 255],    // Tan
            Self::Magnetite => [30, 30, 30, 255],  // Black
            Self::Gold => [255, 215, 0, 255],      // Gold
        }
    }

    /// Typical particle diameter in simulation units (pixels)
    #[inline]
    pub fn typical_diameter(&self) -> f32 {
        match self {
            Self::Mud => 0.5,         // Fine clay/silt particles
            Self::Sand => 2.0,        // Medium sand grains
            Self::Magnetite => 1.5,   // Black sand crystals
            Self::Gold => 0.5,        // Fine gold (high density, small size)
        }
    }

    /// Shape factor C2 for Ferguson-Church settling equation
    /// Higher values = more drag (flakier/less spherical particles)
    pub fn shape_factor(&self) -> f32 {
        match self {
            Self::Mud => 1.2,         // Irregular clay particles
            Self::Sand => 1.0,        // Natural rounded sand
            Self::Magnetite => 1.1,   // Angular crystals
            Self::Gold => 1.8,        // Very flaky (10:1 aspect ratio typical)
        }
    }

    /// Static friction coefficient for bed contact
    pub fn static_friction(&self) -> f32 {
        match self {
            Self::Mud => 0.25,        // Slippery clay
            Self::Sand => 0.5,        // Rough grains
            Self::Magnetite => 0.45,  // Angular but smooth
            Self::Gold => 0.35,       // Smooth metal surface
        }
    }

    /// Dynamic friction coefficient (kinetic friction)
    pub fn dynamic_friction(&self) -> f32 {
        self.static_friction() * 0.8
    }

    /// Critical Shields number for sediment entrainment
    #[inline]
    pub fn shields_critical(&self) -> f32 {
        match self {
            Self::Mud => 0.03,        // Fine particles, easy to suspend
            Self::Sand => 0.045,      // Standard Shields value for sand
            Self::Magnetite => 0.05,  // Slightly harder to move
            Self::Gold => 0.055,      // Heavy particles, harder to entrain
        }
    }

    /// Calculate settling velocity using Ferguson-Church universal equation
    ///
    /// Formula: w = (R * g * D^2) / (C1 * nu + sqrt(0.75 * C2 * R * g * D^3))
    pub fn settling_velocity(&self, diameter: f32, gravity: f32, kinematic_viscosity: f32) -> f32 {
        if diameter <= 0.0 {
            return 0.0;
        }

        const C1: f32 = 18.0; // Stokes constant
        const WATER_DENSITY: f32 = 1.0;

        let density = self.density();
        let c2 = self.shape_factor();

        // Relative submerged density: R = (rho_p - rho_f) / rho_f
        let r = (density - WATER_DENSITY) / WATER_DENSITY;

        // Ferguson-Church equation
        let d = diameter;
        let numerator = r * gravity * d * d;
        let denominator = C1 * kinematic_viscosity
            + (0.75 * c2 * r * gravity * d * d * d).sqrt();

        numerator / denominator
    }

    /// Near-pressure stiffness multiplier (Clavet et al. 2005)
    pub fn near_pressure_stiffness(&self) -> f32 {
        match self {
            Self::Mud => 1.2,
            Self::Sand => 2.0,
            Self::Magnetite => 3.0,
            Self::Gold => 4.0,
        }
    }

    /// Edge sharpness for metaball rendering
    pub fn edge_sharpness(&self) -> f32 {
        match self {
            Self::Mud => 0.05,
            Self::Sand => 0.15,
            Self::Magnetite => 0.20,
            Self::Gold => 0.25,
        }
    }

    /// Density contribution for metaball accumulation
    pub fn density_contribution(&self) -> f32 {
        match self {
            Self::Mud => 0.035,
            Self::Sand => 0.025,
            Self::Magnetite => 0.02,
            Self::Gold => 0.015,
        }
    }

    /// Friction coefficient for sliding on the floor
    pub fn friction_coefficient(&self) -> f32 {
        match self {
            Self::Mud => 0.4,
            Self::Sand => 0.6,
            Self::Magnetite => 0.7,
            Self::Gold => 0.8,
        }
    }
}

// ============================================================================
// PARTICLE STATE (BEDLOAD)
// ============================================================================

/// Particle transport state for friction mechanics
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ParticleState {
    /// Particle is suspended in fluid - follows APIC transfer + settling velocity
    #[default]
    Suspended,
    /// Particle is on the bed - experiences friction, resists flow until Shields exceeded
    Bedload,
}

// ============================================================================
// HINDERED SETTLING (RICHARDSON-ZAKI)
// ============================================================================

/// Richardson-Zaki hindered settling factor
/// Returns multiplier for settling velocity based on concentration
/// concentration: volumetric fraction of solids (0.0 to ~0.6)
pub fn hindered_settling_factor(concentration: f32) -> f32 {
    const N: f32 = 4.0; // Richardson-Zaki exponent for fine particles
    let c = concentration.clamp(0.0, 0.6);
    (1.0 - c).powf(N)
}

/// Convert neighbor count to volumetric concentration estimate
pub fn neighbor_count_to_concentration(neighbor_count: usize, rest_neighbors: f32) -> f32 {
    let ratio = neighbor_count as f32 / rest_neighbors.max(1.0);
    let concentration = 0.5 * (ratio / 2.5).tanh();
    concentration.clamp(0.0, 0.5)
}

// ============================================================================
// SEDIMENT PARTICLE STRUCT
// ============================================================================

use glam::{Mat2, Vec2};

/// A sediment particle with full physical properties
#[derive(Clone, Copy, Debug)]
pub struct SedimentParticle {
    pub position: Vec2,
    pub velocity: Vec2,
    pub affine_velocity: Mat2,
    pub old_grid_velocity: Vec2,
    pub material: SedimentMaterial,
    pub near_density: f32,
    pub diameter: f32,
    pub state: ParticleState,
    pub jam_time: f32,
}

impl SedimentParticle {
    pub fn new(position: Vec2, velocity: Vec2, material: SedimentMaterial) -> Self {
        Self {
            position,
            velocity,
            affine_velocity: Mat2::ZERO,
            old_grid_velocity: Vec2::ZERO,
            material,
            near_density: 0.0,
            diameter: material.typical_diameter(),
            state: ParticleState::Suspended,
            jam_time: 0.0,
        }
    }

    pub fn with_diameter(position: Vec2, velocity: Vec2, material: SedimentMaterial, diameter: f32) -> Self {
        Self {
            position,
            velocity,
            affine_velocity: Mat2::ZERO,
            old_grid_velocity: Vec2::ZERO,
            material,
            near_density: 0.0,
            diameter,
            state: ParticleState::Suspended,
            jam_time: 0.0,
        }
    }

    pub fn effective_diameter(&self) -> f32 {
        if self.diameter > 0.0 {
            self.diameter
        } else {
            self.material.typical_diameter()
        }
    }

    pub fn density(&self) -> f32 {
        self.material.density()
    }

    pub fn color(&self) -> [u8; 4] {
        self.material.color()
    }
}
