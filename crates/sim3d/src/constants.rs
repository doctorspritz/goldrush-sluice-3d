//! Physical constants for the 3D simulation.
//!
//! ## Gravity Conventions
//!
//! This module provides gravity in THREE forms for different use cases:
//!
//! 1. **`GRAVITY`** (-9.81) - Signed scalar for Y-axis calculations
//! 2. **`GRAVITY_MAGNITUDE`** (9.81) - Unsigned scalar for physics formulas
//! 3. **`GRAVITY_VEC`** - Vec3(0, -9.81, 0) for direct vector operations
//!
//! ## Density Conventions
//!
//! This module provides densities in TWO forms:
//!
//! 1. **Absolute density (kg/m³)** - For DEM physics (mass, buoyancy, drag)
//!    - `WATER_DENSITY_KGM3`, `GANGUE_DENSITY_KGM3`, `GOLD_DENSITY_KGM3`
//!
//! 2. **Relative density (water=1.0)** - For FLIP particle density field
//!    - `GANGUE_DENSITY`, `GOLD_DENSITY`
//!
//! FLIP's `Particle3D.density` uses relative density (see particle.rs line 22-23).
//! DEM mass calculations need absolute density in kg/m³.

use glam::Vec3;

// =============================================================================
// GRAVITY CONSTANTS
// =============================================================================

/// Gravity magnitude (m/s²) - always positive, use for physics formulas
/// Example: Shields parameter τ* = τ / (ρ_s - ρ_w) * g * d
pub const GRAVITY_MAGNITUDE: f32 = 9.81;

/// Gravity acceleration on Y-axis (m/s²) - negative (downward)
/// Use for velocity updates: v.y += GRAVITY * dt
pub const GRAVITY: f32 = -GRAVITY_MAGNITUDE;

/// Gravity as a 3D vector pointing downward
/// Use for direct vector operations: velocity += GRAVITY_VEC * dt
pub const GRAVITY_VEC: Vec3 = Vec3::new(0.0, GRAVITY, 0.0);

// =============================================================================
// ABSOLUTE DENSITIES (kg/m³) - For DEM physics (mass, buoyancy, drag)
// =============================================================================

/// Density of water (kg/m³)
pub const WATER_DENSITY: f32 = 1000.0;

/// Density of gangue/sand (kg/m³) - For DEM mass calculation
pub const GANGUE_DENSITY_KGM3: f32 = 2650.0;

/// Density of gold (kg/m³) - For DEM mass calculation
pub const GOLD_DENSITY_KGM3: f32 = 19300.0;

// =============================================================================
// RELATIVE DENSITIES (water=1.0) - For FLIP particle density
// =============================================================================

/// Relative density of gangue/sand (water=1.0)
/// Use this for Particle3D.density and spawn_sediment()
pub const GANGUE_DENSITY: f32 = 2.65;

/// Relative density of gold (water=1.0)
/// Use this for Particle3D.density and spawn_sediment()
pub const GOLD_DENSITY: f32 = 19.3;

/// Default color for gangue [R, G, B, A]
pub const GANGUE_COLOR: [f32; 4] = [0.6, 0.4, 0.2, 1.0];

/// Default color for gold [R, G, B, A]
pub const GOLD_COLOR: [f32; 4] = [0.95, 0.85, 0.2, 1.0];
