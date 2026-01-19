//! Physical constants for the 3D simulation.
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

/// Gravity acceleration (m/s^2) - negative Y direction
pub const GRAVITY: f32 = -9.81;

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
