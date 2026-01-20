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

// =============================================================================
// NUMERICAL THRESHOLDS
// =============================================================================

/// Minimum thickness threshold for layer existence (meters)
/// Used to determine if a terrain layer is effectively depleted
pub const MIN_LAYER_THICKNESS: f32 = 0.001;

/// Minimum water depth threshold (meters)
/// Used to skip shallow water calculations and prevent numerical issues
pub const MIN_WATER_DEPTH: f32 = 0.001;

/// Small epsilon for floating point comparisons (meters)
/// Used for collision detection slop and numerical stability
pub const EPSILON_METERS: f32 = 0.001;

/// Very small epsilon for floating point comparisons (meters)
/// Used for fine-grained numerical checks
pub const EPSILON_FINE: f32 = 0.0001;

/// Maximum restitution coefficient for collision damping
/// Clamped to prevent numerical instability in DEM damping calculations
pub const MAX_RESTITUTION: f32 = 0.999;

// =============================================================================
// SEDIMENT PHYSICS
// =============================================================================

/// Default median grain diameter for fine sediment (meters)
/// 0.1mm fine silt - used in erosion and settling calculations
pub const D50_FINE_SEDIMENT: f32 = 0.0001;

/// Default median grain diameter for coarse sediment (meters)
/// 1mm coarse sand - used for overburden material
pub const D50_COARSE_SEDIMENT: f32 = 0.001;

/// Transition threshold for Stokes settling regime (meters)
/// Particles smaller than this use pure Stokes law
pub const D50_STOKES_MAX: f32 = 0.0001;

/// Transition threshold for turbulent settling regime (meters)
/// Particles larger than this use pure turbulent drag
pub const D50_TURBULENT_MIN: f32 = 0.001;

/// Water dynamic viscosity at 20°C (Pa·s)
pub const WATER_VISCOSITY: f32 = 0.001;

/// Sediment density (quartz) in kg/m³
pub const SEDIMENT_DENSITY: f32 = 2650.0;

// =============================================================================
// DAMPING COEFFICIENTS
// =============================================================================

/// Velocity damping factor for water flow (per frame)
/// 0.99 = 1% damping per frame to prevent numerical oscillations
pub const WATER_VELOCITY_DAMPING: f32 = 0.99;

/// Velocity damping factor for SPH particles
/// 0.999 = 0.1% damping to prevent oscillations without killing velocity
pub const SPH_VELOCITY_DAMPING: f32 = 0.999;

// =============================================================================
// NUMERICAL SOLVER PARAMETERS
// =============================================================================

/// SOR (Successive Over-Relaxation) omega parameter - lower bound
/// Used for iterative pressure solvers in 3D
pub const SOR_OMEGA_MIN: f32 = 1.5;

/// SOR (Successive Over-Relaxation) omega parameter - upper bound
/// Used for iterative pressure solvers in 3D
pub const SOR_OMEGA_MAX: f32 = 1.9;

// =============================================================================
// KERNEL SUPPORT RANGES
// =============================================================================

/// Quadratic B-spline kernel support radius
/// Support range is [-1.5, 1.5] covering 3 grid nodes
pub const BSPLINE_SUPPORT_RADIUS: f32 = 1.5;
