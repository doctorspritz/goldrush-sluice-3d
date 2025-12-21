//! Unified physics constants for the FLIP/PBF simulation.
//!
//! All simulation modules should use these constants instead of defining their own.
//! This prevents drift between subsystems and makes tuning easier.

/// Simulation gravity in pixels/s².
///
/// Used by:
/// - Grid velocity updates (apply_gravity)
/// - Particle settling (Ferguson-Church)
/// - Sediment forces (buoyancy, friction)
/// - PBF external forces
pub const GRAVITY: f32 = 250.0;

/// Reference water density (dimensionless).
///
/// Used as the baseline for specific gravity calculations.
/// Sediment densities are expressed relative to this (e.g., gold = 19.3).
pub const WATER_DENSITY: f32 = 1.0;

/// Kinematic viscosity for Ferguson-Church settling.
///
/// Tuned together with GRAVITY to produce reasonable settling velocities.
/// Higher values = slower settling (more viscous fluid).
pub const KINEMATIC_VISCOSITY: f32 = 1.3;
