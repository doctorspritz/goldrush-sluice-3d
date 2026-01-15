//! Physical constants for the 3D simulation.

/// Gravity acceleration (m/s^2) - negative Y direction
pub const GRAVITY: f32 = -9.81;

/// Density of water (kg/m^3)
pub const WATER_DENSITY: f32 = 1000.0;

/// Density of gangue/sand (kg/m^3)
pub const GANGUE_DENSITY: f32 = 2650.0;

/// Density of gold (kg/m^3)
pub const GOLD_DENSITY: f32 = 19300.0;

/// Default color for gangue [R, G, B, A]
pub const GANGUE_COLOR: [f32; 4] = [0.6, 0.4, 0.2, 1.0];

/// Default color for gold [R, G, B, A]
pub const GOLD_COLOR: [f32; 4] = [0.95, 0.85, 0.2, 1.0];
