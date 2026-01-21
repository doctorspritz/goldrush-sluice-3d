//! Simulation constants for the multigrid solver.

/// Cell size for the simulation grid (2.5cm cells).
pub const SIM_CELL_SIZE: f32 = 0.025;

/// Number of pressure solver iterations for density projection convergence.
pub const SIM_PRESSURE_ITERS: usize = 120;

/// Gravitational acceleration (m/s²).
pub const SIM_GRAVITY: f32 = -9.8;

/// DEM clump radius (8mm clumps).
pub const DEM_CLUMP_RADIUS: f32 = 0.008;

/// Gold density in kg/m³.
pub const DEM_GOLD_DENSITY: f32 = 19300.0;

/// Sand/gangue density in kg/m³.
pub const DEM_SAND_DENSITY: f32 = 2650.0;

/// Water density in kg/m³.
pub const DEM_WATER_DENSITY: f32 = 1000.0;

/// Water drag coefficient.
pub const DEM_DRAG_COEFF: f32 = 5.0;

/// Simple pseudo-random number generator.
/// Uses xorshift-style hash for deterministic randomness.
pub fn rand_float() -> f32 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let seed = COUNTER.fetch_add(1, Ordering::Relaxed);
    // Simple xorshift-style hash
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x = x ^ (x >> 31);
    (x as f32) / (u64::MAX as f32)
}
