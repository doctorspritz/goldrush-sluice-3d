//! Shared erosion computation logic for World and FineRegion.
//!
//! This module consolidates the erosion and sediment transport simulation
//! that was duplicated between `World::update_erosion` and `FineRegion::update_erosion`.

use super::WorldParams;

/// Compute settling velocity for sediment of a given grain size.
///
/// Uses a Stokes-to-turbulent transition model based on grain diameter d50.
pub fn settling_velocity(d50: f32, params: &WorldParams) -> f32 {
    let g = params.gravity;
    let rho_p = params.rho_sediment;
    let rho_f = params.rho_water;
    let mu = params.water_viscosity;

    use crate::constants::D50_TURBULENT_MIN;
    let vs_stokes = g * (rho_p - rho_f) * d50 * d50 / (18.0 * mu);
    let vs_turbulent = (4.0 * g * d50 * (rho_p - rho_f) / (3.0 * rho_f * 0.44)).sqrt();
    let transition = (d50 / D50_TURBULENT_MIN).clamp(0.0, 1.0);
    vs_stokes * (1.0 - transition) + vs_turbulent * transition
}

/// Compute shear stress on the bed using velocity-only formula.
///
/// Assumes friction coefficient cf = 0.003.
pub fn shear_stress_from_velocity(vx: f32, vz: f32, rho_water: f32) -> f32 {
    let cf = 0.003;
    let v_sq = vx * vx + vz * vz;
    rho_water * cf * v_sq
}

/// Helper function to smooth deltas in grid space.
///
/// Applies 3x3 box averaging to reduce artifacts at boundaries.
pub fn smooth_delta(width: usize, depth: usize, input: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; width * depth];
    if width < 3 || depth < 3 {
        output.copy_from_slice(input);
        return output;
    }

    for z in 0..depth {
        for x in 0..width {
            let idx = z * width + x;
            if x == 0 || z == 0 || x + 1 == width || z + 1 == depth {
                output[idx] = input[idx];
                continue;
            }

            let mut sum = 0.0;
            for dz in -1i32..=1 {
                for dx in -1i32..=1 {
                    let nz = (z as i32 + dz) as usize;
                    let nx = (x as i32 + dx) as usize;
                    sum += input[nz * width + nx];
                }
            }
            output[idx] = sum / 9.0;
        }
    }

    output
}

/// Erosion computation parameters shared between grid variants.
#[derive(Clone, Copy, Debug)]
pub struct ErosionParams {
    /// Timestep in seconds.
    pub dt: f32,
    /// Maximum erosion per step (m).
    pub max_erosion_per_step: f32,
    /// Minimum flow speed required for erosion (m/s).
    pub min_erosion_speed: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_settling_velocity_formula() {
        let params = WorldParams::default();
        let d_silt = 0.00005;

        let vs = settling_velocity(d_silt, &params);
        assert!(vs > 0.0);
        assert!(vs < 0.1); // Silt settles slowly
    }

    #[test]
    fn test_shear_stress_from_velocity() {
        use crate::constants::WATER_DENSITY;
        let stress_zero = shear_stress_from_velocity(0.0, 0.0, WATER_DENSITY);
        assert_eq!(stress_zero, 0.0);

        let stress_1ms = shear_stress_from_velocity(1.0, 0.0, WATER_DENSITY);
        assert!(stress_1ms > 0.0);
    }
}
