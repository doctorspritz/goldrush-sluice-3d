//! Physics calculations for sediment transport and fluid mechanics.
//!
//! This module contains pure physics calculations used by the World simulation:
//! - Topographic gradients (bed slope)
//! - Friction-based shear stress
//! - Shields stress for sediment transport
//! - Settling velocities (Stokes and turbulent regimes)
//! - Material property queries

use super::World;

impl World {
    /// Calculate the topographic gradient (bed slope) at a cell.
    /// Uses central differences where possible, forward/backward at boundaries.
    /// Returns the magnitude of the slope gradient (always non-negative).
    pub fn bed_slope(&self, x: usize, z: usize) -> f32 {
        let h_here = self.ground_height(x, z);

        // Calculate slope in X direction (central difference if possible)
        let slope_x = if x > 0 && x < self.width - 1 {
            let h_left = self.ground_height(x - 1, z);
            let h_right = self.ground_height(x + 1, z);
            (h_left - h_right) / (2.0 * self.cell_size)
        } else if x == 0 && self.width > 1 {
            (h_here - self.ground_height(x + 1, z)) / self.cell_size
        } else if x == self.width - 1 && self.width > 1 {
            (self.ground_height(x - 1, z) - h_here) / self.cell_size
        } else {
            0.0
        };

        // Calculate slope in Z direction (central difference if possible)
        let slope_z = if z > 0 && z < self.depth - 1 {
            let h_up = self.ground_height(x, z - 1);
            let h_down = self.ground_height(x, z + 1);
            (h_up - h_down) / (2.0 * self.cell_size)
        } else if z == 0 && self.depth > 1 {
            (h_here - self.ground_height(x, z + 1)) / self.cell_size
        } else if z == self.depth - 1 && self.depth > 1 {
            (self.ground_height(x, z - 1) - h_here) / self.cell_size
        } else {
            0.0
        };

        // Return slope magnitude (always positive)
        // No minimum - returns 0 for truly flat terrain
        (slope_x * slope_x + slope_z * slope_z).sqrt()
    }

    /// Calculate shear velocity at a cell.
    /// Uses velocity-based formula (not combined with gravitational term).
    /// u* = sqrt(Cf × v²)
    /// where Cf ≈ 0.003 is a friction coefficient.
    pub fn shear_velocity(&self, x: usize, z: usize) -> f32 {
        // Velocity-based shear stress formula
        // τ = Cf × ρ × v² where Cf ≈ 0.003 is friction coefficient
        // u*² = τ/ρ = Cf × v²
        let cf = 0.003; // Friction coefficient for turbulent flow
        let flow_x_left = self.water_flow_x[self.flow_x_idx(x, z)];
        let flow_x_right = self.water_flow_x[self.flow_x_idx(x + 1, z)];
        let flow_z_up = self.water_flow_z[self.flow_z_idx(x, z)];
        let flow_z_down = self.water_flow_z[self.flow_z_idx(x, z + 1)];
        let vel_x = (flow_x_left + flow_x_right) * 0.5;
        let vel_z = (flow_z_up + flow_z_down) * 0.5;
        let v_sq = vel_x * vel_x + vel_z * vel_z;

        // Shear velocity
        (cf * v_sq).sqrt()
    }

    /// Calculate bed shear stress at a cell.
    /// τ = ρf × u*²
    pub fn shear_stress(&self, x: usize, z: usize) -> f32 {
        let u_star = self.shear_velocity(x, z);
        self.params.rho_water * u_star * u_star
    }

    /// Calculate Shields stress for a given particle size.
    /// τ* = τ / (g × (ρp - ρf) × d50)
    /// Returns dimensionless Shields parameter.
    pub fn shields_stress(&self, x: usize, z: usize, d50: f32) -> f32 {
        let tau = self.shear_stress(x, z);
        let g = self.params.gravity;
        let rho_diff = self.params.rho_sediment - self.params.rho_water;

        // Avoid division by zero for very fine particles
        let d50_safe = d50.max(1e-6);

        tau / (g * rho_diff * d50_safe)
    }

    /// Calculate settling velocity for a given particle diameter using Stokes law.
    /// vs = (g × (ρp - ρf) × d²) / (18 × μ)
    ///
    /// For larger particles (d > 1mm), uses turbulent settling formula.
    /// Blends between Stokes and turbulent regimes for intermediate sizes.
    pub fn settling_velocity(&self, d50: f32) -> f32 {
        let g = self.params.gravity;
        let rho_p = self.params.rho_sediment;
        let rho_f = self.params.rho_water;
        let mu = self.params.water_viscosity;

        // Stokes settling (valid for small particles, Re < 1)
        let vs_stokes = g * (rho_p - rho_f) * d50 * d50 / (18.0 * mu);

        // Turbulent settling (valid for larger particles, Re > 1000)
        // Using drag coefficient Cd ≈ 0.44 for spheres
        let vs_turbulent = (4.0 * g * d50 * (rho_p - rho_f) / (3.0 * rho_f * 0.44)).sqrt();

        // Blend based on particle size
        // d < 0.1mm: pure Stokes
        // d > 1mm: pure turbulent
        // intermediate: linear blend
        if d50 < 0.0001 {
            vs_stokes
        } else if d50 > 0.001 {
            vs_turbulent
        } else {
            // Linear interpolation
            let t = (d50 - 0.0001) / (0.001 - 0.0001);
            vs_stokes * (1.0 - t) + vs_turbulent * t
        }
    }

    /// Get the particle size (d50) for the top material layer at a cell.
    pub fn top_material_d50(&self, x: usize, z: usize) -> f32 {
        let idx = self.idx(x, z);

        if self.terrain_sediment[idx] > 0.001 {
            self.params.d50_sediment
        } else if self.gravel_thickness[idx] > 0.001 {
            self.params.d50_gravel
        } else if self.overburden_thickness[idx] > 0.001 {
            self.params.d50_overburden
        } else if self.paydirt_thickness[idx] > 0.001 {
            self.params.d50_paydirt
        } else {
            // Bedrock - use large value (won't erode anyway)
            0.1
        }
    }
}
