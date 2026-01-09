#![allow(dead_code)]
//! Profiling and diagnostic methods.
//!
//! Performance measurement and test utilities.
//!
//! NOTE: These methods are currently still defined in mod.rs.
//! This file contains copies ready for activation when mod.rs methods are removed.

// Imports needed when these methods are activated
#[allow(unused_imports)]
use super::FlipSimulation;
#[allow(unused_imports)]
use glam::Vec2;

// =============================================================================
// STAGED IMPLEMENTATIONS - Activate by uncommenting and removing from mod.rs
// =============================================================================

/*
impl FlipSimulation {
    /// Compute neighbor counts for water particles only (legacy no-op)
    /// Now merged into compute_neighbor_counts
    pub fn compute_water_neighbor_counts(&mut self) {
        // No-op, handled by compute_neighbor_counts
    }

    // ========================================================================
    // VORTEX METRICS - For testing and diagnostics
    // ========================================================================

    /// Compute total kinetic energy of particles: KE = 1/2 * sum(|v|^2)
    /// Each particle is assumed to have unit mass for simplicity.
    /// For proper physics, multiply by particle mass (not tracked currently).
    pub fn compute_kinetic_energy(&self) -> f32 {
        self.particles.iter()
            .map(|p| 0.5 * p.velocity.length_squared())
            .sum()
    }

    /// Compute kinetic energy of water particles only
    pub fn compute_water_kinetic_energy(&self) -> f32 {
        self.particles.iter()
            .filter(|p| !p.is_sediment())
            .map(|p| 0.5 * p.velocity.length_squared())
            .sum()
    }

    /// Compute enstrophy from the grid vorticity field
    /// Must call grid.compute_vorticity() first
    pub fn compute_enstrophy(&self) -> f32 {
        self.grid.compute_enstrophy()
    }

    /// Compute and store vorticity, then return enstrophy
    /// Convenience method for tests
    pub fn update_and_compute_enstrophy(&mut self) -> f32 {
        // Ensure cell types are current
        self.classify_cells();
        // Compute vorticity from current grid state
        self.grid.compute_vorticity();
        self.grid.compute_enstrophy()
    }

    /// Get maximum particle velocity (for CFL checking)
    pub fn max_velocity(&self) -> f32 {
        self.particles.iter()
            .map(|p| p.velocity.length())
            .fold(0.0f32, f32::max)
    }

    /// Compute CFL number: CFL = v_max * dt / dx
    /// Should be < 1 for stability, < 0.5 for high-fidelity vortices
    pub fn compute_cfl(&self, dt: f32) -> f32 {
        self.max_velocity() * dt / self.grid.cell_size
    }

    /// Initialize velocity field for Taylor-Green vortex test
    /// u = -cos(pi*x)sin(pi*y), v = sin(pi*x)cos(pi*y)
    /// Domain is assumed to be [0, L] x [0, L] where L = width * cell_size
    pub fn initialize_taylor_green(&mut self) {
        use std::f32::consts::PI;

        let l = self.grid.width as f32 * self.grid.cell_size;
        let cell_size = self.grid.cell_size;
        let width = self.grid.width;
        let height = self.grid.height;

        // Set U velocities
        for j in 0..height {
            for i in 0..=width {
                let x = i as f32 * cell_size;
                let y = (j as f32 + 0.5) * cell_size;

                let u = -f32::cos(PI * x / l) * f32::sin(PI * y / l);
                let idx = self.grid.u_index(i, j);
                self.grid.u[idx] = u;
            }
        }

        // Set V velocities
        for j in 0..=height {
            for i in 0..width {
                let x = (i as f32 + 0.5) * cell_size;
                let y = j as f32 * cell_size;

                let v = f32::sin(PI * x / l) * f32::cos(PI * y / l);
                let idx = self.grid.v_index(i, j);
                self.grid.v[idx] = v;
            }
        }

        // Mark all cells as fluid for the test
        for j in 0..height {
            for i in 0..width {
                let idx = self.grid.cell_index(i, j);
                self.grid.cell_type[idx] = crate::grid::CellType::Fluid;
            }
        }
    }

    /// Initialize solid body rotation: v = omega x r
    /// Creates a rotating disk of fluid
    pub fn initialize_solid_rotation(&mut self, angular_velocity: f32) {
        let cell_size = self.grid.cell_size;
        let width = self.grid.width;
        let height = self.grid.height;
        let cx = width as f32 * cell_size / 2.0;
        let cy = height as f32 * cell_size / 2.0;
        let radius = cx.min(cy) * 0.8; // 80% of domain

        // Set U velocities
        for j in 0..height {
            for i in 0..=width {
                let x = i as f32 * cell_size;
                let y = (j as f32 + 0.5) * cell_size;

                let dx = x - cx;
                let dy = y - cy;
                let r = (dx * dx + dy * dy).sqrt();

                let idx = self.grid.u_index(i, j);
                if r < radius {
                    // u = -omega * (y - cy)
                    self.grid.u[idx] = -angular_velocity * dy;
                } else {
                    self.grid.u[idx] = 0.0;
                }
            }
        }

        // Set V velocities
        for j in 0..=height {
            for i in 0..width {
                let x = (i as f32 + 0.5) * cell_size;
                let y = j as f32 * cell_size;

                let dx = x - cx;
                let dy = y - cy;
                let r = (dx * dx + dy * dy).sqrt();

                let idx = self.grid.v_index(i, j);
                if r < radius {
                    // v = omega * (x - cx)
                    self.grid.v[idx] = angular_velocity * dx;
                } else {
                    self.grid.v[idx] = 0.0;
                }
            }
        }

        // Mark interior cells as fluid
        for j in 0..height {
            for i in 0..width {
                let x = (i as f32 + 0.5) * cell_size;
                let y = (j as f32 + 0.5) * cell_size;
                let dx = x - cx;
                let dy = y - cy;
                let r = (dx * dx + dy * dy).sqrt();

                let idx = self.grid.cell_index(i, j);
                if r < radius {
                    self.grid.cell_type[idx] = crate::grid::CellType::Fluid;
                } else {
                    self.grid.cell_type[idx] = crate::grid::CellType::Air;
                }
            }
        }
    }

    // ========================================================================
    // DIAGNOSTIC METHODS (for testing/debugging)
    // ========================================================================

    /// Run a single update step with per-phase diagnostics
    /// Returns tuple of (phase_name, momentum_after) for each phase
    pub fn update_with_diagnostics(&mut self, dt: f32) -> Vec<(&'static str, f32)> {
        let mut diagnostics = Vec::new();

        let measure = |sim: &Self, _name: &'static str| -> f32 {
            sim.particles.iter()
                .filter(|p| !p.is_sediment())
                .map(|p| p.velocity.length())
                .sum()
        };

        diagnostics.push(("initial", measure(self, "initial")));

        // 1. Classify cells
        self.classify_cells();
        self.grid.compute_sdf();
        diagnostics.push(("after_classify", measure(self, "after_classify")));

        // 2. P2G
        self.particles_to_grid();
        // Measure grid momentum instead of particles (particles unchanged)
        let grid_u_sum: f32 = self.grid.u.iter().sum();
        let grid_v_sum: f32 = self.grid.v.iter().sum();
        diagnostics.push(("grid_after_p2g", (grid_u_sum.powi(2) + grid_v_sum.powi(2)).sqrt()));

        // 3. Store old velocities
        self.store_old_velocities();

        // 4. Apply gravity
        self.grid.apply_gravity(dt);
        let grid_u_sum: f32 = self.grid.u.iter().sum();
        let grid_v_sum: f32 = self.grid.v.iter().sum();
        diagnostics.push(("grid_after_gravity", (grid_u_sum.powi(2) + grid_v_sum.powi(2)).sqrt()));

        // 4b. Vorticity confinement
        {
            let grid = &mut self.grid;
            let pile_height = &self.pile_height;
            grid.apply_vorticity_confinement_with_piles(dt, 0.05, pile_height);
        }
        let grid_u_sum: f32 = self.grid.u.iter().sum();
        let grid_v_sum: f32 = self.grid.v.iter().sum();
        diagnostics.push(("grid_after_vorticity", (grid_u_sum.powi(2) + grid_v_sum.powi(2)).sqrt()));

        // 5. Boundary conditions
        self.grid.enforce_boundary_conditions();
        let grid_u_sum: f32 = self.grid.u.iter().sum();
        let grid_v_sum: f32 = self.grid.v.iter().sum();
        diagnostics.push(("grid_after_boundary", (grid_u_sum.powi(2) + grid_v_sum.powi(2)).sqrt()));

        // 5b. Pressure projection
        self.grid.compute_divergence();
        let div = self.grid.total_divergence();
        diagnostics.push(("divergence", div));

        self.grid.solve_pressure_multigrid(2);
        // Two-way coupling: use mixture density for pressure gradient
        self.apply_pressure_gradient_two_way(dt);
        let grid_u_sum: f32 = self.grid.u.iter().sum();
        let grid_v_sum: f32 = self.grid.v.iter().sum();
        diagnostics.push(("grid_after_pressure", (grid_u_sum.powi(2) + grid_v_sum.powi(2)).sqrt()));

        // 6. G2P
        self.grid_to_particles(dt);
        diagnostics.push(("after_g2p", measure(self, "after_g2p")));

        // 7. Spatial hash & neighbor counts
        self.build_spatial_hash();
        self.compute_neighbor_counts();

        // 8. Legacy sediment forces DISABLED for Phase 2
        // self.apply_sediment_forces(dt);
        diagnostics.push(("after_sediment", measure(self, "after_sediment")));

        // 9. Advection
        self.advect_particles(dt);
        diagnostics.push(("after_advect", measure(self, "after_advect")));

        // 10. Legacy state/pile DISABLED for Phase 2
        // self.update_particle_states(dt);
        // self.compute_pile_heightfield();
        // self.enforce_pile_constraints();
        diagnostics.push(("final", measure(self, "final")));

        // Cleanup
        self.particles.remove_out_of_bounds(
            self.grid.width as f32 * self.grid.cell_size,
            self.grid.height as f32 * self.grid.cell_size,
        );

        self.frame = self.frame.wrapping_add(1);
        diagnostics
    }

    /// Compute total U-weight from P2G transfer buffers
    /// Used to verify partition of unity for B-spline weights
    pub fn get_u_weight_sum(&self) -> f32 {
        self.u_weight.iter().sum()
    }

    /// Compute total V-weight from P2G transfer buffers
    pub fn get_v_weight_sum(&self) -> f32 {
        self.v_weight.iter().sum()
    }

    /// DIAGNOSTIC: Run isolated FLIP cycle WITHOUT any forces
    /// This tests if the P2G -> store_old -> G2P cycle itself causes momentum loss.
    /// If momentum is lost with NO grid modifications, the kernel mismatch is confirmed.
    pub fn run_isolated_flip_cycle(&mut self, dt: f32) -> (f32, f32) {
        // Measure momentum before
        let momentum_before: f32 = self.particles.iter()
            .filter(|p| !p.is_sediment())
            .map(|p| p.velocity.length())
            .sum();

        // 1. Classify cells
        self.classify_cells();

        // 2. P2G transfer
        self.particles_to_grid();

        // 3. Store old velocities (uses bilinear sampling)
        self.store_old_velocities();

        // NO FORCES: Skip gravity, vorticity, boundary conditions, pressure
        // The grid velocity should be UNCHANGED from P2G

        // 4. G2P transfer (uses quadratic B-spline sampling)
        self.grid_to_particles(dt);

        // Measure momentum after
        let momentum_after: f32 = self.particles.iter()
            .filter(|p| !p.is_sediment())
            .map(|p| p.velocity.length())
            .sum();

        (momentum_before, momentum_after)
    }

    /// Compute grid kinetic energy: KE = 1/2 * integral(|v|^2 dV)
    /// This is more accurate than particle KE for grid-based tests
    pub fn compute_grid_kinetic_energy(&self) -> f32 {
        let cell_area = self.grid.cell_size * self.grid.cell_size;
        let mut ke = 0.0f32;

        for j in 0..self.grid.height {
            for i in 0..self.grid.width {
                let idx = self.grid.cell_index(i, j);
                if self.grid.cell_type[idx] != crate::grid::CellType::Fluid {
                    continue;
                }

                // Sample velocity at cell center
                let x = (i as f32 + 0.5) * self.grid.cell_size;
                let y = (j as f32 + 0.5) * self.grid.cell_size;
                let vel = self.grid.sample_velocity(Vec2::new(x, y));

                ke += 0.5 * vel.length_squared() * cell_area;
            }
        }

        ke
    }
}
*/
