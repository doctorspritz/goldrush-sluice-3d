//! APIC (Affine Particle-In-Cell) simulation
//!
//! APIC stores an affine velocity field per particle via the C matrix.
//! This preserves angular momentum and reduces noise compared to FLIP.
//!
//! Algorithm:
//! 1. Classify cells (solid/fluid/air)
//! 2. Transfer particle velocities + affine momentum to grid (P2G)
//! 3. Apply forces (gravity)
//! 4. Pressure projection (creates vortices!)
//! 5. Transfer grid velocities back to particles + reconstruct C matrix (G2P)
//! 6. Advect particles
//!
//! Reference: Jiang et al. 2015 "The Affine Particle-In-Cell Method"

use crate::grid::{apic_d_inverse, quadratic_bspline, quadratic_bspline_1d, CellType, Grid};
use crate::particle::{ParticleMaterial, Particles, ParticleState};
use glam::{Mat2, Vec2};
use rand::Rng;
use rayon::prelude::*;

/// FLIP simulation state
pub struct FlipSimulation {
    pub grid: Grid,
    pub particles: Particles,
    // Pre-allocated P2G transfer buffers (eliminates ~320KB allocation per frame)
    u_sum: Vec<f32>,
    u_weight: Vec<f32>,
    v_sum: Vec<f32>,
    v_weight: Vec<f32>,
    // Volume fraction tracking for two-way coupling
    // Sand volume contribution at each grid face
    sand_volume_u: Vec<f32>,
    water_volume_u: Vec<f32>,
    sand_volume_v: Vec<f32>,
    water_volume_v: Vec<f32>,
    // Pre-allocated spatial hash for particle separation (zero allocation per frame)
    cell_head: Vec<i32>,      // Index of first particle in each cell (-1 = empty)
    particle_next: Vec<i32>,  // Index of next particle in same cell (-1 = end)
    // Pre-allocated impulse buffer for soft separation
    impulse_buffer: Vec<Vec2>,
    // Pre-allocated buffer for near-pressure forces (Clavet stabilization)
    near_force_buffer: Vec<Vec2>,
    // Frame counter for skipping expensive operations
    frame: u32,

    // Pre-allocated buffer for neighbor counts (hindered settling)
    neighbor_counts: Vec<u16>,
    // Pre-allocated buffer for water neighbor counts (in-fluid detection)
    water_neighbor_counts: Vec<u16>,

    // === Pile Heightfield for Stacking ===
    // Bedload particles act as a collision floor for suspended particles
    // pile_height[x] = top of bedload pile at column x (in world coordinates)
    pub pile_height: Vec<f32>,

    // === Sediment Physics Feature Flags ===
    // Toggle these to compare old vs new physics behavior
    /// Use Ferguson-Church settling velocity (vs simple density-based)
    pub use_ferguson_church: bool,
    /// Use Richardson-Zaki hindered settling in concentrated regions
    pub use_hindered_settling: bool,
    /// Use per-particle diameter (vs material typical diameter)
    pub use_variable_diameter: bool,
    /// Size variation factor: diameter = typical * (1 ± variation)
    pub diameter_variation: f32,

    // === Viscosity for Vortex Shedding ===
    // Viscosity creates boundary layer separation needed for vortex formation
    /// Enable viscosity diffusion (creates shear layers at walls)
    pub use_viscosity: bool,
    /// Kinematic viscosity coefficient (higher = more diffusion, slower flow near walls)
    /// Typical values: 0.5-3.0 for visible vortex shedding
    /// Too high: slows entire flow; Too low: no boundary layer separation
    pub viscosity: f32,

    // === Sand Velocity Update ===
    /// PIC/FLIP ratio for sand particles (0.0 = pure FLIP, 1.0 = pure PIC)
    /// Higher PIC = sand follows water velocity directly
    /// Higher FLIP = sand responds to velocity changes, can overshoot
    /// Recommended: 0.5-0.8 for stable sand transport
    pub sand_pic_ratio: f32,
}

impl FlipSimulation {
    pub fn new(width: usize, height: usize, cell_size: f32) -> Self {
        let cell_count = width * height;
        let grid = Grid::new(width, height, cell_size);
        // Pre-allocate P2G buffers based on grid size
        let u_len = grid.u.len();
        let v_len = grid.v.len();
        Self {
            grid,
            particles: Particles::new(),
            u_sum: vec![0.0; u_len],
            u_weight: vec![0.0; u_len],
            v_sum: vec![0.0; v_len],
            v_weight: vec![0.0; v_len],
            // Volume fraction buffers for two-way coupling
            sand_volume_u: vec![0.0; u_len],
            water_volume_u: vec![0.0; u_len],
            sand_volume_v: vec![0.0; v_len],
            water_volume_v: vec![0.0; v_len],
            cell_head: vec![-1; cell_count],
            particle_next: Vec::new(),
            impulse_buffer: Vec::new(),
            near_force_buffer: Vec::new(),
            frame: 0,
            // Neighbor counts for hindered settling
            neighbor_counts: vec![0; cell_count * 9], // Approximation
            // Water neighbor counts for in-fluid detection
            water_neighbor_counts: vec![0; cell_count * 9],
            // Pile heightfield for stacking (one entry per column)
            pile_height: vec![f32::INFINITY; width],
            // Sediment physics feature flags
            use_ferguson_church: true,
            use_hindered_settling: false, // Disabled: neighbor count mismatch crushes settling to 6%
            use_variable_diameter: true,
            diameter_variation: 0.3, // ±30% size variation
            // Viscosity for vortex shedding (disabled by default for comparison)
            use_viscosity: false,
            viscosity: 1.0, // Good starting point for Re ~ 300
            sand_pic_ratio: 0.7, // 70% PIC, 30% FLIP - good balance
        }
    }

    /// Run one simulation step
    pub fn update(&mut self, dt: f32) {
        use std::time::Instant;
        self.frame = self.frame.wrapping_add(1);
        let profile = self.frame % 120 == 0;

        let t0 = Instant::now();

        // 1. Classify cells based on particle positions
        self.classify_cells();
        let t1 = Instant::now();

        // 1b. Update Signed Distance Field for collision
        self.grid.compute_sdf();
        let t2 = Instant::now();

        // 2. Transfer particle velocities to grid (P2G)
        self.particles_to_grid();
        let t3 = Instant::now();

        // 2b. Extrapolate velocities into air cells (1 layer)
        // This ensures store_old_velocities samples valid values everywhere
        self.grid.extrapolate_velocities(1);

        // 3. Store old grid velocities for FLIP blending
        self.store_old_velocities();

        // 4. Apply external forces (gravity)
        self.grid.apply_gravity(dt);

        // 4b. Vorticity confinement - preserves swirl in bulk water
        // Skips cells adjacent to air (free surface) - air is compressible
        // Strength (ε) must be 0.01-0.1 per Fedkiw 2001 - higher causes artificial turbulence
        {
            let grid = &mut self.grid;
            let pile_height = &self.pile_height;
            grid.apply_vorticity_confinement_with_piles(dt, 0.05, pile_height);
        }
        let t4 = Instant::now();

        // 5. Pressure projection - enforces incompressibility
        // CRITICAL: Zero velocities at solid walls BEFORE computing divergence
        self.grid.enforce_boundary_conditions();
        self.grid.compute_divergence();
        let div_before = self.grid.total_divergence();

        // Multigrid pressure solver - 2 V-cycles
        // Testing reduced cycles for faster solve
        self.grid.solve_pressure_multigrid(2);
        // Two-way coupling: use mixture density for pressure gradient
        self.apply_pressure_gradient_two_way(dt);
        let t5 = Instant::now();

        // Diagnostic: compute divergence after for analysis (used by tests)
        self.grid.compute_divergence();
        let _ = self.grid.total_divergence();

        // 5b. Extrapolate after pressure solve for G2P sampling near boundaries
        // With fixed extrapolation (skips fluid-adjacent faces), this is safe
        self.grid.extrapolate_velocities(1);

        // 5c. Surface damping REMOVED - was causing honey-like behavior
        // The damp_surface_vertical() call multiplied v.y by depth factor,
        // killing all vertical velocity near the free surface.
        // If surface noise is a problem, address it differently (surface tension, etc.)

        // 6. Transfer grid velocities back to particles
        // Water: gets FLIP velocity
        // Sediment: stores fluid velocity for drag calculation
        self.grid_to_particles(dt);
        let t6 = Instant::now();

        // 6a. Slope flow correction now handled in Grid::apply_gravity()
        // Gravity is applied tangentially near floors at the grid level.

        // 6b. Build spatial hash for neighbor queries (used by sediment forces + separation)
        self.build_spatial_hash();

        // 6c. Compute neighbor counts for hindered settling AND stickiness
        // (Merged implementation computes both total and water neighbors in parallel)
        self.compute_neighbor_counts();
        let t7 = Instant::now();

        // 7. Legacy sediment forces DISABLED for Phase 2
        // Sand now just follows water via FLIP G2P with 0.95 ratio
        // Two-way coupling happens through mixture density in pressure gradient
        // self.apply_sediment_forces(dt);

        // 8. Advect particles (uses SDF for O(1) collision)
        self.advect_particles(dt);

        // 8b-8d. Legacy bedload system DISABLED for Phase 2
        // Sand stays in Suspended state, no pile mechanics
        // self.update_particle_states(dt);
        // self.compute_pile_heightfield();
        // self.enforce_pile_constraints();

        // Silence unused timing/diagnostic variables (kept for flamegraph profiling)
        let _ = (t0, t1, t2, t3, t4, t5, t6, t7, profile, div_before);

        // 9. Push overlapping particles apart - DISABLED
        // Near-pressure handles incompressibility now. If pressure is strong enough,
        // particles won't overlap and we don't need explicit collision.
        // self.push_particles_apart(1);

        // Clean up particles that left the simulation
        self.particles.remove_out_of_bounds(
            self.grid.width as f32 * self.grid.cell_size,
            self.grid.height as f32 * self.grid.cell_size,
        );
    }

    /// Profiled update - returns timing breakdown in milliseconds
    /// Order: [classify, sdf, p2g, vorticity+pressure, g2p, neighbor, rest]
    pub fn update_profiled(&mut self, dt: f32) -> [f32; 7] {
        use std::time::Instant;
        self.frame = self.frame.wrapping_add(1);

        let t0 = Instant::now();
        self.classify_cells();
        let t1 = Instant::now();

        self.grid.compute_sdf();
        let t2 = Instant::now();

        self.particles_to_grid();
        self.grid.extrapolate_velocities(1);
        self.store_old_velocities();
        let t3 = Instant::now();

        self.grid.apply_gravity(dt);
        {
            let grid = &mut self.grid;
            let pile_height = &self.pile_height;
            grid.apply_vorticity_confinement_with_piles(dt, 0.05, pile_height);
        }
        self.grid.enforce_boundary_conditions();
        self.grid.compute_divergence();
        self.grid.solve_pressure_multigrid(2);
        // Two-way coupling: use mixture density for pressure gradient
        self.apply_pressure_gradient_two_way(dt);
        self.grid.compute_divergence();
        self.grid.extrapolate_velocities(1);
        let t4 = Instant::now();

        self.grid_to_particles(dt);
        let t5 = Instant::now();

        self.build_spatial_hash();
        self.compute_neighbor_counts();
        let t6 = Instant::now();

        // Legacy sediment system DISABLED for Phase 2
        // self.apply_sediment_forces(dt);
        self.advect_particles(dt);
        // self.update_particle_states(dt);
        // self.compute_pile_heightfield();
        // self.enforce_pile_constraints();
        self.particles.remove_out_of_bounds(
            self.grid.width as f32 * self.grid.cell_size,
            self.grid.height as f32 * self.grid.cell_size,
        );
        let t7 = Instant::now();

        [
            (t1 - t0).as_secs_f32() * 1000.0,
            (t2 - t1).as_secs_f32() * 1000.0,
            (t3 - t2).as_secs_f32() * 1000.0,
            (t4 - t3).as_secs_f32() * 1000.0,
            (t5 - t4).as_secs_f32() * 1000.0,
            (t6 - t5).as_secs_f32() * 1000.0,
            (t7 - t6).as_secs_f32() * 1000.0,
        ]
    }

    /// Run isolated FLIP cycle WITH extrapolation for testing
    /// P2G → extrapolate → store_old → G2P (NO forces, NO pressure)
    ///
    /// This tests the FLIP transfer in isolation. If momentum is conserved here,
    /// the core FLIP algorithm is working correctly.
    pub fn run_isolated_flip_cycle_with_extrapolation(&mut self, dt: f32) {
        self.classify_cells();
        self.particles_to_grid();
        self.grid.extrapolate_velocities(1);
        self.store_old_velocities();
        // NO forces - this is an isolated transfer test
        self.grid_to_particles(dt);
    }

    /// Step 1: Classify cells as solid, fluid, or air
    /// Only WATER particles mark cells as fluid (for pressure boundary conditions).
    /// Sediment is purely Lagrangian - it doesn't affect the pressure solve.
    pub fn classify_cells(&mut self) {
        // Reset to air
        for cell in &mut self.grid.cell_type {
            *cell = CellType::Air;
        }

        // Mark solid cells from terrain
    for j in 0..self.grid.height {
        for i in 0..self.grid.width {
            let idx = self.grid.cell_index(i, j);
            
            // Explicitly mark boundaries as solid
            // 0 is Top (Open), Height-1 is Bottom (Floor)
            // 0 and Width-1 are Walls
            let is_boundary = i == 0 || i == self.grid.width - 1 || j == self.grid.height - 1;
            
            if is_boundary || self.grid.is_solid(i, j) {
                self.grid.cell_type[idx] = CellType::Solid;
            }
        }
    }

        // Mark fluid cells - only WATER particles mark cells as fluid
        // Sediment in air should NOT mark cells as fluid
        for particle in self.particles.iter() {
            // Only water marks cells as Fluid
            if particle.material != ParticleMaterial::Water {
                continue;
            }
            let (i, j) = self.grid.pos_to_cell(particle.position);
            let idx = self.grid.cell_index(i, j);
            if self.grid.cell_type[idx] != CellType::Solid {
                self.grid.cell_type[idx] = CellType::Fluid;
            }
        }
    }

    /// Step 2: APIC Particle-to-Grid transfer (P2G)
    ///
    /// Transfers particle velocity + affine momentum to the grid using quadratic B-splines.
    /// The affine velocity matrix C captures local velocity gradients.
    ///
    /// For each grid node i:
    ///   momentum_i += w_ip * (v_p + C_p * (x_i - x_p))
    ///   mass_i += w_ip
    ///
    /// Final grid velocity: v_i = momentum_i / mass_i
    ///
    /// Optimization: Precompute 1D weights to reduce bspline calls from 18 to 12 per particle.
    pub fn particles_to_grid(&mut self) {
        // Clear accumulators
        self.u_sum.fill(0.0);
        self.u_weight.fill(0.0);
        self.v_sum.fill(0.0);
        self.v_weight.fill(0.0);
        // Clear volume fraction accumulators (two-way coupling)
        self.sand_volume_u.fill(0.0);
        self.water_volume_u.fill(0.0);
        self.sand_volume_v.fill(0.0);
        self.water_volume_v.fill(0.0);

        let cell_size = self.grid.cell_size;
        let width = self.grid.width;
        let height = self.grid.height;

        // Particle volume for volume fraction tracking (~4 particles per cell)
        let particle_volume = cell_size * cell_size / 4.0;

        for particle in self.particles.iter() {
            let pos = particle.position;
            let vel = particle.velocity;
            let c_mat = particle.affine_velocity;
            let is_sand = particle.is_sediment();

            // ========== U component (staggered on left edges) ==========
            // U nodes are at (i * cell_size, (j + 0.5) * cell_size) in world coords
            let u_pos = pos / cell_size - Vec2::new(0.0, 0.5);
            let base_i = u_pos.x.floor() as i32;
            let base_j = u_pos.y.floor() as i32;
            let fx = u_pos.x - base_i as f32;
            let fy = u_pos.y - base_j as f32;

            // Precompute 1D weights: wx[di+1], wy[dj+1] for di,dj in {-1,0,1}
            let u_wx = [
                quadratic_bspline_1d(fx + 1.0),  // di = -1
                quadratic_bspline_1d(fx),        // di = 0
                quadratic_bspline_1d(fx - 1.0),  // di = 1
            ];
            let u_wy = [
                quadratic_bspline_1d(fy + 1.0),  // dj = -1
                quadratic_bspline_1d(fy),        // dj = 0
                quadratic_bspline_1d(fy - 1.0),  // dj = 1
            ];

            for dj in -1..=1i32 {
                let nj = base_j + dj;
                if nj < 0 || nj >= height as i32 {
                    continue;
                }
                let wy = u_wy[(dj + 1) as usize];

                for di in -1..=1i32 {
                    let ni = base_i + di;
                    if ni < 0 || ni > width as i32 {
                        continue;
                    }

                    let w = u_wx[(di + 1) as usize] * wy;
                    if w <= 0.0 {
                        continue;
                    }

                    let idx = self.grid.u_index(ni as usize, nj as usize);

                    // Sand does NOT contribute velocity to grid - it's carried passively by water
                    // Two-way coupling happens through mixture density in pressure gradient
                    // Water uses APIC with affine velocity term
                    if !is_sand {
                        // APIC: offset from particle to grid node (in world coords)
                        let offset = Vec2::new(
                            (ni as f32) * cell_size - pos.x,
                            (nj as f32 + 0.5) * cell_size - pos.y,
                        );
                        // Affine velocity contribution: C * offset
                        let affine_vel = c_mat * offset;
                        self.u_sum[idx] += (vel.x + affine_vel.x) * w;
                        self.u_weight[idx] += w;
                    }

                    // Volume fraction tracking for two-way coupling
                    if is_sand {
                        self.sand_volume_u[idx] += w * particle_volume;
                    } else {
                        self.water_volume_u[idx] += w * particle_volume;
                    }
                }
            }

            // ========== V component (staggered on bottom edges) ==========
            // V nodes are at ((i + 0.5) * cell_size, j * cell_size) in world coords
            let v_pos = pos / cell_size - Vec2::new(0.5, 0.0);
            let base_i = v_pos.x.floor() as i32;
            let base_j = v_pos.y.floor() as i32;
            let fx = v_pos.x - base_i as f32;
            let fy = v_pos.y - base_j as f32;

            // Precompute 1D weights for V component
            let v_wx = [
                quadratic_bspline_1d(fx + 1.0),
                quadratic_bspline_1d(fx),
                quadratic_bspline_1d(fx - 1.0),
            ];
            let v_wy = [
                quadratic_bspline_1d(fy + 1.0),
                quadratic_bspline_1d(fy),
                quadratic_bspline_1d(fy - 1.0),
            ];

            for dj in -1..=1i32 {
                let nj = base_j + dj;
                if nj < 0 || nj > height as i32 {
                    continue;
                }
                let wy = v_wy[(dj + 1) as usize];

                for di in -1..=1i32 {
                    let ni = base_i + di;
                    if ni < 0 || ni >= width as i32 {
                        continue;
                    }

                    let w = v_wx[(di + 1) as usize] * wy;
                    if w <= 0.0 {
                        continue;
                    }

                    let idx = self.grid.v_index(ni as usize, nj as usize);

                    // Sand does NOT contribute velocity to grid - it's carried passively by water
                    // Two-way coupling happens through mixture density in pressure gradient
                    // Water uses APIC with affine velocity term
                    if !is_sand {
                        // Offset from particle to grid node (in world coords)
                        let offset = Vec2::new(
                            (ni as f32 + 0.5) * cell_size - pos.x,
                            (nj as f32) * cell_size - pos.y,
                        );
                        // Affine velocity contribution
                        let affine_vel = c_mat * offset;
                        self.v_sum[idx] += (vel.y + affine_vel.y) * w;
                        self.v_weight[idx] += w;
                    }

                    // Volume fraction tracking for two-way coupling
                    if is_sand {
                        self.sand_volume_v[idx] += w * particle_volume;
                    } else {
                        self.water_volume_v[idx] += w * particle_volume;
                    }
                }
            }
        }

        // Normalize: velocity = momentum / mass
        for i in 0..self.grid.u.len() {
            if self.u_weight[i] > 0.0 {
                self.grid.u[i] = self.u_sum[i] / self.u_weight[i];
            } else {
                self.grid.u[i] = 0.0;
            }
        }
        for i in 0..self.grid.v.len() {
            if self.v_weight[i] > 0.0 {
                self.grid.v[i] = self.v_sum[i] / self.v_weight[i];
            } else {
                self.grid.v[i] = 0.0;
            }
        }
    }

    /// Apply pressure gradient with mixture density for two-way coupling
    ///
    /// Higher sand fraction → higher mixture density → smaller acceleration
    /// This is what causes sand-laden flow to move slower than clear water
    fn apply_pressure_gradient_two_way(&mut self, _dt: f32) {
        let scale = 1.0 / self.grid.cell_size;

        // Density constants
        const WATER_DENSITY: f32 = 1.0;
        const SAND_DENSITY: f32 = 2.65;

        // Update U velocities (horizontal)
        for j in 0..self.grid.height {
            for i in 1..self.grid.width {
                let idx_left = self.grid.cell_index(i - 1, j);
                let idx_right = self.grid.cell_index(i, j);

                let left_type = self.grid.cell_type[idx_left];
                let right_type = self.grid.cell_type[idx_right];

                let u_idx = self.grid.u_index(i, j);
                if left_type == CellType::Solid || right_type == CellType::Solid {
                    self.grid.u[u_idx] = 0.0;
                } else if left_type == CellType::Fluid || right_type == CellType::Fluid {
                    // Compute sand fraction at this face
                    let sand_vol = self.sand_volume_u[u_idx];
                    let water_vol = self.water_volume_u[u_idx];
                    let total_vol = sand_vol + water_vol;

                    // Mixture density: ρ_mix = ρ_water * (1 - φ) + ρ_sand * φ
                    // Capped at 1.5 so water can still push through sand accumulations
                    let rho_mix = if total_vol > 0.0 {
                        let sand_frac = sand_vol / total_vol;
                        let raw_rho = WATER_DENSITY * (1.0 - sand_frac) + SAND_DENSITY * sand_frac;
                        raw_rho.min(1.5)  // Cap at 1.5x water density
                    } else {
                        WATER_DENSITY  // Default to water if no particles
                    };

                    let grad = (self.grid.pressure[idx_right] - self.grid.pressure[idx_left]) * scale;
                    self.grid.u[u_idx] -= grad / rho_mix;
                }
            }
        }

        // Update V velocities (vertical)
        for j in 1..self.grid.height {
            for i in 0..self.grid.width {
                let idx_bottom = self.grid.cell_index(i, j - 1);
                let idx_top = self.grid.cell_index(i, j);

                let bottom_type = self.grid.cell_type[idx_bottom];
                let top_type = self.grid.cell_type[idx_top];

                let v_idx = self.grid.v_index(i, j);
                if bottom_type == CellType::Solid || top_type == CellType::Solid {
                    self.grid.v[v_idx] = 0.0;
                } else if bottom_type == CellType::Fluid || top_type == CellType::Fluid {
                    // Compute sand fraction at this face
                    let sand_vol = self.sand_volume_v[v_idx];
                    let water_vol = self.water_volume_v[v_idx];
                    let total_vol = sand_vol + water_vol;

                    // Mixture density (capped at 1.5)
                    let rho_mix = if total_vol > 0.0 {
                        let sand_frac = sand_vol / total_vol;
                        let raw_rho = WATER_DENSITY * (1.0 - sand_frac) + SAND_DENSITY * sand_frac;
                        raw_rho.min(1.5)
                    } else {
                        WATER_DENSITY
                    };

                    let grad = (self.grid.pressure[idx_top] - self.grid.pressure[idx_bottom]) * scale;
                    self.grid.v[v_idx] -= grad / rho_mix;
                }
            }
        }
    }

    /// Step 3: Store old grid velocities for FLIP calculation
    ///
    /// CRITICAL: Must use quadratic B-spline sampling (same as G2P) to avoid
    /// kernel mismatch that causes phantom velocity deltas.
    /// See: plans/flip-damping-diagnosis.md
    ///
    /// Optimizations:
    /// - Parallelized with rayon (particles are independent)
    /// - Precomputed 1D weights (reduces bspline calls from 18 to 12)
    fn store_old_velocities(&mut self) {
        let cell_size = self.grid.cell_size;
        let width = self.grid.width;
        let height = self.grid.height;
        let grid = &self.grid;

        self.particles.list.par_iter_mut().for_each(|particle| {
            let pos = particle.position;
            let mut velocity = Vec2::ZERO;
            let mut u_weight_sum = 0.0f32;
            let mut v_weight_sum = 0.0f32;

            // ========== Sample U component (quadratic B-spline) ==========
            // U nodes are at (i * cell_size, (j + 0.5) * cell_size)
            let u_pos = pos / cell_size - Vec2::new(0.0, 0.5);
            let base_i = u_pos.x.floor() as i32;
            let base_j = u_pos.y.floor() as i32;
            let fx = u_pos.x - base_i as f32;
            let fy = u_pos.y - base_j as f32;

            // Precompute 1D weights for U
            let u_wx = [
                quadratic_bspline_1d(fx + 1.0),
                quadratic_bspline_1d(fx),
                quadratic_bspline_1d(fx - 1.0),
            ];
            let u_wy = [
                quadratic_bspline_1d(fy + 1.0),
                quadratic_bspline_1d(fy),
                quadratic_bspline_1d(fy - 1.0),
            ];

            for dj in -1..=1i32 {
                let nj = base_j + dj;
                if nj < 0 || nj >= height as i32 {
                    continue;
                }
                let wy = u_wy[(dj + 1) as usize];

                for di in -1..=1i32 {
                    let ni = base_i + di;
                    if ni < 0 || ni > width as i32 {
                        continue;
                    }

                    let w = u_wx[(di + 1) as usize] * wy;
                    if w <= 0.0 {
                        continue;
                    }

                    let u_idx = grid.u_index(ni as usize, nj as usize);
                    velocity.x += w * grid.u[u_idx];
                    u_weight_sum += w;
                }
            }

            // ========== Sample V component (quadratic B-spline) ==========
            // V nodes are at ((i + 0.5) * cell_size, j * cell_size)
            let v_pos = pos / cell_size - Vec2::new(0.5, 0.0);
            let base_i = v_pos.x.floor() as i32;
            let base_j = v_pos.y.floor() as i32;
            let fx = v_pos.x - base_i as f32;
            let fy = v_pos.y - base_j as f32;

            // Precompute 1D weights for V
            let v_wx = [
                quadratic_bspline_1d(fx + 1.0),
                quadratic_bspline_1d(fx),
                quadratic_bspline_1d(fx - 1.0),
            ];
            let v_wy = [
                quadratic_bspline_1d(fy + 1.0),
                quadratic_bspline_1d(fy),
                quadratic_bspline_1d(fy - 1.0),
            ];

            for dj in -1..=1i32 {
                let nj = base_j + dj;
                if nj < 0 || nj > height as i32 {
                    continue;
                }
                let wy = v_wy[(dj + 1) as usize];

                for di in -1..=1i32 {
                    let ni = base_i + di;
                    if ni < 0 || ni >= width as i32 {
                        continue;
                    }

                    let w = v_wx[(di + 1) as usize] * wy;
                    if w <= 0.0 {
                        continue;
                    }

                    let v_idx = grid.v_index(ni as usize, nj as usize);
                    velocity.y += w * grid.v[v_idx];
                    v_weight_sum += w;
                }
            }

            // Normalize by weight sum to handle boundary clipping
            if u_weight_sum > 0.0 {
                velocity.x /= u_weight_sum;
            }
            if v_weight_sum > 0.0 {
                velocity.y /= v_weight_sum;
            }

            particle.old_grid_velocity = velocity;
        });
    }

    /// Step 5: APIC Grid-to-Particle transfer (G2P)
    ///
    /// For each particle:
    ///   v_p = Σ_i w_ip * v_i                    (velocity from grid)
    ///   C_p = D_inv * Σ_i w_ip * v_i ⊗ (x_i - x_p)  (affine velocity matrix)
    ///
    /// Where D_inv = 4/Δx² for quadratic B-splines.
    /// The C matrix captures local velocity gradients for angular momentum conservation.
    ///
    /// TWO DIFFERENT BEHAVIORS:
    /// - Water: Gets grid velocity and updates C matrix (APIC)
    /// - Sediment: Only samples fluid velocity for drag calculation (Lagrangian)
    /// - Water: Gets grid velocity and updates C matrix (APIC)
    /// - Sediment: Only samples fluid velocity for drag calculation (Lagrangian)
    fn grid_to_particles(&mut self, dt: f32) {
        let cell_size = self.grid.cell_size;
        let width = self.grid.width;
        let height = self.grid.height;
        let d_inv = apic_d_inverse(cell_size);
        let sand_pic_ratio = self.sand_pic_ratio;

        let grid = &self.grid;

        self.particles.list.par_iter_mut().for_each(|particle| {
            let pos = particle.position;

            // Sand uses high FLIP ratio so it's carried by the flow
            // Skip APIC C matrix update (sand is discrete solid, not continuous fluid)
            if particle.is_sediment() {
                // CRITICAL: Use B-spline sampling to match store_old_velocities kernel
                // Bilinear vs B-spline mismatch caused phantom velocity deltas!
                let v_grid = grid.sample_velocity_bspline(pos);
                use crate::physics::GRAVITY;
                use std::sync::atomic::{AtomicU32, Ordering};
                static DIAG_COUNTER: AtomicU32 = AtomicU32::new(0);

                // Check if sand is in a Fluid cell (has meaningful water velocity to follow)
                let (ci, cj) = (
                    (pos.x / cell_size) as usize,
                    (pos.y / cell_size) as usize,
                );
                let cell_idx = cj * width + ci;
                let cell_type = if cell_idx < grid.cell_type.len() {
                    grid.cell_type[cell_idx]
                } else {
                    CellType::Air
                };

                if cell_type == CellType::Fluid {
                    // IN FLUID: Blend between PIC (follow grid) and FLIP (respond to changes)
                    // Sand should mostly follow the water (PIC), with some FLIP for momentum
                    // High FLIP causes sand to overshoot when water slows down

                    let old_grid_vel = particle.old_grid_velocity;
                    let grid_delta = v_grid - old_grid_vel;

                    // PIC: velocity = v_grid
                    // FLIP: velocity = velocity + delta
                    // Blend: velocity = PIC_RATIO * v_grid + (1 - PIC_RATIO) * (velocity + delta)
                    let pic_vel = v_grid;
                    let flip_vel = particle.velocity + grid_delta;
                    particle.velocity = sand_pic_ratio * pic_vel + (1.0 - sand_pic_ratio) * flip_vel;

                    particle.old_grid_velocity = v_grid;
                } else {
                    // IN AIR: Don't use FLIP (no meaningful water velocity)
                    // Just maintain current velocity - gravity and settling applied below
                    // Reset old_grid_velocity so next time we enter fluid, delta is clean
                    particle.old_grid_velocity = Vec2::ZERO;
                }

                // DIAGNOSTIC: Only print if sand has significant negative velocity
                let new_vel = particle.velocity;
                let count = DIAG_COUNTER.fetch_add(1, Ordering::Relaxed);
                if new_vel.x < -10.0 && count < 20 {
                    eprintln!(
                        "NEG SAND: pos=({:.0},{:.0}) vel=({:.1},{:.1}) v_grid=({:.1},{:.1})",
                        pos.x, pos.y, new_vel.x, new_vel.y, v_grid.x, v_grid.y
                    );
                }

                // Simple settling: sand is denser than water, so it settles
                // Settling velocity based on density difference: (ρ_sand - ρ_water) / ρ_sand * g * dt
                // For sand (2.65) vs water (1.0): factor ≈ 0.62
                const SETTLING_FACTOR: f32 = 0.62;
                particle.velocity.y += GRAVITY * SETTLING_FACTOR * dt;
                return;
            }

            // Store old particle velocity for FLIP blend
            let old_particle_velocity = particle.velocity;

            // ========== APIC for water particles ==========
            let mut new_velocity = Vec2::ZERO;
            let mut new_c = Mat2::ZERO;
            let mut u_weight_sum = 0.0f32;
            let mut v_weight_sum = 0.0f32;

            // Sample U component
            let u_pos = pos / cell_size - Vec2::new(0.0, 0.5);
            let base_i = u_pos.x.floor() as i32;
            let base_j = u_pos.y.floor() as i32;
            let fx = u_pos.x - base_i as f32;
            let fy = u_pos.y - base_j as f32;

            for dj in -1..=1i32 {
                for di in -1..=1i32 {
                    let ni = base_i + di;
                    let nj = base_j + dj;

                    if ni < 0 || ni > width as i32 || nj < 0 || nj >= height as i32 {
                        continue;
                    }

                    let delta = Vec2::new(fx - di as f32, fy - dj as f32);
                    let w = quadratic_bspline(delta);
                    if w <= 0.0 {
                        continue;
                    }

                    let u_idx = grid.u_index(ni as usize, nj as usize);
                    let u_val = grid.u[u_idx];

                    // Velocity contribution
                    new_velocity.x += w * u_val;
                    u_weight_sum += w;

                    // C matrix contribution: v_i ⊗ (x_i - x_p) * w * D_inv
                    let offset = Vec2::new(
                        (ni as f32) * cell_size - pos.x,
                        (nj as f32 + 0.5) * cell_size - pos.y,
                    );
                    // outer_product(u_component, offset) contributes to C's first column
                    new_c.x_axis += offset * (w * u_val * d_inv);
                }
            }

            // Sample V component
            let v_pos = pos / cell_size - Vec2::new(0.5, 0.0);
            let base_i = v_pos.x.floor() as i32;
            let base_j = v_pos.y.floor() as i32;
            let fx = v_pos.x - base_i as f32;
            let fy = v_pos.y - base_j as f32;

            for dj in -1..=1i32 {
                for di in -1..=1i32 {
                    let ni = base_i + di;
                    let nj = base_j + dj;

                    if ni < 0 || ni >= width as i32 || nj < 0 || nj > height as i32 {
                        continue;
                    }

                    let delta = Vec2::new(fx - di as f32, fy - dj as f32);
                    let w = quadratic_bspline(delta);
                    if w <= 0.0 {
                        continue;
                    }

                    let v_idx = grid.v_index(ni as usize, nj as usize);
                    let v_val = grid.v[v_idx];

                    // Velocity contribution
                    new_velocity.y += w * v_val;
                    v_weight_sum += w;

                    // C matrix contribution (second column from V)
                    let offset = Vec2::new(
                        (ni as f32 + 0.5) * cell_size - pos.x,
                        (nj as f32) * cell_size - pos.y,
                    );
                    new_c.y_axis += offset * (w * v_val * d_inv);
                }
            }

            // Normalize by weight sum to handle boundary clipping
            if u_weight_sum > 0.0 {
                new_velocity.x /= u_weight_sum;
                new_c.x_axis /= u_weight_sum;
            }
            if v_weight_sum > 0.0 {
                new_velocity.y /= v_weight_sum;
                new_c.y_axis /= v_weight_sum;
            }

            // FLIP/PIC blend for velocity update
            // FLIP: preserve particle velocity, add grid delta from forces/pressure
            // PIC: take grid velocity directly (dissipative but stable)
            //
            // FLIP formula: v_p^{n+1} = v_p^n + (v_grid^{n+1} - v_grid^n)
            // The grid delta captures: gravity + pressure projection + vorticity
            // This preserves particle momentum while adding external forces
            //
            // We stored old_grid_v in store_old_velocities() BEFORE forces were applied
            let grid_delta = new_velocity - particle.old_grid_velocity;

            // CLAMPING: Prevent energy explosions by limiting the FLIP delta
            // This is "non-negotiable" for stability.
            // Limit delta to 5 cells per frame (heuristic)
            let max_dv = 5.0 * cell_size / dt; 
            let clamped_delta = if grid_delta.length_squared() > max_dv * max_dv {
                grid_delta.normalize() * max_dv
            } else {
                grid_delta
            };

            let flip_velocity = old_particle_velocity + clamped_delta;
            let pic_velocity = new_velocity;

            // Blend: mostly FLIP (0.95) to preserve horizontal momentum
            // EXCEPTION: Bedload particles should ignore grid updates to stay "stuck"
            // unless they are transitioning out. But apply_sediment_forces handles adhesion.
            // Problem: If FLIP adds a large grid_delta (pressure push), it might unstick.
            // Solution: For Bedload, mostly use particle velocity (which is zero) + very small delta
            if particle.state == ParticleState::Bedload {
                 // For bedload, we want to IGNORE the grid velocity update that would push them
                 // We only update if they are explicitly being moved by forces we want (like lifting)
                 // But apply_sediment_forces is responsible for mobility.
                 // So we keep them at their current velocity (zero) until sediment forces act.
                 // However, we DO need to update affine velocity for consistency.
                 particle.velocity = old_particle_velocity;
            } else {
                 // Uniform FLIP/PIC blend (industry standard: 0.95-0.99)
                 // Higher FLIP = more momentum preservation, slight noise
                 // Lower FLIP = more smoothing, momentum loss
                 const FLIP_RATIO: f32 = 0.97;
                 particle.velocity = FLIP_RATIO * flip_velocity + (1.0 - FLIP_RATIO) * pic_velocity;
            }
            particle.affine_velocity = new_c;

            // Safety clamp
            // Increased to 2000.0 to allow tunneling test to reproduce (and eventually be fixed by micro-steps)
            const MAX_VELOCITY: f32 = 2000.0;
            let speed = particle.velocity.length();
            if speed > MAX_VELOCITY {
                particle.velocity *= MAX_VELOCITY / speed;
            }
        });
    }

    /// Step 7: Apply forces to sediment particles (Lagrangian sediment transport)
    ///
    /// Uses drift-flux model with Ferguson-Church (2004) settling velocity:
    /// - Particles are advected by fluid velocity
    /// - Plus a "slip" velocity toward the bottom (settling)
    /// - Slip velocity computed from Ferguson-Church universal equation
    ///   which accounts for particle size, density, and shape factor
    /// - Richardson-Zaki hindered settling reduces velocity in concentrated regions
    ///
    /// BEDLOAD vs SUSPENDED behavior:
    /// - Suspended: Normal settling + drag toward fluid velocity
    /// - Bedload: Friction-dominated, resists motion until Shields exceeded
    ///
    /// Feature flags control which physics are active:
    /// - use_ferguson_church: Use Ferguson-Church (true) or simple density-based (false)
    /// - use_hindered_settling: Apply Richardson-Zaki correction in concentrated regions
    /// - use_variable_diameter: Use per-particle diameter (true) or material typical (false)
    fn apply_sediment_forces(&mut self, dt: f32) {
        use crate::particle::{hindered_settling_factor, neighbor_count_to_concentration};

        // Drag rate controls how fast particles approach target velocity
        // Higher = particles quickly match fluid + slip (more responsive)
        // Lower = particles maintain momentum longer (more inertial)
        const BASE_DRAG_RATE: f32 = 5.0;

        // Expected neighbors at "rest" (dilute) conditions
        // ~8 particles per 3x3 cell neighborhood is typical for dilute flow
        const REST_NEIGHBORS: f32 = 8.0;

        use crate::physics::GRAVITY;

        // Minimum water neighbors to be considered "in fluid"
        const MIN_WATER_NEIGHBORS: u16 = 3;

        // Borrow neighbor_counts as a slice for parallel access
        let neighbor_counts = &self.neighbor_counts;
        let water_neighbor_counts = &self.water_neighbor_counts;
        let grid = &self.grid;

        // Copy feature flags for closure capture
        let use_ferguson_church = self.use_ferguson_church;
        let use_hindered_settling = self.use_hindered_settling;
        let use_variable_diameter = self.use_variable_diameter;

        self.particles
            .list
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, particle)| {
                // Only apply to sediment - water is handled by APIC
                if !particle.is_sediment() {
                    return;
                }

                // === TASK 4: BEDLOAD PARTICLES ARE FORCE-FREE ===
                // Jammed/bedload particles do not receive gravity, drag, or any integration.
                // They remain static until unjammed by update_particle_states().
                // This is critical for stable pile formation.
                if particle.state == ParticleState::Bedload {
                    particle.velocity = Vec2::ZERO;
                    return;
                }

                // Check if particle is in water or air
                let water_neighbors = water_neighbor_counts.get(idx).copied().unwrap_or(0);
                let in_water = water_neighbors >= MIN_WATER_NEIGHBORS;

                if !in_water {
                    // IN AIR: Apply gravity directly (free fall)
                    // No settling, no fluid drag - just gravity + air resistance
                    particle.velocity.y += GRAVITY * dt;

                    // Light air drag to prevent runaway velocities
                    particle.velocity *= 0.995;

                    // Safety clamp
                    const MAX_VELOCITY: f32 = 200.0; // Higher limit for free fall
                    let speed = particle.velocity.length();
                    if speed > MAX_VELOCITY {
                        particle.velocity *= MAX_VELOCITY / speed;
                    }
                    return;
                }



                // IN WATER (Suspended): Apply drift-flux settling physics
                let density = particle.density();

                // Get diameter: per-particle if enabled, otherwise material typical
                let diameter = if use_variable_diameter {
                    particle.effective_diameter()
                } else {
                    particle.material.typical_diameter()
                };

                // ========== FIRST: Check if in air - applies to ALL states ==========
                // If in air, just fall. No settling physics, no friction, just gravity.
                let (ci, cj) = grid.pos_to_cell(particle.position);
                let cell_idx = grid.cell_index(ci, cj);
                let cell_type = grid.cell_type[cell_idx];
                let in_air = cell_type == CellType::Air;

                if in_air {
                    // IN AIR: Free fall under gravity - no other physics
                    particle.velocity.y += GRAVITY * dt;
                    // Reset to suspended if we were bedload (can't be bedload in air)
                    particle.state = ParticleState::Suspended;
                    return;
                }

                // ========== IN FLUID: Apply settling physics (Suspended only) ==========
                // Note: Bedload particles are skipped at the top of this function
                // Settling through water
                let v_fluid = particle.old_grid_velocity;

                // Compute base settling velocity
                let base_settling = if use_ferguson_church {
                    particle.material.settling_velocity(diameter)
                } else {
                    let r = (density - 1.0) / 1.0;
                    if r > 0.0 && diameter > 0.0 {
                        (r * GRAVITY * diameter).sqrt() / density.sqrt()
                    } else {
                        0.0
                    }
                };

                // Apply hindered settling if enabled
                let settling_velocity = if use_hindered_settling {
                    let neighbor_count = neighbor_counts.get(idx).copied().unwrap_or(0) as usize;
                    let concentration = neighbor_count_to_concentration(neighbor_count, REST_NEIGHBORS);
                    let hindered_factor = hindered_settling_factor(concentration);
                    base_settling * hindered_factor
                } else {
                    base_settling
                };

                // Target velocity: fluid velocity + downward settling slip
                let slip = glam::Vec2::new(0.0, settling_velocity);
                let target_velocity = v_fluid + slip;

                // Drag toward target velocity
                // RATE physics:
                // Settling is driven by effective gravity (gravity - buoyancy).
                // Terminal velocity is reached when Drag = Effective Gravity.
                // Drag Force = m * drag_rate * v.
                // m * drag_rate * v_term = m * g_eff.
                // So drag_rate = g_eff / v_term.
                // This ensures initial acceleration matches gravity.

                let effective_gravity = GRAVITY * (density - 1.0) / density;
                let drag_rate = if base_settling > 0.001 {
                    // Physics-based rate
                    effective_gravity / base_settling
                } else {
                    // Fallback for neutral buoyancy
                    BASE_DRAG_RATE
                };

                let blend = (drag_rate * dt).clamp(0.0, 1.0);
                particle.velocity = particle.velocity.lerp(target_velocity, blend);

                // Safety clamp
                const MAX_VELOCITY: f32 = 120.0;
                let speed = particle.velocity.length();
                if speed > MAX_VELOCITY {
                    particle.velocity *= MAX_VELOCITY / speed;
                }
            });
    }

    /// Step 7b: Update particle states (Suspended ↔ Bedload)
    ///
    /// State transitions based on:
    /// - Enter Bedload: velocity < threshold AND near floor (SDF < radius)
    /// - Exit Bedload: Shields number > unjam threshold AND jam_time > MIN_JAM_TIME
    ///
    /// Hysteresis prevents rapid flickering between states:
    /// - UNJAM_THRESHOLD is 2-3x JAM_THRESHOLD
    /// - MIN_JAM_TIME requires particles to be bedload for some time before unjamming
    /// - Shear deadzone shields bedload particles from small velocity fluctuations
    fn update_particle_states(&mut self, dt: f32) {
        // Thresholds in normalized units (cells-per-frame)
        // Lowered to make sand easier to re-entrain by water flow
        const JAM_THRESHOLD: f32 = 0.08;    // Was 0.15 - slower before jamming
        const UNJAM_THRESHOLD: f32 = 0.12;  // Was 0.40 - easier to unjam
        const SHEAR_DEADZONE: f32 = 0.02;   // Was 0.08 - more responsive
        const MIN_JAM_TIME: f32 = 0.10;     // Was 0.20 - faster unjam
        const JAM_VEL_MAX: f32 = 0.03;      // Was 0.05 - stricter jam

        // Per-material shear resistance multiplier
        // Higher = harder to entrain (requires more shear)
        fn material_shear_factor(mat: ParticleMaterial) -> f32 {
            match mat {
                ParticleMaterial::Sand => 1.0,
                ParticleMaterial::Water => 1.0, // N/A for water
            }
        }

        let grid = &self.grid;
        let cell_size = grid.cell_size;
        let pile_height = &self.pile_height;
        let pile_width = pile_height.len();

        // Velocity scale: "1 cell per frame" in velocity units
        // This normalizes velocities to cells-per-frame for threshold comparison
        let v_scale = cell_size / dt;

        // Near floor distance: 1.5x cell size to catch particles resting on floor
        let near_floor_distance = cell_size * 1.5;

        self.particles.list.par_iter_mut().for_each(|particle| {
            // Only update sediment states
            if !particle.is_sediment() {
                return;
            }

            // Compute shear as velocity gradient (difference between layers)
            // shear = (u(x,y) - u(x,y-dx)).length()
            // where y-dx is one cell below (toward bed)
            let fluid_vel_here = particle.old_grid_velocity;
            let pos_below = particle.position + Vec2::new(0.0, cell_size);
            let fluid_vel_below = grid.sample_velocity(pos_below);
            let shear = (fluid_vel_here - fluid_vel_below).length();

            // Apply deadzone and normalize to cells-per-frame
            let effective_shear = (shear - SHEAR_DEADZONE).max(0.0);
            let shear_n = effective_shear / v_scale;

            // Relative velocity: particle motion relative to local fluid (normalized)
            let rel_vel = (particle.velocity - fluid_vel_here).length();
            let rel_vel_n = rel_vel / v_scale;

            // Material-aware thresholds
            let factor = material_shear_factor(particle.material);
            let jam_th = JAM_THRESHOLD * factor;
            let unjam_th = UNJAM_THRESHOLD * factor;

            // === SUPPORT CHECK ===
            // Particles can only jam if they're supported by something:
            // 1. On a solid floor (SDF surface pointing up)
            // 2. On top of a pile (bedload particles below)
            let sdf_dist = grid.sample_sdf(particle.position);
            let near_solid = sdf_dist < near_floor_distance;
            let is_floor_surface = if near_solid {
                let grad = grid.sdf_gradient(particle.position);
                grad.y < -0.5 // Floor has gradient pointing upward (negative y)
            } else {
                false
            };
            let on_solid_floor = near_solid && is_floor_surface;

            // Check if supported by pile
            // pile_top is the smallest Y (highest point) of bedload in this column
            // Particle is "on pile" if its bottom is CLOSE to pile_top (within 1 cell)
            let col = ((particle.position.x / cell_size) as usize).min(pile_width.saturating_sub(1));
            let pile_top = pile_height[col];
            let particle_radius = cell_size * 0.5;
            let particle_bottom = particle.position.y + particle_radius;
            // Particle is "on pile" if:
            // 1. pile exists (pile_top < INFINITY)
            // 2. particle_bottom >= pile_top (particle is AT or BELOW pile surface, not floating above)
            // 3. particle_bottom is close to pile_top (within 1 cell)
            let on_pile = pile_top < f32::INFINITY
                && particle_bottom >= pile_top  // not floating above the pile
                && (particle_bottom - pile_top) < cell_size;  // within 1 cell of pile top

            let has_support = on_solid_floor || on_pile;

            match particle.state {
                ParticleState::Suspended => {
                    // JAM: must have support AND low normalized shear AND low relative velocity
                    if has_support && shear_n < jam_th && rel_vel_n < JAM_VEL_MAX {
                        particle.state = ParticleState::Bedload;
                        particle.jam_time = 0.0;
                    }
                }
                ParticleState::Bedload => {
                    // 1. LOSS-OF-SUPPORT UNJAM (hard constraint)
                    // Bedload MUST unjam if it loses support - no floating bedload allowed
                    if !has_support {
                        particle.state = ParticleState::Suspended;
                        particle.velocity = fluid_vel_here; // Resume with local fluid velocity
                        particle.jam_time = 0.0;
                        return; // Skip shear check
                    }

                    // 2. SHEAR-BASED UNJAM (physics)
                    // Increment jam time
                    particle.jam_time += dt;

                    // UNJAM: if normalized shear exceeds threshold after minimum jam time
                    if shear_n > unjam_th && particle.jam_time > MIN_JAM_TIME {
                        particle.state = ParticleState::Suspended;
                    }
                }
            }
        });
    }

    /// Step 8c: Compute pile heightfield from bedload particles
    ///
    /// Builds a column-based heightfield where pile_height[x] is the top
    /// of the bedload pile at column x. This allows suspended particles
    /// to land on existing piles and stack upward.
    fn compute_pile_heightfield(&mut self) {
        let cell_size = self.grid.cell_size;
        let width = self.pile_height.len();

        // Reset heightfield to positive infinity (no pile)
        // With +Y = DOWN, smaller Y = higher up, so we use INFINITY as "no pile"
        for h in &mut self.pile_height {
            *h = f32::INFINITY;
        }

        // Find highest bedload particle in each column (smallest Y = highest point)
        for particle in self.particles.list.iter() {
            if particle.state == ParticleState::Bedload && particle.is_sediment() {
                let col = ((particle.position.x / cell_size) as usize).min(width.saturating_sub(1));
                let particle_top = particle.position.y - cell_size * 0.5; // Top of particle (smaller Y)
                self.pile_height[col] = self.pile_height[col].min(particle_top); // min = highest point
            }
        }

        // Smooth over ±1 column to avoid stair-stepping
        // This helps particles settle smoothly onto pile edges
        let mut smoothed = self.pile_height.clone();
        for i in 1..(width - 1) {
            let left = self.pile_height[i - 1];
            let center = self.pile_height[i];
            let right = self.pile_height[i + 1];

            // Only smooth if all three columns have piles
            if left < f32::INFINITY && center < f32::INFINITY && right < f32::INFINITY {
                smoothed[i] = (left + 2.0 * center + right) / 4.0;
            }
        }
        self.pile_height = smoothed;
    }

    /// Step 8d: Enforce pile constraints on all non-bedload particles
    ///
    /// After pile heightfield is computed, ensure no particle is below the pile.
    /// This catches particles that:
    /// - Drifted sideways under a pile
    /// - Were at same level as bedload when it became stuck
    /// - Fell through gaps between columns
    fn enforce_pile_constraints(&mut self) {
        let cell_size = self.grid.cell_size;
        let pile_height = &self.pile_height;
        let pile_width = pile_height.len();

        self.particles.list.par_iter_mut().for_each(|p| {
            // Bedload particles ARE the pile, don't push them
            if p.state == ParticleState::Bedload {
                return;
            }

            let col = ((p.position.x / cell_size) as usize).min(pile_width.saturating_sub(1));

            // Check current and adjacent columns for pile
            let mut floor_y = pile_height[col];
            if col > 0 {
                floor_y = floor_y.min(pile_height[col - 1]);
            }
            if col + 1 < pile_width {
                floor_y = floor_y.min(pile_height[col + 1]);
            }

            // If there's a pile and particle is below it, push up
            if floor_y < f32::INFINITY {
                let particle_radius = cell_size * 0.5;
                let particle_bottom = p.position.y + particle_radius;

                if particle_bottom > floor_y {
                    p.position.y = floor_y - particle_radius;
                    if p.velocity.y > 0.0 {
                        p.velocity.y = 0.0;
                    }
                }
            }
        });
    }


    /// Step 8: Advect particles with SDF-based collision detection
    /// Uses precomputed signed distance field for O(1) collision queries
    /// Also projects suspended particles onto pile heightfield (pile as floor constraint)
    fn advect_particles(&mut self, dt: f32) {
        let cell_size = self.grid.cell_size;
        let width = self.grid.width;
        let height = self.grid.height;
        let margin = cell_size;
        let max_x = width as f32 * cell_size - margin;
        let max_y = height as f32 * cell_size - margin;

        let grid = &self.grid;
        let pile_height = &self.pile_height;
        let pile_width = pile_height.len();

        self.particles.list.par_iter_mut().for_each(|particle| {
            // Optimization: Skip advection for settled particles
            // This allows large sediment beds with zero cost
            if particle.state == ParticleState::Bedload {
                return;
            }

            // Micro-stepped advection with collision checking per substep
            // This prevents tunneling through thin walls at high velocities
            particle.advect_micro_stepped(dt, cell_size, |p| {
                // Simple SDF collision: only push out if inside or touching solid
                let sdf_dist = grid.sample_sdf(p.position);

                // If inside or very close to solid, push out and apply friction
                if sdf_dist < cell_size * 0.5 {
                    let grad = grid.sdf_gradient(p.position);

                    // Push out to safe distance
                    let push_dist = cell_size * 0.5 - sdf_dist;
                    p.position += grad * push_dist;

                    // Decompose velocity into normal and tangential components
                    let v_dot_n = p.velocity.dot(grad);
                    let v_normal = grad * v_dot_n;
                    let v_tangent = p.velocity - v_normal;

                    // Apply friction to tangential velocity (sediment only)
                    let v_tangent_damped = if p.is_sediment() {
                        let friction = if p.state == ParticleState::Bedload {
                            p.material.static_friction()
                        } else {
                            p.material.dynamic_friction() * 0.5
                        };
                        v_tangent * (1.0 - friction)
                    } else {
                        v_tangent
                    };

                    // Remove velocity component into solid, keep damped tangential
                    let v_normal_clamped = if v_dot_n < 0.0 { Vec2::ZERO } else { v_normal };
                    p.velocity = v_tangent_damped + v_normal_clamped;
                }

                // === PILE AS SOLID FLOOR ===
                // Piles act as solid floors for ALL particles (water and sediment)
                // Check current column AND adjacent columns to prevent slipping through gaps
                {
                    let col = ((p.position.x / cell_size) as usize).min(pile_width.saturating_sub(1));

                    // Find the highest pile (smallest Y) among current and adjacent columns
                    let mut floor_y = pile_height[col];
                    if col > 0 {
                        floor_y = floor_y.min(pile_height[col - 1]);
                    }
                    if col + 1 < pile_width {
                        floor_y = floor_y.min(pile_height[col + 1]);
                    }

                    // Only project if there's a pile nearby
                    if floor_y < f32::INFINITY {
                        let particle_radius = cell_size * 0.5;
                        let particle_bottom = p.position.y + particle_radius;

                        // If particle bottom is below pile top, push it up
                        if particle_bottom > floor_y {
                            p.position.y = floor_y - particle_radius;

                            // Zero vertical velocity (landed on pile)
                            if p.velocity.y > 0.0 {
                                p.velocity.y = 0.0;
                            }
                        }
                    }
                }

                // Final bounds clamp (per substep to keep it contained)
                p.position.x = p.position.x.clamp(margin, max_x);
                p.position.y = p.position.y.clamp(margin, max_y);

                if p.position.x <= margin || p.position.x >= max_x {
                    p.velocity.x = 0.0;
                }
                if p.position.y <= margin || p.position.y >= max_y {
                    p.velocity.y = 0.0;
                }
            });
        });
    }

    /// Step 9: Push overlapping particles apart (FLIP-native separation)
    ///
    /// Simple overlap check that only does work when particles actually overlap.
    /// Standard approach used by production FLIP solvers like Houdini.
    ///
    /// Algorithm from Matthias Müller:
    /// - For each particle pair within 2*radius, compute penetration depth
    /// - Push both particles apart along their center line by half the overlap
    ///
    /// Collision with solids is handled by advect_particles, not here.
    fn push_particles_apart(&mut self, iterations: usize) {
        let cell_size = self.grid.cell_size;
        let width = self.grid.width;
        let height = self.grid.height;

        // Particle radius based on visual size (particles render as ~2x2 blocks)
        let particle_radius = 1.25;
        let min_dist = particle_radius * 2.0;
        let min_dist_sq = min_dist * min_dist;

        let particle_count = self.particles.len();
        if self.impulse_buffer.len() < particle_count {
            self.impulse_buffer.resize(particle_count, Vec2::ZERO);
        }

        for _ in 0..iterations {
            self.build_spatial_hash();

            // Clear impulse buffer
            for imp in self.impulse_buffer.iter_mut().take(particle_count) {
                *imp = Vec2::ZERO;
            }

            // Check all particle pairs and accumulate corrections
            for idx in 0..particle_count {
                let pos_i = self.particles.list[idx].position;
                let gi = ((pos_i.x / cell_size) as i32).max(0).min(width as i32 - 1);
                let gj = ((pos_i.y / cell_size) as i32).max(0).min(height as i32 - 1);

                // Check 3x3 neighborhood
                for dj in -1i32..=1 {
                    for di in -1i32..=1 {
                        let ni = gi + di;
                        let nj = gj + dj;

                        if ni < 0 || nj < 0 || ni >= width as i32 || nj >= height as i32 {
                            continue;
                        }

                        let cell_idx = (nj as usize) * width + (ni as usize);
                        let mut j = self.cell_head[cell_idx];

                        while j >= 0 {
                            let j_idx = j as usize;
                            // Only process each pair once (idx < j)
                            if j_idx > idx {
                                let pos_j = self.particles.list[j_idx].position;
                                let diff = pos_i - pos_j;
                                let dist_sq = diff.length_squared();

                                // Only push apart if overlapping
                                if dist_sq < min_dist_sq && dist_sq > 0.0001 {
                                    let dist = dist_sq.sqrt();
                                    let overlap = min_dist - dist;
                                    let dir = diff / dist;

                                    // Very gentle push - only 10% of overlap to avoid jitter
                                    let correction = dir * overlap * 0.1;
                                    self.impulse_buffer[idx] += correction;
                                    self.impulse_buffer[j_idx] -= correction;
                                }
                            }
                            j = self.particle_next[j_idx];
                        }
                    }
                }
            }

            // Apply corrections (with SDF check to avoid pushing into solids)
            let grid = &self.grid;
            for idx in 0..particle_count {
                let correction = self.impulse_buffer[idx];
                if correction.length_squared() > 0.0001 {
                    let particle = &mut self.particles.list[idx];
                    let new_pos = particle.position + correction;

                    // Only apply if new position is outside solids
                    let sdf_dist = grid.sample_sdf(new_pos);
                    if sdf_dist > 0.0 {
                        particle.position = new_pos;
                    }
                    // If would push into solid, skip this correction
                }
            }
        }
    }

    /// Build linked-cell list for spatial hashing (zero allocations after warmup)
    fn build_spatial_hash(&mut self) {
        let cell_size = self.grid.cell_size;
        let width = self.grid.width;
        let height = self.grid.height;

        // Clear cell heads (no allocation - just fill!)
        self.cell_head.fill(-1);

        // Resize particle_next only if particle count increased
        let particle_count = self.particles.len();
        if self.particle_next.len() < particle_count {
            self.particle_next.resize(particle_count, -1);
        }

        // Build linked lists - insert each particle at head of its cell's list
        for (idx, particle) in self.particles.list.iter().enumerate() {
            let gi = ((particle.position.x / cell_size) as usize).min(width - 1);
            let gj = ((particle.position.y / cell_size) as usize).min(height - 1);
            let cell_idx = gj * width + gi;

            // Insert at head of list
            self.particle_next[idx] = self.cell_head[cell_idx];
            self.cell_head[cell_idx] = idx as i32;
        }
    }

    /// Compute neighbor counts for each particle (used for hindered settling and stickiness)
    /// Uses the spatial hash to count particles in same + adjacent cells.
    /// OPTIMIZATION: Only computes for sediment particles (water doesn't need it).
    /// Parallelized for performance.
    fn compute_neighbor_counts(&mut self) {
        use crate::particle::ParticleMaterial;
        use rayon::prelude::*;

        let cell_size = self.grid.cell_size;
        let width = self.grid.width;
        let height = self.grid.height;
        let particle_count = self.particles.len();

        // Resize buffers (zero-init is fine, water particles just read 0)
        if self.neighbor_counts.len() < particle_count {
            self.neighbor_counts.resize(particle_count, 0);
        }
        if self.water_neighbor_counts.len() < particle_count {
            self.water_neighbor_counts.resize(particle_count, 0);
        }

        // Check if we have any sediment - skip entirely if water-only
        let has_sediment = self.particles.list.iter().any(|p| p.is_sediment());
        if !has_sediment {
            return; // No sediment = no neighbor counts needed
        }

        // Build index list of sediment particles only
        let sediment_indices: Vec<usize> = self.particles.list.iter()
            .enumerate()
            .filter(|(_, p)| p.is_sediment())
            .map(|(i, _)| i)
            .collect();

        // Capture immutable references for parallel iteration
        let positions: Vec<Vec2> = self.particles.list.iter().map(|p| p.position).collect();
        let materials: Vec<ParticleMaterial> = self.particles.list.iter().map(|p| p.material).collect();
        let cell_head = &self.cell_head;
        let particle_next = &self.particle_next;

        // Compute counts in parallel - ONLY for sediment particles
        let counts: Vec<(usize, u16, u16)> = sediment_indices
            .into_par_iter()
            .map(|idx| {
                let pos = positions[idx];
                let gi = ((pos.x / cell_size) as i32).clamp(0, width as i32 - 1) as usize;
                let gj = ((pos.y / cell_size) as i32).clamp(0, height as i32 - 1) as usize;

                let mut total_count: u16 = 0;
                let mut water_count: u16 = 0;

                // Check 3x3 neighborhood
                for dj in -1i32..=1 {
                    for di in -1i32..=1 {
                        let ni = gi as i32 + di;
                        let nj = gj as i32 + dj;

                        if ni < 0 || ni >= width as i32 || nj < 0 || nj >= height as i32 {
                            continue;
                        }

                        let cell_idx = nj as usize * width + ni as usize;
                        let mut p_idx = cell_head[cell_idx];

                        while p_idx >= 0 {
                            let neighbor_idx = p_idx as usize;
                            total_count = total_count.saturating_add(1);

                            if materials[neighbor_idx] == ParticleMaterial::Water {
                                water_count = water_count.saturating_add(1);
                            }

                            p_idx = particle_next[neighbor_idx];
                        }
                    }
                }
                (idx, total_count, water_count)
            })
            .collect();

        // Write back to self (only sediment indices updated)
        for (idx, total, water) in counts {
            self.neighbor_counts[idx] = total;
            self.water_neighbor_counts[idx] = water;
        }
    }

    // Legacy function removed (merged into compute_neighbor_counts)
    fn compute_water_neighbor_counts(&mut self) {
        // No-op, handled by compute_neighbor_counts
    }

    /// Check if a position is safe for spawning (not inside solid)
    #[inline]
    fn is_spawn_safe(&self, x: f32, y: f32) -> bool {
        let (i, j) = self.grid.pos_to_cell(Vec2::new(x, y));
        !self.grid.is_solid(i, j) && self.grid.sample_sdf(Vec2::new(x, y)) > 0.0
    }

    /// Spawn water particles at a position with initial velocity
    pub fn spawn_water(&mut self, x: f32, y: f32, vx: f32, vy: f32, count: usize) {
        let mut rng = rand::thread_rng();
        for _ in 0..count {
            let offset_x = (rng.gen::<f32>() - 0.5) * self.grid.cell_size;
            let offset_y = (rng.gen::<f32>() - 0.5) * self.grid.cell_size;
            let px = x + offset_x;
            let py = y + offset_y;
            // Only spawn if position is not inside solid
            if self.is_spawn_safe(px, py) {
                self.particles.spawn_water(
                    px,
                    py,
                    vx + (rng.gen::<f32>() - 0.5) * 10.0,
                    vy + (rng.gen::<f32>() - 0.5) * 10.0,
                );
            }
        }
    }

    /// Spawn sand particles (light sediment, carried by flow)
    pub fn spawn_sand(&mut self, x: f32, y: f32, vx: f32, vy: f32, count: usize) {
        use crate::particle::ParticleMaterial;
        let mut rng = rand::thread_rng();
        let use_variation = self.use_variable_diameter;
        let variation = self.diameter_variation;
        for _ in 0..count {
            let offset_x = (rng.gen::<f32>() - 0.5) * self.grid.cell_size;
            let offset_y = (rng.gen::<f32>() - 0.5) * self.grid.cell_size;
            let px = x + offset_x;
            let py = y + offset_y;
            if self.is_spawn_safe(px, py) {
                let final_vx = vx + (rng.gen::<f32>() - 0.5) * 5.0;
                let final_vy = vy + (rng.gen::<f32>() - 0.5) * 5.0;
                if use_variation {
                    self.particles.spawn_with_variation(
                        px, py, final_vx, final_vy,
                        ParticleMaterial::Sand, variation, rng.gen(),
                    );
                } else {
                    self.particles.spawn_sand(px, py, final_vx, final_vy);
                }
            }
        }
    }

    // ========================================================================
    // VORTEX METRICS - For testing and diagnostics
    // ========================================================================

    /// Compute total kinetic energy of particles: KE = ½Σ|v|²
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
    /// u = -cos(πx)sin(πy), v = sin(πx)cos(πy)
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

    /// Initialize solid body rotation: v = ω × r
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
                    // u = -ω * (y - cy)
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
                    // v = ω * (x - cx)
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

        let measure = |sim: &Self, name: &'static str| -> f32 {
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
    /// This tests if the P2G → store_old → G2P cycle itself causes momentum loss.
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

    /// Compute grid kinetic energy: KE = ½∫|v|² dV
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
