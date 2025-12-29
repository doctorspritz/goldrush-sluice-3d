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

// Submodules (methods will be moved here incrementally)
mod advection;
mod diagnostics;
mod pile;
mod pressure;
mod sediment;
mod spawning;
mod transfer;

// Re-exports for backwards compatibility
// (will be populated as methods are moved)

use crate::dem::DemSimulation;
use crate::grid::{apic_d_inverse, quadratic_bspline, quadratic_bspline_1d, CellType, Grid};
use crate::particle::{ParticleMaterial, Particles, ParticleState};
use crate::physics::GRAVITY;
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

    // === Deposition System (Multi-Material) ===
    // Tracks accumulated sediment mass per grid cell BY MATERIAL TYPE
    // When total mass exceeds threshold, cell converts to solid with composition
    deposited_mass_mud: Vec<f32>,
    deposited_mass_sand: Vec<f32>,
    deposited_mass_magnetite: Vec<f32>,
    deposited_mass_gold: Vec<f32>,

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

    // === Unified DEM for Granular Materials ===
    /// Handles both bedload in water and dry particle physics
    pub dem: DemSimulation,
    /// Use unified DEM instead of legacy apply_dem_settling
    pub use_unified_dem: bool,
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
            // Deposition mass accumulators per material type
            deposited_mass_mud: vec![0.0; cell_count],
            deposited_mass_sand: vec![0.0; cell_count],
            deposited_mass_magnetite: vec![0.0; cell_count],
            deposited_mass_gold: vec![0.0; cell_count],
            // Sediment physics feature flags
            use_ferguson_church: true,
            use_hindered_settling: false, // Disabled: neighbor count mismatch crushes settling to 6%
            use_variable_diameter: true,
            diameter_variation: 0.3, // ±30% size variation
            // Viscosity for vortex shedding (disabled by default for comparison)
            use_viscosity: false,
            viscosity: 1.0, // Good starting point for Re ~ 300
            sand_pic_ratio: 0.7, // 70% PIC, 30% FLIP - good balance
            // Unified DEM
            dem: DemSimulation::new(),
            use_unified_dem: true, // Enable unified DEM by default
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

        // Multigrid pressure solver - 4 V-cycles for better convergence
        self.grid.solve_pressure_multigrid(4);
        // Two-way coupling: use mixture density for pressure gradient
        self.apply_pressure_gradient_two_way(dt);

        // 5a. Porosity-based drag: water slows in dense particle regions
        // This replaces rigid cell-based deposition with continuous resistance
        self.apply_porosity_drag(dt);
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

        // 8e. DEM settling: particle contacts for pile formation
        // Unified DEM handles both bedload in water and dry granular materials
        let has_water = self.particles.iter().any(|p| p.material == ParticleMaterial::Water);
        if self.use_unified_dem {
            self.dem.update(
                &mut self.particles,
                &self.grid,
                &self.cell_head,
                &self.particle_next,
                dt,
                GRAVITY,
                has_water,
            );
        } else {
            // Legacy DEM (for comparison/fallback)
            self.apply_dem_settling(dt);
        }

        // 8f-8h. Cell-based deposition DISABLED for porosity model
        // Particles stay as particles - piles affect water through porosity drag
        // self.deposit_settled_sediment(dt);
        // self.entrain_deposited_sediment(dt);
        // self.collapse_deposited_sediment();

        // 8b-8d. Legacy bedload system DISABLED for Phase 2
        // Sand stays in Suspended state, no pile mechanics
        // self.update_particle_states(dt);
        // self.compute_pile_heightfield();
        // self.enforce_pile_constraints();

        // Silence unused timing/diagnostic variables (kept for flamegraph profiling)
        let _ = (t0, t1, t2, t3, t4, t5, t6, t7, profile, div_before);

        // 9. Push overlapping particles apart - DISABLED
        // Causes brownian jitter. DEM settling handles pile formation for slow particles.
        // Pressure solver handles incompressibility for flowing particles.
        // self.push_particles_apart(2);

        // Clean up particles that left the simulation
        self.particles.remove_out_of_bounds(
            self.grid.width as f32 * self.grid.cell_size,
            self.grid.height as f32 * self.grid.cell_size,
        );
    }

    /// Check if simulation has any water particles
    pub fn has_water(&self) -> bool {
        self.particles.iter().any(|p| p.material == ParticleMaterial::Water)
    }

    /// Update for dry-only simulations (no water particles)
    ///
    /// Skips FLIP grid operations entirely - pure DEM granular physics.
    /// Much faster than full update when only sediment is present.
    pub fn update_dry(&mut self, dt: f32) {
        self.frame = self.frame.wrapping_add(1);

        // 1. Update SDF for collision
        self.grid.compute_sdf();

        // 2. Build spatial hash for neighbor queries
        self.build_spatial_hash();

        // 3. Run unified DEM with full gravity (no buoyancy)
        self.dem.update(
            &mut self.particles,
            &self.grid,
            &self.cell_head,
            &self.particle_next,
            dt,
            GRAVITY,
            false, // in_water = false
        );

        // 4. Advect non-sleeping particles
        for p in self.particles.iter_mut() {
            if p.state != ParticleState::Bedload {
                p.position += p.velocity * dt;

                // SDF collision
                let sdf = self.grid.sample_sdf(p.position);
                if sdf < 0.0 {
                    let grad = self.grid.sdf_gradient(p.position);
                    p.position -= grad * sdf;
                }
            }
        }

        // 5. Cleanup
        self.particles.remove_out_of_bounds(
            self.grid.width as f32 * self.grid.cell_size,
            self.grid.height as f32 * self.grid.cell_size,
        );
    }

    /// Get count of sleeping particles (for performance metrics)
    pub fn sleeping_particle_count(&self) -> usize {
        self.dem.sleeping_count()
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
        self.grid.solve_pressure_multigrid(4);
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
        self.particles_to_grid_impl();
    }

    /// Apply pressure gradient with mixture density for two-way coupling
    ///
    /// Higher sand fraction → higher mixture density → smaller acceleration
    /// This is what causes sand-laden flow to move slower than clear water
    fn apply_pressure_gradient_two_way(&mut self, dt: f32) {
        self.apply_pressure_gradient_two_way_impl(dt);
    }

    /// Apply porosity-based drag to grid velocities
    ///
    /// In dense particle regions, water experiences resistance (Darcy flow).
    /// This replaces rigid cell-based deposition with continuous drag:
    /// - Low particle density → water flows freely
    /// - High particle density → water slows/stops (pile acts like porous wall)
    ///
    /// The drag is applied exponentially: v *= exp(-drag_rate * sand_fraction * dt)
    /// At high sand fraction (~0.6), velocity decays rapidly toward zero.
    fn apply_porosity_drag(&mut self, dt: f32) {
        self.apply_porosity_drag_impl(dt);
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
        self.store_old_velocities_impl();
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
        self.grid_to_particles_impl(dt);
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
        self.apply_sediment_forces_impl(dt);
    }

    /// Step 8e: Apply DEM contact forces for ALL sediment particles
    ///
    /// All sediment particles get collision detection to prevent overlap.
    /// This replaces cell-based deposition with continuous particle piles.
    /// Water particles are excluded (handled by pressure solver).
    fn apply_dem_settling(&mut self, dt: f32) {
        self.apply_dem_settling_impl(dt);
    }

    /// Step 8f: Deposit settled sediment onto terrain (Multi-Material)
    ///
    /// Detects sediment particles that have truly settled (very low velocity,
    /// in dense regions near floor) and converts cells to solid when enough
    /// particles accumulate. Tracks material composition for mixed beds.
    ///
    /// This implements deposition with material tracking for multi-sediment support.
    fn deposit_settled_sediment(&mut self, dt: f32) {
        self.deposit_settled_sediment_impl(dt);
    }

    /// Step 8g: Entrain deposited sediment when flow velocity exceeds threshold (Single-Material)
    ///
    /// Checks each deposited cell for high velocity flow above it.
    /// Uses the cell's material Shields parameter for entrainment threshold.
    /// Spawns all particles of the cell's single material type.
    ///
    /// This enables correct selective entrainment:
    /// - Sand cells (Shields 0.045) wash away first
    /// - Magnetite cells (Shields 0.07) resist 56% more flow
    /// - Gold cells (Shields 0.09) resist 2× more than sand
    fn entrain_deposited_sediment(&mut self, dt: f32) {
        self.entrain_deposited_sediment_impl(dt);
    }

    /// Step 8h: Collapse and avalanche deposited sediment
    ///
    /// Ensures deposited cells have support underneath (won't float) and
    /// spread according to angle of repose (no steep cliffs).
    /// This fixes gaps and artifacts in sediment piles.
    /// Preserves material composition when moving cells.
    fn collapse_deposited_sediment(&mut self) {
        self.collapse_deposited_sediment_impl();
    }

    /// Count deposited cells in a column (from bottom up)
    fn count_column_deposited(&self, i: usize) -> usize {
        self.count_column_deposited_impl(i)
    }

    /// Find the topmost deposited cell in a column (lowest j value)
    fn find_top_deposited_in_column(&self, i: usize) -> Option<usize> {
        self.find_top_deposited_in_column_impl(i)
    }

    /// Find landing j position for a falling/avalanching cell
    fn find_landing_j(&self, i: usize) -> usize {
        self.find_landing_j_impl(i)
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
        self.update_particle_states_impl(dt);
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
        self.advect_particles_impl(dt);
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
        self.push_particles_apart_impl(iterations);
    }

    /// Build linked-cell list for spatial hashing (zero allocations after warmup)
    fn build_spatial_hash(&mut self) {
        self.build_spatial_hash_impl();
    }

    /// Compute neighbor counts for each particle (used for hindered settling and stickiness)
    /// Uses the spatial hash to count particles in same + adjacent cells.
    /// OPTIMIZATION: Only computes for sediment particles (water doesn't need it).
    /// Parallelized for performance.
    fn compute_neighbor_counts(&mut self) {
        self.compute_neighbor_counts_impl();
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

    /// Spawn magnetite particles (black sand - heavy sediment, settles fast, hard to entrain)
    pub fn spawn_magnetite(&mut self, x: f32, y: f32, vx: f32, vy: f32, count: usize) {
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
                        ParticleMaterial::Magnetite, variation, rng.gen(),
                    );
                } else {
                    self.particles.spawn_magnetite(px, py, final_vx, final_vy);
                }
            }
        }
    }

    /// Spawn gold particles (heaviest sediment - settles fastest, hardest to entrain)
    pub fn spawn_gold(&mut self, x: f32, y: f32, vx: f32, vy: f32, count: usize) {
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
                        ParticleMaterial::Gold, variation, rng.gen(),
                    );
                } else {
                    self.particles.spawn_gold(px, py, final_vx, final_vy);
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
