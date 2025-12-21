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

use crate::grid::{apic_d_inverse, quadratic_bspline, CellType, Grid};
use crate::particle::Particles;
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
    // Pre-allocated spatial hash for particle separation (zero allocation per frame)
    cell_head: Vec<i32>,      // Index of first particle in each cell (-1 = empty)
    particle_next: Vec<i32>,  // Index of next particle in same cell (-1 = end)
    // Pre-allocated impulse buffer for soft separation
    impulse_buffer: Vec<Vec2>,
    // Pre-allocated buffer for near-pressure forces (Clavet stabilization)
    near_force_buffer: Vec<Vec2>,
    // Frame counter for skipping expensive operations
    frame: u32,

    // Async near-pressure computation (spread over multiple frames)
    // This decouples computation from application:
    // - Snapshot positions once, compute forces over N frames, apply atomically
    near_pressure_snapshot: Vec<Vec2>,      // Particle positions when computation started
    near_pressure_densities: Vec<(f32, f32)>, // (density, near_density) for each particle
    near_pressure_forces: Vec<Vec2>,        // Accumulated forces to apply
    near_pressure_phase: u32,               // Current phase in computation cycle

    // Tunable near-pressure parameters (press 1-9 to adjust in game)
    pub near_pressure_h: f32,       // Interaction radius (default 4.0)
    pub near_pressure_rest: f32,    // Rest density (default 0.8)

    // Pre-allocated buffer for neighbor counts (hindered settling)
    neighbor_counts: Vec<u16>,

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
            cell_head: vec![-1; cell_count],
            particle_next: Vec::new(),
            impulse_buffer: Vec::new(),
            near_force_buffer: Vec::new(),
            frame: 0,
            // Async near-pressure buffers
            near_pressure_snapshot: Vec::new(),
            near_pressure_densities: Vec::new(),
            near_pressure_forces: Vec::new(),
            near_pressure_phase: 0,
            // Tunable parameters with defaults
            near_pressure_h: 4.0,
            near_pressure_rest: 0.8,
            // Neighbor counts for hindered settling
            neighbor_counts: Vec::new(),
            // Sediment physics feature flags (all enabled by default)
            use_ferguson_church: true,
            use_hindered_settling: true,
            use_variable_diameter: true,
            diameter_variation: 0.3, // ±30% size variation
        }
    }

    /// Run one simulation step
    pub fn update(&mut self, dt: f32) {
        self.frame = self.frame.wrapping_add(1);

        // 1. Classify cells based on particle positions
        self.classify_cells();

        // 2. Transfer particle velocities to grid
        self.particles_to_grid();

        // 3. Store old grid velocities for FLIP
        self.store_old_velocities();

        // 4. Apply external forces (gravity + vorticity)
        // These forces create divergence which is then removed by pressure projection
        self.grid.apply_gravity(dt);
        // Vorticity confinement: ε < 0.1 per literature to avoid artificial turbulence
        // OPTIMIZATION: Run every 2 frames (less critical than pressure)
        if self.frame % 2 == 0 {
            self.grid.apply_vorticity_confinement(dt * 2.0, 0.05);
        }

        // 5. Pressure projection - enforces incompressibility
        // CRITICAL: Zero velocities at solid walls BEFORE computing divergence
        self.grid.enforce_boundary_conditions();
        self.grid.compute_divergence();
        // Fewer iterations for speed (trades accuracy for FPS)
        // 10 iterations is sufficient for visual quality
        self.grid.solve_pressure(10);
        self.grid.apply_pressure_gradient(dt);

        // 6. Transfer grid velocities back to particles
        // Water: gets FLIP velocity
        // Sediment: stores fluid velocity for drag calculation
        self.grid_to_particles();

        // 6b. Build spatial hash for neighbor queries (used by sediment forces + separation)
        self.build_spatial_hash();

        // 6c. Compute neighbor counts for hindered settling
        self.compute_neighbor_counts();

        // 7. Apply forces to sediment (drag + buoyancy + hindered settling)
        // This is proper Lagrangian sediment transport with Richardson-Zaki correction
        self.apply_sediment_forces(dt);

        // 8. Advect particles (uses SDF for O(1) collision)
        self.advect_particles(dt);

        // 9. Push overlapping particles apart (FLIP-native separation)
        // This replaces Clavet SPH near-pressure with a simpler, faster approach.
        // Unlike Clavet's O(n × neighbors × kernels), this is O(n × overlaps).
        // See docs/plans/flip-particle-separation-plan.md
        self.push_particles_apart(2); // 2 iterations as recommended by Houdini FLIP

        // Clean up particles that left the simulation
        self.particles.remove_out_of_bounds(
            self.grid.width as f32 * self.grid.cell_size,
            self.grid.height as f32 * self.grid.cell_size,
        );
    }

    /// Step 1: Classify cells as solid, fluid, or air
    /// ALL particles mark cells as fluid (for pressure boundary conditions).
    /// Sediment doesn't contribute to P2G velocity, but it occupies space.
    fn classify_cells(&mut self) {
        // Reset to air
        for cell in &mut self.grid.cell_type {
            *cell = CellType::Air;
        }

        // Mark solid cells from terrain
        for j in 0..self.grid.height {
            for i in 0..self.grid.width {
                if self.grid.is_solid(i, j) {
                    let idx = self.grid.cell_index(i, j);
                    self.grid.cell_type[idx] = CellType::Solid;
                }
            }
        }

        // Mark fluid cells - ALL particles mark cells as fluid
        // This ensures proper pressure boundaries at water-sediment interface
        for particle in self.particles.iter() {
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
    fn particles_to_grid(&mut self) {
        // Clear accumulators
        self.u_sum.fill(0.0);
        self.u_weight.fill(0.0);
        self.v_sum.fill(0.0);
        self.v_weight.fill(0.0);

        let cell_size = self.grid.cell_size;
        let width = self.grid.width;
        let height = self.grid.height;

        for particle in self.particles.iter() {
            // Sediment contributes with reduced weight (occupies space but sparse)
            let weight_scale = if particle.is_sediment() { 0.5 } else { 1.0 };

            let pos = particle.position;
            let vel = particle.velocity;
            let c_mat = particle.affine_velocity;

            // ========== U component (staggered on left edges) ==========
            // U nodes are at (i * cell_size, (j + 0.5) * cell_size) in world coords
            // Convert to cell-space for quadratic B-spline
            let u_pos = pos / cell_size - Vec2::new(0.0, 0.5);
            let base_i = u_pos.x.floor() as i32;
            let base_j = u_pos.y.floor() as i32;
            let fx = u_pos.x - base_i as f32;
            let fy = u_pos.y - base_j as f32;

            for dj in -1..=1i32 {
                for di in -1..=1i32 {
                    let ni = base_i + di;
                    let nj = base_j + dj;

                    // Bounds check for U grid: ni in [0, width], nj in [0, height-1]
                    if ni < 0 || ni > width as i32 || nj < 0 || nj >= height as i32 {
                        continue;
                    }

                    // Distance from particle to this node (in cell units)
                    let delta = Vec2::new(fx - di as f32, fy - dj as f32);
                    let w = quadratic_bspline(delta);
                    if w <= 0.0 {
                        continue;
                    }

                    // APIC: offset from particle to grid node (in world coords)
                    let offset = Vec2::new(
                        (ni as f32) * cell_size - pos.x,
                        (nj as f32 + 0.5) * cell_size - pos.y,
                    );

                    // Affine velocity contribution: C * offset
                    let affine_vel = c_mat * offset;

                    let idx = self.grid.u_index(ni as usize, nj as usize);
                    let scaled_w = w * weight_scale;

                    // APIC: momentum includes affine term
                    self.u_sum[idx] += (vel.x + affine_vel.x) * scaled_w;
                    self.u_weight[idx] += scaled_w;
                }
            }

            // ========== V component (staggered on bottom edges) ==========
            // V nodes are at ((i + 0.5) * cell_size, j * cell_size) in world coords
            let v_pos = pos / cell_size - Vec2::new(0.5, 0.0);
            let base_i = v_pos.x.floor() as i32;
            let base_j = v_pos.y.floor() as i32;
            let fx = v_pos.x - base_i as f32;
            let fy = v_pos.y - base_j as f32;

            for dj in -1..=1i32 {
                for di in -1..=1i32 {
                    let ni = base_i + di;
                    let nj = base_j + dj;

                    // Bounds check for V grid: ni in [0, width-1], nj in [0, height]
                    if ni < 0 || ni >= width as i32 || nj < 0 || nj > height as i32 {
                        continue;
                    }

                    let delta = Vec2::new(fx - di as f32, fy - dj as f32);
                    let w = quadratic_bspline(delta);
                    if w <= 0.0 {
                        continue;
                    }

                    // Offset from particle to grid node (in world coords)
                    let offset = Vec2::new(
                        (ni as f32 + 0.5) * cell_size - pos.x,
                        (nj as f32) * cell_size - pos.y,
                    );

                    // Affine velocity contribution
                    let affine_vel = c_mat * offset;

                    let idx = self.grid.v_index(ni as usize, nj as usize);
                    let scaled_w = w * weight_scale;

                    self.v_sum[idx] += (vel.y + affine_vel.y) * scaled_w;
                    self.v_weight[idx] += scaled_w;
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

    /// Step 3: Store old grid velocities for FLIP calculation
    fn store_old_velocities(&mut self) {
        for particle in self.particles.iter_mut() {
            particle.old_grid_velocity = self.grid.sample_velocity(particle.position);
        }
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
    fn grid_to_particles(&mut self) {
        let cell_size = self.grid.cell_size;
        let width = self.grid.width;
        let height = self.grid.height;
        let d_inv = apic_d_inverse(cell_size);

        let grid = &self.grid;

        self.particles.list.par_iter_mut().for_each(|particle| {
            let pos = particle.position;

            // Sample velocity from grid using bilinear interpolation for sediment
            let v_grid = grid.sample_velocity(pos);

            // Sediment doesn't participate in APIC - it's Lagrangian
            // Store fluid velocity for drag calculation in apply_sediment_forces
            if particle.is_sediment() {
                particle.old_grid_velocity = v_grid;
                return;
            }

            // ========== APIC for water particles ==========
            let mut new_velocity = Vec2::ZERO;
            let mut new_c = Mat2::ZERO;

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

                    // C matrix contribution (second column from V)
                    let offset = Vec2::new(
                        (ni as f32 + 0.5) * cell_size - pos.x,
                        (nj as f32) * cell_size - pos.y,
                    );
                    new_c.y_axis += offset * (w * v_val * d_inv);
                }
            }

            // Update particle state
            particle.velocity = new_velocity;
            particle.affine_velocity = new_c;

            // Spray dampening: reduce upward velocity (negative y = up in screen coords)
            if particle.velocity.y < -40.0 {
                particle.velocity.y *= 0.7;
            }

            // Safety clamp: CFL requires v*dt < cell_size
            const MAX_VELOCITY: f32 = 100.0;
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
    /// The Ferguson-Church equation naturally handles:
    /// - Fine particles (Stokes regime): settling ∝ d²
    /// - Coarse particles (Newton regime): settling ∝ √d
    /// - Shape effects: flaky gold settles slower than spherical sand of equal mass
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

        // Simple density-based settling constant (used when Ferguson-Church is disabled)
        const SIMPLE_GRAVITY: f32 = 150.0;

        // Borrow neighbor_counts as a slice for parallel access
        let neighbor_counts = &self.neighbor_counts;

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

                let density = particle.density();

                // Get diameter: per-particle if enabled, otherwise material typical
                let diameter = if use_variable_diameter {
                    particle.effective_diameter()
                } else {
                    particle.material.typical_diameter()
                };

                // Compute base settling velocity
                let base_settling = if use_ferguson_church {
                    // Ferguson-Church universal equation (handles all Reynolds regimes)
                    particle.material.settling_velocity(diameter)
                } else {
                    // Simple density-based settling: v = sqrt((ρ-1) * g * d) / ρ
                    // This is a basic Stokes-like approximation
                    let r = (density - 1.0) / 1.0; // Relative submerged density
                    if r > 0.0 && diameter > 0.0 {
                        (r * SIMPLE_GRAVITY * diameter).sqrt() / density.sqrt()
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

                // Sample fluid velocity at particle position (stored during G2P)
                let v_fluid = particle.old_grid_velocity;

                // Target velocity: fluid velocity + downward settling slip
                // The settling velocity is the terminal velocity in still water
                let slip = glam::Vec2::new(0.0, settling_velocity);
                let target_velocity = v_fluid + slip;

                // Drag toward target velocity
                // Heavier particles have more inertia (respond slower)
                let drag_rate = BASE_DRAG_RATE / density;
                let blend = (drag_rate * dt).clamp(0.0, 1.0);

                // Exponential approach to target for stability
                particle.velocity = particle.velocity.lerp(target_velocity, blend);

                // Safety clamp
                const MAX_VELOCITY: f32 = 120.0;
                let speed = particle.velocity.length();
                if speed > MAX_VELOCITY {
                    particle.velocity *= MAX_VELOCITY / speed;
                }
            });
    }

    /// Step 7b: Apply near-pressure forces (Clavet et al. 2005)
    ///
    /// Simple SPH-style collision avoidance for ALL particles.
    /// This prevents overlap without encoding material properties.
    /// Material behavior is handled by drag/buoyancy in apply_sediment_forces().
    ///
    /// Uses rayon for parallel computation of densities and forces.
    fn apply_near_pressure(&mut self, dt: f32) {
        // Clavet parameters - H and REST_DENSITY are tunable
        let h = self.near_pressure_h;
        let rest_density = self.near_pressure_rest;
        const K_STIFFNESS: f32 = 0.5; // Volume preservation (gentle)
        const K_NEAR: f32 = 2.0;      // Overlap prevention

        let h2 = h * h;
        let cell_size = self.grid.cell_size;
        let width = self.grid.width;
        let height = self.grid.height;

        // Build spatial hash for neighbor queries
        self.build_spatial_hash();

        let particle_count = self.particles.len();
        if particle_count == 0 {
            return;
        }

        // Ensure buffers are sized
        if self.near_force_buffer.len() < particle_count {
            self.near_force_buffer.resize(particle_count, Vec2::ZERO);
        }
        // Also need a buffer for densities (to avoid aliasing issues with parallel writes)
        if self.near_pressure_densities.len() < particle_count {
            self.near_pressure_densities.resize(particle_count, (0.0, 0.0));
        }

        // Capture references for parallel closure
        let positions: Vec<Vec2> = self.particles.list.iter().map(|p| p.position).collect();
        let cell_head = &self.cell_head;
        let particle_next = &self.particle_next;

        // Pass 1: Compute densities in parallel
        let densities: Vec<(f32, f32)> = (0..particle_count)
            .into_par_iter()
            .map(|idx| {
                let pos_i = positions[idx];
                let gi = ((pos_i.x / cell_size) as i32).max(0).min(width as i32 - 1);
                let gj = ((pos_i.y / cell_size) as i32).max(0).min(height as i32 - 1);

                let mut density = 0.0f32;
                let mut near_density = 0.0f32;

                for dj in -1i32..=1 {
                    for di in -1i32..=1 {
                        let ni = gi + di;
                        let nj = gj + dj;

                        if ni < 0 || nj < 0 || ni >= width as i32 || nj >= height as i32 {
                            continue;
                        }

                        let cell_idx = (nj as usize) * width + (ni as usize);
                        let mut j = cell_head[cell_idx];

                        while j >= 0 {
                            let j_idx = j as usize;
                            let pos_j = positions[j_idx];
                            let r2 = (pos_i - pos_j).length_squared();

                            if r2 < h2 && r2 > 0.0001 {
                                let q = 1.0 - r2.sqrt() / h;
                                density += q * q;
                                near_density += q * q * q;
                            }

                            j = particle_next[j_idx];
                        }
                    }
                }
                (density, near_density)
            })
            .collect();

        // Copy densities to particles (needed for other systems)
        for (idx, (density, near_density)) in densities.iter().enumerate() {
            self.particles.list[idx].near_density = *near_density;
            self.near_force_buffer[idx] = Vec2::new(*density, 0.0);
        }

        // Pass 2: Compute forces in parallel
        let forces: Vec<Vec2> = (0..particle_count)
            .into_par_iter()
            .map(|idx| {
                let (density_i, near_density_i) = densities[idx];
                let pos_i = positions[idx];
                let pressure_i = K_STIFFNESS * (density_i - rest_density);
                let near_pressure_i = K_NEAR * near_density_i;

                let gi = ((pos_i.x / cell_size) as i32).max(0).min(width as i32 - 1);
                let gj = ((pos_i.y / cell_size) as i32).max(0).min(height as i32 - 1);

                let mut force = Vec2::ZERO;

                for dj in -1i32..=1 {
                    for di in -1i32..=1 {
                        let ni = gi + di;
                        let nj = gj + dj;

                        if ni < 0 || nj < 0 || ni >= width as i32 || nj >= height as i32 {
                            continue;
                        }

                        let cell_idx = (nj as usize) * width + (ni as usize);
                        let mut j = cell_head[cell_idx];

                        while j >= 0 {
                            let j_idx = j as usize;
                            if j_idx != idx {
                                let pos_j = positions[j_idx];
                                let r_vec = pos_i - pos_j;
                                let r2 = r_vec.length_squared();

                                if r2 < h2 && r2 > 0.0001 {
                                    let r = r2.sqrt();
                                    let q = 1.0 - r / h;
                                    let dir = r_vec / r;

                                    let (density_j, near_density_j) = densities[j_idx];
                                    let pressure_j = K_STIFFNESS * (density_j - rest_density);
                                    let near_pressure_j = K_NEAR * near_density_j;

                                    let shared_p = (pressure_i + pressure_j) * 0.5;
                                    let shared_p_near = (near_pressure_i + near_pressure_j) * 0.5;

                                    force += dir * (shared_p * q + shared_p_near * q * q);
                                }
                            }
                            j = particle_next[j_idx];
                        }
                    }
                }
                force
            })
            .collect();

        // Apply forces to particles
        for (idx, force) in forces.into_iter().enumerate() {
            self.particles.list[idx].velocity += force * dt;

            // Clamp velocity
            const MAX_VELOCITY: f32 = 100.0;
            let speed = self.particles.list[idx].velocity.length();
            if speed > MAX_VELOCITY {
                self.particles.list[idx].velocity *= MAX_VELOCITY / speed;
            }
        }
    }

    /// Step 7b (async): Near-pressure forces computed over multiple frames
    ///
    /// This decouples computation from application:
    /// - Phase 0: Snapshot positions, build spatial hash from snapshot
    /// - Phases 1-3: Compute densities for 1/3 of particles each
    /// - Phases 4-6: Compute forces for 1/3 of particles each
    /// - Phase 7: Apply all forces atomically, reset for new cycle
    ///
    /// Total cycle: 8 frames. Each frame does ~1/6 of the work.
    /// Forces are computed from consistent snapshot, preserving momentum.
    fn near_pressure_async(&mut self, dt: f32) {
        const CYCLE_LENGTH: u32 = 8;
        const H: f32 = 4.0;
        const K_STIFFNESS: f32 = 0.5;
        const K_NEAR: f32 = 2.0;
        const REST_DENSITY: f32 = 0.8;

        let phase = self.near_pressure_phase % CYCLE_LENGTH;
        let particle_count = self.particles.len();

        match phase {
            0 => {
                // Snapshot positions and initialize buffers
                self.near_pressure_snapshot.clear();
                self.near_pressure_snapshot.extend(self.particles.iter().map(|p| p.position));
                self.near_pressure_densities.resize(particle_count, (0.0, 0.0));
                self.near_pressure_forces.resize(particle_count, Vec2::ZERO);
                for f in &mut self.near_pressure_forces {
                    *f = Vec2::ZERO;
                }
                // Build spatial hash based on snapshot positions
                self.build_spatial_hash_from_snapshot();
            }
            1 | 2 | 3 => {
                // Compute densities for 1/3 of particles
                let third = particle_count / 3 + 1;
                let start = (phase as usize - 1) * third;
                let end = ((phase as usize) * third).min(particle_count);
                self.compute_densities_range(start, end, H);
            }
            4 | 5 | 6 => {
                // Compute forces for 1/3 of particles
                let third = particle_count / 3 + 1;
                let start = (phase as usize - 4) * third;
                let end = ((phase as usize - 3) * third).min(particle_count);
                self.compute_forces_range(start, end, H, K_STIFFNESS, K_NEAR, REST_DENSITY);
            }
            7 => {
                // Apply all accumulated forces
                let dt_scaled = dt * CYCLE_LENGTH as f32;
                for (idx, force) in self.near_pressure_forces.iter().enumerate() {
                    if idx < self.particles.len() {
                        self.particles.list[idx].velocity += *force * dt_scaled;

                        // Clamp velocity
                        const MAX_VELOCITY: f32 = 100.0;
                        let speed = self.particles.list[idx].velocity.length();
                        if speed > MAX_VELOCITY {
                            self.particles.list[idx].velocity *= MAX_VELOCITY / speed;
                        }
                    }
                }
            }
            _ => {}
        }

        self.near_pressure_phase = self.near_pressure_phase.wrapping_add(1);
    }

    /// Build spatial hash from snapshot positions (not current positions)
    fn build_spatial_hash_from_snapshot(&mut self) {
        let cell_size = self.grid.cell_size;
        let width = self.grid.width;
        let count = self.near_pressure_snapshot.len();

        // Ensure buffers sized
        if self.particle_next.len() < count {
            self.particle_next.resize(count, -1);
        }
        for head in &mut self.cell_head {
            *head = -1;
        }
        for next in &mut self.particle_next[..count] {
            *next = -1;
        }

        for idx in 0..count {
            let pos = self.near_pressure_snapshot[idx];
            let i = (pos.x / cell_size) as usize;
            let j = (pos.y / cell_size) as usize;
            if i < width && j < self.grid.height {
                let cell_idx = j * width + i;
                self.particle_next[idx] = self.cell_head[cell_idx];
                self.cell_head[cell_idx] = idx as i32;
            }
        }
    }

    /// Compute densities for particles in range [start, end)
    fn compute_densities_range(&mut self, start: usize, end: usize, h: f32) {
        let h2 = h * h;
        let cell_size = self.grid.cell_size;
        let width = self.grid.width;
        let height = self.grid.height;
        let snapshot = &self.near_pressure_snapshot;

        for idx in start..end.min(snapshot.len()) {
            let pos_i = snapshot[idx];
            let gi = ((pos_i.x / cell_size) as i32).max(0).min(width as i32 - 1);
            let gj = ((pos_i.y / cell_size) as i32).max(0).min(height as i32 - 1);

            let mut density = 0.0f32;
            let mut near_density = 0.0f32;

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
                        let pos_j = snapshot[j_idx];
                        let r2 = (pos_i - pos_j).length_squared();

                        if r2 < h2 && r2 > 0.0001 {
                            let q = 1.0 - r2.sqrt() / h;
                            density += q * q;
                            near_density += q * q * q;
                        }
                        j = self.particle_next[j_idx];
                    }
                }
            }
            self.near_pressure_densities[idx] = (density, near_density);
        }
    }

    /// Compute forces for particles in range [start, end)
    fn compute_forces_range(&mut self, start: usize, end: usize, h: f32, k_stiff: f32, k_near: f32, rest_density: f32) {
        let h2 = h * h;
        let cell_size = self.grid.cell_size;
        let width = self.grid.width;
        let height = self.grid.height;
        let snapshot = &self.near_pressure_snapshot;
        let densities = &self.near_pressure_densities;

        for idx in start..end.min(snapshot.len()) {
            let pos_i = snapshot[idx];
            let (density_i, near_density_i) = densities[idx];
            let pressure_i = k_stiff * (density_i - rest_density);
            let near_pressure_i = k_near * near_density_i;

            let gi = ((pos_i.x / cell_size) as i32).max(0).min(width as i32 - 1);
            let gj = ((pos_i.y / cell_size) as i32).max(0).min(height as i32 - 1);

            let mut force = Vec2::ZERO;

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
                        if j_idx != idx {
                            let pos_j = snapshot[j_idx];
                            let r_vec = pos_i - pos_j;
                            let r2 = r_vec.length_squared();

                            if r2 < h2 && r2 > 0.0001 {
                                let r = r2.sqrt();
                                let q = 1.0 - r / h;
                                let dir = r_vec / r;

                                let (density_j, near_density_j) = densities[j_idx];
                                let pressure_j = k_stiff * (density_j - rest_density);
                                let near_pressure_j = k_near * near_density_j;

                                let shared_p = (pressure_i + pressure_j) * 0.5;
                                let shared_near = (near_pressure_i + near_pressure_j) * 0.5;

                                force += dir * (shared_p * q + shared_near * q * q);
                            }
                        }
                        j = self.particle_next[j_idx];
                    }
                }
            }
            self.near_pressure_forces[idx] = force;
        }
    }

    /// Step 8: Advect particles with SDF-based collision detection
    /// Uses precomputed signed distance field for O(1) collision queries
    fn advect_particles(&mut self, dt: f32) {
        let cell_size = self.grid.cell_size;
        let width = self.grid.width;
        let height = self.grid.height;
        let margin = cell_size;
        let max_x = width as f32 * cell_size - margin;
        let max_y = height as f32 * cell_size - margin;

        let grid = &self.grid;

        self.particles.list.par_iter_mut().for_each(|particle| {
            // Move particle
            particle.position += particle.velocity * dt;

            // SDF collision: sample distance to nearest solid
            let sdf_dist = grid.sample_sdf(particle.position);

            // If inside or very close to solid, push out
            if sdf_dist < cell_size * 0.5 {
                // Get gradient (points away from solid)
                let grad = grid.sdf_gradient(particle.position);

                // Push out to safe distance
                let push_dist = cell_size * 0.5 - sdf_dist;
                particle.position += grad * push_dist;

                // Remove velocity component into solid
                let v_dot_n = particle.velocity.dot(grad);
                if v_dot_n < 0.0 {
                    particle.velocity -= grad * v_dot_n;
                }
            }

            // SAFETY NET: Direct is_solid check to catch edge cases where SDF
            // bilinear interpolation returns positive values for particles at
            // the boundary of solid cells.
            let (ci, cj) = grid.pos_to_cell(particle.position);
            if grid.is_solid(ci, cj) {
                // Particle is in a solid cell despite SDF check passing.
                // Find nearest non-solid cell and push there.
                let mut best_dir = Vec2::ZERO;
                let mut found_escape = false;

                // Check 8 neighbors for escape direction
                for &(di, dj) in &[(-1i32, 0i32), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)] {
                    let ni = (ci as i32 + di).clamp(0, width as i32 - 1) as usize;
                    let nj = (cj as i32 + dj).clamp(0, height as i32 - 1) as usize;
                    if !grid.is_solid(ni, nj) {
                        best_dir = Vec2::new(di as f32, dj as f32).normalize();
                        found_escape = true;
                        break;
                    }
                }

                if found_escape {
                    // Push particle out of solid cell towards non-solid neighbor
                    particle.position += best_dir * cell_size;
                    // Kill velocity into solid
                    let v_dot_d = particle.velocity.dot(best_dir);
                    if v_dot_d < 0.0 {
                        particle.velocity -= best_dir * v_dot_d;
                    }
                } else {
                    // Surrounded by solids - try pushing up (most common escape)
                    particle.position.y -= cell_size;
                    particle.velocity.y = particle.velocity.y.min(0.0);
                }
            }

            // Final bounds clamp
            particle.position.x = particle.position.x.clamp(margin, max_x);
            particle.position.y = particle.position.y.clamp(margin, max_y);

            if particle.position.x <= margin || particle.position.x >= max_x {
                particle.velocity.x = 0.0;
            }
            if particle.position.y <= margin || particle.position.y >= max_y {
                particle.velocity.y = 0.0;
            }
        });
    }

    /// Step 9: Push overlapping particles apart (FLIP-native separation)
    ///
    /// Unlike Clavet SPH which computes densities and pressures for all neighbors,
    /// this is a simple overlap check that only does work when particles actually overlap.
    /// This is the standard approach used by production FLIP solvers like Houdini.
    ///
    /// Algorithm from Matthias Müller:
    /// - For each particle pair within 2*radius, compute penetration depth
    /// - Push both particles apart along their center line by half the overlap
    ///
    /// WALL-AWARE MODIFICATION: When one particle is near a solid boundary, we push
    /// the other particle more to avoid pushing into the wall. This prevents compression
    /// against walls/floor.
    ///
    /// Complexity: O(n × overlaps) vs Clavet's O(n × neighbors × kernels)
    fn push_particles_apart(&mut self, iterations: usize) {
        let cell_size = self.grid.cell_size;
        let width = self.grid.width;
        let height = self.grid.height;

        // Particle radius based on visual size (particles render as ~2x2 blocks)
        // Separation distance = 2 * radius
        let particle_radius = 1.25;
        let min_dist = particle_radius * 2.0;
        let min_dist_sq = min_dist * min_dist;

        // Distance threshold for "near solid" - particles closer than this to a wall
        // get reduced push toward the wall
        let near_solid_threshold = cell_size * 1.0;

        // Ensure impulse buffer is sized
        let particle_count = self.particles.len();
        if self.impulse_buffer.len() < particle_count {
            self.impulse_buffer.resize(particle_count, Vec2::ZERO);
        }

        for _ in 0..iterations {
            // Build spatial hash for this iteration
            self.build_spatial_hash();

            // Clear impulse buffer
            for imp in self.impulse_buffer.iter_mut().take(particle_count) {
                *imp = Vec2::ZERO;
            }

            // Check all particle pairs and accumulate corrections
            for idx in 0..particle_count {
                let pos_i = self.particles.list[idx].position;
                let sdf_i = self.grid.sample_sdf(pos_i);
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
                                    let dir = diff / dist;  // Points from j to i

                                    // Sample SDF for j
                                    let sdf_j = self.grid.sample_sdf(pos_j);

                                    // Calculate push ratios based on wall proximity
                                    // If near wall, don't push toward wall - push the other particle more
                                    let i_near_wall = sdf_i < near_solid_threshold;
                                    let j_near_wall = sdf_j < near_solid_threshold;

                                    let (ratio_i, ratio_j) = match (i_near_wall, j_near_wall) {
                                        // Neither near wall: split evenly
                                        (false, false) => (0.5, 0.5),
                                        // i near wall: push j more (away from i)
                                        (true, false) => (0.2, 0.8),
                                        // j near wall: push i more (away from j)
                                        (false, true) => (0.8, 0.2),
                                        // Both near wall: still try to separate, but gently
                                        (true, true) => (0.5, 0.5),
                                    };

                                    // Get wall normals (points away from solid)
                                    let grad_i = if i_near_wall { self.grid.sdf_gradient(pos_i) } else { Vec2::ZERO };
                                    let grad_j = if j_near_wall { self.grid.sdf_gradient(pos_j) } else { Vec2::ZERO };

                                    // Correction for i (pushed away from j)
                                    let mut corr_i = dir * overlap * ratio_i;
                                    // If pushing i toward its nearby wall, redirect along wall tangent
                                    if i_near_wall {
                                        let toward_wall = corr_i.dot(grad_i);
                                        if toward_wall < 0.0 {
                                            // Remove wall-facing component and add tangent component
                                            corr_i -= grad_i * toward_wall;
                                            // Also add a small push away from wall
                                            corr_i += grad_i * overlap * 0.3;
                                        }
                                    }

                                    // Correction for j (pushed away from i, so opposite dir)
                                    let mut corr_j = -dir * overlap * ratio_j;
                                    // If pushing j toward its nearby wall, redirect along wall tangent
                                    if j_near_wall {
                                        let toward_wall = corr_j.dot(grad_j);
                                        if toward_wall < 0.0 {
                                            // Remove wall-facing component and add tangent component
                                            corr_j -= grad_j * toward_wall;
                                            // Also add a small push away from wall
                                            corr_j += grad_j * overlap * 0.3;
                                        }
                                    }

                                    self.impulse_buffer[idx] += corr_i;
                                    self.impulse_buffer[j_idx] += corr_j;
                                }
                            }
                            j = self.particle_next[j_idx];
                        }
                    }
                }
            }

            // Apply corrections (with SDF collision check)
            let grid = &self.grid;
            for idx in 0..particle_count {
                let correction = self.impulse_buffer[idx];
                if correction.length_squared() > 0.0001 {
                    let particle = &mut self.particles.list[idx];
                    let new_pos = particle.position + correction;

                    // Check if new position is inside solid
                    let sdf_dist = grid.sample_sdf(new_pos);
                    if sdf_dist > cell_size * 0.25 {
                        // Safe to apply full correction (well outside solid)
                        particle.position = new_pos;
                    } else if sdf_dist > 0.0 {
                        // Near solid surface - apply correction but ensure we stay outside
                        particle.position = new_pos;
                        // Push slightly away from solid to maintain margin
                        let grad = grid.sdf_gradient(new_pos);
                        let safe_dist = cell_size * 0.3;
                        if sdf_dist < safe_dist {
                            particle.position += grad * (safe_dist - sdf_dist);
                        }
                    } else {
                        // New position would be inside solid - DON'T apply this correction
                        // Instead, try to slide along the solid surface
                        let grad = grid.sdf_gradient(particle.position);
                        // Project correction onto tangent plane
                        let tangent_corr = correction - grad * correction.dot(grad);
                        if tangent_corr.length_squared() > 0.01 {
                            let tangent_pos = particle.position + tangent_corr;
                            let tangent_sdf = grid.sample_sdf(tangent_pos);
                            if tangent_sdf > 0.0 {
                                particle.position = tangent_pos;
                            }
                        }
                        // If we can't slide, just don't move
                    }
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

    /// Compute neighbor counts for each particle (used for hindered settling)
    /// Uses the spatial hash to count particles in same + adjacent cells
    fn compute_neighbor_counts(&mut self) {
        let cell_size = self.grid.cell_size;
        let width = self.grid.width;
        let height = self.grid.height;
        let particle_count = self.particles.len();

        // Resize neighbor counts buffer if needed
        if self.neighbor_counts.len() < particle_count {
            self.neighbor_counts.resize(particle_count, 0);
        }

        // Count neighbors for each particle (same cell + 8 adjacent cells)
        for (idx, particle) in self.particles.list.iter().enumerate() {
            let gi = ((particle.position.x / cell_size) as i32).clamp(0, width as i32 - 1) as usize;
            let gj = ((particle.position.y / cell_size) as i32).clamp(0, height as i32 - 1) as usize;

            let mut count: u16 = 0;

            // Check 3x3 neighborhood
            for dj in -1i32..=1 {
                for di in -1i32..=1 {
                    let ni = gi as i32 + di;
                    let nj = gj as i32 + dj;

                    if ni < 0 || ni >= width as i32 || nj < 0 || nj >= height as i32 {
                        continue;
                    }

                    let cell_idx = nj as usize * width + ni as usize;
                    let mut p_idx = self.cell_head[cell_idx];

                    // Walk linked list counting particles
                    while p_idx >= 0 {
                        count = count.saturating_add(1);
                        p_idx = self.particle_next[p_idx as usize];
                    }
                }
            }

            self.neighbor_counts[idx] = count;
        }
    }

    /// Step 9: Soft particle separation - applies velocity impulses instead of position snaps
    fn separate_particles(&mut self) {
        let cell_size = self.grid.cell_size;
        let width = self.grid.width;
        let height = self.grid.height;
        // Particles render as 2x2 blocks, so min distance should prevent visual overlap
        let min_dist = 2.5; // Slightly more than 2 pixels to prevent overlap
        let min_dist_sq = min_dist * min_dist;

        // Build spatial hash once
        self.build_spatial_hash();

        // Use pre-allocated impulse buffer for position corrections
        let particle_count = self.particles.len();
        if self.impulse_buffer.len() < particle_count {
            self.impulse_buffer.resize(particle_count, Vec2::ZERO);
        }
        // Clear impulses
        for imp in self.impulse_buffer.iter_mut().take(particle_count) {
            *imp = Vec2::ZERO;
        }

        for idx in 0..self.particles.len() {
            let pos = self.particles.list[idx].position;
            let gi = ((pos.x / cell_size) as i32).max(0).min(width as i32 - 1);
            let gj = ((pos.y / cell_size) as i32).max(0).min(height as i32 - 1);

            // Check 3x3 neighborhood using linked-cell list
            for dj in -1i32..=1 {
                for di in -1i32..=1 {
                    let ni = gi + di;
                    let nj = gj + dj;

                    if ni < 0 || nj < 0 || ni >= width as i32 || nj >= height as i32 {
                        continue;
                    }

                    let cell_idx = (nj as usize) * width + (ni as usize);

                    // Walk linked list for this cell
                    let mut other_idx = self.cell_head[cell_idx];
                    while other_idx >= 0 {
                        let other_idx_usize = other_idx as usize;
                        if other_idx_usize > idx {
                            // Only process each pair once (idx < other)
                            let other_pos = self.particles.list[other_idx_usize].position;
                            let diff = pos - other_pos;
                            let dist_sq = diff.length_squared();

                            if dist_sq < min_dist_sq && dist_sq > 0.0001 {
                                let dist = dist_sq.sqrt();
                                let overlap = min_dist - dist;
                                let dir = diff / dist;
                                // Position correction - push each particle half the overlap
                                let correction = dir * overlap * 0.5;
                                self.impulse_buffer[idx] += correction;
                                self.impulse_buffer[other_idx_usize] -= correction;
                            }
                        }
                        other_idx = self.particle_next[other_idx_usize];
                    }
                }
            }
        }

        // Apply position corrections, but don't push into solids
        for idx in 0..particle_count {
            let particle = &mut self.particles.list[idx];
            let new_pos = particle.position + self.impulse_buffer[idx];

            // Check if new position is in a solid cell
            let ni = (new_pos.x / cell_size) as usize;
            let nj = (new_pos.y / cell_size) as usize;

            if ni < width && nj < height && !self.grid.is_solid(ni, nj) {
                particle.position = new_pos;
            } else {
                // Try just horizontal correction
                let horiz_pos = Vec2::new(new_pos.x, particle.position.y);
                let hi = (horiz_pos.x / cell_size) as usize;
                let hj = (horiz_pos.y / cell_size) as usize;
                if hi < width && hj < height && !self.grid.is_solid(hi, hj) {
                    particle.position.x = new_pos.x;
                }
                // Try just vertical correction
                let vert_pos = Vec2::new(particle.position.x, new_pos.y);
                let vi = (vert_pos.x / cell_size) as usize;
                let vj = (vert_pos.y / cell_size) as usize;
                if vi < width && vj < height && !self.grid.is_solid(vi, vj) {
                    particle.position.y = new_pos.y;
                }
            }
        }

        // Final pass: push any particles inside solids back out
        self.resolve_solid_penetration();
    }

    /// Push particles that are inside solid cells back to the nearest non-solid position
    fn resolve_solid_penetration(&mut self) {
        let cell_size = self.grid.cell_size;
        let width = self.grid.width;
        let height = self.grid.height;

        for particle in self.particles.list.iter_mut() {
            let i = (particle.position.x / cell_size) as usize;
            let j = (particle.position.y / cell_size) as usize;

            if i >= width || j >= height {
                continue;
            }

            if !self.grid.is_solid(i, j) {
                continue; // Not in solid, nothing to do
            }

            // Particle is inside solid - find nearest non-solid cell
            let cell_center_x = (i as f32 + 0.5) * cell_size;
            let cell_center_y = (j as f32 + 0.5) * cell_size;

            // Check 4 cardinal neighbors and move to nearest non-solid
            let neighbors = [
                (i.wrapping_sub(1), j, -1.0f32, 0.0f32), // left
                (i + 1, j, 1.0, 0.0),                     // right
                (i, j.wrapping_sub(1), 0.0, -1.0),       // up
                (i, j + 1, 0.0, 1.0),                     // down
            ];

            for (ni, nj, dx, dy) in neighbors {
                if ni < width && nj < height && !self.grid.is_solid(ni, nj) {
                    // Move particle to edge of this neighbor cell
                    let edge_x = if dx < 0.0 {
                        i as f32 * cell_size - 0.5
                    } else if dx > 0.0 {
                        (i + 1) as f32 * cell_size + 0.5
                    } else {
                        particle.position.x
                    };
                    let edge_y = if dy < 0.0 {
                        j as f32 * cell_size - 0.5
                    } else if dy > 0.0 {
                        (j + 1) as f32 * cell_size + 0.5
                    } else {
                        particle.position.y
                    };

                    particle.position.x = edge_x;
                    particle.position.y = edge_y;
                    // Zero velocity component going into solid
                    if dx != 0.0 { particle.velocity.x = 0.0; }
                    if dy != 0.0 { particle.velocity.y = 0.0; }
                    break;
                }
            }
        }
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

    /// Spawn mud particles at a position
    pub fn spawn_mud(&mut self, x: f32, y: f32, vx: f32, vy: f32, count: usize) {
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
                let final_vx = vx + (rng.gen::<f32>() - 0.5) * 10.0;
                let final_vy = vy + (rng.gen::<f32>() - 0.5) * 10.0;
                if use_variation {
                    self.particles.spawn_with_variation(
                        px, py, final_vx, final_vy,
                        ParticleMaterial::Mud, variation, rng.gen(),
                    );
                } else {
                    self.particles.spawn_mud(px, py, final_vx, final_vy);
                }
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

    /// Spawn magnetite particles (black sand indicator)
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

    /// Spawn gold particles (heavy, settles fast through water)
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
}

/// Result of raycasting through the grid
enum RaycastResult {
    NoHit,
    Hit { position: Vec2, normal: Vec2 },
}

/// Raycast from start to end, detecting first solid cell collision
/// Uses DDA (Digital Differential Analyzer) for efficient grid traversal
fn raycast_to_solid(grid: &Grid, start: Vec2, end: Vec2, cell_size: f32) -> RaycastResult {
    let dir = end - start;
    let dist = dir.length();

    // Skip raycast for very short movements
    if dist < 0.001 {
        let (i, j) = grid.pos_to_cell(end);
        if grid.is_solid(i, j) {
            return RaycastResult::Hit {
                position: start,
                normal: Vec2::new(0.0, -1.0), // Default up
            };
        }
        return RaycastResult::NoHit;
    }

    let dir_norm = dir / dist;

    // DDA setup
    let mut x = start.x / cell_size;
    let mut y = start.y / cell_size;
    let end_x = end.x / cell_size;
    let end_y = end.y / cell_size;

    let step_x: i32 = if dir_norm.x >= 0.0 { 1 } else { -1 };
    let step_y: i32 = if dir_norm.y >= 0.0 { 1 } else { -1 };

    // Distance to next cell boundary
    let t_delta_x = if dir_norm.x.abs() > 0.0001 {
        (cell_size / dir_norm.x.abs())
    } else {
        f32::MAX
    };
    let t_delta_y = if dir_norm.y.abs() > 0.0001 {
        (cell_size / dir_norm.y.abs())
    } else {
        f32::MAX
    };

    // Distance to first cell boundary
    let mut t_max_x = if dir_norm.x > 0.0 {
        ((x.floor() + 1.0) - x) * t_delta_x
    } else if dir_norm.x < 0.0 {
        (x - x.floor()) * t_delta_x
    } else {
        f32::MAX
    };
    let mut t_max_y = if dir_norm.y > 0.0 {
        ((y.floor() + 1.0) - y) * t_delta_y
    } else if dir_norm.y < 0.0 {
        (y - y.floor()) * t_delta_y
    } else {
        f32::MAX
    };

    let mut cell_x = x.floor() as i32;
    let mut cell_y = y.floor() as i32;
    let end_cell_x = end_x.floor() as i32;
    let end_cell_y = end_y.floor() as i32;

    // Walk through cells
    let max_steps = ((end_cell_x - cell_x).abs() + (end_cell_y - cell_y).abs() + 2) as usize;
    for _ in 0..max_steps {
        // Check current cell
        if cell_x >= 0
            && cell_y >= 0
            && (cell_x as usize) < grid.width
            && (cell_y as usize) < grid.height
        {
            if grid.is_solid(cell_x as usize, cell_y as usize) {
                // Hit! Calculate position just before entering this cell
                let hit_t = (t_max_x.min(t_max_y) - t_delta_x.min(t_delta_y)).max(0.0);
                let hit_pos = start + dir_norm * hit_t * 0.95; // Back off slightly

                // Determine normal based on which boundary we crossed
                let normal = if t_max_x < t_max_y {
                    Vec2::new(-step_x as f32, 0.0)
                } else {
                    Vec2::new(0.0, -step_y as f32)
                };

                return RaycastResult::Hit {
                    position: hit_pos,
                    normal,
                };
            }
        }

        // Check if we've reached the end
        if cell_x == end_cell_x && cell_y == end_cell_y {
            break;
        }

        // Step to next cell
        if t_max_x < t_max_y {
            t_max_x += t_delta_x;
            cell_x += step_x;
        } else {
            t_max_y += t_delta_y;
            cell_y += step_y;
        }
    }

    RaycastResult::NoHit
}
