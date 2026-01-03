//! Unified Discrete Element Method (DEM) for granular materials
//!
//! Handles both:
//! - **Bedload particles in water**: Buoyancy-reduced gravity, friction on bed
//! - **Dry particles**: Full gravity, stacking, digging mechanics
//!
//! Key features:
//! - Spring-damper contact model for particle-particle and particle-wall collisions
//! - Static/dynamic friction with Coulomb model
//! - Sleep system for performance (stable particles skip physics)
//! - Neighbor-based support detection for pile stability
//!
//! Reference: Cundall & Strack 1979 "A discrete numerical model for granular assemblies"

use crate::grid::Grid;
use crate::particle::{ParticleMaterial, ParticleState, Particles};
use glam::Vec2;

/// DEM simulation parameters
#[derive(Clone, Copy, Debug)]
pub struct DemParams {
    /// Spring stiffness for contacts (N/m equivalent)
    pub contact_stiffness: f32,
    /// Damping ratio (0-1, >0.7 for overdamped)
    pub damping_ratio: f32,
    /// Coulomb friction coefficient (legacy, now uses material properties)
    pub friction_coeff: f32,
    /// Tangential stiffness as ratio of normal stiffness (k_t = ratio * k_n)
    pub tangent_stiffness_ratio: f32,
    /// Static friction velocity threshold (px/s) - below this, particles "stick"
    pub static_friction_threshold: f32,
    /// Sleep velocity threshold (px/s) - below this with support, particle sleeps
    pub sleep_threshold: f32,
    /// Wake velocity threshold (px/s) - above this, sleeping particle wakes
    pub wake_threshold: f32,
    /// Particle radius in cells (legacy, now uses material.typical_diameter())
    pub particle_radius_cells: f32,
    /// Minimum neighbors below for sleep eligibility
    pub min_support_neighbors: usize,
    /// Maximum DEM substeps per frame (prevents tunneling at high speeds)
    pub max_substeps: usize,
    /// Max displacement per substep as a fraction of particle radius
    pub max_displacement_ratio: f32,
    /// Global velocity damping per frame (0.95 = 5% reduction/frame)
    /// Dissipates wave energy even when not in contact
    pub velocity_damping: f32,
    /// Enable surface roughness jitter for vertical contacts
    pub use_jitter: bool,
    /// Use per-material properties (radius, mass, friction)
    pub use_material_properties: bool,
}

impl Default for DemParams {
    fn default() -> Self {
        Self {
            contact_stiffness: 20000.0,      // Must overpower gravity for stable packing
            damping_ratio: 0.7,              // Critically damped
            friction_coeff: 0.6,             // Legacy fallback
            tangent_stiffness_ratio: 0.4,    // k_t = 0.4 * k_n
            static_friction_threshold: 5.0,  // Below this: static friction regime
            sleep_threshold: 1.0,            // Very low - aggressive sleep
            wake_threshold: 3.0,             // Lower wake threshold for responsiveness
            particle_radius_cells: 0.5,      // Legacy fallback
            min_support_neighbors: 1,        // Only need 1 neighbor to sleep
            velocity_damping: 0.95,          // 5% per frame instead of 2% - faster settling
            use_jitter: true,                // Enable surface roughness
            use_material_properties: true,   // Use per-material radius/mass/friction
            max_substeps: 8,                 // Adaptive substeps limit
            max_displacement_ratio: 0.25,    // Substep when displacement > 0.25 * radius
        }
    }
}

/// DEM simulation state and methods
pub struct DemSimulation {
    /// Parameters
    pub params: DemParams,
    /// Pre-allocated force buffer
    forces: Vec<Vec2>,
    /// Pre-allocated support count buffer (neighbors below each particle)
    support_counts: Vec<u16>,
    /// Pre-allocated sleeping flag buffer
    is_sleeping: Vec<bool>,
    /// Frame counter for time-varying jitter
    frame: u32,
    /// Wake flags for this frame (set when sleeping particle is impacted)
    wake_flags: Vec<bool>,
    /// Pre-collision positions (for PBD velocity derivation)
    pre_solve_positions: Vec<Vec2>,
}

impl DemSimulation {
    pub fn new() -> Self {
        Self {
            params: DemParams::default(),
            forces: Vec::new(),
            support_counts: Vec::new(),
            is_sleeping: Vec::new(),
            frame: 0,
            wake_flags: Vec::new(),
            pre_solve_positions: Vec::new(),
        }
    }

    /// Compute effective gravity for a particle based on environment
    ///
    /// - In water: g_eff = g * (1 - ρ_water / ρ_particle) [buoyancy-reduced]
    /// - Dry: g_eff = g [full gravity]
    #[inline]
    pub fn effective_gravity(base_gravity: f32, material: ParticleMaterial, in_water: bool) -> f32 {
        if in_water && material.is_sediment() {
            let density = material.density();
            // Buoyancy reduces effective weight: (ρ_p - ρ_w) / ρ_p
            let buoyancy_factor = (density - 1.0) / density;
            base_gravity * buoyancy_factor
        } else {
            base_gravity
        }
    }

    /// Run DEM physics for all sediment particles
    ///
    /// This handles:
    /// - Contact forces (spring-damper)
    /// - Friction (static/dynamic Coulomb)
    /// - Floor collision
    /// - Sleep detection and management
    ///
    /// # Arguments
    /// - `particles`: The particle system
    /// - `grid`: Grid for SDF collision queries
    /// - `cell_head`/`particle_next`: Spatial hash for neighbor queries
    /// - `dt`: Time step
    /// - `base_gravity`: Gravity magnitude (applied in +Y direction, screen coords)
    /// - `in_water`: Whether particles are submerged (affects effective gravity)
    pub fn update(
        &mut self,
        particles: &mut Particles,
        grid: &Grid,
        cell_head: &[i32],
        particle_next: &[i32],
        dt: f32,
        base_gravity: f32,
        in_water: bool,
    ) {
        let water_level = if in_water { f32::MAX } else { -1.0 };
        self.update_with_water_level(particles, grid, cell_head, particle_next, dt, base_gravity, water_level);
    }

    /// Run DEM physics with a water surface level
    ///
    /// Particles below water_level (higher Y in screen coords) experience buoyancy.
    /// Pass -1.0 for no water, f32::MAX for fully submerged.
    pub fn update_with_water_level(
        &mut self,
        particles: &mut Particles,
        grid: &Grid,
        cell_head: &[i32],
        particle_next: &[i32],
        dt: f32,
        base_gravity: f32,
        water_level: f32,
    ) {
        // No FLIP coupling - use water_level based detection
        self.update_coupled(particles, grid, cell_head, particle_next, dt, base_gravity, Some(water_level), false);
    }

    /// Run DEM physics fully coupled with FLIP water simulation
    ///
    /// Uses the grid's velocity field (u, v) to apply drag to sediment particles.
    /// Particles in fluid cells experience buoyancy and drag toward water velocity.
    ///
    /// # Arguments
    /// - `water_level`: Optional explicit water level. If None, uses grid cell types.
    /// - `use_flip_velocities`: If true, sample grid.u/v for water velocity drag.
    pub fn update_coupled(
        &mut self,
        particles: &mut Particles,
        grid: &Grid,
        cell_head: &[i32],
        particle_next: &[i32],
        dt: f32,
        base_gravity: f32,
        water_level: Option<f32>,
        use_flip_velocities: bool,
    ) {
        self.frame = self.frame.wrapping_add(1);

        let particle_count = particles.len();
        if particle_count == 0 {
            return;
        }

        let cell_size = grid.cell_size;
        let width = grid.width;
        let height = grid.height;
        let particle_radius = self.params.particle_radius_cells * cell_size;
        let contact_dist = particle_radius * 2.0;
        let contact_dist_sq = contact_dist * contact_dist;

        // Ensure buffers are sized
        self.forces.resize(particle_count, Vec2::ZERO);
        self.support_counts.resize(particle_count, 0);
        self.is_sleeping.resize(particle_count, false);
        self.wake_flags.resize(particle_count, false);
        self.pre_solve_positions.resize(particle_count, Vec2::ZERO);

        // Reset forces and wake flags
        self.forces.fill(Vec2::ZERO);
        self.support_counts.fill(0);
        self.wake_flags.fill(false);

        // === Pass 1: Compute support counts (neighbors below) ===
        self.compute_support_counts(
            particles,
            grid,
            cell_head,
            particle_next,
            contact_dist,
        );

        // === Pass 2: Determine sleep states ===
        self.update_sleep_states(particles, grid, particle_radius);

        // === Pass 3: Simple integration - gravity, move, collide ===
        self.integrate_and_collide_coupled(
            particles,
            grid,
            cell_head,
            particle_next,
            dt,
            base_gravity,
            water_level,
            use_flip_velocities,
        );

        // === Pass 6: Apply wake flags ===
        self.apply_wake_flags();
    }

    /// Count neighbors below each particle (for support/sleep detection)
    fn compute_support_counts(
        &mut self,
        particles: &Particles,
        grid: &Grid,
        cell_head: &[i32],
        particle_next: &[i32],
        contact_dist: f32,
    ) {
        let cell_size = grid.cell_size;
        let width = grid.width;
        let height = grid.height;
        let _ = (width, height); // Used in nested loops

        for idx in 0..particles.len() {
            let p = &particles.list[idx];
            if !p.is_sediment() {
                continue;
            }

            let pos = p.position;
            let gi = ((pos.x / cell_size) as i32).clamp(0, width as i32 - 1);
            let gj = ((pos.y / cell_size) as i32).clamp(0, height as i32 - 1);

            let mut support_count: u16 = 0;

            // Check cells at same level and below (dj = 0, 1 in screen coords where +Y is down)
            for dj in 0i32..=1 {
                for di in -1i32..=1 {
                    let ni = gi + di;
                    let nj = gj + dj;

                    if ni < 0 || ni >= width as i32 || nj < 0 || nj >= height as i32 {
                        continue;
                    }

                    let cell_idx = (nj as usize) * width + (ni as usize);
                    let mut j = cell_head[cell_idx];

                    while j >= 0 {
                        let j_idx = j as usize;
                        if j_idx != idx {
                            let other_pos = particles.list[j_idx].position;
                            // Check if neighbor is below (higher Y in screen coords)
                            if other_pos.y > pos.y {
                                let dist = (other_pos - pos).length();
                                if dist < contact_dist * 1.2 {
                                    support_count += 1;
                                }
                            }
                        }
                        j = particle_next[j_idx];
                    }
                }
            }

            // Floor also counts as support
            let sdf = grid.sample_sdf(pos);
            if sdf < cell_size * 1.5 {
                let grad = grid.sdf_gradient(pos);
                // Floor normal points up (negative Y in screen coords)
                if grad.y < -0.5 {
                    support_count += 2; // Floor is strong support
                }
            }

            self.support_counts[idx] = support_count;
        }
    }

    /// Update particle sleep states based on velocity and support
    fn update_sleep_states(
        &mut self,
        particles: &Particles,
        grid: &Grid,
        particle_radius: f32,
    ) {
        for idx in 0..particles.len() {
            let p = &particles.list[idx];
            if !p.is_sediment() {
                self.is_sleeping[idx] = false;
                continue;
            }

            let speed = p.velocity.length();
            let has_support = self.support_counts[idx] >= self.params.min_support_neighbors as u16;

            // Near floor check
            let sdf = grid.sample_sdf(p.position);
            let near_floor = sdf < particle_radius * 2.0;

            if self.is_sleeping[idx] {
                // Currently sleeping - check for wake condition
                // Wake if: velocity too high OR flagged for wake by impact OR lost support
                let lost_support = !has_support && !near_floor;
                if speed > self.params.wake_threshold || self.wake_flags[idx] || lost_support {
                    self.is_sleeping[idx] = false;
                }
            } else {
                // Currently awake - check for sleep condition
                // More aggressive: use very low threshold
                if speed < self.params.sleep_threshold && (has_support || near_floor) {
                    self.is_sleeping[idx] = true;
                }
            }
        }
    }

    /// Apply wake flags after force computation (called after integrate)
    fn apply_wake_flags(&mut self) {
        for idx in 0..self.wake_flags.len() {
            if self.wake_flags[idx] && self.is_sleeping[idx] {
                self.is_sleeping[idx] = false;
            }
        }
    }

    /// OLD: Position-based collision resolution (PBD style)
    #[allow(dead_code)]
    fn resolve_collisions_pbd(
        &mut self,
        particles: &mut Particles,
        grid: &Grid,
        _cell_head: &[i32],
        _particle_next: &[i32],
        particle_radius: f32,
        iterations: usize,
    ) {
        let contact_dist = particle_radius * 2.0;
        let contact_dist_sq = contact_dist * contact_dist;
        let n = particles.len();

        for _iter in 0..iterations {
            // Brute-force O(n²) for debugging - TODO: use spatial hash
            for idx in 0..n {
                if !particles.list[idx].is_sediment() {
                    continue;
                }

                for j_idx in (idx + 1)..n {
                    if !particles.list[j_idx].is_sediment() {
                        continue;
                    }

                    // Get CURRENT positions
                    let pos_i = particles.list[idx].position;
                    let pos_j = particles.list[j_idx].position;
                    let diff = pos_i - pos_j;
                    let dist_sq = diff.length_squared();

                    if dist_sq >= contact_dist_sq || dist_sq < 0.0001 {
                        continue;
                    }

                    let dist = dist_sq.sqrt();
                    let overlap = contact_dist - dist;

                    if overlap > 0.0 {
                        let normal = diff / dist;
                        let density_i = particles.list[idx].material.density();
                        let density_j = particles.list[j_idx].material.density();

                        // Mass-weighted position correction
                        // ratios sum to 1.0, so each particle gets its share of the full overlap
                        let total_density = density_i + density_j;
                        let ratio_i = density_j / total_density;  // lighter particle moves more
                        let ratio_j = density_i / total_density;

                        // Full overlap correction - mass ratios distribute it
                        // With equal masses: each moves overlap * 0.5, total = overlap (full separation)
                        particles.list[idx].position += normal * overlap * ratio_i;
                        particles.list[j_idx].position -= normal * overlap * ratio_j;

                        // Wake sleeping particle if hit by moving particle
                        if self.is_sleeping[j_idx] && !self.is_sleeping[idx] {
                            self.wake_flags[j_idx] = true;
                        }
                        if self.is_sleeping[idx] && !self.is_sleeping[j_idx] {
                            self.wake_flags[idx] = true;
                        }
                    }
                }
            }

            // Floor collision (SDF-based)
            for idx in 0..particles.len() {
                let p = &mut particles.list[idx];
                if !p.is_sediment() {
                    continue;
                }

                let sdf = grid.sample_sdf(p.position);
                if sdf < particle_radius {
                    let grad = grid.sdf_gradient(p.position);
                    if grad.length_squared() > 0.001 {
                        let push_dist = particle_radius - sdf + 0.2;
                        let push_dir = grad.normalize();
                        p.position += push_dir * push_dist;

                        // Zero velocity into floor
                        let normal_vel = p.velocity.dot(push_dir);
                        if normal_vel < 0.0 {
                            p.velocity -= push_dir * normal_vel * 0.9; // Keep tiny bit for settling
                        }

                        // Apply friction - reduce tangent velocity
                        let tangent = Vec2::new(-push_dir.y, push_dir.x);
                        let tangent_vel = p.velocity.dot(tangent);
                        p.velocity -= tangent * tangent_vel * self.params.friction_coeff;
                    }
                }
            }
        }
    }

    /// Proper DEM integration with FLIP coupling
    /// Handles gravity, water drag, spring-damper contacts with friction
    fn integrate_and_collide_coupled(
        &mut self,
        particles: &mut Particles,
        grid: &Grid,
        cell_head: &[i32],
        particle_next: &[i32],
        dt: f32,
        base_gravity: f32,
        water_level: Option<f32>,
        use_flip_velocities: bool,
    ) {
        let n = particles.len();
        let cell_size = grid.cell_size;
        let width = grid.width;
        let height = grid.height;

        // DEM parameters
        let k_n = self.params.contact_stiffness;
        let k_t = k_n * self.params.tangent_stiffness_ratio;
        let use_material = self.params.use_material_properties;

        // Drag coefficient for FLIP coupling
        let flip_drag_coeff = 8.0;

        // Helper to get particle radius
        let get_radius = |p: &crate::particle::Particle| -> f32 {
            if use_material {
                p.material.typical_diameter() * 0.5 * cell_size
            } else {
                self.params.particle_radius_cells * cell_size
            }
        };

        // Helper to get particle mass (density * sphere volume)
        let get_mass = |p: &crate::particle::Particle, radius: f32| -> f32 {
            if use_material {
                let volume = (4.0 / 3.0) * std::f32::consts::PI * radius.powi(3);
                p.material.density() * volume
            } else {
                1.0 // Unit mass fallback
            }
        };

        // Step 1: Apply gravity, water drag, and integrate
        for idx in 0..n {
            let p = &mut particles.list[idx];
            if !p.is_sediment() {
                continue;
            }

            let radius = get_radius(p);

            // Sleeping particles on floor: still apply settling force for stratification
            // Heavy particles (gold) should sink through lighter particles (sand)
            if self.is_sleeping[idx] {
                let sdf = grid.sample_sdf(p.position);
                if sdf < radius * 1.5 {
                    // On floor - stop horizontal motion but allow vertical settling
                    p.velocity.x = 0.0;
                    p.state = ParticleState::Bedload;

                    // Apply tiny settling velocity based on density difference from average
                    // This allows heavy particles to slowly sink through lighter ones
                    let in_water = if let Some(wl) = water_level {
                        p.position.y > wl
                    } else {
                        true // assume in water for sluice
                    };
                    if in_water {
                        let density = p.material.density();
                        // Average sediment density ~5 (mix of sand 2.65, magnetite 5.2, gold 19.3)
                        const AVG_SEDIMENT_DENSITY: f32 = 5.0;
                        // Settling rate: heavier than average sinks, lighter than average rises
                        // Scale: gold (19.3) gets +0.5 px/s, sand (2.65) gets -0.2 px/s
                        let settling_rate = (density - AVG_SEDIMENT_DENSITY) * 0.035;
                        p.velocity.y = settling_rate;
                        p.position.y += p.velocity.y * dt;
                    }
                    continue;
                } else {
                    self.is_sleeping[idx] = false;
                }
            }

            // Determine if particle is in water
            let in_water = if let Some(wl) = water_level {
                p.position.y > wl
            } else {
                let gi = (p.position.x / cell_size) as usize;
                let gj = (p.position.y / cell_size) as usize;
                gi < grid.width && gj < grid.height &&
                    grid.cell_type[gj * grid.width + gi] == crate::grid::CellType::Fluid
            };

            // Gravity with buoyancy
            let g_eff = Self::effective_gravity(base_gravity, p.material, in_water);
            p.velocity.y += g_eff * dt;

            // Water drag OR global damping (not both)
            if in_water {
                if use_flip_velocities {
                    let water_vel = grid.sample_velocity(p.position);
                    let density = p.material.density();
                    let drag_factor = flip_drag_coeff / density;
                    let vel_diff = water_vel - p.velocity;
                    p.velocity += vel_diff * drag_factor * dt;
                }
                // Light damping in water
                p.velocity *= 0.995;
                // Natural stratification emerges from Ferguson-Church settling velocity
                // differences applied in G2P transfer (transfer.rs)
            } else {
                // Global velocity damping only for DRY sediments (not in water)
                p.velocity *= self.params.velocity_damping;

                // Support-based damping: particles with support below them settle faster
                let has_support = self.support_counts[idx] >= self.params.min_support_neighbors as u16;
                if has_support {
                    let speed = p.velocity.length();
                    if speed < 5.0 {
                        p.velocity *= 0.5;
                    } else if speed < 15.0 {
                        p.velocity *= 0.8;
                    } else if speed < 30.0 {
                        p.velocity *= 0.9;  // Light damping for faster particles too
                    }
                }
            }

            // Move
            p.position += p.velocity * dt;
        }

        // Step 2: Resolve particle-particle collisions with proper DEM
        // Use spatial hash for O(n) instead of O(n²)
        let relaxation = 0.5;

        for _iter in 0..4 {
            for idx in 0..n {
                if !particles.list[idx].is_sediment() {
                    continue;
                }

                let i_sleeping = self.is_sleeping[idx];
                let pos_i = particles.list[idx].position;
                let vel_i = particles.list[idx].velocity;
                let mat_i = particles.list[idx].material;
                let radius_i = get_radius(&particles.list[idx]);
                let mass_i = get_mass(&particles.list[idx], radius_i);

                // Grid cell for this particle
                let gi = ((pos_i.x / cell_size) as i32).clamp(0, width as i32 - 1);
                let gj = ((pos_i.y / cell_size) as i32).clamp(0, height as i32 - 1);

                // Check 3x3 neighborhood using spatial hash
                for dj in -1i32..=1 {
                    for di in -1i32..=1 {
                        let ni = gi + di;
                        let nj = gj + dj;

                        if ni < 0 || ni >= width as i32 || nj < 0 || nj >= height as i32 {
                            continue;
                        }

                        let cell_idx = (nj as usize) * width + (ni as usize);
                        let mut j = cell_head[cell_idx];

                        while j >= 0 {
                            let j_idx = j as usize;
                            j = particle_next[j_idx];

                            // Only process each pair once (idx < j_idx)
                            if j_idx <= idx {
                                continue;
                            }

                            if !particles.list[j_idx].is_sediment() {
                                continue;
                            }

                            // Skip intra-clump collisions (particles in same clump don't collide)
                            let clump_i = particles.list[idx].clump_id;
                            let clump_j = particles.list[j_idx].clump_id;
                            if clump_i != 0 && clump_i == clump_j {
                                continue;
                            }

                            let j_sleeping = self.is_sleeping[j_idx];

                            // Skip if BOTH are sleeping
                            if i_sleeping && j_sleeping {
                                continue;
                            }

                            let pos_j = particles.list[j_idx].position;
                            let vel_j = particles.list[j_idx].velocity;
                            let mat_j = particles.list[j_idx].material;
                            let radius_j = get_radius(&particles.list[j_idx]);
                            let mass_j = get_mass(&particles.list[j_idx], radius_j);

                            // Contact distance is sum of radii
                            let contact_dist = radius_i + radius_j;
                            let diff = pos_i - pos_j;
                            let dist_sq = diff.length_squared();

                            if dist_sq >= contact_dist * contact_dist || dist_sq < 0.0001 {
                                continue;
                            }

                            let dist = dist_sq.sqrt();
                            let overlap = contact_dist - dist;

                            if overlap > 0.0 {
                                let normal = diff / dist;
                                let tangent = Vec2::new(-normal.y, normal.x);

                                // Relative velocity
                                let rel_vel = vel_i - vel_j;
                                let v_n = rel_vel.dot(normal);
                                let v_t = rel_vel.dot(tangent);

                                // Effective mass for collision
                                let m_eff = (mass_i * mass_j) / (mass_i + mass_j);

                                // Normal force: spring-damper
                                let c_n = self.params.damping_ratio * 2.0 * (k_n * m_eff).sqrt();
                                let f_n = (k_n * overlap - c_n * v_n).max(0.0);

                                // Tangential friction (Coulomb model)
                                let mu = if use_material {
                                    0.5 * (mat_i.static_friction() + mat_j.static_friction())
                                } else {
                                    self.params.friction_coeff
                                };
                                let f_t_max = mu * f_n;

                                // Coulomb friction with static threshold
                                // Problem: pure viscous damping (k_t * v_t) gives tiny forces for slow particles,
                                // allowing them to creep forever. True static friction locks slow particles.
                                const STATIC_VELOCITY_THRESHOLD: f32 = 0.5; // cells/frame - below this, stop completely

                                let v_t_mag = v_t.abs();
                                let f_t = if v_t_mag < 1e-6 {
                                    0.0  // No relative motion
                                } else if v_t_mag < STATIC_VELOCITY_THRESHOLD {
                                    // Static friction regime: apply stopping force
                                    // Force needed to zero velocity in one timestep: F = m * v / dt
                                    // But we're computing force, then multiplying by dt for impulse,
                                    // so stopping_force = m_eff * v_t / dt, impulse = stopping_force * dt = m_eff * v_t
                                    // To avoid dt dependency here, just use large damping
                                    let stopping_force = k_t * STATIC_VELOCITY_THRESHOLD; // Same as if at threshold
                                    let friction_force = stopping_force.min(f_t_max);
                                    -friction_force * v_t.signum()
                                } else {
                                    // Dynamic friction: viscous damping clamped by μ*F_n
                                    let damping_force = k_t * v_t_mag;
                                    let friction_force = damping_force.min(f_t_max);
                                    -friction_force * v_t.signum()
                                };

                                // Total impulse
                                let impulse_n = normal * f_n * dt;
                                let impulse_t = tangent * f_t * dt;
                                let total_impulse = impulse_n + impulse_t;

                                // Apply position correction with relaxation
                                let push = overlap * relaxation;
                                let ratio_i = mass_j / (mass_i + mass_j);
                                let ratio_j = mass_i / (mass_i + mass_j);
                                particles.list[idx].position += normal * push * ratio_i;
                                particles.list[j_idx].position -= normal * push * ratio_j;

                                // Apply velocity changes (mass-weighted)
                                particles.list[idx].velocity += total_impulse / mass_i;
                                particles.list[j_idx].velocity -= total_impulse / mass_j;

                                // Wake sleeping particles on impact
                                if j_sleeping && !i_sleeping {
                                    self.wake_flags[j_idx] = true;
                                }
                                if i_sleeping && !j_sleeping {
                                    self.wake_flags[idx] = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Step 3: Floor collision with friction
        for idx in 0..n {
            let p = &mut particles.list[idx];
            if !p.is_sediment() {
                continue;
            }

            let radius = get_radius(p);
            let sdf = grid.sample_sdf(p.position);

            if sdf < radius {
                let grad = grid.sdf_gradient(p.position);
                if grad.length_squared() > 0.001 {
                    let push_dir = grad.normalize();
                    let push_dist = radius - sdf + 0.1;
                    p.position += push_dir * push_dist;

                    // Zero velocity into floor
                    let normal_vel = p.velocity.dot(push_dir);
                    if normal_vel < 0.0 {
                        p.velocity -= push_dir * normal_vel;
                    }

                    // Floor friction using material property with static threshold
                    // Same fix as particle-particle: slow particles get stopped completely
                    let mu = if use_material {
                        p.material.static_friction()
                    } else {
                        self.params.friction_coeff
                    };
                    let tangent = Vec2::new(-push_dir.y, push_dir.x);
                    let tangent_vel = p.velocity.dot(tangent);

                    const STATIC_VELOCITY_THRESHOLD: f32 = 0.5;
                    if tangent_vel.abs() < STATIC_VELOCITY_THRESHOLD {
                        // Static friction: stop completely
                        p.velocity -= tangent * tangent_vel;
                    } else {
                        // Dynamic friction: reduce by mu fraction
                        p.velocity -= tangent * tangent_vel * mu;
                    }
                }
            }

            // Update state and sleep
            // Only zero horizontal velocity - allow vertical settling for stratification
            let speed = p.velocity.length();
            if speed < 0.5 {
                // Stop horizontal sliding but keep vertical for density stratification
                p.velocity.x = 0.0;
                // Apply density-based settling for bedload particles
                let density = p.material.density();
                const AVG_SEDIMENT_DENSITY: f32 = 5.0;
                let settling_rate = (density - AVG_SEDIMENT_DENSITY) * 0.035;
                p.velocity.y = settling_rate;
            }

            let sdf = grid.sample_sdf(p.position);
            let on_floor = sdf < radius * 1.5;
            let has_support = self.support_counts[idx] >= self.params.min_support_neighbors as u16;

            if on_floor || has_support {
                p.state = ParticleState::Bedload;
                if speed < self.params.sleep_threshold {
                    self.is_sleeping[idx] = true;
                }
            } else {
                p.state = ParticleState::Suspended;
            }
        }
    }

    /// OLD: PBD Step 1: Apply gravity and predict positions
    #[allow(dead_code)]
    fn integrate_pbd(
        &mut self,
        particles: &mut Particles,
        grid: &Grid,
        dt: f32,
        base_gravity: f32,
        in_water: bool,
        particle_radius: f32,
    ) {
        for idx in 0..particles.len() {
            let p = &mut particles.list[idx];
            if !p.is_sediment() {
                continue;
            }

            // Sleeping particles on floor: skip
            if self.is_sleeping[idx] {
                let sdf = grid.sample_sdf(p.position);
                if sdf < particle_radius * 1.5 {
                    p.velocity = Vec2::ZERO;
                    p.state = ParticleState::Bedload;
                    continue;
                } else {
                    // Floating sleeper - wake it
                    self.is_sleeping[idx] = false;
                }
            }

            // Apply gravity
            let g_eff = Self::effective_gravity(base_gravity, p.material, in_water);
            p.velocity.y += g_eff * dt;

            // Apply velocity damping
            p.velocity *= self.params.velocity_damping;

            // Integrate position
            p.position += p.velocity * dt;
        }
    }

    /// OLD: PBD Step 3: Finalize - derive velocities from position changes, update states
    #[allow(dead_code)]
    fn finalize_pbd(
        &mut self,
        particles: &mut Particles,
        grid: &Grid,
        particle_radius: f32,
        dt: f32,
    ) {
        for idx in 0..particles.len() {
            let p = &mut particles.list[idx];
            if !p.is_sediment() {
                continue;
            }

            // Core PBD: derive velocity from position change
            // This ensures velocity is consistent with where particle ended up
            let old_pos = self.pre_solve_positions[idx];
            let pos_delta = p.position - old_pos;
            p.velocity = pos_delta / dt;

            // Apply damping after deriving velocity
            p.velocity *= self.params.velocity_damping;

            let speed = p.velocity.length();

            // Snap very slow to zero
            if speed < 0.5 {
                p.velocity = Vec2::ZERO;
            }

            // Update state
            let sdf = grid.sample_sdf(p.position);
            let on_floor = sdf < particle_radius * 1.5;
            let has_support = self.support_counts[idx] >= self.params.min_support_neighbors as u16;

            if on_floor || has_support {
                p.state = ParticleState::Bedload;
                if speed < self.params.sleep_threshold {
                    self.is_sleeping[idx] = true;
                }
            } else {
                p.state = ParticleState::Suspended;
            }
        }
    }

    /// OLD: Apply gravity and integrate velocities/positions (kept for reference)
    #[allow(dead_code)]
    fn integrate(
        &mut self,
        particles: &mut Particles,
        grid: &Grid,
        dt: f32,
        base_gravity: f32,
        in_water: bool,
        particle_radius: f32,
    ) {
        let cell_size = grid.cell_size;
        let max_vel = cell_size / dt * 0.4; // CFL-like velocity limit

        for idx in 0..particles.len() {
            let p = &mut particles.list[idx];
            if !p.is_sediment() {
                continue;
            }

            // Sleeping particles on floor: zero velocity, skip integration
            // Sleeping particles NOT on floor: apply gravity (they'll fall and wake on impact)
            let sdf = grid.sample_sdf(p.position);
            let on_floor = sdf < particle_radius * 1.5;

            if self.is_sleeping[idx] {
                if on_floor {
                    // Truly at rest - zero velocity
                    p.velocity = Vec2::ZERO;
                    p.state = ParticleState::Bedload;
                    continue;
                } else {
                    // Sleeping but floating - wake up!
                    self.is_sleeping[idx] = false;
                    // Fall through to normal integration
                }
            }

            // Apply gravity (effective gravity accounts for buoyancy)
            let g_eff = Self::effective_gravity(base_gravity, p.material, in_water);
            let gravity_force = Vec2::new(0.0, g_eff);

            // Total acceleration
            let accel = self.forces[idx] + gravity_force;

            // Integrate velocity
            p.velocity += accel * dt;

            // Global velocity damping - dissipates wave energy even without contact
            // This is crucial for settling: 0.95 = 5% velocity reduction per frame
            p.velocity *= self.params.velocity_damping;

            // Clamp velocity for stability
            let speed = p.velocity.length();
            if speed > max_vel {
                p.velocity *= max_vel / speed;
            }

            // Snap very slow velocities to zero to prevent micro-drift
            if speed < 0.5 {
                p.velocity = Vec2::ZERO;
            }

            // Integrate position
            p.position += p.velocity * dt;

            // Resolve floor penetration - robust collision
            let sdf = grid.sample_sdf(p.position);
            if sdf < particle_radius {
                let grad = grid.sdf_gradient(p.position);
                if grad.length_squared() > 0.001 {
                    // Gradient points AWAY from solid, so ADD to push particle out
                    let push_dist = particle_radius - sdf + 0.1; // Small margin
                    let push_dir = grad.normalize();
                    p.position += push_dir * push_dist;

                    // Also zero out velocity component into the wall
                    let normal_vel = p.velocity.dot(push_dir);
                    if normal_vel < 0.0 {
                        p.velocity -= push_dir * normal_vel;
                    }
                }
            }

            // Update state based on velocity and floor contact
            let final_sdf = grid.sample_sdf(p.position);
            let final_on_floor = final_sdf < particle_radius * 1.5;
            let final_speed = p.velocity.length();

            if final_on_floor && final_speed < self.params.sleep_threshold * 2.0 {
                p.state = ParticleState::Bedload;
            } else {
                p.state = ParticleState::Suspended;
            }
        }
    }

    /// Get sleep state for a particle (for diagnostics/rendering)
    pub fn is_particle_sleeping(&self, idx: usize) -> bool {
        self.is_sleeping.get(idx).copied().unwrap_or(false)
    }

    /// Count sleeping particles (for performance metrics)
    pub fn sleeping_count(&self) -> usize {
        self.is_sleeping.iter().filter(|&&s| s).count()
    }
}

impl Default for DemSimulation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effective_gravity_dry() {
        let g = DemSimulation::effective_gravity(100.0, ParticleMaterial::Sand, false);
        assert_eq!(g, 100.0, "Dry particles should get full gravity");
    }

    #[test]
    fn test_effective_gravity_submerged() {
        // Sand: density 2.65, buoyancy factor = (2.65 - 1) / 2.65 ≈ 0.623
        let g = DemSimulation::effective_gravity(100.0, ParticleMaterial::Sand, true);
        assert!((g - 62.3).abs() < 1.0, "Sand should have ~62% effective gravity in water");

        // Gold: density 19.3, buoyancy factor = (19.3 - 1) / 19.3 ≈ 0.948
        let g = DemSimulation::effective_gravity(100.0, ParticleMaterial::Gold, true);
        assert!((g - 94.8).abs() < 1.0, "Gold should have ~95% effective gravity in water");
    }

    #[test]
    fn test_water_no_dem() {
        let g = DemSimulation::effective_gravity(100.0, ParticleMaterial::Water, true);
        assert_eq!(g, 100.0, "Water should not have buoyancy reduction");
    }
}
