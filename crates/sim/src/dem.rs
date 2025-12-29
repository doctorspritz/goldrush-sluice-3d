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
    /// Coulomb friction coefficient
    pub friction_coeff: f32,
    /// Static friction velocity threshold (px/s) - below this, particles "stick"
    pub static_friction_threshold: f32,
    /// Sleep velocity threshold (px/s) - below this with support, particle sleeps
    pub sleep_threshold: f32,
    /// Wake velocity threshold (px/s) - above this, sleeping particle wakes
    pub wake_threshold: f32,
    /// Particle radius in cells
    pub particle_radius_cells: f32,
    /// Minimum neighbors below for sleep eligibility
    pub min_support_neighbors: usize,
    /// Global velocity damping per frame (0.95 = 5% reduction/frame)
    /// Dissipates wave energy even when not in contact
    pub velocity_damping: f32,
    /// Enable surface roughness jitter for vertical contacts
    pub use_jitter: bool,
}

impl Default for DemParams {
    fn default() -> Self {
        Self {
            contact_stiffness: 2000.0,       // Stiffer for faster response
            damping_ratio: 2.5,              // More overdamped - very fast settling
            friction_coeff: 0.6,
            static_friction_threshold: 2.0,  // Below this: static friction regime
            sleep_threshold: 1.0,            // Very low - aggressive sleep
            wake_threshold: 3.0,             // Lower wake threshold for responsiveness
            particle_radius_cells: 0.5,
            min_support_neighbors: 1,        // Only need 1 neighbor to sleep
            velocity_damping: 0.90,          // 10% velocity reduction per frame - faster settling
            use_jitter: true,                // Enable surface roughness
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
            dt,
            base_gravity,
            water_level,
            use_flip_velocities,
            particle_radius,
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

    /// Simple integration with FLIP coupling support
    /// Handles gravity, water drag, FLIP velocity sampling, and collisions
    fn integrate_and_collide_coupled(
        &mut self,
        particles: &mut Particles,
        grid: &Grid,
        dt: f32,
        base_gravity: f32,
        water_level: Option<f32>,
        use_flip_velocities: bool,
        particle_radius: f32,
    ) {
        let contact_dist = particle_radius * 2.0;
        let n = particles.len();
        let cell_size = grid.cell_size;

        // Drag coefficient for FLIP coupling
        // Higher = sediment follows water more closely
        let flip_drag_coeff = 8.0;

        // Step 1: Apply gravity, water drag, and integrate
        for idx in 0..n {
            let p = &mut particles.list[idx];
            if !p.is_sediment() {
                continue;
            }

            // Skip sleeping particles on floor
            if self.is_sleeping[idx] {
                let sdf = grid.sample_sdf(p.position);
                if sdf < particle_radius * 1.5 {
                    p.velocity = Vec2::ZERO;
                    p.state = ParticleState::Bedload;
                    continue;
                } else {
                    self.is_sleeping[idx] = false;
                }
            }

            // Determine if particle is in water
            let in_water = if let Some(wl) = water_level {
                p.position.y > wl
            } else {
                // Use grid cell type - check if nearby cells have water
                let gi = (p.position.x / cell_size) as usize;
                let gj = (p.position.y / cell_size) as usize;
                gi < grid.width && gj < grid.height &&
                    grid.cell_type[gj * grid.width + gi] == crate::grid::CellType::Fluid
            };

            // Gravity with buoyancy
            let g_eff = Self::effective_gravity(base_gravity, p.material, in_water);
            p.velocity.y += g_eff * dt;

            // Water drag - either from FLIP velocities or basic damping
            if in_water {
                if use_flip_velocities {
                    // Sample water velocity from FLIP grid
                    let water_vel = grid.sample_velocity(p.position);

                    // Apply drag toward water velocity
                    // Lighter particles (lower density) follow water more
                    let density = p.material.density();
                    let drag_factor = flip_drag_coeff / density;

                    let vel_diff = water_vel - p.velocity;
                    p.velocity += vel_diff * drag_factor * dt;
                }

                // Extra damping in water
                p.velocity *= self.params.velocity_damping * 0.85;
            } else {
                // Air damping
                p.velocity *= self.params.velocity_damping;
            }

            // Move
            p.position += p.velocity * dt;
        }

        // Step 2: Resolve particle-particle collisions (multiple passes)
        // Relaxation prevents overcorrection jitter in dense piles
        let relaxation = 0.6;

        for _iter in 0..4 {
            for idx in 0..n {
                if !particles.list[idx].is_sediment() {
                    continue;
                }

                let i_sleeping = self.is_sleeping[idx];

                for j_idx in (idx + 1)..n {
                    if !particles.list[j_idx].is_sediment() {
                        continue;
                    }

                    let j_sleeping = self.is_sleeping[j_idx];

                    // Skip if BOTH are sleeping - they're stable
                    if i_sleeping && j_sleeping {
                        continue;
                    }

                    let pos_i = particles.list[idx].position;
                    let pos_j = particles.list[j_idx].position;
                    let diff = pos_i - pos_j;
                    let dist_sq = diff.length_squared();

                    if dist_sq >= contact_dist * contact_dist || dist_sq < 0.0001 {
                        continue;
                    }

                    let dist = dist_sq.sqrt();
                    let overlap = contact_dist - dist;

                    if overlap > 0.0 {
                        let normal = diff / dist;

                        // Push apart with relaxation to prevent jitter
                        let push = overlap * 0.5 * relaxation;
                        particles.list[idx].position += normal * push;
                        particles.list[j_idx].position -= normal * push;

                        // Kill relative velocity along collision normal
                        let vel_i = particles.list[idx].velocity;
                        let vel_j = particles.list[j_idx].velocity;
                        let rel_vel = vel_i - vel_j;
                        let rel_normal = rel_vel.dot(normal);

                        // Only if approaching
                        if rel_normal < 0.0 {
                            // Inelastic collision - remove approaching velocity
                            particles.list[idx].velocity -= normal * rel_normal * 0.5;
                            particles.list[j_idx].velocity += normal * rel_normal * 0.5;
                        }

                        // Wake sleeping particle if hit by moving particle
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

        // Step 3: Floor collision
        for idx in 0..n {
            let p = &mut particles.list[idx];
            if !p.is_sediment() {
                continue;
            }

            let sdf = grid.sample_sdf(p.position);
            if sdf < particle_radius {
                let grad = grid.sdf_gradient(p.position);
                if grad.length_squared() > 0.001 {
                    let push_dir = grad.normalize();
                    let push_dist = particle_radius - sdf + 0.1;
                    p.position += push_dir * push_dist;

                    // Zero velocity into floor
                    let normal_vel = p.velocity.dot(push_dir);
                    if normal_vel < 0.0 {
                        p.velocity -= push_dir * normal_vel;
                    }

                    // Friction
                    let tangent = Vec2::new(-push_dir.y, push_dir.x);
                    let tangent_vel = p.velocity.dot(tangent);
                    p.velocity -= tangent * tangent_vel * self.params.friction_coeff;
                }
            }

            // Update state and sleep
            let speed = p.velocity.length();
            if speed < 0.5 {
                p.velocity = Vec2::ZERO;
            }

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
