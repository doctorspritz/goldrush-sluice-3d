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
            contact_stiffness: 1500.0,       // Moderate stiffness
            damping_ratio: 2.0,              // Overdamped (>1) - no oscillation, fast settling
            friction_coeff: 0.6,
            static_friction_threshold: 3.0,  // Below this: static friction regime
            sleep_threshold: 2.0,            // Below this + support: sleep
            wake_threshold: 5.0,             // Above this: wake up
            particle_radius_cells: 0.5,
            min_support_neighbors: 2,        // Need 2+ neighbors below to sleep
            velocity_damping: 0.95,          // 5% velocity reduction per frame
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
}

impl DemSimulation {
    pub fn new() -> Self {
        Self {
            params: DemParams::default(),
            forces: Vec::new(),
            support_counts: Vec::new(),
            is_sleeping: Vec::new(),
            frame: 0,
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

        // Reset forces
        self.forces.fill(Vec2::ZERO);
        self.support_counts.fill(0);

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

        // === Pass 3: Compute contact forces ===
        self.compute_contact_forces(
            particles,
            grid,
            cell_head,
            particle_next,
            contact_dist_sq,
            particle_radius,
        );

        // === Pass 4: Apply gravity and integrate ===
        self.integrate(particles, grid, dt, base_gravity, in_water, particle_radius);
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
                if speed > self.params.wake_threshold {
                    self.is_sleeping[idx] = false;
                }
            } else {
                // Currently awake - check for sleep condition
                if speed < self.params.sleep_threshold && (has_support || near_floor) {
                    self.is_sleeping[idx] = true;
                }
            }
        }
    }

    /// Compute contact forces between particles and with floor
    fn compute_contact_forces(
        &mut self,
        particles: &Particles,
        grid: &Grid,
        cell_head: &[i32],
        particle_next: &[i32],
        contact_dist_sq: f32,
        particle_radius: f32,
    ) {
        let cell_size = grid.cell_size;
        let width = grid.width;
        let height = grid.height;
        let contact_dist = contact_dist_sq.sqrt();

        let stiffness = self.params.contact_stiffness;
        let damping = self.params.damping_ratio;
        let friction = self.params.friction_coeff;
        let static_threshold = self.params.static_friction_threshold;

        for idx in 0..particles.len() {
            let p_i = &particles.list[idx];
            if !p_i.is_sediment() {
                continue;
            }

            // Sleeping particles still receive forces (to detect wake conditions)
            // but we can skip some computations

            let pos_i = p_i.position;
            let vel_i = if self.is_sleeping[idx] { Vec2::ZERO } else { p_i.velocity };
            let density_i = p_i.material.density();

            let gi = ((pos_i.x / cell_size) as i32).clamp(0, width as i32 - 1);
            let gj = ((pos_i.y / cell_size) as i32).clamp(0, height as i32 - 1);

            // === Particle-particle contacts ===
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

                        // Only process pairs once
                        if j_idx <= idx {
                            continue;
                        }

                        let p_j = &particles.list[j_idx];
                        if !p_j.is_sediment() {
                            continue;
                        }

                        let pos_j = p_j.position;
                        let vel_j = if self.is_sleeping[j_idx] { Vec2::ZERO } else { p_j.velocity };
                        let density_j = p_j.material.density();

                        let diff = pos_i - pos_j;
                        let dist_sq = diff.length_squared();

                        if dist_sq >= contact_dist_sq || dist_sq < 0.0001 {
                            continue;
                        }

                        let dist = dist_sq.sqrt();
                        let overlap = contact_dist - dist;
                        let mut normal = diff / dist;

                        // Time-varying jitter for vertical contacts
                        // Prevents perfect horizontal sliding on stacked particles
                        if self.params.use_jitter && normal.y.abs() > 0.95 {
                            let seed = (self.frame as usize)
                                .wrapping_mul(17)
                                .wrapping_add(idx.wrapping_mul(7))
                                .wrapping_add(j_idx.wrapping_mul(13));
                            let jitter = ((seed % 100) as f32 / 100.0 - 0.5) * 0.3;
                            let jittered = Vec2::new(normal.x + jitter, normal.y);
                            normal = jittered.normalize();
                        }

                        // Relative velocity
                        let rel_vel = vel_i - vel_j;
                        let normal_vel = rel_vel.dot(normal);
                        let tangent_vel = rel_vel - normal * normal_vel;
                        let tangent_speed = tangent_vel.length();

                        // Spring-damper normal force
                        let spring_f = stiffness * overlap;
                        let damp_f = damping * 2.0 * stiffness.sqrt() * normal_vel;
                        let normal_force_mag = (spring_f - damp_f).max(0.0);

                        // Friction force (static vs dynamic regime)
                        let max_friction = friction * normal_force_mag;
                        let friction_force = if tangent_speed < static_threshold {
                            // Static friction: spring-like "sticking"
                            let stick_force = -tangent_vel * stiffness * 0.3;
                            let stick_mag = stick_force.length();
                            if stick_mag > max_friction {
                                stick_force * (max_friction / stick_mag)
                            } else {
                                stick_force
                            }
                        } else if tangent_speed > 0.001 {
                            // Dynamic friction: Coulomb sliding
                            -tangent_vel.normalize() * max_friction
                        } else {
                            Vec2::ZERO
                        };

                        let total_force = normal * normal_force_mag + friction_force;

                        // Mass-weighted distribution (lighter moves more)
                        let total_density = density_i + density_j;
                        let ratio_i = density_j / total_density;
                        let ratio_j = density_i / total_density;

                        self.forces[idx] += total_force * ratio_i;
                        self.forces[j_idx] -= total_force * ratio_j;
                    }
                }
            }

            // === Floor contact (SDF-based) ===
            let sdf = grid.sample_sdf(pos_i);
            if sdf < particle_radius && sdf > -particle_radius * 2.0 {
                let floor_normal = -grid.sdf_gradient(pos_i);
                let floor_overlap = particle_radius - sdf;

                if floor_overlap > 0.0 && floor_normal.length_squared() > 0.001 {
                    let floor_normal = floor_normal.normalize();

                    // Normal force
                    let normal_vel = vel_i.dot(floor_normal);
                    let spring_f = stiffness * 1.5 * floor_overlap; // Stiffer floor
                    let damp_f = damping * 2.0 * (stiffness * 1.5).sqrt() * normal_vel;
                    let floor_force_mag = (spring_f - damp_f).max(0.0);

                    self.forces[idx] += floor_normal * floor_force_mag;

                    // Floor friction
                    let tangent = Vec2::new(-floor_normal.y, floor_normal.x);
                    let tangent_vel = vel_i.dot(tangent);
                    let tangent_speed = tangent_vel.abs();

                    // Material-specific friction
                    let mat_friction = p_i.material.friction_coefficient();
                    let max_friction = mat_friction * floor_force_mag;

                    let friction_force = if tangent_speed < static_threshold {
                        // Static: resist all motion
                        -tangent * tangent_vel * stiffness * 0.3
                    } else {
                        // Dynamic: Coulomb
                        -tangent * tangent_vel.signum() * max_friction
                    };

                    // Clamp friction to not exceed tangent velocity elimination
                    let friction_mag = friction_force.length();
                    if friction_mag > max_friction {
                        self.forces[idx] += friction_force * (max_friction / friction_mag);
                    } else {
                        self.forces[idx] += friction_force;
                    }
                }
            }
        }
    }

    /// Apply gravity and integrate velocities/positions
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

            // Sleeping particles: zero velocity, skip integration
            if self.is_sleeping[idx] {
                p.velocity = Vec2::ZERO;
                // Update state to Bedload (settled)
                p.state = ParticleState::Bedload;
                continue;
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

            // Integrate position
            p.position += p.velocity * dt;

            // Resolve floor penetration
            let sdf = grid.sample_sdf(p.position);
            if sdf < particle_radius * 0.5 {
                let grad = grid.sdf_gradient(p.position);
                if grad.length_squared() > 0.001 {
                    let push_out = (particle_radius * 0.5 - sdf) * grad.normalize();
                    p.position -= push_out;
                }
            }

            // Update state based on velocity
            let sdf = grid.sample_sdf(p.position);
            let on_floor = sdf < particle_radius * 1.5;

            if on_floor && speed < self.params.sleep_threshold * 2.0 {
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
