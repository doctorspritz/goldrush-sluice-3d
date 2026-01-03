//! Sediment transport physics.
//!
//! Forces, settling, deposition, and entrainment.

#![allow(dead_code)]

use super::FlipSimulation;
use crate::grid::CellType;
use crate::particle::{
    hindered_settling_factor, neighbor_count_to_concentration, ParticleMaterial, ParticleState,
};
use crate::physics::GRAVITY;
use glam::Vec2;
use rand::Rng;
use rayon::prelude::*;

impl FlipSimulation {
    /// Step 7a: Apply sediment forces (settling + drag)
    ///
    /// Two regimes:
    /// 1. IN AIR: Free fall under gravity with light air resistance
    /// 2. IN WATER: Drift-flux settling toward terminal velocity
    ///
    /// Uses Ferguson-Church settling when enabled (more accurate for fine particles).
    /// Hindered settling reduces velocity in dense regions.
    pub fn apply_sediment_forces_impl(&mut self, dt: f32) {
        // Drag rate controls how fast particles approach target velocity
        // Higher = particles quickly match fluid + slip (more responsive)
        // Lower = particles maintain momentum longer (more inertial)
        const BASE_DRAG_RATE: f32 = 5.0;

        // Expected neighbors at "rest" (dilute) conditions
        // ~8 particles per 3x3 cell neighborhood is typical for dilute flow
        const REST_NEIGHBORS: f32 = 8.0;

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

    /// Step 8e: Apply DEM contact forces for ALL sediment particles
    ///
    /// All sediment particles get collision detection to prevent overlap.
    /// This replaces cell-based deposition with continuous particle piles.
    /// Water particles are excluded (handled by pressure solver).
    pub fn apply_dem_settling_impl(&mut self, dt: f32) {
        // === Contact Parameters ===
        // Contact spring stiffness (higher = stiffer contacts, less overlap)
        const CONTACT_STIFFNESS: f32 = 3000.0;
        // Damping ratio (0-1, higher = more energy dissipation, less bounce)
        const DAMPING_RATIO: f32 = 0.7;
        // Base friction coefficient (tangential resistance)
        const FRICTION_COEFF: f32 = 0.5;
        // Particle radius for contact detection
        const PARTICLE_RADIUS_CELLS: f32 = 0.5;
        // Velocity threshold for friction scaling (slow = more friction)
        const FRICTION_VELOCITY_THRESHOLD: f32 = 20.0; // px/s
        // Floor distance for enhanced friction
        const FLOOR_PROXIMITY_CELLS: f32 = 4.0;

        let cell_size = self.grid.cell_size;
        let width = self.grid.width;
        let height = self.grid.height;

        let particle_radius = PARTICLE_RADIUS_CELLS * cell_size;
        let contact_dist = particle_radius * 2.0;
        let contact_dist_sq = contact_dist * contact_dist;

        // Pre-allocate force buffer
        let particle_count = self.particles.list.len();
        let mut forces: Vec<Vec2> = vec![Vec2::ZERO; particle_count];

        // Compute contact forces between sediment particles
        for idx in 0..particle_count {
            let p_i = &self.particles.list[idx];

            // Only sediment particles participate in DEM
            if !p_i.is_sediment() {
                continue;
            }

            let pos_i = p_i.position;
            let vel_i = p_i.velocity;
            let speed_i = vel_i.length();

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
                        j = self.particle_next[j_idx];

                        // Only process pairs once and skip self
                        if j_idx <= idx {
                            continue;
                        }

                        // Other particle must also be sediment
                        let p_j = &self.particles.list[j_idx];
                        if !p_j.is_sediment() {
                            continue;
                        }
                        let pos_j = p_j.position;
                        let vel_j = p_j.velocity;

                        let diff = pos_i - pos_j;
                        let dist_sq = diff.length_squared();

                        if dist_sq >= contact_dist_sq || dist_sq < 0.0001 {
                            continue;
                        }

                        let dist = dist_sq.sqrt();
                        let overlap = contact_dist - dist;
                        let normal = diff / dist; // Points from j to i

                        // Relative velocity
                        let rel_vel = vel_i - vel_j;
                        let normal_vel = rel_vel.dot(normal);
                        let tangent_vel = rel_vel - normal * normal_vel;

                        // Spring-damper normal force - ALWAYS applies (collision)
                        let spring_force = CONTACT_STIFFNESS * overlap;
                        let damping_force =
                            DAMPING_RATIO * 2.0 * (CONTACT_STIFFNESS).sqrt() * normal_vel;
                        let normal_force_mag = (spring_force - damping_force).max(0.0);
                        let normal_force = normal * normal_force_mag;

                        // Friction scales with velocity: slow = grip, fast = slide
                        let avg_speed = (speed_i + p_j.velocity.length()) * 0.5;
                        let friction_scale =
                            (1.0 - (avg_speed / FRICTION_VELOCITY_THRESHOLD).min(1.0)).max(0.1);
                        let effective_friction = FRICTION_COEFF * friction_scale;

                        // Friction (Coulomb model)
                        let max_friction = effective_friction * normal_force_mag;
                        let tangent_speed = tangent_vel.length();
                        let friction_force = if tangent_speed > 0.001 {
                            let friction_dir = -tangent_vel / tangent_speed;
                            friction_dir * max_friction.min(tangent_speed * CONTACT_STIFFNESS * 0.5)
                        } else {
                            Vec2::ZERO
                        };

                        let total_force = normal_force + friction_force;

                        // Get densities for mass-weighted force distribution
                        let density_i = p_i.material.density();
                        let density_j = p_j.material.density();
                        let total_density = density_i + density_j;

                        // Distribute force inversely by mass (lighter particles move more)
                        // This makes heavier particles "sink through" lighter ones
                        let ratio_i = density_j / total_density; // lighter i -> bigger ratio
                        let ratio_j = density_i / total_density; // lighter j -> bigger ratio

                        // Collision forces always apply at full strength
                        forces[idx] += total_force * ratio_i;
                        forces[j_idx] -= total_force * ratio_j;

                        // Buoyancy-like sinking: heavy particles push down on light ones
                        // When density_diff > 0 (i heavier than j):
                        // - Particle i should sink DOWN (+Y direction)
                        // - Particle j should get pushed UP (-Y direction)
                        let density_diff = density_i - density_j;
                        if density_diff.abs() > 0.1 {
                            let buoyancy_strength = 50.0;
                            // +Y = down (gravity direction), so positive pushes heavy down
                            let buoyancy =
                                Vec2::new(0.0, buoyancy_strength * density_diff * overlap);
                            forces[idx] += buoyancy * ratio_i;
                            forces[j_idx] -= buoyancy * ratio_j;
                        }
                    }
                }
            }

            // Floor contact (SDF-based) - always applied
            let sdf_dist = self.grid.sample_sdf(pos_i);
            if sdf_dist < particle_radius && sdf_dist > -particle_radius {
                let floor_normal = -self.grid.sdf_gradient(pos_i); // Points away from solid
                let floor_overlap = particle_radius - sdf_dist;

                if floor_overlap > 0.0 {
                    // Normal force against floor - always full strength
                    let floor_normal_vel = vel_i.dot(floor_normal);
                    let spring_f = CONTACT_STIFFNESS * floor_overlap * 1.5; // Stiffer floor
                    let damp_f =
                        DAMPING_RATIO * 2.0 * (CONTACT_STIFFNESS * 1.5).sqrt() * floor_normal_vel;
                    let floor_force_mag = (spring_f - damp_f).max(0.0);

                    forces[idx] += floor_normal * floor_force_mag;

                    // Floor friction - scales with velocity (slow = grip, fast = slide)
                    // Also stronger when closer to floor
                    let floor_dist_cells = sdf_dist / cell_size;
                    let proximity_factor =
                        (1.0 - (floor_dist_cells / FLOOR_PROXIMITY_CELLS).max(0.0)).max(0.0);
                    let velocity_factor =
                        (1.0 - (speed_i / FRICTION_VELOCITY_THRESHOLD).min(1.0)).max(0.2);
                    let friction_strength = proximity_factor * velocity_factor;

                    let tangent = Vec2::new(-floor_normal.y, floor_normal.x);
                    let tangent_vel = vel_i.dot(tangent);
                    let floor_friction = FRICTION_COEFF * floor_force_mag * friction_strength;
                    if tangent_vel.abs() > 0.001 {
                        forces[idx] -= tangent
                            * tangent_vel.signum()
                            * floor_friction.min(tangent_vel.abs() * CONTACT_STIFFNESS * friction_strength);
                    }
                }
            }
        }

        // Apply forces to velocities for all sediment particles
        let mass = 1.0; // Normalized mass
        let max_vel = cell_size / dt * 0.5; // Max velocity for stability

        for idx in 0..particle_count {
            if forces[idx] != Vec2::ZERO && self.particles.list[idx].is_sediment() {
                let accel = forces[idx] / mass;
                self.particles.list[idx].velocity += accel * dt;

                // Clamp velocity to prevent instability
                let speed = self.particles.list[idx].velocity.length();
                if speed > max_vel {
                    self.particles.list[idx].velocity *= max_vel / speed;
                }
            }
        }
    }

    /// Step 8f: Deposit settled sediment onto terrain (Multi-Material)
    ///
    /// Detects sediment particles that have truly settled (very low velocity,
    /// in dense regions near floor) and converts cells to solid when enough
    /// particles accumulate. Tracks material composition for mixed beds.
    ///
    /// This implements deposition with material tracking for multi-sediment support.
    pub fn deposit_settled_sediment_impl(&mut self, dt: f32) {
        use std::sync::atomic::{AtomicUsize, Ordering};

        // === Thresholds ===
        // Velocity threshold: must be truly at rest (very low)
        const VELOCITY_THRESHOLD: f32 = 0.05; // cells/frame - tighter for stable deposits
        // Distance to floor for deposition eligibility
        const FLOOR_DISTANCE_CELLS: f32 = 1.0; // Must be right on floor
        // Minimum neighbors for "packed" state (prevents isolated deposits)
        const MIN_NEIGHBORS: u16 = 3;
        // Mass required to convert one cell to solid
        const MASS_PER_CELL: f32 = 4.0; // Tighter packing with DEM

        let cell_size = self.grid.cell_size;
        let width = self.grid.width;
        let height = self.grid.height;
        let v_scale = cell_size / dt; // Convert cells/frame to velocity units

        // Atomic counters for particle count per cell PER MATERIAL TYPE (parallel-safe)
        let mud_counts: Vec<AtomicUsize> = (0..width * height)
            .map(|_| AtomicUsize::new(0))
            .collect();
        let sand_counts: Vec<AtomicUsize> = (0..width * height)
            .map(|_| AtomicUsize::new(0))
            .collect();
        let magnetite_counts: Vec<AtomicUsize> = (0..width * height)
            .map(|_| AtomicUsize::new(0))
            .collect();
        let gold_counts: Vec<AtomicUsize> = (0..width * height)
            .map(|_| AtomicUsize::new(0))
            .collect();

        // Track which particles to remove (by index)
        let remove_flags: Vec<std::sync::atomic::AtomicBool> = (0..self.particles.list.len())
            .map(|_| std::sync::atomic::AtomicBool::new(false))
            .collect();

        let grid = &self.grid;
        let neighbor_counts = &self.neighbor_counts;

        // Phase 1: Identify settled particles and count per cell BY MATERIAL
        self.particles
            .list
            .par_iter()
            .enumerate()
            .for_each(|(idx, particle)| {
                // Only deposit sediment particles
                if !particle.is_sediment() {
                    return;
                }

                // Check velocity (normalized to cells/frame)
                let vel_normalized = particle.velocity.length() / v_scale;
                if vel_normalized > VELOCITY_THRESHOLD {
                    return; // Moving too fast
                }

                // Check distance to floor
                let sdf_dist = grid.sample_sdf(particle.position);
                let floor_dist = FLOOR_DISTANCE_CELLS * cell_size;
                if sdf_dist > floor_dist {
                    return; // Too far from floor
                }

                // Check floor surface (gradient pointing up = floor, not wall)
                let grad = grid.sdf_gradient(particle.position);
                if grad.y > -0.3 {
                    return; // Wall or ceiling, not floor
                }

                // Find the cell to deposit into (cell containing particle)
                let ci = ((particle.position.x / cell_size) as usize).min(width - 1);
                let cj = ((particle.position.y / cell_size) as usize).min(height - 1);
                let cell_idx = cj * width + ci;

                // Don't deposit into already-solid cells
                if grid.is_solid(ci, cj) {
                    return;
                }

                // Check neighbor count (use pre-computed from compute_neighbor_counts)
                // Only deposit if particle is in a packed region
                if idx < neighbor_counts.len() && neighbor_counts[idx] < MIN_NEIGHBORS {
                    return; // Not packed enough
                }

                // Count this particle BY MATERIAL TYPE and mark for potential removal
                match particle.material {
                    ParticleMaterial::Mud => {
                        mud_counts[cell_idx].fetch_add(1, Ordering::Relaxed);
                    }
                    ParticleMaterial::Sand => {
                        sand_counts[cell_idx].fetch_add(1, Ordering::Relaxed);
                    }
                    ParticleMaterial::Magnetite => {
                        magnetite_counts[cell_idx].fetch_add(1, Ordering::Relaxed);
                    }
                    ParticleMaterial::Gold => {
                        gold_counts[cell_idx].fetch_add(1, Ordering::Relaxed);
                    }
                    ParticleMaterial::Water | ParticleMaterial::Gravel => {} // Not depositing materials
                }
                remove_flags[idx].store(true, Ordering::Relaxed);
            });

        // Phase 2: Convert cells with enough particles to solid (single dominant material)
        // Collect which cells to convert (to avoid borrow issues)
        let mut cells_to_convert: Vec<(usize, usize, usize, ParticleMaterial)> = Vec::new();
        for j in 0..height {
            for i in 0..width {
                let idx = j * width + i;

                // Add this frame's counts to accumulated mass
                let mud_count = mud_counts[idx].load(Ordering::Relaxed) as f32;
                let sand_count = sand_counts[idx].load(Ordering::Relaxed) as f32;
                let magnetite_count = magnetite_counts[idx].load(Ordering::Relaxed) as f32;
                let gold_count = gold_counts[idx].load(Ordering::Relaxed) as f32;

                self.deposited_mass_mud[idx] += mud_count;
                self.deposited_mass_sand[idx] += sand_count;
                self.deposited_mass_magnetite[idx] += magnetite_count;
                self.deposited_mass_gold[idx] += gold_count;

                // Check if we have enough total mass to solidify
                let total_mass = self.deposited_mass_mud[idx]
                    + self.deposited_mass_sand[idx]
                    + self.deposited_mass_magnetite[idx]
                    + self.deposited_mass_gold[idx];

                if total_mass >= MASS_PER_CELL {
                    // Pick the DOMINANT material (highest mass fraction)
                    // This enables correct selective entrainment:
                    // - Sand cells have Shields 0.045 (easy to entrain)
                    // - Magnetite cells have Shields 0.07 (harder)
                    // - Gold cells have Shields 0.09 (hardest)
                    let masses = [
                        (self.deposited_mass_mud[idx], ParticleMaterial::Mud),
                        (self.deposited_mass_sand[idx], ParticleMaterial::Sand),
                        (
                            self.deposited_mass_magnetite[idx],
                            ParticleMaterial::Magnetite,
                        ),
                        (self.deposited_mass_gold[idx], ParticleMaterial::Gold),
                    ];
                    let dominant_material = masses
                        .iter()
                        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                        .map(|(_, mat)| *mat)
                        .unwrap_or(ParticleMaterial::Sand);

                    cells_to_convert.push((i, j, idx, dominant_material));

                    // Clear ALL accumulated mass for this cell (clean slate)
                    self.deposited_mass_mud[idx] = 0.0;
                    self.deposited_mass_sand[idx] = 0.0;
                    self.deposited_mass_magnetite[idx] = 0.0;
                    self.deposited_mass_gold[idx] = 0.0;
                }
            }
        }

        // Convert cells to deposited with single material
        for (i, j, _idx, material) in &cells_to_convert {
            self.grid.set_deposited_with_material(*i, *j, *material);
        }

        let cells_converted = cells_to_convert.len();

        // Phase 3: Remove particles that were in converted cells
        if cells_converted > 0 {
            // Build set of converted cell indices for fast lookup
            let converted_set: std::collections::HashSet<usize> =
                cells_to_convert.iter().map(|(_, _, idx, _)| *idx).collect();

            // Remove particles that were marked AND are in converted cells
            let particles = &mut self.particles.list;
            let mut write_idx = 0;
            for read_idx in 0..particles.len() {
                let should_remove = if remove_flags[read_idx].load(Ordering::Relaxed) {
                    let p = &particles[read_idx];
                    let ci = ((p.position.x / cell_size) as usize).min(width - 1);
                    let cj = ((p.position.y / cell_size) as usize).min(height - 1);
                    let cell_idx = cj * width + ci;
                    converted_set.contains(&cell_idx)
                } else {
                    false
                };

                if !should_remove {
                    if write_idx != read_idx {
                        particles.swap(write_idx, read_idx);
                    }
                    write_idx += 1;
                }
            }
            particles.truncate(write_idx);
        }

        // Debug output (every 120 frames)
        if self.frame % 120 == 0 && cells_converted > 0 {
            eprintln!(
                "[Deposition] Converted {} cells to solid (single-material)",
                cells_converted
            );
        }
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
    /// - Gold cells (Shields 0.09) resist 2x more than sand
    pub fn entrain_deposited_sediment_impl(&mut self, dt: f32) {
        // === Thresholds ===
        const BASE_CRITICAL_VELOCITY: f32 = 0.5; // cells/frame for sand (Shields = 0.045)
        const BASE_SHIELDS: f32 = 0.045; // Reference Shields for sand
        const BASE_ENTRAINMENT_RATE: f32 = 1.0; // probability scaling - max
        const MAX_PROBABILITY: f32 = 0.95; // cap per frame - nearly instant
        const PARTICLES_PER_CELL: usize = 4; // matches MASS_PER_CELL

        let cell_size = self.grid.cell_size;
        let width = self.grid.width;
        let height = self.grid.height;
        let v_scale = cell_size / dt; // Convert to cells/frame

        // Store cells and their material for spawning
        struct CellToEntrain {
            i: usize,
            j: usize,
            material: ParticleMaterial,
            vel_above: Vec2,
        }
        let mut cells_to_clear: Vec<CellToEntrain> = Vec::new();

        let mut rng = rand::thread_rng();

        // Iterate deposited cells
        for j in 1..height - 1 {
            for i in 1..width - 1 {
                if !self.grid.is_deposited(i, j) {
                    continue;
                }

                // Get cell's material for exact Shields threshold
                let composition = self.grid.deposited[self.grid.cell_index(i, j)];
                let material = match composition.get_material() {
                    Some(mat) => mat,
                    None => continue, // No material, skip
                };
                let effective_shields = material.shields_critical();

                // Scale critical velocity by Shields ratio
                // Higher Shields = harder to entrain = need faster velocity
                let critical_velocity = BASE_CRITICAL_VELOCITY * (effective_shields / BASE_SHIELDS);

                // Sample velocity just above the cell
                let sample_pos = Vec2::new(
                    (i as f32 + 0.5) * cell_size,
                    (j as f32 - 0.5) * cell_size, // Half cell above
                );
                let vel_above = self.grid.sample_velocity(sample_pos);
                let speed = vel_above.length() / v_scale; // cells/frame

                if speed <= critical_velocity {
                    continue; // Not fast enough for this material
                }

                // Compute entrainment probability
                let excess_ratio = speed / critical_velocity - 1.0;
                let probability = (BASE_ENTRAINMENT_RATE * excess_ratio).min(MAX_PROBABILITY);

                // Stochastic check
                if rng.gen::<f32>() >= probability {
                    continue; // Not entrained this frame
                }

                // Check support: don't entrain if it would leave floating cells above
                let has_deposit_above = j > 0 && self.grid.is_deposited(i, j - 1);
                if has_deposit_above {
                    let left_support = i > 0 && self.grid.is_solid(i - 1, j - 1);
                    let right_support = i < width - 1 && self.grid.is_solid(i + 1, j - 1);
                    if !left_support && !right_support {
                        continue; // Would create floating deposit
                    }
                }

                // Store cell info for entrainment
                cells_to_clear.push(CellToEntrain {
                    i,
                    j,
                    material,
                    vel_above,
                });
            }
        }

        // Clear cells and spawn particles with the cell's material type
        for cell in &cells_to_clear {
            self.grid.clear_deposited(cell.i, cell.j);

            // Reset accumulated mass for this cell
            let idx = cell.j * width + cell.i;
            self.deposited_mass_mud[idx] = 0.0;
            self.deposited_mass_sand[idx] = 0.0;
            self.deposited_mass_magnetite[idx] = 0.0;
            self.deposited_mass_gold[idx] = 0.0;

            // Spawn particles of the cell's single material type
            let cell_center = Vec2::new(
                (cell.i as f32 + 0.5) * cell_size,
                (cell.j as f32 + 0.5) * cell_size,
            );

            // Entrained particles get flow velocity with small random perturbation
            // No violent upward kick - just inherit the flow that eroded them
            let base_vel = cell.vel_above * 0.7; // Inherit 70% of flow velocity

            // Spawn all particles of the same material
            for _ in 0..PARTICLES_PER_CELL {
                let jitter = Vec2::new(
                    (rng.gen::<f32>() - 0.5) * cell_size * 0.6,
                    (rng.gen::<f32>() - 0.5) * cell_size * 0.6,
                );
                let pos = cell_center + jitter;
                let vel_jitter = Vec2::new(
                    (rng.gen::<f32>() - 0.5) * 10.0,
                    (rng.gen::<f32>() - 0.5) * 10.0,
                );
                let vel = base_vel + vel_jitter;

                match cell.material {
                    ParticleMaterial::Mud => self.particles.spawn_mud(pos.x, pos.y, vel.x, vel.y),
                    ParticleMaterial::Sand => self.particles.spawn_sand(pos.x, pos.y, vel.x, vel.y),
                    ParticleMaterial::Magnetite => {
                        self.particles.spawn_magnetite(pos.x, pos.y, vel.x, vel.y)
                    }
                    ParticleMaterial::Gold => self.particles.spawn_gold(pos.x, pos.y, vel.x, vel.y),
                    ParticleMaterial::Water | ParticleMaterial::Gravel => {} // Should never happen
                }
            }
        }

        // Update SDF if terrain changed
        if !cells_to_clear.is_empty() {
            self.grid.compute_sdf();
            self.grid.compute_bed_heights();

            // Debug output
            if self.frame % 60 == 0 {
                eprintln!(
                    "[Entrainment] Eroded {} cells (single-material)",
                    cells_to_clear.len()
                );
            }
        }
    }

    /// Step 8h: Collapse and avalanche deposited sediment
    ///
    /// Ensures deposited cells have support underneath (won't float) and
    /// spread according to angle of repose (no steep cliffs).
    /// This fixes gaps and artifacts in sediment piles.
    /// Preserves material composition when moving cells.
    pub fn collapse_deposited_sediment_impl(&mut self) {
        const MAX_ITERATIONS: usize = 50;
        // Height difference (in cells) before avalanche triggers
        // Higher = steeper piles allowed. 2.0 means 2 cells taller than neighbor before spreading
        const MAX_HEIGHT_DIFF: f32 = 2.0;
        // Minimum pile height before avalanche logic applies (let small piles accumulate)
        const MIN_PILE_HEIGHT: usize = 3;

        let width = self.grid.width;
        let height = self.grid.height;

        let mut cells_changed = true;
        let mut iteration = 0;
        let mut total_collapses = 0;
        let mut total_avalanches = 0;

        while cells_changed && iteration < MAX_ITERATIONS {
            cells_changed = false;
            iteration += 1;

            // Phase 1: Support check (bottom-up to handle chains)
            // Scan from bottom to top so collapsed cells can support ones above
            for j in (1..height - 1).rev() {
                for i in 1..width - 1 {
                    if !self.grid.is_deposited(i, j) {
                        continue;
                    }

                    // Check support: cell below must be solid
                    if !self.grid.is_solid(i, j + 1) {
                        // Find landing position (first supported cell below)
                        let mut target_j = j + 1;
                        while target_j + 1 < height && !self.grid.is_solid(i, target_j + 1) {
                            target_j += 1;
                        }

                        // Only collapse if we have a valid landing spot
                        if target_j < height && !self.grid.is_solid(i, target_j) {
                            // Preserve material when moving
                            let source_idx = self.grid.cell_index(i, j);
                            let composition = self.grid.deposited[source_idx];
                            if let Some(material) = composition.get_material() {
                                self.grid.clear_deposited(i, j);
                                self.grid.set_deposited_with_material(i, target_j, material);
                                total_collapses += 1;
                                cells_changed = true;
                            }
                        }
                        // If no valid landing spot, leave the cell where it is
                    }
                }
            }

            // Phase 2: Angle of repose check
            // Move material from steep piles to the LOWEST neighbor only
            // Process each column once, not each cell
            for i in 1..width - 1 {
                let my_height = self.count_column_deposited_impl(i);

                // Skip if pile is too small (let it accumulate first)
                if my_height < MIN_PILE_HEIGHT {
                    continue;
                }

                // Check both neighbors, avalanche to the LOWEST one only
                let left_height = if i > 1 {
                    self.count_column_deposited_impl(i - 1)
                } else {
                    usize::MAX // Can't avalanche left at edge
                };

                let right_height = if i < width - 2 {
                    self.count_column_deposited_impl(i + 1)
                } else {
                    usize::MAX // Can't avalanche right at edge
                };

                // Find the lowest neighbor
                let (target_i, target_height) = if left_height <= right_height {
                    (i - 1, left_height)
                } else {
                    (i + 1, right_height)
                };

                // Check if height difference exceeds threshold
                if target_height < usize::MAX
                    && my_height as f32 - target_height as f32 > MAX_HEIGHT_DIFF
                {
                    // Avalanche one cell to the lower neighbor
                    if let Some(top_j) = self.find_top_deposited_in_column_impl(i) {
                        let land_j = self.find_landing_j_impl(target_i);
                        if land_j < height && !self.grid.is_solid(target_i, land_j) {
                            // Preserve material when avalanching
                            let source_idx = self.grid.cell_index(i, top_j);
                            let composition = self.grid.deposited[source_idx];
                            if let Some(material) = composition.get_material() {
                                self.grid.clear_deposited(i, top_j);
                                self.grid.set_deposited_with_material(target_i, land_j, material);
                                total_avalanches += 1;
                                cells_changed = true;
                            }
                        }
                    }
                }
            }
        }

        // Update SDF if anything changed
        if total_collapses > 0 || total_avalanches > 0 {
            self.grid.compute_sdf();
            self.grid.compute_bed_heights();
        }

        // Suppress unused variable warnings in release builds
        let _ = (iteration, total_collapses, total_avalanches);
    }

    /// Count deposited cells in a column (from bottom up)
    pub fn count_column_deposited_impl(&self, i: usize) -> usize {
        let mut count = 0;
        for j in 0..self.grid.height {
            if self.grid.is_deposited(i, j) {
                count += 1;
            }
        }
        count
    }

    /// Find the topmost deposited cell in a column (lowest j value)
    pub fn find_top_deposited_in_column_impl(&self, i: usize) -> Option<usize> {
        for j in 0..self.grid.height {
            if self.grid.is_deposited(i, j) {
                return Some(j);
            }
        }
        None
    }

    /// Find landing j position for a falling/avalanching cell
    pub fn find_landing_j_impl(&self, i: usize) -> usize {
        // Start from bottom and find first empty cell with support below
        for j in (0..self.grid.height).rev() {
            if !self.grid.is_solid(i, j) {
                // Check if supported (at bottom or has solid below)
                if j + 1 >= self.grid.height || self.grid.is_solid(i, j + 1) {
                    return j;
                }
            }
        }
        self.grid.height - 1 // Bottom of grid
    }

    /// Step 7b: Update particle states (Suspended <-> Bedload) using Rouse number
    ///
    /// The Rouse number determines transport mode based on physics:
    /// - Rouse = w_s / (κ * u*) where:
    ///   - w_s = settling velocity (from Ferguson-Church)
    ///   - κ = von Karman constant (0.4)
    ///   - u* = shear velocity (derived from velocity gradient)
    ///
    /// Transport regimes:
    /// - Rouse > 2.5: Bedload (particles stay near bottom)
    /// - Rouse < 0.8: Suspended (particles distributed through depth)
    /// - 0.8 - 2.5: Transitional (blend of both behaviors)
    ///
    /// This naturally handles density differentiation:
    /// - Gold (high settling velocity) → high Rouse → bedload
    /// - Sand (low settling velocity) → low Rouse in fast flow → suspended
    pub fn update_particle_states_impl(&mut self, dt: f32) {
        // === Rouse Number Constants ===
        const VON_KARMAN: f32 = 0.4; // κ - von Karman constant
        const ROUSE_BEDLOAD: f32 = 2.5; // Above this → definitely bedload
        const ROUSE_SUSPENDED: f32 = 0.8; // Below this → definitely suspended

        // Hysteresis to prevent rapid state flickering
        const ROUSE_JAM_HYSTERESIS: f32 = 0.3; // Extra margin to enter bedload
        const ROUSE_UNJAM_HYSTERESIS: f32 = 0.5; // Extra margin to leave bedload
        const MIN_JAM_TIME: f32 = 0.15; // Minimum time before unjamming

        // Minimum shear velocity to prevent divide-by-zero
        // Also prevents false bedload in stagnant water
        const MIN_SHEAR_VELOCITY: f32 = 0.5; // pixels/s

        // Velocity threshold for jamming (must be nearly stopped)
        const JAM_VEL_THRESHOLD: f32 = 5.0; // pixels/s

        let grid = &self.grid;
        let cell_size = grid.cell_size;
        let pile_height = &self.pile_height;
        let pile_width = pile_height.len();

        // Near floor distance: 1.5x cell size to catch particles resting on floor
        let near_floor_distance = cell_size * 1.5;

        // Copy flags for closure
        let use_variable_diameter = self.use_variable_diameter;

        self.particles.list.par_iter_mut().for_each(|particle| {
            // Only update sediment states
            if !particle.is_sediment() {
                return;
            }

            // === COMPUTE SHEAR VELOCITY ===
            // Shear velocity u* = sqrt(τ_b / ρ) ≈ velocity gradient at bed
            // We use velocity difference over one cell as approximation
            let fluid_vel_here = particle.old_grid_velocity;
            let pos_below = particle.position + Vec2::new(0.0, cell_size);
            let fluid_vel_below = grid.sample_velocity(pos_below);
            let velocity_gradient = (fluid_vel_here - fluid_vel_below).length();

            // Shear velocity estimate (simple approximation)
            // For log-law velocity profile: u* = κ * du/dz * z
            // Simplified: u* ≈ velocity_gradient * 0.5
            let shear_velocity = (velocity_gradient * 0.5).max(MIN_SHEAR_VELOCITY);

            // === COMPUTE SETTLING VELOCITY ===
            let diameter = if use_variable_diameter {
                particle.effective_diameter()
            } else {
                particle.material.typical_diameter()
            };
            let settling_velocity = particle.material.settling_velocity(diameter);

            // === COMPUTE ROUSE NUMBER ===
            // Rouse = w_s / (κ * u*)
            // Higher Rouse = more likely to be bedload (heavy particles in slow flow)
            // Lower Rouse = more likely to be suspended (light particles in fast flow)
            let rouse = settling_velocity / (VON_KARMAN * shear_velocity);

            // Relative velocity for jam check
            let rel_vel = (particle.velocity - fluid_vel_here).length();

            // === SUPPORT CHECK ===
            // Particles can only be bedload if they're supported by something:
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
            let col = ((particle.position.x / cell_size) as usize).min(pile_width.saturating_sub(1));
            let pile_top = pile_height[col];
            let particle_radius = cell_size * 0.5;
            let particle_bottom = particle.position.y + particle_radius;
            let on_pile = pile_top < f32::INFINITY
                && particle_bottom >= pile_top
                && (particle_bottom - pile_top) < cell_size;

            let has_support = on_solid_floor || on_pile;

            match particle.state {
                ParticleState::Suspended => {
                    // === JAM (Enter Bedload) ===
                    // Conditions:
                    // 1. Has support (on floor or pile)
                    // 2. Rouse number indicates bedload regime (with hysteresis)
                    // 3. Particle is moving slowly (near rest)
                    let rouse_threshold = ROUSE_BEDLOAD - ROUSE_JAM_HYSTERESIS;
                    if has_support && rouse > rouse_threshold && rel_vel < JAM_VEL_THRESHOLD {
                        particle.state = ParticleState::Bedload;
                        particle.jam_time = 0.0;
                    }
                }
                ParticleState::Bedload => {
                    // === LOSS-OF-SUPPORT UNJAM (hard constraint) ===
                    // Bedload MUST unjam if it loses support - no floating bedload
                    if !has_support {
                        particle.state = ParticleState::Suspended;
                        particle.velocity = fluid_vel_here; // Resume with local fluid velocity
                        particle.jam_time = 0.0;
                        return;
                    }

                    // Increment jam time
                    particle.jam_time += dt;

                    // === ROUSE-BASED UNJAM ===
                    // If Rouse number drops below suspended threshold, flow is strong
                    // enough to entrain this particle
                    let unjam_threshold = ROUSE_SUSPENDED + ROUSE_UNJAM_HYSTERESIS;
                    if rouse < unjam_threshold && particle.jam_time > MIN_JAM_TIME {
                        particle.state = ParticleState::Suspended;
                        // Give particle some of the fluid velocity to help it move
                        particle.velocity = fluid_vel_here * 0.5;
                    }
                }
            }
        });
    }
}
