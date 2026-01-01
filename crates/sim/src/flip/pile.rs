#![allow(dead_code)]
//! Pile heightfield and particle state management.
//!
//! Bedload pile tracking and state transitions.
//!
//! NOTE: These methods are currently still defined in mod.rs.
//! This file contains copies ready for activation when mod.rs methods are removed.

// Imports needed when these methods are activated
#[allow(unused_imports)]
use super::FlipSimulation;
#[allow(unused_imports)]
use crate::particle::{ParticleMaterial, ParticleState};
#[allow(unused_imports)]
use glam::Vec2;
#[allow(unused_imports)]
use rayon::prelude::*;

// =============================================================================
// STAGED IMPLEMENTATIONS - Activate by uncommenting and removing from mod.rs
// =============================================================================

/*
impl FlipSimulation {
    /// Step 8: Update particle states (Suspended <-> Bedload)
    ///
    /// State transitions based on:
    /// - Enter Bedload: velocity < threshold AND near floor (SDF < radius)
    /// - Exit Bedload: Shields number > unjam threshold AND jam_time > MIN_JAM_TIME
    ///
    /// Hysteresis prevents rapid flickering between states:
    /// - UNJAM_THRESHOLD is 2-3x JAM_THRESHOLD
    /// - MIN_JAM_TIME requires particles to be bedload for some time before unjamming
    /// - Shear deadzone shields bedload particles from small velocity fluctuations
    pub fn update_particle_states(&mut self, dt: f32) {
        // Thresholds in normalized units (cells-per-frame)
        // Lowered to make sand easier to re-entrain by water flow
        const JAM_THRESHOLD: f32 = 0.08;    // Was 0.15 - slower before jamming
        const UNJAM_THRESHOLD: f32 = 0.12;  // Was 0.40 - easier to unjam
        const SHEAR_DEADZONE: f32 = 0.02;   // Was 0.08 - more responsive
        const MIN_JAM_TIME: f32 = 0.10;     // Was 0.20 - faster unjam
        const JAM_VEL_MAX: f32 = 0.03;      // Was 0.05 - stricter jam

        // Per-material shear resistance multiplier
        // Higher = harder to entrain (requires more shear)
        // Relative to sand's Shields parameter (0.045)
        fn material_shear_factor(mat: ParticleMaterial) -> f32 {
            match mat {
                ParticleMaterial::Water => 1.0,     // N/A for water
                ParticleMaterial::Mud => 0.67,      // Shields 0.03 / 0.045 - easier to entrain
                ParticleMaterial::Sand => 1.0,      // Reference material
                ParticleMaterial::Magnetite => 1.56, // Shields 0.07 / 0.045 - much harder to entrain
                ParticleMaterial::Gold => 2.0,      // Shields 0.09 / 0.045 - hardest to entrain
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
    pub fn compute_pile_heightfield(&mut self) {
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

        // Smooth over +-1 column to avoid stair-stepping
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
    pub fn enforce_pile_constraints(&mut self) {
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

    /// Step 9: Push overlapping particles apart (FLIP-native separation)
    ///
    /// Simple overlap check that only does work when particles actually overlap.
    /// Standard approach used by production FLIP solvers like Houdini.
    ///
    /// Algorithm from Matthias Muller:
    /// - For each particle pair within 2*radius, compute penetration depth
    /// - Push both particles apart along their center line by half the overlap
    ///
    /// Collision with solids is handled by advect_particles, not here.
    pub fn push_particles_apart(&mut self, iterations: usize) {
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
}
*/
