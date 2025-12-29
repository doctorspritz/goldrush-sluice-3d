//! Particle advection and spatial hashing.
//!
//! Particle position updates, collision handling, and neighbor queries.

#![allow(dead_code)]

use super::FlipSimulation;
use crate::particle::{ParticleMaterial, ParticleState};
use glam::Vec2;
use rayon::prelude::*;

impl FlipSimulation {
    /// Step 8: Advect particles with SDF-based collision detection
    /// Uses precomputed signed distance field for O(1) collision queries
    /// Also projects suspended particles onto pile heightfield (pile as floor constraint)
    pub fn advect_particles_impl(&mut self, dt: f32) {
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
    /// Algorithm from Matthias Muller:
    /// - For each particle pair within 2*radius, compute penetration depth
    /// - Push both particles apart along their center line by half the overlap
    ///
    /// Collision with solids is handled by advect_particles, not here.
    pub fn push_particles_apart_impl(&mut self, iterations: usize) {
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
    pub fn build_spatial_hash_impl(&mut self) {
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
    pub fn compute_neighbor_counts_impl(&mut self) {
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
}
