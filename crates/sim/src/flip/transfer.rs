#![allow(dead_code)]
//! Particle-grid transfer (P2G, G2P).
//!
//! Core APIC transfer operations using quadratic B-spline kernels.
//!
//! ## Methods
//! - `particles_to_grid`: P2G with APIC affine momentum
//! - `store_old_velocities`: Save pre-force grid velocities for FLIP delta
//! - `grid_to_particles`: G2P with FLIP/PIC blending

use super::FlipSimulation;
use crate::grid::{apic_d_inverse, quadratic_bspline, quadratic_bspline_1d, CellType};
use crate::particle::ParticleState;
use glam::{Mat2, Vec2};
use rayon::prelude::*;

impl FlipSimulation {
    /// Step 1: APIC Particle-to-Grid transfer (P2G)
    ///
    /// For each particle, transfer momentum + affine terms to nearby grid nodes.
    /// Uses quadratic B-spline with 3x3 stencil (9 nodes per component).
    ///
    /// APIC transfer formula:
    ///   momentum_i += w_ip * (v_p + C_p * (x_i - x_p))
    ///   mass_i += w_ip
    ///
    /// Final grid velocity: v_i = momentum_i / mass_i
    ///
    /// Optimization: Precompute 1D weights to reduce bspline calls from 18 to 12 per particle.
    pub fn particles_to_grid_impl(&mut self) {
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
        // Clear weighted density accumulators (material-aware mixture density)
        self.sediment_density_sum_u.fill(0.0);
        self.sediment_density_sum_v.fill(0.0);

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
                        // Track weighted density for material-aware mixture density
                        self.sediment_density_sum_u[idx] += w * particle_volume * particle.material.density();
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
                        // Track weighted density for material-aware mixture density
                        self.sediment_density_sum_v[idx] += w * particle_volume * particle.material.density();
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

    /// Step 3: Store old grid velocities for FLIP calculation
    ///
    /// CRITICAL: Must use quadratic B-spline sampling (same as G2P) to avoid
    /// kernel mismatch that causes phantom velocity deltas.
    /// See: plans/flip-damping-diagnosis.md
    ///
    /// Optimizations:
    /// - Parallelized with rayon (particles are independent)
    /// - Precomputed 1D weights (reduces bspline calls from 18 to 12)
    pub fn store_old_velocities_impl(&mut self) {
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
    pub fn grid_to_particles_impl(&mut self, dt: f32) {
        let cell_size = self.grid.cell_size;
        let width = self.grid.width;
        let height = self.grid.height;
        let d_inv = apic_d_inverse(cell_size);
        // Note: sand_pic_ratio no longer used - replaced with drag-based coupling

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
                    // IN FLUID: Drag-based coupling with settling
                    //
                    // Physics: Instead of overriding velocity with PIC blend, we apply
                    // drag forces that pull the particle toward the water velocity.
                    // Heavy particles (gold) have low drag → resist water, settle despite flow.
                    // Light particles (sand) have high drag → follow water, wash away.
                    //
                    // This preserves settling velocity and friction effects that would
                    // otherwise be erased by the 70% PIC override every frame.

                    // Drag coefficient inversely proportional to density ratio
                    // water/gold ≈ 0.05 → gold resists water strongly
                    // water/sand ≈ 0.38 → sand follows water more readily
                    const RHO_WATER: f32 = 1.0;
                    const DRAG_STRENGTH: f32 = 10.0; // Tunable: controls coupling timescale
                    let rho_particle = particle.material.density();
                    let drag_coeff = RHO_WATER / rho_particle;

                    // Drag rate: fraction of velocity difference applied per frame
                    // Clamped to prevent over-damping (max 0.9 per frame for stability)
                    let drag_rate = (drag_coeff * DRAG_STRENGTH * dt).min(0.9);

                    // Apply drag toward water velocity (HORIZONTAL ONLY)
                    // Heavy particles (low drag_rate) resist horizontal flow → stay in riffles
                    // Light particles (high drag_rate) follow horizontal flow → wash away
                    //
                    // VERTICAL motion is handled ONLY by the vorticity settling section below.
                    // This ensures heavy particles settle faster regardless of drag_rate.
                    particle.velocity.x += (v_grid.x - particle.velocity.x) * drag_rate;
                    // NO vertical drag here - settling is material-specific in vorticity section

                    particle.old_grid_velocity = v_grid;
                } else {
                    // IN AIR: Don't apply water drag (no meaningful water velocity)
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

                // ========== VORTICITY-BASED SUSPENSION ==========
                // High vorticity regions have turbulent eddies that carry particles upward
                // This creates realistic suspension where sand swirls in fast-moving water
                // instead of just dragging along the bottom.
                //
                // Physics: Rouse number P = ws / (κ * u*) determines suspension
                // - P < 2.5: suspended load (turbulence can support particle)
                // - P > 2.5: bedload (particle settles despite turbulence)
                // We use vorticity magnitude as a proxy for turbulence intensity.

                // Tunable parameters for vorticity suspension
                const VORT_LIFT_SCALE: f32 = 0.3;      // How much vorticity counters settling
                const VORT_SWIRL_SCALE: f32 = 0.05;    // How much vorticity adds tangential motion

                // Material-specific settling using Ferguson-Church formula
                // Different materials settle at different rates based on density and size
                let settling_velocity = particle.material.settling_velocity(particle.effective_diameter());

                // Convert terminal velocity to acceleration factor
                // settling_velocity is terminal velocity (px/s), we want fraction of gravity
                // Factor = v_terminal / (g * tau) where tau is relaxation time
                // For simplicity, normalize so that reference settling gives reasonable behavior
                const REFERENCE_SETTLING: f32 = 28.0;  // Sand at typical diameter
                const BASE_FACTOR: f32 = 0.62;         // Original tuned value for sand
                let settling_factor = BASE_FACTOR * (settling_velocity / REFERENCE_SETTLING);

                // Sample vorticity at particle position
                let vorticity = grid.sample_vorticity(pos);
                let vort_magnitude = vorticity.abs();

                // 1. LIFT: Reduce settling in high-vorticity regions
                // Higher vorticity → less settling (particle stays suspended)
                // Only apply lift if vorticity is significant (reduces noise-driven suspension)
                const MIN_VORT_FOR_LIFT: f32 = 0.5; // Threshold to filter noise
                let base_lift = if vort_magnitude > MIN_VORT_FOR_LIFT {
                    ((vort_magnitude - MIN_VORT_FOR_LIFT) * VORT_LIFT_SCALE).min(1.0)
                } else {
                    0.0
                };

                // Scale lift by inverse density (Rouse number effect):
                // Heavy particles (gold ρ=19.3): lift reduced by 1/19.3 ≈ 0.05 → settles despite turbulence
                // Light particles (sand ρ=2.65): lift reduced by 1/2.65 ≈ 0.38 → can be suspended
                let density_lift_scale = (1.0 / particle.material.density()).min(0.5);
                let lift_factor = base_lift * density_lift_scale;
                let effective_settling = settling_factor * (1.0 - lift_factor);

                // 2. SWIRL: Add velocity perpendicular to particle motion
                // Only apply swirl if there's actual flow (not in calm water)
                // Requires both: particle moving AND meaningful vorticity
                const MIN_VORT_FOR_SWIRL: f32 = 1.0; // Higher threshold for swirl
                let speed = particle.velocity.length();
                let swirl_velocity = if speed > 5.0 && vort_magnitude > MIN_VORT_FOR_SWIRL {
                    let v_normalized = particle.velocity / speed;
                    let v_perp = Vec2::new(-v_normalized.y, v_normalized.x);
                    // Signed vorticity: positive ω → CCW rotation → add CCW perpendicular
                    v_perp * vorticity * VORT_SWIRL_SCALE
                } else {
                    Vec2::ZERO
                };

                // Apply modified settling + swirl
                particle.velocity.y += GRAVITY * effective_settling * dt;
                particle.velocity += swirl_velocity * dt;
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

    /// G2P for sediment particles only (called when GPU handles water G2P)
    ///
    /// This preserves all sediment physics:
    /// - FLIP/PIC blend with sand_pic_ratio
    /// - old_grid_velocity tracking for FLIP delta
    /// - Vorticity-based suspension (lift + swirl)
    /// - Material-specific settling velocity
    ///
    /// Water particles are SKIPPED - they should be handled by GPU G2P.
    pub fn grid_to_particles_sediment_only(&mut self, dt: f32) {
        let cell_size = self.grid.cell_size;
        let width = self.grid.width;
        // Note: sand_pic_ratio no longer used - replaced with drag-based coupling

        let grid = &self.grid;

        self.particles.list.par_iter_mut().for_each(|particle| {
            // ONLY process sediment particles
            if !particle.is_sediment() {
                return;
            }

            let pos = particle.position;

            // CRITICAL: Use B-spline sampling to match store_old_velocities kernel
            let v_grid = grid.sample_velocity_bspline(pos);
            use crate::physics::GRAVITY;

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
                // IN FLUID: Drag-based coupling with settling (same as main G2P)
                const RHO_WATER: f32 = 1.0;
                const DRAG_STRENGTH: f32 = 10.0;
                let rho_particle = particle.material.density();
                let drag_coeff = RHO_WATER / rho_particle;
                let drag_rate = (drag_coeff * DRAG_STRENGTH * dt).min(0.9);

                // Apply drag toward water velocity (HORIZONTAL ONLY)
                // Vertical settling is material-specific in vorticity section below
                particle.velocity.x += (v_grid.x - particle.velocity.x) * drag_rate;

                particle.old_grid_velocity = v_grid;
            } else {
                // IN AIR: Don't apply water drag
                particle.old_grid_velocity = Vec2::ZERO;
            }

            // ========== VORTICITY-BASED SUSPENSION ==========
            const VORT_LIFT_SCALE: f32 = 0.3;
            const VORT_SWIRL_SCALE: f32 = 0.05;

            // Material-specific settling using Ferguson-Church formula
            let settling_velocity = particle.material.settling_velocity(particle.effective_diameter());
            const REFERENCE_SETTLING: f32 = 28.0;
            const BASE_FACTOR: f32 = 0.62;
            let settling_factor = BASE_FACTOR * (settling_velocity / REFERENCE_SETTLING);

            // Sample vorticity at particle position
            let vorticity = grid.sample_vorticity(pos);
            let vort_magnitude = vorticity.abs();

            // 1. LIFT: Reduce settling in high-vorticity regions
            const MIN_VORT_FOR_LIFT: f32 = 0.5;
            let base_lift = if vort_magnitude > MIN_VORT_FOR_LIFT {
                ((vort_magnitude - MIN_VORT_FOR_LIFT) * VORT_LIFT_SCALE).min(1.0)
            } else {
                0.0
            };
            // Scale lift by inverse density (heavy particles resist suspension)
            let density_lift_scale = (1.0 / particle.material.density()).min(0.5);
            let lift_factor = base_lift * density_lift_scale;
            let effective_settling = settling_factor * (1.0 - lift_factor);

            // 2. SWIRL: Add velocity perpendicular to particle motion
            const MIN_VORT_FOR_SWIRL: f32 = 1.0;
            let speed = particle.velocity.length();
            let swirl_velocity = if speed > 5.0 && vort_magnitude > MIN_VORT_FOR_SWIRL {
                let v_normalized = particle.velocity / speed;
                let v_perp = Vec2::new(-v_normalized.y, v_normalized.x);
                v_perp * vorticity * VORT_SWIRL_SCALE
            } else {
                Vec2::ZERO
            };

            // Apply modified settling + swirl
            particle.velocity.y += GRAVITY * effective_settling * dt;
            particle.velocity += swirl_velocity * dt;
        });
    }
}
