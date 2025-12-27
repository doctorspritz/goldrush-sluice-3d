//! APIC (Affine Particle-In-Cell) water simulation
//!
//! WATER-ONLY VERSION - Sediment code archived in archive/sediment_archive.rs
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
    // Pre-allocated P2G transfer buffers
    u_sum: Vec<f32>,
    u_weight: Vec<f32>,
    v_sum: Vec<f32>,
    v_weight: Vec<f32>,
    // Pre-allocated spatial hash for particle separation
    cell_head: Vec<i32>,
    particle_next: Vec<i32>,
    // Pre-allocated impulse buffer for soft separation
    impulse_buffer: Vec<Vec2>,
    // Pre-allocated buffer for near-pressure forces
    near_force_buffer: Vec<Vec2>,
    // Frame counter
    frame: u32,

    // Async near-pressure computation
    near_pressure_snapshot: Vec<Vec2>,
    near_pressure_densities: Vec<(f32, f32)>,
    near_pressure_forces: Vec<Vec2>,
    near_pressure_phase: u32,

    // Tunable near-pressure parameters
    pub near_pressure_h: f32,
    pub near_pressure_rest: f32,

    /// FLIP blend ratio (0 = pure PIC, 1 = pure FLIP)
    /// Higher values preserve more energy but can be noisier
    pub flip_ratio: f32,
}

impl FlipSimulation {
    pub fn new(width: usize, height: usize, cell_size: f32) -> Self {
        let cell_count = width * height;
        let grid = Grid::new(width, height, cell_size);
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
            near_pressure_snapshot: Vec::new(),
            near_pressure_densities: Vec::new(),
            near_pressure_forces: Vec::new(),
            near_pressure_phase: 0,
            near_pressure_h: 4.0,
            near_pressure_rest: 0.8,
            // Near 100% FLIP for lively water - PIC dampens too much
            flip_ratio: 0.99,
        }
    }

    /// Run one simulation step
    pub fn update(&mut self, dt: f32) {
        self.frame = self.frame.wrapping_add(1);

        // 1. Classify cells based on particle positions
        self.classify_cells();

        // 1b. Update SDF for collision
        self.grid.compute_sdf();

        // 2. Transfer particle velocities to grid (P2G)
        self.particles_to_grid();

        // 3. Store old grid velocities for FLIP blending
        self.store_old_velocities();

        // 4. Apply external forces (gravity)
        self.grid.apply_gravity(dt);

        // 4b. Vorticity confinement - preserves swirl in bulk water
        // Strength 0.05 per Fedkiw 2001 (0.01-0.1 range)
        self.grid.apply_vorticity_confinement(dt, 0.05);

        // 5. Pressure projection - enforces incompressibility
        self.grid.enforce_boundary_conditions();
        self.grid.compute_divergence();
        self.grid.solve_pressure(15);
        self.grid.apply_pressure_gradient(dt);

        // NOTE: No surface vertical damping - let water flow naturally!

        // 6. Transfer grid velocities back to particles (G2P)
        self.grid_to_particles(dt);

        // 7. Build spatial hash for near-pressure
        self.build_spatial_hash();

        // 8. Apply near-pressure for particle-particle repulsion
        self.apply_near_pressure(dt);

        // 9. Advect particles
        self.advect_particles(dt);

        // Clean up particles that left the simulation
        self.particles.remove_out_of_bounds(
            self.grid.width as f32 * self.grid.cell_size,
            self.grid.height as f32 * self.grid.cell_size,
        );
    }

    /// Step 1: Classify cells as solid, fluid, or air
    fn classify_cells(&mut self) {
        // Reset to air
        for cell in &mut self.grid.cell_type {
            *cell = CellType::Air;
        }

        // Mark solid cells from terrain and boundaries
        for j in 0..self.grid.height {
            for i in 0..self.grid.width {
                let idx = self.grid.cell_index(i, j);
                let is_boundary = i == 0 || i == self.grid.width - 1 || j == self.grid.height - 1;

                if is_boundary || self.grid.is_solid(i, j) {
                    self.grid.cell_type[idx] = CellType::Solid;
                }
            }
        }

        // Mark fluid cells from particles
        for particle in self.particles.iter() {
            let (i, j) = self.grid.pos_to_cell(particle.position);
            let idx = self.grid.cell_index(i, j);
            if self.grid.cell_type[idx] != CellType::Solid {
                self.grid.cell_type[idx] = CellType::Fluid;
            }
        }
    }

    /// Step 2: APIC Particle-to-Grid transfer (P2G)
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
            let pos = particle.position;
            let vel = particle.velocity;
            let c_mat = particle.affine_velocity;

            // ========== U component (staggered on left edges) ==========
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

                    let offset = Vec2::new(
                        (ni as f32) * cell_size - pos.x,
                        (nj as f32 + 0.5) * cell_size - pos.y,
                    );
                    let affine_vel = c_mat * offset;

                    let idx = self.grid.u_index(ni as usize, nj as usize);
                    self.u_sum[idx] += (vel.x + affine_vel.x) * w;
                    self.u_weight[idx] += w;
                }
            }

            // ========== V component (staggered on bottom edges) ==========
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

                    let offset = Vec2::new(
                        (ni as f32 + 0.5) * cell_size - pos.x,
                        (nj as f32) * cell_size - pos.y,
                    );
                    let affine_vel = c_mat * offset;

                    let idx = self.grid.v_index(ni as usize, nj as usize);
                    self.v_sum[idx] += (vel.y + affine_vel.y) * w;
                    self.v_weight[idx] += w;
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
    /// Uses standard FLIP blend - NO anisotropic surface hacks!
    fn grid_to_particles(&mut self, dt: f32) {
        let cell_size = self.grid.cell_size;
        let width = self.grid.width;
        let height = self.grid.height;
        let d_inv = apic_d_inverse(cell_size);
        let flip_ratio = self.flip_ratio;

        let grid = &self.grid;

        self.particles.list.par_iter_mut().for_each(|particle| {
            let pos = particle.position;
            let old_particle_velocity = particle.velocity;

            // ========== APIC velocity and C matrix reconstruction ==========
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

                    new_velocity.x += w * u_val;

                    let offset = Vec2::new(
                        (ni as f32) * cell_size - pos.x,
                        (nj as f32 + 0.5) * cell_size - pos.y,
                    );
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

                    new_velocity.y += w * v_val;

                    let offset = Vec2::new(
                        (ni as f32 + 0.5) * cell_size - pos.x,
                        (nj as f32) * cell_size - pos.y,
                    );
                    new_c.y_axis += offset * (w * v_val * d_inv);
                }
            }

            // FLIP/PIC blend
            // FLIP: preserve particle velocity, add grid delta
            // PIC: take grid velocity directly
            let grid_delta = new_velocity - particle.old_grid_velocity;

            // Safety clamp on delta to prevent explosions
            let max_dv = 5.0 * cell_size / dt;
            let clamped_delta = if grid_delta.length_squared() > max_dv * max_dv {
                grid_delta.normalize() * max_dv
            } else {
                grid_delta
            };

            let flip_velocity = old_particle_velocity + clamped_delta;
            let pic_velocity = new_velocity;

            // Standard isotropic FLIP blend - same ratio for x and y
            particle.velocity = flip_ratio * flip_velocity + (1.0 - flip_ratio) * pic_velocity;
            particle.affine_velocity = new_c;

            // Safety clamp on velocity magnitude
            const MAX_VELOCITY: f32 = 2000.0;
            let speed = particle.velocity.length();
            if speed > MAX_VELOCITY {
                particle.velocity *= MAX_VELOCITY / speed;
            }
        });
    }

    /// Step 8: Apply near-pressure for particle-particle repulsion
    fn apply_near_pressure(&mut self, dt: f32) {
        let h = self.near_pressure_h;
        let rest_density = self.near_pressure_rest;
        const K_STIFFNESS: f32 = 5.0;
        const K_NEAR: f32 = 80.0;

        let h2 = h * h;
        let width = self.grid.width;
        let cell_size = self.grid.cell_size;

        // Resize force buffer
        self.near_force_buffer.resize(self.particles.len(), Vec2::ZERO);
        self.near_force_buffer.fill(Vec2::ZERO);

        // Phase 1: Compute densities
        let densities: Vec<(f32, f32)> = self.particles.list.par_iter().enumerate().map(|(_idx, p)| {
            let pos = p.position;
            let ci = (pos.x / cell_size) as i32;
            let cj = (pos.y / cell_size) as i32;

            let mut density = 0.0f32;
            let mut near_density = 0.0f32;

            // 3x3 neighbor search
            for dj in -1..=1i32 {
                for di in -1..=1i32 {
                    let ni = ci + di;
                    let nj = cj + dj;
                    if ni < 0 || nj < 0 || ni >= width as i32 || nj >= self.grid.height as i32 {
                        continue;
                    }

                    let cell_idx = nj as usize * width + ni as usize;
                    let mut head = self.cell_head[cell_idx];

                    while head >= 0 {
                        let neighbor_idx = head as usize;
                        if neighbor_idx < self.particles.len() {
                            let q = &self.particles.list[neighbor_idx];
                            let d = pos - q.position;
                            let r2 = d.length_squared();

                            if r2 < h2 && r2 > 0.0001 {
                                let r = r2.sqrt();
                                let q_val = 1.0 - r / h;
                                density += q_val * q_val;
                                near_density += q_val * q_val * q_val;
                            }
                        }

                        head = self.particle_next.get(neighbor_idx).copied().unwrap_or(-1);
                    }
                }
            }

            (density, near_density)
        }).collect();

        // Phase 2: Compute forces
        let forces: Vec<Vec2> = self.particles.list.par_iter().enumerate().map(|(idx, p)| {
            let pos = p.position;
            let ci = (pos.x / cell_size) as i32;
            let cj = (pos.y / cell_size) as i32;

            let (density, near_density) = densities[idx];
            let pressure = K_STIFFNESS * (density - rest_density);
            let near_pressure = K_NEAR * near_density;

            let mut force = Vec2::ZERO;

            // 3x3 neighbor search for forces
            for dj in -1..=1i32 {
                for di in -1..=1i32 {
                    let ni = ci + di;
                    let nj = cj + dj;
                    if ni < 0 || nj < 0 || ni >= width as i32 || nj >= self.grid.height as i32 {
                        continue;
                    }

                    let cell_idx = nj as usize * width + ni as usize;
                    let mut head = self.cell_head[cell_idx];

                    while head >= 0 {
                        let neighbor_idx = head as usize;
                        if neighbor_idx < self.particles.len() && neighbor_idx != idx {
                            let q = &self.particles.list[neighbor_idx];
                            let d = pos - q.position;
                            let r2 = d.length_squared();

                            if r2 < h2 && r2 > 0.0001 {
                                let r = r2.sqrt();
                                let q_val = 1.0 - r / h;
                                let dir = d / r;

                                // Pressure force
                                let (q_density, q_near_density) = densities[neighbor_idx];
                                let q_pressure = K_STIFFNESS * (q_density - rest_density);
                                let q_near_pressure = K_NEAR * q_near_density;

                                let avg_pressure = (pressure + q_pressure) * 0.5;
                                let avg_near_pressure = (near_pressure + q_near_pressure) * 0.5;

                                force += dir * (avg_pressure * q_val + avg_near_pressure * q_val * q_val);
                            }
                        }

                        head = self.particle_next.get(neighbor_idx).copied().unwrap_or(-1);
                    }
                }
            }

            force
        }).collect();

        // Apply forces
        for (idx, force) in forces.into_iter().enumerate() {
            self.particles.list[idx].velocity += force * dt;
        }
    }

    /// Step 9: Advect particles
    fn advect_particles(&mut self, dt: f32) {
        let cell_size = self.grid.cell_size;
        let width = self.grid.width;
        let height = self.grid.height;
        let margin = cell_size;
        let max_x = width as f32 * cell_size - margin;
        let max_y = height as f32 * cell_size - margin;

        let grid = &self.grid;

        self.particles.list.par_iter_mut().for_each(|particle| {
            // Micro-stepped advection with collision checking
            particle.advect_micro_stepped(dt, cell_size, |p| {
                // SDF collision
                let sdf_dist = grid.sample_sdf(p.position);

                if sdf_dist < cell_size * 0.5 {
                    let grad = grid.sdf_gradient(p.position);

                    // Push out
                    let push_dist = cell_size * 0.5 - sdf_dist;
                    p.position += grad * push_dist;

                    // Reflect velocity - water bounces, no friction
                    let v_dot_n = p.velocity.dot(grad);
                    if v_dot_n < 0.0 {
                        // Remove velocity into solid (simple reflection)
                        p.velocity -= grad * v_dot_n;
                    }
                }

                // Bounds clamp
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

    /// Build spatial hash for near-pressure calculation
    fn build_spatial_hash(&mut self) {
        let width = self.grid.width;
        let height = self.grid.height;
        let cell_size = self.grid.cell_size;

        // Reset cell heads
        self.cell_head.fill(-1);

        // Resize particle_next if needed
        self.particle_next.resize(self.particles.len(), -1);
        self.particle_next.fill(-1);

        // Insert particles into cells
        for (idx, particle) in self.particles.list.iter().enumerate() {
            let ci = (particle.position.x / cell_size) as usize;
            let cj = (particle.position.y / cell_size) as usize;

            if ci < width && cj < height {
                let cell_idx = cj * width + ci;
                self.particle_next[idx] = self.cell_head[cell_idx];
                self.cell_head[cell_idx] = idx as i32;
            }
        }
    }

    fn is_spawn_safe(&self, x: f32, y: f32) -> bool {
        let (i, j) = self.grid.pos_to_cell(Vec2::new(x, y));
        let idx = self.grid.cell_index(i, j);
        self.grid.cell_type[idx] != CellType::Solid
    }

    /// Spawn water particles in a block pattern
    pub fn spawn_water(&mut self, x: f32, y: f32, vx: f32, vy: f32, count: usize) {
        let cell_size = self.grid.cell_size;
        let spacing = cell_size * 0.5;
        let side = (count as f32).sqrt().ceil() as usize;
        let mut rng = rand::thread_rng();

        for i in 0..side {
            for j in 0..side {
                if i * side + j >= count {
                    break;
                }
                let jitter_x: f32 = rng.gen_range(-0.2..0.2) * spacing;
                let jitter_y: f32 = rng.gen_range(-0.2..0.2) * spacing;
                let px = x + (i as f32) * spacing + jitter_x;
                let py = y + (j as f32) * spacing + jitter_y;
                if self.is_spawn_safe(px, py) {
                    self.particles.spawn_water(px, py, vx, vy);
                }
            }
        }
    }

    /// Compute total kinetic energy
    pub fn compute_kinetic_energy(&self) -> f32 {
        self.particles.iter()
            .map(|p| 0.5 * p.velocity.length_squared())
            .sum()
    }

    /// Max particle velocity
    pub fn max_velocity(&self) -> f32 {
        self.particles.iter()
            .map(|p| p.velocity.length())
            .fold(0.0f32, f32::max)
    }

    /// Compute CFL number
    pub fn compute_cfl(&self, dt: f32) -> f32 {
        self.max_velocity() * dt / self.grid.cell_size
    }

    /// Compute grid kinetic energy
    pub fn compute_grid_kinetic_energy(&self) -> f32 {
        let mut energy = 0.0f32;

        for j in 0..self.grid.height {
            for i in 0..self.grid.width {
                let idx = self.grid.cell_index(i, j);
                if self.grid.cell_type[idx] == CellType::Fluid {
                    // Average velocities at cell center
                    let u_left = self.grid.u[self.grid.u_index(i, j)];
                    let u_right = self.grid.u[self.grid.u_index(i + 1, j)];
                    let v_bottom = self.grid.v[self.grid.v_index(i, j)];
                    let v_top = self.grid.v[self.grid.v_index(i, j + 1)];

                    let u = (u_left + u_right) * 0.5;
                    let v = (v_bottom + v_top) * 0.5;

                    energy += 0.5 * (u * u + v * v);
                }
            }
        }

        energy
    }

    /// Compute enstrophy (vorticity magnitude)
    pub fn compute_enstrophy(&self) -> f32 {
        self.grid.compute_enstrophy()
    }

    /// Update vorticity field and compute enstrophy
    pub fn update_and_compute_enstrophy(&mut self) -> f32 {
        self.grid.compute_vorticity();
        self.grid.compute_enstrophy()
    }

    /// Initialize Taylor-Green vortex test pattern
    pub fn initialize_taylor_green(&mut self) {
        let width = self.grid.width;
        let height = self.grid.height;
        let cs = self.grid.cell_size;

        // Fill domain with particles
        let spacing = cs * 0.5;
        for j in 2..height - 2 {
            for i in 2..width - 2 {
                let x = (i as f32 + 0.5) * cs;
                let y = (j as f32 + 0.5) * cs;

                // 2x2 particles per cell
                for di in 0..2 {
                    for dj in 0..2 {
                        let px = x - spacing * 0.5 + di as f32 * spacing;
                        let py = y - spacing * 0.5 + dj as f32 * spacing;
                        self.particles.spawn_water(px, py, 0.0, 0.0);
                    }
                }
            }
        }

        // Set grid velocities for Taylor-Green vortex
        let lx = width as f32 * cs;
        let ly = height as f32 * cs;
        let kx = 2.0 * std::f32::consts::PI / lx;
        let ky = 2.0 * std::f32::consts::PI / ly;
        let u0 = 50.0;

        // U velocities
        for j in 0..height {
            for i in 0..=width {
                let x = i as f32 * cs;
                let y = (j as f32 + 0.5) * cs;
                let u = u0 * (kx * x).sin() * (ky * y).cos();
                let idx = self.grid.u_index(i, j);
                self.grid.u[idx] = u;
            }
        }

        // V velocities
        for j in 0..=height {
            for i in 0..width {
                let x = (i as f32 + 0.5) * cs;
                let y = j as f32 * cs;
                let v = -u0 * (kx * x).cos() * (ky * y).sin();
                let idx = self.grid.v_index(i, j);
                self.grid.v[idx] = v;
            }
        }

        // Transfer grid velocities to particles
        for particle in self.particles.iter_mut() {
            particle.velocity = self.grid.sample_velocity(particle.position);
        }
    }

    /// Initialize solid body rotation test
    pub fn initialize_solid_rotation(&mut self, angular_velocity: f32) {
        let width = self.grid.width;
        let height = self.grid.height;
        let cs = self.grid.cell_size;

        let cx = width as f32 * cs * 0.5;
        let cy = height as f32 * cs * 0.5;
        let radius = (width.min(height) as f32 * cs * 0.4).min(100.0);

        // Fill circular region with particles
        let spacing = cs * 0.5;
        for j in 2..height - 2 {
            for i in 2..width - 2 {
                let x = (i as f32 + 0.5) * cs;
                let y = (j as f32 + 0.5) * cs;
                let dx = x - cx;
                let dy = y - cy;
                let r = (dx * dx + dy * dy).sqrt();

                if r < radius {
                    for di in 0..2 {
                        for dj in 0..2 {
                            let px = x - spacing * 0.5 + di as f32 * spacing;
                            let py = y - spacing * 0.5 + dj as f32 * spacing;
                            self.particles.spawn_water(px, py, 0.0, 0.0);
                        }
                    }
                }
            }
        }

        // Set grid velocities for solid rotation
        for j in 0..height {
            for i in 0..=width {
                let x = i as f32 * cs;
                let y = (j as f32 + 0.5) * cs;
                let dy = y - cy;
                let r = ((x - cx).powi(2) + dy.powi(2)).sqrt();
                let idx = self.grid.u_index(i, j);
                if r < radius {
                    self.grid.u[idx] = -angular_velocity * dy;
                } else {
                    self.grid.u[idx] = 0.0;
                }
            }
        }

        for j in 0..=height {
            for i in 0..width {
                let x = (i as f32 + 0.5) * cs;
                let y = j as f32 * cs;
                let dx = x - cx;
                let r = (dx.powi(2) + (y - cy).powi(2)).sqrt();
                let idx = self.grid.v_index(i, j);
                if r < radius {
                    self.grid.v[idx] = angular_velocity * dx;
                } else {
                    self.grid.v[idx] = 0.0;
                }
            }
        }

        // Transfer to particles
        for particle in self.particles.iter_mut() {
            particle.velocity = self.grid.sample_velocity(particle.position);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spawn_water() {
        let mut sim = FlipSimulation::new(20, 20, 5.0);
        sim.spawn_water(50.0, 50.0, 10.0, 0.0, 16);
        assert!(sim.particles.len() >= 10);
    }

    #[test]
    fn test_update_does_not_panic() {
        let mut sim = FlipSimulation::new(20, 20, 5.0);
        sim.spawn_water(50.0, 50.0, 0.0, 0.0, 16);
        for _ in 0..10 {
            sim.update(0.016);
        }
    }

    #[test]
    fn test_energy_conservation() {
        let mut sim = FlipSimulation::new(30, 30, 5.0);
        sim.initialize_solid_rotation(2.0);

        let initial_energy = sim.compute_kinetic_energy();

        for _ in 0..100 {
            sim.update(0.01);
        }

        let final_energy = sim.compute_kinetic_energy();

        // APIC should preserve most energy (allow 30% loss over 100 steps)
        assert!(
            final_energy > initial_energy * 0.7,
            "Energy dropped too much: {} -> {}",
            initial_energy,
            final_energy
        );
    }
}
