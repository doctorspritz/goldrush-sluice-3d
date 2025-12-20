//! PIC/FLIP (Particle-In-Cell / Fluid-Implicit-Particle) simulation
//!
//! This module ties together particles and grid for fluid simulation:
//! 1. Classify cells (solid/fluid/air)
//! 2. Transfer particle velocities to grid (P2G)
//! 3. Store old grid velocities
//! 4. Apply forces (gravity)
//! 5. Pressure projection (creates vortices!)
//! 6. Transfer grid velocities back to particles (G2P with PIC/FLIP blend)
//! 7. Apply density-based settling
//! 8. Advect particles

use crate::grid::{CellType, Grid};
use crate::particle::{Particle, Particles};
use glam::Vec2;
use rand::Rng;

/// FLIP simulation state
pub struct FlipSimulation {
    pub grid: Grid,
    pub particles: Particles,
    // Pre-allocated spatial hash for particle separation (zero allocation per frame)
    cell_head: Vec<i32>,      // Index of first particle in each cell (-1 = empty)
    particle_next: Vec<i32>,  // Index of next particle in same cell (-1 = end)
    // Pre-allocated impulse buffer for soft separation
    impulse_buffer: Vec<Vec2>,
    // Frame counter for skipping expensive operations
    frame: u32,
}

impl FlipSimulation {
    pub fn new(width: usize, height: usize, cell_size: f32) -> Self {
        let cell_count = width * height;
        Self {
            grid: Grid::new(width, height, cell_size),
            particles: Particles::new(),
            cell_head: vec![-1; cell_count],
            particle_next: Vec::new(),
            impulse_buffer: Vec::new(),
            frame: 0,
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

        // 4. Apply forces (gravity)
        self.grid.apply_gravity(dt);

        // 5. Pressure projection - this creates vortices and enforces incompressibility
        self.grid.compute_divergence();
        self.grid.solve_pressure(20); // 20 Red-Black GS iterations (reduced for perf)
        self.grid.apply_pressure_gradient(dt);

        // 5b. Vorticity confinement - skip every other frame for perf
        if self.frame % 2 == 0 {
            self.grid.apply_vorticity_confinement(dt * 2.0, 5.0);
        }

        // 6. Transfer grid velocities back to particles
        self.grid_to_particles();

        // 7. Apply density-based settling (mud sinks)
        self.apply_settling(dt);

        // 8. Advect particles
        self.advect_particles(dt);

        // 9. Separate overlapping particles - every other frame
        if self.frame % 2 == 0 {
            self.separate_particles();
        }

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

        // Mark solid cells from terrain
        for j in 0..self.grid.height {
            for i in 0..self.grid.width {
                if self.grid.is_solid(i, j) {
                    let idx = self.grid.cell_index(i, j);
                    self.grid.cell_type[idx] = CellType::Solid;
                }
            }
        }

        // Mark fluid cells (contain particles)
        for particle in self.particles.iter() {
            let (i, j) = self.grid.pos_to_cell(particle.position);
            let idx = self.grid.cell_index(i, j);
            if self.grid.cell_type[idx] != CellType::Solid {
                self.grid.cell_type[idx] = CellType::Fluid;
            }
        }
    }

    /// Step 2: Transfer particle velocities to grid (P2G)
    fn particles_to_grid(&mut self) {
        // Clear accumulators
        let mut u_sum = vec![0.0f32; self.grid.u.len()];
        let mut u_weight = vec![0.0f32; self.grid.u.len()];
        let mut v_sum = vec![0.0f32; self.grid.v.len()];
        let mut v_weight = vec![0.0f32; self.grid.v.len()];

        let cell_size = self.grid.cell_size;

        for particle in self.particles.iter() {
            // U component (staggered - sample at left edges)
            let u_pos = particle.position - Vec2::new(cell_size * 0.5, 0.0);
            let (i, j, weights) = self.grid.get_interp_weights(u_pos);

            for (di, dj, w) in weights {
                let ni = (i as i32 + di) as usize;
                let nj = (j as i32 + dj) as usize;
                if ni <= self.grid.width && nj < self.grid.height {
                    let idx = self.grid.u_index(ni, nj);
                    u_sum[idx] += particle.velocity.x * w;
                    u_weight[idx] += w;
                }
            }

            // V component (staggered - sample at bottom edges)
            let v_pos = particle.position - Vec2::new(0.0, cell_size * 0.5);
            let (i, j, weights) = self.grid.get_interp_weights(v_pos);

            for (di, dj, w) in weights {
                let ni = (i as i32 + di) as usize;
                let nj = (j as i32 + dj) as usize;
                if ni < self.grid.width && nj <= self.grid.height {
                    let idx = self.grid.v_index(ni, nj);
                    v_sum[idx] += particle.velocity.y * w;
                    v_weight[idx] += w;
                }
            }
        }

        // Normalize
        for i in 0..self.grid.u.len() {
            if u_weight[i] > 0.0 {
                self.grid.u[i] = u_sum[i] / u_weight[i];
            } else {
                self.grid.u[i] = 0.0;
            }
        }
        for i in 0..self.grid.v.len() {
            if v_weight[i] > 0.0 {
                self.grid.v[i] = v_sum[i] / v_weight[i];
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

    /// Step 6: Transfer grid velocities back to particles (G2P)
    /// Uses PIC/FLIP blend: 15% PIC (stable but dissipative) + 85% FLIP (preserves momentum)
    /// Increased PIC for less bounciness
    fn grid_to_particles(&mut self) {
        const ALPHA: f32 = 0.15; // 15% PIC, 85% FLIP (more stable, less bouncy)

        for particle in self.particles.iter_mut() {
            let v_grid = self.grid.sample_velocity(particle.position);

            // PIC: use grid velocity directly
            let v_pic = v_grid;

            // FLIP: add velocity change to particle
            let delta_v = v_grid - particle.old_grid_velocity;
            let v_flip = particle.velocity + delta_v;

            // Blend
            particle.velocity = v_pic * ALPHA + v_flip * (1.0 - ALPHA);
        }
    }

    /// Step 7: Apply density-based settling (mud sinks through water)
    fn apply_settling(&mut self, dt: f32) {
        const WATER_DENSITY: f32 = 1.0;

        for particle in self.particles.iter_mut() {
            if particle.density > WATER_DENSITY {
                // Settling velocity based on density difference (simplified Stokes law)
                let settling = (particle.density - WATER_DENSITY) * 2.0;
                particle.velocity.y += settling * dt;
            }
        }
    }

    /// Step 8: Advect particles with ray-based collision detection
    fn advect_particles(&mut self, dt: f32) {
        let cell_size = self.grid.cell_size;
        let width = self.grid.width;
        let height = self.grid.height;
        // Only use expensive raycast if particle moves more than half a cell
        let raycast_threshold = cell_size * 0.5;

        for particle in self.particles.iter_mut() {
            let displacement = particle.velocity * dt;
            let dist = displacement.length();

            if dist > raycast_threshold {
                // Fast particle - use raycast
                let start = particle.position;
                let end = start + displacement;

                let result = raycast_to_solid(&self.grid, start, end, cell_size);

                match result {
                    RaycastResult::NoHit => {
                        particle.position = end;
                    }
                    RaycastResult::Hit { position, normal } => {
                        particle.position = position;
                        let v_dot_n = particle.velocity.dot(normal);
                        if v_dot_n > 0.0 {
                            particle.velocity -= normal * v_dot_n;
                        }
                    }
                }
            } else {
                // Slow particle - simple move + check
                particle.position += displacement;

                // Quick solid check at new position
                let (i, j) = self.grid.pos_to_cell(particle.position);
                if self.grid.is_solid(i, j) {
                    // Push out of solid
                    particle.position -= displacement;
                    // Kill velocity into solid
                    if displacement.y > 0.0 {
                        particle.velocity.y = 0.0;
                    }
                }
            }

            // Final bounds clamp
            let margin = cell_size;
            let max_x = width as f32 * cell_size - margin;
            let max_y = height as f32 * cell_size - margin;

            particle.position.x = particle.position.x.clamp(margin, max_x);
            particle.position.y = particle.position.y.clamp(margin, max_y);

            if particle.position.x <= margin || particle.position.x >= max_x {
                particle.velocity.x = 0.0;
            }
            if particle.position.y <= margin || particle.position.y >= max_y {
                particle.velocity.y = 0.0;
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

    /// Step 9: Soft particle separation - applies velocity impulses instead of position snaps
    fn separate_particles(&mut self) {
        let cell_size = self.grid.cell_size;
        let width = self.grid.width;
        let height = self.grid.height;
        let min_dist = cell_size * 0.5;
        let min_dist_sq = min_dist * min_dist;
        let stiffness = 200.0;

        // Build spatial hash once
        self.build_spatial_hash();

        // Use pre-allocated impulse buffer
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
                        if other_idx_usize != idx {
                            let other_pos = self.particles.list[other_idx_usize].position;
                            let diff = pos - other_pos;
                            let dist_sq = diff.length_squared();

                            if dist_sq < min_dist_sq && dist_sq > 0.0001 {
                                let dist = dist_sq.sqrt();
                                let overlap = min_dist - dist;
                                let dir = diff / dist;
                                // Soft force instead of hard position correction
                                self.impulse_buffer[idx] += dir * overlap * stiffness;
                            }
                        }
                        other_idx = self.particle_next[other_idx_usize];
                    }
                }
            }
        }

        // Apply velocity impulses (dt=1/60)
        let dt = 1.0 / 60.0;
        for idx in 0..particle_count {
            self.particles.list[idx].velocity += self.impulse_buffer[idx] * dt;
        }
    }

    /// Spawn water particles at a position with initial velocity
    pub fn spawn_water(&mut self, x: f32, y: f32, vx: f32, vy: f32, count: usize) {
        let mut rng = rand::thread_rng();
        for _ in 0..count {
            let offset_x = (rng.gen::<f32>() - 0.5) * self.grid.cell_size;
            let offset_y = (rng.gen::<f32>() - 0.5) * self.grid.cell_size;
            self.particles.spawn_water(
                x + offset_x,
                y + offset_y,
                vx + (rng.gen::<f32>() - 0.5) * 10.0,
                vy + (rng.gen::<f32>() - 0.5) * 10.0,
            );
        }
    }

    /// Spawn mud particles at a position
    pub fn spawn_mud(&mut self, x: f32, y: f32, vx: f32, vy: f32, count: usize) {
        let mut rng = rand::thread_rng();
        for _ in 0..count {
            let offset_x = (rng.gen::<f32>() - 0.5) * self.grid.cell_size;
            let offset_y = (rng.gen::<f32>() - 0.5) * self.grid.cell_size;
            self.particles.spawn_mud(
                x + offset_x,
                y + offset_y,
                vx + (rng.gen::<f32>() - 0.5) * 10.0,
                vy + (rng.gen::<f32>() - 0.5) * 10.0,
            );
        }
    }
}

/// Resolve collision with solid cells (free function to avoid borrow issues)
fn resolve_solid_collision(
    particle: &mut Particle,
    grid: &Grid,
    cell_size: f32,
    width: usize,
    height: usize,
) {
    // Multiple iterations to handle corners
    for _ in 0..3 {
        let (i, j) = grid.pos_to_cell(particle.position);

        if !grid.is_solid(i, j) {
            break;
        }

        // Try to find the nearest non-solid cell
        // Priority: up (most common for floor collision), then left/right, then down
        let checks = [
            (0i32, -1i32), // up
            (-1, 0),       // left
            (1, 0),        // right
            (0, 1),        // down (last resort)
        ];

        let mut pushed = false;
        for (di, dj) in checks {
            let ni = (i as i32 + di).clamp(0, width as i32 - 1) as usize;
            let nj = (j as i32 + dj).clamp(0, height as i32 - 1) as usize;

            if !grid.is_solid(ni, nj) {
                // Push to the edge of this cell towards the non-solid neighbor
                if di < 0 {
                    particle.position.x = i as f32 * cell_size - 0.5;
                    particle.velocity.x = particle.velocity.x.min(0.0);
                } else if di > 0 {
                    particle.position.x = (i + 1) as f32 * cell_size + 0.5;
                    particle.velocity.x = particle.velocity.x.max(0.0);
                }
                if dj < 0 {
                    particle.position.y = j as f32 * cell_size - 0.5;
                    particle.velocity.y = particle.velocity.y.min(0.0);
                } else if dj > 0 {
                    particle.position.y = (j + 1) as f32 * cell_size + 0.5;
                    particle.velocity.y = particle.velocity.y.max(0.0);
                }
                pushed = true;
                break;
            }
        }

        // If completely stuck, just push up
        if !pushed {
            particle.position.y -= cell_size;
            particle.velocity.y = particle.velocity.y.min(0.0);
        }
    }

    // Clamp to bounds
    let margin = cell_size;
    let max_x = width as f32 * cell_size - margin;
    let max_y = height as f32 * cell_size - margin;

    if particle.position.x < margin {
        particle.position.x = margin;
        particle.velocity.x = particle.velocity.x.max(0.0);
    }
    if particle.position.x > max_x {
        particle.position.x = max_x;
        particle.velocity.x = particle.velocity.x.min(0.0);
    }
    if particle.position.y < margin {
        particle.position.y = margin;
        particle.velocity.y = particle.velocity.y.max(0.0);
    }
    if particle.position.y > max_y {
        particle.position.y = max_y;
        particle.velocity.y = particle.velocity.y.min(0.0);
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
