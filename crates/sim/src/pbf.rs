use glam::Vec2;
use rayon::prelude::*;
use std::f32::consts::PI;

const GRAVITY: Vec2 = Vec2::new(0.0, 400.0);
const SOLVER_ITERATIONS: usize = 4;
const REST_DENSITY: f32 = 0.05; 
const MAX_PARTICLES: usize = 10000;
const EPSILON: f32 = 1.0; 
const H_SCALE: f32 = 1.5; 
const VORTICITY_EPSILON: f32 = 30.0; // Tune this for swirl strength

#[derive(Clone, Copy, Debug, Default)] 
pub struct PbfParticle {
    pub position: Vec2,
    pub old_position: Vec2,
    pub velocity: Vec2,
    pub density: f32,
    pub lambda: f32,
    pub phase: i32,
    pub vorticity: f32, // New scalar vorticity (2D Curl)
}

impl PbfParticle {
    pub fn is_mud(&self) -> bool {
        self.phase == 1
    }
}

pub struct PbfSimulation {
    pub particles: Vec<PbfParticle>,
    pub width: f32,
    pub height: f32,
    pub cell_size: f32,
    // Spatial hash grid
    grid_heads: Vec<i32>,
    grid_next: Vec<i32>,
    pub grid_solid: Vec<bool>, // [NEW] Collision map
}

impl PbfSimulation {
    pub fn new(width: f32, height: f32, cell_size: f32) -> Self {
        let grid_cells_x = (width / cell_size).ceil() as usize;
        let grid_cells_y = (height / cell_size).ceil() as usize;
        let num_cells = grid_cells_x * grid_cells_y;
        
        Self {
            particles: Vec::with_capacity(MAX_PARTICLES),
            width,
            height,
            cell_size,
            grid_heads: vec![-1; num_cells],
            grid_next: Vec::with_capacity(MAX_PARTICLES),
            grid_solid: vec![false; num_cells],
        }
    }
    
    // Helper to set collision
    pub fn set_solid(&mut self, x: usize, y: usize) {
        let w = (self.width / self.cell_size).ceil() as usize;
        if x < w {
            let idx = y * w + x;
            if idx < self.grid_solid.len() {
                self.grid_solid[idx] = true;
            }
        }
    }

    pub fn update(&mut self, dt: f32) {
        let grid_w = (self.width / self.cell_size).ceil() as usize;
        let grid_h = (self.height / self.cell_size).ceil() as usize;
        
        // 1. Prediction (Parallel)
        // Clone grid_solid to avoid borrow conflict with build_spatial_hash
        let grid_solid = self.grid_solid.clone();
        let cs = self.cell_size;
        
        self.particles.par_iter_mut().for_each(|p| {
            p.velocity += GRAVITY * dt;
            p.old_position = p.position;
            
            let mut new_pos = p.position + p.velocity * dt;
            
            // Simple Boundary/Solid Collision
            // Map pos to grid
            let cx = (new_pos.x / cs) as i32;
            let cy = (new_pos.y / cs) as i32;
            
            let mut collided = false;
            
            // Wall Clamping (Screen bounds)
            if new_pos.x < cs { new_pos.x = cs; collided = true; }
            if new_pos.x > self.width - cs { new_pos.x = self.width - cs; collided = true; }
            if new_pos.y < cs { new_pos.y = cs; collided = true; }
            if new_pos.y > self.height - cs { new_pos.y = self.height - cs; collided = true; }
            
            // Internal Solid Grid Collision
            if !collided && cx >= 0 && cy >= 0 && (cx as usize) < grid_w && (cy as usize) < grid_h {
                 let idx = (cy as usize) * grid_w + (cx as usize);
                 if grid_solid[idx] {
                     // Hit a solid cell!
                     // Rudimentary resolve: push back to old position
                     // Logic: if we moved into solid, revert to old_pos which (presumably) was Free.
                     // And kill velocity (sticky) or reflect?
                     // Let's try "Sticky" for mud/sluice behavior
                     new_pos = p.old_position;
                     p.velocity *= 0.0; // Stop dead
                 }
            }

            p.position = new_pos;
        });

        // 2. Neighbors
        self.build_spatial_hash();

        // 3. Solve (Iterative)
        let grid_width = grid_w as i32;
        let grid_height = grid_h as i32;
        let h = self.cell_size * H_SCALE;
        let h2 = h * h;

        for _ in 0..SOLVER_ITERATIONS {
            // Lambda Pass
            let lambdas: Vec<f32> = self.particles.par_iter().enumerate().map(|(i, p_i)| {
                let mut density = 0.0;
                let mut sum_grad_sq = 0.0;
                let mut grad_i = Vec2::ZERO;
                let cx = (p_i.position.x / self.cell_size) as i32;
                let cy = (p_i.position.y / self.cell_size) as i32;

                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let nx = cx + dx;
                        let ny = cy + dy;
                        if nx >= 0 && nx < grid_width && ny >= 0 && ny < grid_height {
                             let cell = (ny * grid_width + nx) as usize;
                             if cell < self.grid_heads.len() {
                                 let mut j = self.grid_heads[cell];
                                 while j != -1 {
                                     let j_idx = j as usize;
                                     if j_idx < self.particles.len() {
                                         let p_j = &self.particles[j_idx];
                                         let r_vec = p_i.position - p_j.position;
                                         let r2 = r_vec.length_squared();
                                         if r2 < h2 {
                                             density += poly6_kernel_2d(r2, h);
                                              if i != j_idx {
                                                 let r = r2.sqrt();
                                                 let grad_j = spiky_kernel_gradient_2d(r_vec, r, h) * (1.0 / REST_DENSITY);
                                                 sum_grad_sq += grad_j.length_squared();
                                                 grad_i += grad_j;
                                              }
                                         }
                                     }
                                     j = self.grid_next[j_idx];
                                 }
                             }
                        }
                    }
                }
                
                sum_grad_sq += grad_i.length_squared();
                let c = density / REST_DENSITY - 1.0;
                if c > 0.0 { -c / (sum_grad_sq + EPSILON) } else { 0.0 }
            }).collect();

            // Apply Lambdas
            self.particles.par_iter_mut().zip(lambdas).for_each(|(p, l)| p.lambda = l);
            
            // Delta Pos Pass
            let deltas: Vec<Vec2> = self.particles.par_iter().enumerate().map(|(i, p_i)| {
                let mut delta = Vec2::ZERO;
                let cx = (p_i.position.x / self.cell_size) as i32;
                let cy = (p_i.position.y / self.cell_size) as i32;

                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let nx = cx + dx;
                        let ny = cy + dy;
                        if nx >= 0 && nx < grid_width && ny >= 0 && ny < grid_height {
                             let cell = (ny * grid_width + nx) as usize;
                             if cell < self.grid_heads.len() {
                                 let mut j = self.grid_heads[cell];
                                 while j != -1 {
                                     let j_idx = j as usize;
                                      if i != j_idx && j_idx < self.particles.len() {
                                         let p_j = &self.particles[j_idx];
                                         let r_vec = p_i.position - p_j.position;
                                         let r2 = r_vec.length_squared();
                                         if r2 < h2 {
                                             let r = r2.sqrt();
                                             let w_dq = poly6_kernel_2d((0.2 * h).powi(2), h);
                                             let w_p = poly6_kernel_2d(r2, h);
                                             let s_corr = -0.1 * (w_p / w_dq).powi(4);
                                             
                                             let grad = spiky_kernel_gradient_2d(r_vec, r, h);
                                             delta += grad * (p_i.lambda + p_j.lambda + s_corr);
                                         }
                                      }
                                     j = self.grid_next[j_idx];
                                 }
                             }
                        }
                    }
                }
                delta / REST_DENSITY
            }).collect();

            // Apply Deltas (with Solid Collision Check)
            self.particles.par_iter_mut().zip(deltas).for_each(|(p, d)| {
                let mut new_pos = p.position + d;
                
                // Re-check Collision (Constraint projection could push us into walls)
                 let cx = (new_pos.x / cs) as i32;
                 let cy = (new_pos.y / cs) as i32;
                 let mut collided = false;
                 
                if new_pos.x < cs { new_pos.x = cs; collided = true; }
                if new_pos.x > self.width - cs { new_pos.x = self.width - cs; collided = true; }
                if new_pos.y < cs { new_pos.y = cs; collided = true; }
                if new_pos.y > self.height - cs { new_pos.y = self.height - cs; collided = true; }
                
                 if !collided && cx >= 0 && cy >= 0 && (cx as usize) < grid_w && (cy as usize) < grid_h {
                     let idx = (cy as usize) * grid_w + (cx as usize);
                     if grid_solid[idx] {
                         // Push back to pre-delta position (simple fix)
                         new_pos = p.position;
                     }
                 }
                p.position = new_pos;
            });
        }

        // 4. Update Velocities
        self.particles.par_iter_mut().for_each(|p| {
            p.velocity = (p.position - p.old_position) / dt;
            p.velocity *= 0.999;
        });

        // 5. Vorticity Confinement (New)
        // Pass A: Calculate Vorticity (Curl)
        let vorticities: Vec<f32> = self.particles.par_iter().enumerate().map(|(i, p_i)| {
            let mut curl = 0.0;
            let cx = (p_i.position.x / self.cell_size) as i32;
            let cy = (p_i.position.y / self.cell_size) as i32;
            
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let nx = cx + dx;
                    let ny = cy + dy;
                     if nx >= 0 && nx < grid_width && ny >= 0 && ny < grid_height {
                        let cell = (ny * grid_width + nx) as usize;
                        if cell < self.grid_heads.len() {
                            let mut j = self.grid_heads[cell];
                            while j != -1 {
                                let j_idx = j as usize;
                                 if i != j_idx && j_idx < self.particles.len() {
                                    let p_j = &self.particles[j_idx];
                                    let r_vec = p_i.position - p_j.position;
                                    let r2 = r_vec.length_squared();
                                    if r2 < h2 {
                                        let r = r2.sqrt();
                                        let grad = spiky_kernel_gradient_2d(r_vec, r, h);
                                        // Curl_2D = (v_diff x grad_W) = v_diff.x * grad.y - v_diff.y * grad.x
                                        let v_diff = p_j.velocity - p_i.velocity;
                                        curl += v_diff.x * grad.y - v_diff.y * grad.x;
                                    }
                                 }
                                j = self.grid_next[j_idx];
                            }
                        }
                    }
                }
            }
            curl
        }).collect();
        
        // Apply curl back to particles
        self.particles.par_iter_mut().zip(vorticities).for_each(|(p, v)| p.vorticity = v);

        // Pass B: Confinement Force
        let vorticity_forces: Vec<Vec2> = self.particles.par_iter().enumerate().map(|(i, p_i)| {
            let mut force = Vec2::ZERO;
            let mut grad_mag_omega = Vec2::ZERO; // Gradient of vorticity magnitude
            
            let cx = (p_i.position.x / self.cell_size) as i32;
            let cy = (p_i.position.y / self.cell_size) as i32;

             for dy in -1..=1 {
                for dx in -1..=1 {
                    let nx = cx + dx;
                    let ny = cy + dy;
                     if nx >= 0 && nx < grid_width && ny >= 0 && ny < grid_height {
                        let cell = (ny * grid_width + nx) as usize;
                        if cell < self.grid_heads.len() {
                            let mut j = self.grid_heads[cell];
                            while j != -1 {
                                let j_idx = j as usize;
                                 if i != j_idx && j_idx < self.particles.len() {
                                    let p_j = &self.particles[j_idx];
                                    let r_vec = p_i.position - p_j.position;
                                    let r2 = r_vec.length_squared();
                                    if r2 < h2 {
                                        let r = r2.sqrt();
                                        let grad = spiky_kernel_gradient_2d(r_vec, r, h);
                                        let omega_len_diff = p_j.vorticity.abs() - p_i.vorticity.abs();
                                        grad_mag_omega += grad * omega_len_diff;
                                    }
                                 }
                                j = self.grid_next[j_idx];
                            }
                        }
                    }
                }
            }
            
            let eps = 1e-5;
            let len_sq = grad_mag_omega.length_squared();
            if len_sq > eps {
                let N = grad_mag_omega / len_sq.sqrt();
                // F = epsilon * h * (N x omega_i)
                // In 2D: N=(Nx, Ny), omega scalar. Cross => (Ny * omega, -Nx * omega)
                force = Vec2::new(N.y * p_i.vorticity, -N.x * p_i.vorticity) * VORTICITY_EPSILON;
            }
            force
        }).collect();

        // Apply Forces & Damping
        self.particles.par_iter_mut().zip(vorticity_forces).for_each(|(p, f_v)| {
            p.velocity += f_v * dt;
            // XSPH Viscosity could go here, but simple damping is cleaner for now
            p.velocity *= 0.999;
        });
    }

    fn build_spatial_hash(&mut self) {
        self.grid_heads.fill(-1);
        self.grid_next.clear();
        self.grid_next.resize(self.particles.len(), -1);

        let grid_width = (self.width / self.cell_size).ceil() as usize;

        // Serial build for now, but fast enough for 4000
        for (i, p) in self.particles.iter().enumerate() {
            let cx = (p.position.x / self.cell_size) as usize;
            let cy = (p.position.y / self.cell_size) as usize;
            if cx < grid_width && cy * grid_width + cx < self.grid_heads.len() {
                let cell = cy * grid_width + cx;
                self.grid_next[i] = self.grid_heads[cell];
                self.grid_heads[cell] = i as i32;
            }
        }
    }
    
    pub fn spawn_particle(&mut self, pos: Vec2, vel: Vec2) {
        if self.particles.len() < MAX_PARTICLES {
            self.particles.push(PbfParticle {
                position: pos,
                old_position: pos,
                velocity: vel,
                density: 0.0,
                lambda: 0.0,
                phase: 0,
                vorticity: 0.0,
            });
        }
    }

    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }
}

// 2D Kernels
// Poly6 2D: (4 / (pi * h^8)) * (h^2 - r^2)^3
fn poly6_kernel_2d(r2: f32, h: f32) -> f32 {
    let h2 = h * h;
    if r2 >= h2 { return 0.0; }
    let term = h2 - r2;
    (4.0 / (PI * h.powi(8))) * term.powi(3)
}

// Spiky Gradient 2D: -(30 / (pi * h^5)) * (h - r)^2 * r_norm
fn spiky_kernel_gradient_2d(r_vec: Vec2, r: f32, h: f32) -> Vec2 {
    if r >= h || r <= 1e-5 { return Vec2::ZERO; }
    let term = h - r;
    let scalar = -30.0 / (PI * h.powi(5)) * term * term;
    r_vec * (scalar / r)
}
