use glam::Vec2;
use std::f32::consts::PI;

use sim::{Particles, ParticleMaterial, physics};

/// Gravity vector derived from unified physics constant.
const GRAVITY_VEC: Vec2 = Vec2::new(0.0, physics::GRAVITY);

// Core PBF Parameters
const SOLVER_ITERATIONS: usize = 4; // Default iterations (adapted dynamically)
const REST_DENSITY: f32 = 1.0;  // Normalized density for SPH
const H_SCALE: f32 = 1.5; // Radius of support relative to cell size
const EPSILON: f32 = 0.0001; // Avoid div by zero
const MAX_PARTICLES_CAP: usize = 3000; // Hard cap for performance safety

pub struct DfsphSimulation {
    /// Core particles (position, velocity, material) - shared with renderer
    pub particles: Particles,

    // --- Auxiliary State (Structure of Arrays) ---
    // These must be kept in sync with particles.list
    pub old_positions: Vec<Vec2>,
    pub densities: Vec<f32>,
    pub lambdas: Vec<f32>,

    // --- Solver Buffers (pre-allocated to avoid per-iteration allocation) ---
    positions_buffer: Vec<Vec2>,
    deltas_buffer: Vec<Vec2>,

    // Simulation Bounds
    pub width: f32,
    pub height: f32,
    pub cell_size: f32,
    
    // Spatial Hash Grid
    grid_heads: Vec<i32>,
    grid_next: Vec<i32>,
    pub grid_solid: Vec<bool>,
    
    // Simulation Parameters
    pub viscosity: f32,
    pub near_pressure_h: f32,
    pub near_pressure_rest: f32,
    pub use_viscosity: bool,

    // Precomputed SPH kernel coefficients (avoid powi in hot path)
    poly6_coeff: f32,   // 4.0 / (PI * h^8)
    spiky_coeff: f32,   // -30.0 / (PI * h^5)
    h_squared: f32,     // h * h
}

impl DfsphSimulation {
    pub fn new(width: usize, height: usize, cell_size: f32) -> Self {
        let w_f = width as f32 * cell_size;
        let h_f = height as f32 * cell_size;
        
        let num_cells = width * height;

        // Precompute kernel coefficients
        let h = cell_size * H_SCALE;
        let poly6_coeff = 4.0 / (PI * h.powi(8));
        let spiky_coeff = -30.0 / (PI * h.powi(5));
        let h_squared = h * h;

        Self {
            particles: Particles::new(),
            old_positions: Vec::with_capacity(MAX_PARTICLES_CAP),
            densities: Vec::with_capacity(MAX_PARTICLES_CAP),
            lambdas: Vec::with_capacity(MAX_PARTICLES_CAP),

            // Initialize solver buffers with capacity
            positions_buffer: Vec::with_capacity(MAX_PARTICLES_CAP),
            deltas_buffer: Vec::with_capacity(MAX_PARTICLES_CAP),

            width: w_f,
            height: h_f,
            cell_size,

            grid_heads: vec![-1; num_cells],
            grid_next: Vec::with_capacity(MAX_PARTICLES_CAP),
            grid_solid: vec![false; num_cells],

            // Default tunings
            viscosity: 0.5,
            near_pressure_h: cell_size * 2.0,
            near_pressure_rest: 1.0,
            use_viscosity: true,

            // Precomputed kernel coefficients
            poly6_coeff,
            spiky_coeff,
            h_squared,
        }
    }
    
    pub fn spawn_particles(
        &mut self,
        x: f32,
        y: f32,
        vx: f32,
        vy: f32,
        count: usize,
        material: ParticleMaterial,
        jitter: f32,
    ) {
        for _ in 0..count {
            let pos = Vec2::new(
                x + (rand::random::<f32>() - 0.5) * jitter,
                y + (rand::random::<f32>() - 0.5) * jitter,
            );
            self.spawn_particle_internal(pos, Vec2::new(vx, vy), material);
        }
    }

    pub fn spawn_particle_internal(&mut self, pos: Vec2, vel: Vec2, mat: ParticleMaterial) {
        if self.particles.len() >= MAX_PARTICLES_CAP { return; }
        
        // Push to core particles
        match mat {
            ParticleMaterial::Water => self.particles.spawn_water(pos.x, pos.y, vel.x, vel.y),
            ParticleMaterial::Mud => self.particles.spawn_mud(pos.x, pos.y, vel.x, vel.y),
            ParticleMaterial::Sand => self.particles.spawn_sand(pos.x, pos.y, vel.x, vel.y),
            ParticleMaterial::Magnetite => self.particles.spawn_magnetite(pos.x, pos.y, vel.x, vel.y),
            ParticleMaterial::Gold => self.particles.spawn_gold(pos.x, pos.y, vel.x, vel.y),
        }
        
        // Sync aux vectors
        self.old_positions.push(pos);
        self.densities.push(0.0);
        self.lambdas.push(0.0);
    }
    
    fn sync_aux_vectors(&mut self) {
        // Ensure aux vectors match particle count (incase particles were removed/added externally or Desync)
        let len = self.particles.len();
        self.old_positions.resize(len, Vec2::ZERO);
        self.densities.resize(len, 0.0);
        self.lambdas.resize(len, 0.0);
    }

    pub fn update(&mut self, dt: f32) {
        debug_assert!(dt > 0.0 && dt.is_finite(), "Invalid timestep: {}", dt);
        if dt <= 0.0 || !dt.is_finite() {
            return;
        }

        self.sync_aux_vectors();
        // 0. CFL: Determine subtops
        // Max displacement should be ~0.9 * particle_radius per step (PBF is stable).
        // H ~ 1.5 * cell_size. Radius approx cell_size/2.
        // Limit: 0.9 * (cell_size * 0.5) 
        let limit = 0.9 * self.cell_size * 0.5;
        
        let max_vel_sq: f32 = self.particles.list.iter()
            .map(|p| p.velocity.length_squared())
            .fold(0.0f32, |a, b| a.max(b));
            
        let max_vel = max_vel_sq.sqrt();
        
        let mut num_substeps = 1;
        if max_vel > 0.001 {
            let min_dt = limit / max_vel;
            num_substeps = (dt / min_dt).ceil() as usize;
        }
        
        // Safety clamps
        if num_substeps < 1 { num_substeps = 1; }
        if num_substeps > 4 { num_substeps = 4; } // Cap at 4 for performance (was 8)
        
        let dt_sub = dt / (num_substeps as f32);

        for _sub in 0..num_substeps {
            self.perform_substep(dt_sub, num_substeps);
        }
        
        // 5. XSPH Viscosity (Applied once per frame for performance)
        // This is an approximation but sufficient for visual stability.
        if self.use_viscosity {
             self.apply_xsph(dt);
        }
    }
    
    fn apply_xsph(&mut self, _dt: f32) {
         let cs = self.cell_size;
         let h2 = self.h_squared;
         let poly6_coeff = self.poly6_coeff;
         let grid_w = (self.width / self.cell_size).ceil() as usize;
         let grid_h = (self.height / self.cell_size).ceil() as usize;
         let heads = &self.grid_heads;
         let next = &self.grid_next;

         let particles_slice = &mut self.particles.list;
         let positions: Vec<Vec2> = particles_slice.iter().map(|p| p.position).collect();
         let velocities: Vec<Vec2> = particles_slice.iter().map(|p| p.velocity).collect();
         let c = 0.05; // Viscosity

         let vis_deltas: Vec<Vec2> = velocities.iter().enumerate().map(|(i, v_i)| {
             let mut v_delta = Vec2::ZERO;
             let pos_i = positions[i];
             let cx = (pos_i.x / cs) as i32;
             let cy = (pos_i.y / cs) as i32;
             
              for dy in -1..=1 {
                for dx in -1..=1 {
                    let nx = cx + dx;
                    let ny = cy + dy;
                    if nx >= 0 && nx < grid_w as i32 && ny >= 0 && ny < grid_h as i32 {
                        let cell = (ny * (grid_w as i32) + nx) as usize;
                        let mut j = heads[cell];
                        while j != -1 {
                            let j_idx = j as usize;
                            if j_idx < velocities.len() && i != j_idx {
                                let v_j = velocities[j_idx];
                                let pos_j = positions[j_idx];
                                let r_vec = pos_i - pos_j;
                                let r2 = r_vec.length_squared();
                                if r2 < h2 {
                                    let w = poly6_kernel_2d(r2, h2, poly6_coeff);
                                    v_delta += (v_j - *v_i) * w;
                                }
                            }
                            j = next[j_idx];
                        }
                    }
                }
             }
             v_delta * c
         }).collect();
         
         // Apply
         particles_slice.iter_mut().zip(vis_deltas.iter()).for_each(|(p, d)| {
             p.velocity += *d;
         });
    }

    // Pass num_substeps to optimize iterations
    fn perform_substep(&mut self, dt: f32, total_substeps: usize) {

        let grid_w = (self.width / self.cell_size).ceil() as usize;
        let grid_h = (self.height / self.cell_size).ceil() as usize;
        
        // 1. Prediction (Apply Gravity & Predict Position)
        {
             let particles_slice = &mut self.particles.list;
             let old_pos_slice = &mut self.old_positions;
             let grid_solid = &self.grid_solid;
             let w_f = self.width;
             let h_f = self.height;
             let cs = self.cell_size;

             particles_slice.iter_mut().zip(old_pos_slice.iter_mut()).for_each(|(p, old_pos)| {
                // Apply Gravity!
                p.velocity += GRAVITY_VEC * dt;
                
                // Damping (keeping small air drag)
                p.velocity *= 0.999; 

                *old_pos = p.position;
                
                // Prediction: integration
                let mut temp_pos = p.position + p.velocity * dt;
                
                // --- CONTINUOUS COLLISION DETECTION (CCD) ---
                let is_solid = |pos: Vec2| -> bool {
                    let cx = (pos.x / cs) as i32;
                    let cy = (pos.y / cs) as i32;
                     if cx >= 0 && cy >= 0 && (cx as usize) < grid_w && (cy as usize) < grid_h {
                         let idx = (cy as usize) * grid_w + (cx as usize);
                         return idx < grid_solid.len() && grid_solid[idx];
                     }
                     false
                };

                // Wall Bounds (Screen)
                let restitution = 0.5;
                
                if temp_pos.x < cs { temp_pos.x = cs; p.velocity.x *= -restitution; }
                if temp_pos.x > w_f - cs { temp_pos.x = w_f - cs; p.velocity.x *= -restitution; }
                if temp_pos.y < cs { temp_pos.y = cs; p.velocity.y *= -restitution; }
                if temp_pos.y > h_f - cs { temp_pos.y = h_f - cs; p.velocity.y *= -restitution; }
                
                // Grid Solids
                if is_solid(temp_pos) {
                    let start_cell_x = ((*old_pos).x / cs) as i32;
                    let start_cell_y = ((*old_pos).y / cs) as i32;
                    let end_cell_x = (temp_pos.x / cs) as i32;
                    let end_cell_y = (temp_pos.y / cs) as i32;

                    if start_cell_x != end_cell_x {
                        // X Crossing
                        p.velocity.x *= -restitution; 
                        temp_pos.x = (*old_pos).x; 
                    }
                    
                    if start_cell_y != end_cell_y {
                        // Y Crossing
                         p.velocity.y *= -restitution;
                         temp_pos.y = (*old_pos).y;
                    }
                    
                    if is_solid(temp_pos) {
                        temp_pos = *old_pos;
                        p.velocity = Vec2::ZERO; 
                    }
                }
                
                p.position = temp_pos;
            });
        }
        
        // 2. Build Neighbors
        self.build_spatial_hash();
        
        // 3. Solver Loop
        // Adapt iterations: If we step 4 times, 1 iter is enough (total 4).
        // If we step 1 time, 3 iters needed.
        let iterations = if total_substeps >= 4 { 1 } else if total_substeps >= 2 { 2 } else { 3 };
        
        let cs = self.cell_size;
        let h = self.cell_size * H_SCALE;
        let h2 = self.h_squared;
        let poly6_coeff = self.poly6_coeff;
        let spiky_coeff = self.spiky_coeff;
        // Constants for closures
        let heads = &self.grid_heads;
        let next = &self.grid_next;
        let grid_solid = &self.grid_solid;
        let w_f = self.width;
        let h_f = self.height;

        {
            let particles_slice = &mut self.particles.list;
            let lambdas_slice = &mut self.lambdas;

            // Use persistent buffer instead of allocating new Vec
            self.positions_buffer.clear();
            self.positions_buffer.extend(particles_slice.iter().map(|p| p.position));

            for _ in 0..iterations {
                // Lambda Pass
                lambdas_slice.iter_mut().enumerate().for_each(|(i, lambda)| {
                    let pos_i = self.positions_buffer[i];
                    let mut density = 0.0;
                    let mut sum_grad_sq = 0.0;
                    let mut grad_i = Vec2::ZERO;
                    
                    let cx = (pos_i.x / cs) as i32;
                    let cy = (pos_i.y / cs) as i32;
                    
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let nx = cx + dx;
                            let ny = cy + dy;
                            if nx >= 0 && nx < grid_w as i32 && ny >= 0 && ny < grid_h as i32 {
                                let cell = (ny * (grid_w as i32) + nx) as usize;
                                let mut j = heads[cell];
                                while j != -1 {
                                    let j_idx = j as usize;
                                    if j_idx < self.positions_buffer.len() {
                                        let pos_j = self.positions_buffer[j_idx];
                                        let r_vec = pos_i - pos_j;
                                        let r2 = r_vec.length_squared();
                                        if r2 < h2 {
                                            density += poly6_kernel_2d(r2, h2, poly6_coeff);
                                            if i != j_idx {
                                                let r = r2.sqrt();
                                                let grad = spiky_kernel_gradient_2d(r_vec, r, h, h2, spiky_coeff) * (1.0 / REST_DENSITY);
                                                sum_grad_sq += grad.length_squared();
                                                grad_i += grad;
                                            }
                                        }
                                    }
                                    j = next[j_idx];
                                }
                            }
                        }
                    }
                    sum_grad_sq += grad_i.length_squared();
                    let c = density / REST_DENSITY - 1.0;
                    *lambda = if c > 0.0 { -c / (sum_grad_sq + EPSILON) } else { 0.0 };
                });
                
                // Delta Pos Pass - reuse deltas_buffer instead of allocating
                let lambdas_read = &*lambdas_slice; // Re-borrow

                self.deltas_buffer.clear();
                self.deltas_buffer.extend(self.positions_buffer.iter().enumerate().map(|(i, pos_i)| {
                    let mut delta = Vec2::ZERO;
                    let cx = (pos_i.x / cs) as i32;
                    let cy = (pos_i.y / cs) as i32;
                    
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let nx = cx + dx;
                            let ny = cy + dy;
                            if nx >= 0 && nx < grid_w as i32 && ny >= 0 && ny < grid_h as i32 {
                                let cell = (ny * (grid_w as i32) + nx) as usize;
                                let mut j = heads[cell];
                                while j != -1 {
                                    let j_idx = j as usize;
                                    if j_idx < self.positions_buffer.len() && i != j_idx {
                                        let pos_j = self.positions_buffer[j_idx];
                                        let r_vec = *pos_i - pos_j;
                                        let r2 = r_vec.length_squared();
                                        if r2 < h2 {
                                            let r = r2.sqrt();
                                            // Tensile instability correction
                                            // Note: (0.2 * h)^2 = 0.04 * h^2 is precomputed inline
                                            let dq_dist_sq = 0.04 * h2;
                                            let w_dq = poly6_kernel_2d(dq_dist_sq, h2, poly6_coeff);
                                            let w_p = poly6_kernel_2d(r2, h2, poly6_coeff);
                                            let s_corr = -0.1 * (w_p / w_dq).powi(4);

                                            let grad = spiky_kernel_gradient_2d(r_vec, r, h, h2, spiky_coeff);
                                            let lambda_i = lambdas_read[i];
                                            let lambda_j = lambdas_read[j_idx];
                                            delta += grad * (lambda_i + lambda_j + s_corr);
                                        }
                                    }
                                    j = next[j_idx];
                                }
                            }
                        }
                    }
                    delta / REST_DENSITY
                }));

                // Apply deltas to particles and update positions_buffer in-place
                particles_slice.iter_mut().enumerate().zip(self.deltas_buffer.iter()).for_each(|((i, p), d)| {
                    let mut new_pos = p.position + *d;
                    let cx = (new_pos.x / cs) as i32;
                    let cy = (new_pos.y / cs) as i32;
                    let mut collided = false;
                    if new_pos.x < cs { new_pos.x = cs; collided = true; }
                    if new_pos.x > w_f - cs { new_pos.x = w_f - cs; collided = true; }
                    if new_pos.y < cs { new_pos.y = cs; collided = true; }
                    if new_pos.y > h_f - cs { new_pos.y = h_f - cs; collided = true; }
                    
                    if !collided && cx >= 0 && cy >= 0 && (cx as usize) < grid_w && (cy as usize) < grid_h {
                        let idx = (cy as usize) * grid_w + (cx as usize);
                        if idx < grid_solid.len() && grid_solid[idx] {
                            new_pos = p.position;
                        }
                    }
                    p.position = new_pos;
                    self.positions_buffer[i] = new_pos; // Update buffer in-place instead of re-collecting
                });
            }
        }
        
        // 4. Update Velocity
        {
             let particles_slice = &mut self.particles.list;
             let old_pos_slice = &mut self.old_positions;

             particles_slice.iter_mut().zip(old_pos_slice.iter()).for_each(|(p, old_pos)| {
                p.velocity = (p.position - *old_pos) / dt;
            });
        }
    }
    
    fn build_spatial_hash(&mut self) {
        self.grid_heads.fill(-1);
        self.grid_next.clear();
        self.grid_next.resize(self.particles.len(), -1);
        let grid_w = (self.width / self.cell_size).ceil() as usize;

        for (i, p) in self.particles.iter().enumerate() {
            let cx = (p.position.x / self.cell_size) as usize;
            let cy = (p.position.y / self.cell_size) as usize;
            if cx < grid_w {
                let cell = cy * grid_w + cx;
                if cell < self.grid_heads.len() {
                    self.grid_next[i] = self.grid_heads[cell];
                    self.grid_heads[cell] = i as i32;
                }
            }
        }
    }
}

// Optimized Kernels (no powi in hot path)
#[inline(always)]
fn poly6_kernel_2d(r2: f32, h2: f32, coeff: f32) -> f32 {
    if r2 >= h2 { return 0.0; }
    let term = h2 - r2;
    coeff * term * term * term
}

#[inline(always)]
fn spiky_kernel_gradient_2d(r_vec: Vec2, r: f32, h: f32, _h2: f32, coeff: f32) -> Vec2 {
    if r >= h || r <= 1e-5 { return Vec2::ZERO; }
    let term = h - r;
    r_vec * (coeff * term * term / r)
}
