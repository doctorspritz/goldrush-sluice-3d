//! Particle-Grid transfer functions for 3D FLIP/APIC.
//!
//! P2G: Scatter particle momentum to grid using quadratic B-spline kernels.
//! G2P: Gather grid velocity to particles with APIC affine velocity reconstruction.

use glam::{Mat3, Vec3};

use crate::grid::Grid3D;
use crate::kernels::{apic_d_inverse, quadratic_bspline_1d};
use crate::particle::Particles3D;

/// Pre-allocated buffers for P2G transfer (avoids allocation each frame).
pub struct TransferBuffers {
    pub u_sum: Vec<f32>,
    pub u_weight: Vec<f32>,
    pub v_sum: Vec<f32>,
    pub v_weight: Vec<f32>,
    pub w_sum: Vec<f32>,
    pub w_weight: Vec<f32>,
}

impl TransferBuffers {
    /// Create buffers sized for the given grid.
    pub fn new(grid: &Grid3D) -> Self {
        let u_count = (grid.width + 1) * grid.height * grid.depth;
        let v_count = grid.width * (grid.height + 1) * grid.depth;
        let w_count = grid.width * grid.height * (grid.depth + 1);

        Self {
            u_sum: vec![0.0; u_count],
            u_weight: vec![0.0; u_count],
            v_sum: vec![0.0; v_count],
            v_weight: vec![0.0; v_count],
            w_sum: vec![0.0; w_count],
            w_weight: vec![0.0; w_count],
        }
    }

    /// Clear all buffers to zero.
    pub fn clear(&mut self) {
        self.u_sum.fill(0.0);
        self.u_weight.fill(0.0);
        self.v_sum.fill(0.0);
        self.v_weight.fill(0.0);
        self.w_sum.fill(0.0);
        self.w_weight.fill(0.0);
    }
}

/// Transfer particle velocities to grid (P2G) using APIC.
///
/// Each velocity component uses a 3x3x3 stencil centered on the particle.
/// APIC adds the affine velocity term: momentum = (v + C*(x_node - x_particle)) * weight
pub fn particles_to_grid(
    grid: &mut Grid3D,
    particles: &Particles3D,
    buffers: &mut TransferBuffers,
) {
    buffers.clear();

    let cell_size = grid.cell_size;
    let inv_cell_size = 1.0 / cell_size;

    for particle in &particles.list {
        let pos = particle.position;
        let vel = particle.velocity;
        let c_mat = particle.affine_velocity;

        // ===== U component (on left YZ faces) =====
        // U nodes at (i * dx, (j + 0.5) * dx, (k + 0.5) * dx)
        let u_pos = pos * inv_cell_size - Vec3::new(0.0, 0.5, 0.5);
        let u_base = u_pos.floor();
        let u_frac = u_pos - u_base;
        let u_base_i = u_base.x as i32;
        let u_base_j = u_base.y as i32;
        let u_base_k = u_base.z as i32;

        // Precompute 1D weights
        let u_wx = [
            quadratic_bspline_1d(u_frac.x + 1.0),
            quadratic_bspline_1d(u_frac.x),
            quadratic_bspline_1d(u_frac.x - 1.0),
        ];
        let u_wy = [
            quadratic_bspline_1d(u_frac.y + 1.0),
            quadratic_bspline_1d(u_frac.y),
            quadratic_bspline_1d(u_frac.y - 1.0),
        ];
        let u_wz = [
            quadratic_bspline_1d(u_frac.z + 1.0),
            quadratic_bspline_1d(u_frac.z),
            quadratic_bspline_1d(u_frac.z - 1.0),
        ];

        // 3x3x3 stencil for U
        for dk in -1i32..=1 {
            for dj in -1i32..=1 {
                for di in -1i32..=1 {
                    let ni = u_base_i + di;
                    let nj = u_base_j + dj;
                    let nk = u_base_k + dk;

                    // Bounds check for U array: (width+1) x height x depth
                    if ni < 0
                        || ni > grid.width as i32
                        || nj < 0
                        || nj >= grid.height as i32
                        || nk < 0
                        || nk >= grid.depth as i32
                    {
                        continue;
                    }

                    let w =
                        u_wx[(di + 1) as usize] * u_wy[(dj + 1) as usize] * u_wz[(dk + 1) as usize];

                    if w < 1e-10 {
                        continue;
                    }

                    // APIC: compute affine velocity contribution
                    let node_pos = grid.u_position(ni as usize, nj as usize, nk as usize);
                    let offset = node_pos - pos;
                    let affine_vel = c_mat * offset;

                    let idx = grid.u_index(ni as usize, nj as usize, nk as usize);
                    buffers.u_sum[idx] += (vel.x + affine_vel.x) * w;
                    buffers.u_weight[idx] += w;
                }
            }
        }

        // ===== V component (on bottom XZ faces) =====
        // V nodes at ((i + 0.5) * dx, j * dx, (k + 0.5) * dx)
        let v_pos = pos * inv_cell_size - Vec3::new(0.5, 0.0, 0.5);
        let v_base = v_pos.floor();
        let v_frac = v_pos - v_base;
        let v_base_i = v_base.x as i32;
        let v_base_j = v_base.y as i32;
        let v_base_k = v_base.z as i32;

        let v_wx = [
            quadratic_bspline_1d(v_frac.x + 1.0),
            quadratic_bspline_1d(v_frac.x),
            quadratic_bspline_1d(v_frac.x - 1.0),
        ];
        let v_wy = [
            quadratic_bspline_1d(v_frac.y + 1.0),
            quadratic_bspline_1d(v_frac.y),
            quadratic_bspline_1d(v_frac.y - 1.0),
        ];
        let v_wz = [
            quadratic_bspline_1d(v_frac.z + 1.0),
            quadratic_bspline_1d(v_frac.z),
            quadratic_bspline_1d(v_frac.z - 1.0),
        ];

        for dk in -1i32..=1 {
            for dj in -1i32..=1 {
                for di in -1i32..=1 {
                    let ni = v_base_i + di;
                    let nj = v_base_j + dj;
                    let nk = v_base_k + dk;

                    // Bounds check for V array: width x (height+1) x depth
                    if ni < 0
                        || ni >= grid.width as i32
                        || nj < 0
                        || nj > grid.height as i32
                        || nk < 0
                        || nk >= grid.depth as i32
                    {
                        continue;
                    }

                    let w =
                        v_wx[(di + 1) as usize] * v_wy[(dj + 1) as usize] * v_wz[(dk + 1) as usize];

                    if w < 1e-10 {
                        continue;
                    }

                    let node_pos = grid.v_position(ni as usize, nj as usize, nk as usize);
                    let offset = node_pos - pos;
                    let affine_vel = c_mat * offset;

                    let idx = grid.v_index(ni as usize, nj as usize, nk as usize);
                    buffers.v_sum[idx] += (vel.y + affine_vel.y) * w;
                    buffers.v_weight[idx] += w;
                }
            }
        }

        // ===== W component (on back XY faces) =====
        // W nodes at ((i + 0.5) * dx, (j + 0.5) * dx, k * dx)
        let w_pos = pos * inv_cell_size - Vec3::new(0.5, 0.5, 0.0);
        let w_base = w_pos.floor();
        let w_frac = w_pos - w_base;
        let w_base_i = w_base.x as i32;
        let w_base_j = w_base.y as i32;
        let w_base_k = w_base.z as i32;

        let w_wx = [
            quadratic_bspline_1d(w_frac.x + 1.0),
            quadratic_bspline_1d(w_frac.x),
            quadratic_bspline_1d(w_frac.x - 1.0),
        ];
        let w_wy = [
            quadratic_bspline_1d(w_frac.y + 1.0),
            quadratic_bspline_1d(w_frac.y),
            quadratic_bspline_1d(w_frac.y - 1.0),
        ];
        let w_wz = [
            quadratic_bspline_1d(w_frac.z + 1.0),
            quadratic_bspline_1d(w_frac.z),
            quadratic_bspline_1d(w_frac.z - 1.0),
        ];

        for dk in -1i32..=1 {
            for dj in -1i32..=1 {
                for di in -1i32..=1 {
                    let ni = w_base_i + di;
                    let nj = w_base_j + dj;
                    let nk = w_base_k + dk;

                    // Bounds check for W array: width x height x (depth+1)
                    if ni < 0
                        || ni >= grid.width as i32
                        || nj < 0
                        || nj >= grid.height as i32
                        || nk < 0
                        || nk > grid.depth as i32
                    {
                        continue;
                    }

                    let w =
                        w_wx[(di + 1) as usize] * w_wy[(dj + 1) as usize] * w_wz[(dk + 1) as usize];

                    if w < 1e-10 {
                        continue;
                    }

                    let node_pos = grid.w_position(ni as usize, nj as usize, nk as usize);
                    let offset = node_pos - pos;
                    let affine_vel = c_mat * offset;

                    let idx = grid.w_index(ni as usize, nj as usize, nk as usize);
                    buffers.w_sum[idx] += (vel.z + affine_vel.z) * w;
                    buffers.w_weight[idx] += w;
                }
            }
        }
    }

    // Normalize: velocity = momentum / mass
    for i in 0..grid.u.len() {
        grid.u[i] = if buffers.u_weight[i] > 1e-10 {
            buffers.u_sum[i] / buffers.u_weight[i]
        } else {
            0.0
        };
    }

    for i in 0..grid.v.len() {
        grid.v[i] = if buffers.v_weight[i] > 1e-10 {
            buffers.v_sum[i] / buffers.v_weight[i]
        } else {
            0.0
        };
    }

    for i in 0..grid.w.len() {
        grid.w[i] = if buffers.w_weight[i] > 1e-10 {
            buffers.w_sum[i] / buffers.w_weight[i]
        } else {
            0.0
        };
    }
}

/// Transfer grid velocities to particles (G2P) with APIC.
///
/// Reconstructs the affine velocity matrix C for angular momentum conservation.
/// Uses FLIP/PIC blending for stability.
///
/// IMPORTANT: Normalizes by weight sum to handle boundary particles correctly.
/// Near boundaries, some stencil nodes are out-of-bounds, so weights don't sum to 1.0.
/// Without normalization, velocity gets artificially dampened.
pub fn grid_to_particles(grid: &Grid3D, particles: &mut Particles3D, flip_ratio: f32) {
    let cell_size = grid.cell_size;
    let inv_cell_size = 1.0 / cell_size;
    let d_inv = apic_d_inverse(cell_size);

    for particle in &mut particles.list {
        let pos = particle.position;
        let old_vel = particle.velocity;

        let mut new_vel = Vec3::ZERO;
        let mut new_c = Mat3::ZERO;
        let mut u_delta = 0.0f32;
        let mut v_delta = 0.0f32;
        let mut w_delta = 0.0f32;

        // ===== Sample U component =====
        let u_pos = pos * inv_cell_size - Vec3::new(0.0, 0.5, 0.5);
        let u_base = u_pos.floor();
        let u_frac = u_pos - u_base;
        let u_base_i = u_base.x as i32;
        let u_base_j = u_base.y as i32;
        let u_base_k = u_base.z as i32;

        let u_wx = [
            quadratic_bspline_1d(u_frac.x + 1.0),
            quadratic_bspline_1d(u_frac.x),
            quadratic_bspline_1d(u_frac.x - 1.0),
        ];
        let u_wy = [
            quadratic_bspline_1d(u_frac.y + 1.0),
            quadratic_bspline_1d(u_frac.y),
            quadratic_bspline_1d(u_frac.y - 1.0),
        ];
        let u_wz = [
            quadratic_bspline_1d(u_frac.z + 1.0),
            quadratic_bspline_1d(u_frac.z),
            quadratic_bspline_1d(u_frac.z - 1.0),
        ];

        let mut u_sum = 0.0f32;
        let mut u_old_sum = 0.0f32;
        let mut u_weight_sum = 0.0f32;
        let mut u_c_axis = Vec3::ZERO;

        for dk in -1i32..=1 {
            for dj in -1i32..=1 {
                for di in -1i32..=1 {
                    let ni = u_base_i + di;
                    let nj = u_base_j + dj;
                    let nk = u_base_k + dk;

                    // Bounds check for U array: (width+1) x height x depth
                    if ni < 0
                        || ni > grid.width as i32
                        || nj < 0
                        || nj >= grid.height as i32
                        || nk < 0
                        || nk >= grid.depth as i32
                    {
                        continue;
                    }

                    // CRITICAL: Skip solid faces - they have zero velocity and would
                    // drag down the weighted average, causing velocity decay
                    if grid.is_u_face_solid(ni, nj, nk) {
                        continue;
                    }

                    let w =
                        u_wx[(di + 1) as usize] * u_wy[(dj + 1) as usize] * u_wz[(dk + 1) as usize];

                    if w < 1e-10 {
                        continue;
                    }

                    let idx = grid.u_index(ni as usize, nj as usize, nk as usize);
                    let u_val = grid.u[idx];
                    let u_old_val = grid.u_old[idx];

                    u_sum += u_val * w;
                    u_old_sum += u_old_val * w;
                    u_weight_sum += w;

                    // APIC C matrix contribution (first column for U)
                    let node_pos = grid.u_position(ni as usize, nj as usize, nk as usize);
                    let offset = node_pos - pos;
                    u_c_axis += d_inv * w * u_val * offset;
                }
            }
        }

        // Normalize U by weight sum (now excludes solid faces)
        if u_weight_sum > 1e-6 {
            new_vel.x = u_sum / u_weight_sum;
            u_delta = (u_sum - u_old_sum) / u_weight_sum;
            new_c.x_axis = u_c_axis / u_weight_sum;
        }

        // ===== Sample V component =====
        let v_pos = pos * inv_cell_size - Vec3::new(0.5, 0.0, 0.5);
        let v_base = v_pos.floor();
        let v_frac = v_pos - v_base;
        let v_base_i = v_base.x as i32;
        let v_base_j = v_base.y as i32;
        let v_base_k = v_base.z as i32;

        let v_wx = [
            quadratic_bspline_1d(v_frac.x + 1.0),
            quadratic_bspline_1d(v_frac.x),
            quadratic_bspline_1d(v_frac.x - 1.0),
        ];
        let v_wy = [
            quadratic_bspline_1d(v_frac.y + 1.0),
            quadratic_bspline_1d(v_frac.y),
            quadratic_bspline_1d(v_frac.y - 1.0),
        ];
        let v_wz = [
            quadratic_bspline_1d(v_frac.z + 1.0),
            quadratic_bspline_1d(v_frac.z),
            quadratic_bspline_1d(v_frac.z - 1.0),
        ];

        let mut v_sum = 0.0f32;
        let mut v_old_sum = 0.0f32;
        let mut v_weight_sum = 0.0f32;
        let mut v_c_axis = Vec3::ZERO;

        for dk in -1i32..=1 {
            for dj in -1i32..=1 {
                for di in -1i32..=1 {
                    let ni = v_base_i + di;
                    let nj = v_base_j + dj;
                    let nk = v_base_k + dk;

                    // Bounds check for V array: width x (height+1) x depth
                    if ni < 0
                        || ni >= grid.width as i32
                        || nj < 0
                        || nj > grid.height as i32
                        || nk < 0
                        || nk >= grid.depth as i32
                    {
                        continue;
                    }

                    // Skip solid faces
                    if grid.is_v_face_solid(ni, nj, nk) {
                        continue;
                    }

                    let w =
                        v_wx[(di + 1) as usize] * v_wy[(dj + 1) as usize] * v_wz[(dk + 1) as usize];

                    if w < 1e-10 {
                        continue;
                    }

                    let idx = grid.v_index(ni as usize, nj as usize, nk as usize);
                    let v_val = grid.v[idx];
                    let v_old_val = grid.v_old[idx];

                    v_sum += v_val * w;
                    v_old_sum += v_old_val * w;
                    v_weight_sum += w;

                    // APIC C matrix contribution (second column for V)
                    let node_pos = grid.v_position(ni as usize, nj as usize, nk as usize);
                    let offset = node_pos - pos;
                    v_c_axis += d_inv * w * v_val * offset;
                }
            }
        }

        // Normalize V by weight sum (excludes solid faces)
        if v_weight_sum > 1e-6 {
            new_vel.y = v_sum / v_weight_sum;
            v_delta = (v_sum - v_old_sum) / v_weight_sum;
            new_c.y_axis = v_c_axis / v_weight_sum;
        }

        // ===== Sample W component =====
        let w_pos = pos * inv_cell_size - Vec3::new(0.5, 0.5, 0.0);
        let w_base = w_pos.floor();
        let w_frac = w_pos - w_base;
        let w_base_i = w_base.x as i32;
        let w_base_j = w_base.y as i32;
        let w_base_k = w_base.z as i32;

        let w_wx = [
            quadratic_bspline_1d(w_frac.x + 1.0),
            quadratic_bspline_1d(w_frac.x),
            quadratic_bspline_1d(w_frac.x - 1.0),
        ];
        let w_wy = [
            quadratic_bspline_1d(w_frac.y + 1.0),
            quadratic_bspline_1d(w_frac.y),
            quadratic_bspline_1d(w_frac.y - 1.0),
        ];
        let w_wz = [
            quadratic_bspline_1d(w_frac.z + 1.0),
            quadratic_bspline_1d(w_frac.z),
            quadratic_bspline_1d(w_frac.z - 1.0),
        ];

        let mut w_sum = 0.0f32;
        let mut w_old_sum = 0.0f32;
        let mut w_weight_sum = 0.0f32;
        let mut w_c_axis = Vec3::ZERO;

        for dk in -1i32..=1 {
            for dj in -1i32..=1 {
                for di in -1i32..=1 {
                    let ni = w_base_i + di;
                    let nj = w_base_j + dj;
                    let nk = w_base_k + dk;

                    // Bounds check for W array: width x height x (depth+1)
                    if ni < 0
                        || ni >= grid.width as i32
                        || nj < 0
                        || nj >= grid.height as i32
                        || nk < 0
                        || nk > grid.depth as i32
                    {
                        continue;
                    }

                    // Skip solid faces
                    if grid.is_w_face_solid(ni, nj, nk) {
                        continue;
                    }

                    let w =
                        w_wx[(di + 1) as usize] * w_wy[(dj + 1) as usize] * w_wz[(dk + 1) as usize];

                    if w < 1e-10 {
                        continue;
                    }

                    let idx = grid.w_index(ni as usize, nj as usize, nk as usize);
                    let w_val = grid.w[idx];
                    let w_old_val = grid.w_old[idx];

                    w_sum += w_val * w;
                    w_old_sum += w_old_val * w;
                    w_weight_sum += w;

                    // APIC C matrix contribution (third column for W)
                    let node_pos = grid.w_position(ni as usize, nj as usize, nk as usize);
                    let offset = node_pos - pos;
                    w_c_axis += d_inv * w * w_val * offset;
                }
            }
        }

        // Normalize W by weight sum (excludes solid faces)
        if w_weight_sum > 1e-6 {
            new_vel.z = w_sum / w_weight_sum;
            w_delta = (w_sum - w_old_sum) / w_weight_sum;
            new_c.z_axis = w_c_axis / w_weight_sum;
        }

        // FLIP/PIC blend
        let grid_delta = Vec3::new(u_delta, v_delta, w_delta);
        let flip_velocity = old_vel + grid_delta;
        let pic_velocity = new_vel;

        let mut final_vel = flip_ratio * flip_velocity + (1.0 - flip_ratio) * pic_velocity;

        // Clamp velocity to prevent numerical explosion
        const MAX_VELOCITY: f32 = 20.0; // m/s - reasonable max for water flow
        let speed = final_vel.length();
        if speed > MAX_VELOCITY {
            final_vel = final_vel * (MAX_VELOCITY / speed);
        }

        // Safety: NaN/Inf check
        if !final_vel.is_finite() {
            final_vel = Vec3::ZERO;
        }

        particle.velocity = final_vel;
        particle.affine_velocity = new_c;
        particle.old_grid_velocity = new_vel;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::Particle3D;

    #[test]
    fn test_p2g_single_particle_at_node() {
        let mut grid = Grid3D::new(4, 4, 4, 1.0);
        let mut particles = Particles3D::new();
        let mut buffers = TransferBuffers::new(&grid);

        // Place particle exactly at U node (1, 0.5, 0.5)
        particles.list.push(Particle3D::new(
            Vec3::new(1.0, 0.5, 0.5),
            Vec3::new(1.0, 0.0, 0.0),
        ));

        particles_to_grid(&mut grid, &particles, &mut buffers);

        // U at (1,0,0) should have the highest weight contribution
        let u_idx = grid.u_index(1, 0, 0);
        assert!(
            grid.u[u_idx].abs() > 0.5,
            "Expected significant U velocity at node, got {}",
            grid.u[u_idx]
        );
    }

    #[test]
    fn test_g2p_updates_particle() {
        let mut grid = Grid3D::new(4, 4, 4, 1.0);
        let mut particles = Particles3D::new();

        // Set uniform grid velocity
        grid.u.fill(1.0);
        grid.v.fill(2.0);
        grid.w.fill(3.0);
        grid.u_old.fill(0.0);
        grid.v_old.fill(0.0);
        grid.w_old.fill(0.0);

        // Place particle in center
        particles
            .list
            .push(Particle3D::new(Vec3::new(2.0, 2.0, 2.0), Vec3::ZERO));

        grid_to_particles(&grid, &mut particles, 0.97);

        let p = &particles.list[0];
        // Should pick up grid velocity (FLIP delta = new - old = 1,2,3)
        assert!(p.velocity.x > 0.5, "Expected positive X velocity");
        assert!(p.velocity.y > 1.0, "Expected positive Y velocity");
        assert!(p.velocity.z > 2.0, "Expected positive Z velocity");
    }
}
