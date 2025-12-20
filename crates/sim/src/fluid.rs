//! Fluid Solver implementing Jos Stam's "Stable Fluids".
//!
//! Operates on the Velocity Grid within a Chunk to create realistic flows (vortices).

use crate::chunk::{Chunk, CHUNK_SIZE};

const ITERATIONS: usize = 4; // Back to 4 for better pressure solving
const DT: f32 = 0.1; // Time step
const VISCOSITY: f32 = 0.0; // Zero for inviscid water

/// Solve fluid dynamics for a single frame.
/// Uses chunk's scratch buffers to avoid heap allocation.
pub fn solve_fluid(chunk: &mut Chunk) {
    // Use scratch buffers instead of allocating
    // scratch_a = vx_prev, scratch_b = vy_prev
    chunk.scratch_a.copy_from_slice(&*chunk.vel_x);
    chunk.scratch_b.copy_from_slice(&*chunk.vel_y);

    // 1. Diffuse (Viscosity) - Apply to velocity
    diffuse(1, &mut *chunk.vel_x, &*chunk.scratch_a, VISCOSITY, DT);
    diffuse(2, &mut *chunk.vel_y, &*chunk.scratch_b, VISCOSITY, DT);

    // 2. Project (Enforce Mass Conservation / Incompressibility)
    project(&mut *chunk.vel_x, &mut *chunk.vel_y, &mut *chunk.scratch_a, &mut *chunk.scratch_b);

    // 3. Advect (Self-Advection: Velocity moves Velocity)
    chunk.scratch_a.copy_from_slice(&*chunk.vel_x);
    chunk.scratch_b.copy_from_slice(&*chunk.vel_y);

    advect(1, &mut *chunk.vel_x, &*chunk.scratch_a, &*chunk.scratch_a, &*chunk.scratch_b, DT);
    advect(2, &mut *chunk.vel_y, &*chunk.scratch_b, &*chunk.scratch_a, &*chunk.scratch_b, DT);

    // 4. Project again (to keep it clean)
    project(&mut *chunk.vel_x, &mut *chunk.vel_y, &mut *chunk.scratch_a, &mut *chunk.scratch_b);
}

/// Linear backtrace advection.
/// d: destination array
/// d0: source array
/// u, v: velocity field
fn advect(b: usize, d: &mut [f32], d0: &[f32], u: &[f32], v: &[f32], dt: f32) {
    let dt0 = dt * CHUNK_SIZE as f32;
    
    for j in 1..CHUNK_SIZE-1 {
        for i in 1..CHUNK_SIZE-1 {
            let idx = Chunk::index(i, j);
            
            // Backtrace
            let x = i as f32 - dt0 * u[idx];
            let y = j as f32 - dt0 * v[idx];
            
            // Clamp
            let x = x.clamp(0.5, CHUNK_SIZE as f32 - 1.5);
            let y = y.clamp(0.5, CHUNK_SIZE as f32 - 1.5);
            
            // Bilinear Interpolation
            let i0 = x as usize;
            let i1 = i0 + 1;
            let j0 = y as usize;
            let j1 = j0 + 1;
            
            let s1 = x - i0 as f32;
            let s0 = 1.0 - s1;
            let t1 = y - j0 as f32;
            let t0 = 1.0 - t1;
            
            let i0j0 = Chunk::index(i0, j0);
            let i0j1 = Chunk::index(i0, j1);
            let i1j0 = Chunk::index(i1, j0);
            let i1j1 = Chunk::index(i1, j1);
            
            d[idx] = s0 * (t0 * d0[i0j0] + t1 * d0[i0j1]) +
                     s1 * (t0 * d0[i1j0] + t1 * d0[i1j1]);
        }
    }
    
    set_bnd(b, d);
}

/// Diffuse density/velocity (Viscosity).
fn diffuse(b: usize, x: &mut [f32], x0: &[f32], diff: f32, dt: f32) {
    let a = dt * diff * (CHUNK_SIZE - 2) as f32 * (CHUNK_SIZE - 2) as f32;
    
    for _ in 0..ITERATIONS {
        for j in 1..CHUNK_SIZE-1 {
            for i in 1..CHUNK_SIZE-1 {
                let idx = Chunk::index(i, j);
                x[idx] = (x0[idx] + a * (
                    x[Chunk::index(i-1, j)] + x[Chunk::index(i+1, j)] +
                    x[Chunk::index(i, j-1)] + x[Chunk::index(i, j+1)]
                )) / (1.0 + 4.0 * a);
            }
        }
        set_bnd(b, x);
    }
}

/// Enforce incompressibility (divergence-free field).
fn project(u: &mut [f32], v: &mut [f32], p: &mut [f32], div: &mut [f32]) {
    let h = 1.0 / CHUNK_SIZE as f32;

    // Calculate Divergence
    for j in 1..CHUNK_SIZE-1 {
        for i in 1..CHUNK_SIZE-1 {
            let idx = Chunk::index(i, j);
            div[idx] = -0.5 * h * (
                u[Chunk::index(i+1, j)] - u[Chunk::index(i-1, j)] +
                v[Chunk::index(i, j+1)] - v[Chunk::index(i, j-1)]
            );
            p[idx] = 0.0;
        }
    }
    
    set_bnd(0, div);
    set_bnd(0, p);

    // Solve Poisson Equation (Gauss-Seidel)
    for _ in 0..ITERATIONS {
        for j in 1..CHUNK_SIZE-1 {
            for i in 1..CHUNK_SIZE-1 {
                let idx = Chunk::index(i, j);
                p[idx] = (div[idx] + 
                    p[Chunk::index(i-1, j)] + p[Chunk::index(i+1, j)] +
                    p[Chunk::index(i, j-1)] + p[Chunk::index(i, j+1)]
                ) / 4.0;
            }
        }
        set_bnd(0, p);
    }
    
    // Subtract Gradient
    for j in 1..CHUNK_SIZE-1 {
        for i in 1..CHUNK_SIZE-1 {
            let idx = Chunk::index(i, j);
            u[idx] -= 0.5 * (p[Chunk::index(i+1, j)] - p[Chunk::index(i-1, j)]) / h;
            v[idx] -= 0.5 * (p[Chunk::index(i, j+1)] - p[Chunk::index(i, j-1)]) / h;
        }
    }
    
    set_bnd(1, u);
    set_bnd(2, v);
}

/// Handle boundaries (walls reflect/block).
fn set_bnd(b: usize, x: &mut [f32]) {
    // Walls
    for i in 1..CHUNK_SIZE-1 {
        // Vertical walls (Left/Right)
        x[Chunk::index(0, i)] = if b == 1 { -x[Chunk::index(1, i)] } else { x[Chunk::index(1, i)] };
        x[Chunk::index(CHUNK_SIZE-1, i)] = if b == 1 { -x[Chunk::index(CHUNK_SIZE-2, i)] } else { x[Chunk::index(CHUNK_SIZE-2, i)] };
        
        // Horizontal walls (Top/Bottom)
        x[Chunk::index(i, 0)] = if b == 2 { -x[Chunk::index(i, 1)] } else { x[Chunk::index(i, 1)] };
        x[Chunk::index(i, CHUNK_SIZE-1)] = if b == 2 { -x[Chunk::index(i, CHUNK_SIZE-2)] } else { x[Chunk::index(i, CHUNK_SIZE-2)] };
    }
    
    // Corners
    x[Chunk::index(0, 0)] = 0.5 * (x[Chunk::index(1, 0)] + x[Chunk::index(0, 1)]);
    x[Chunk::index(0, CHUNK_SIZE-1)] = 0.5 * (x[Chunk::index(1, CHUNK_SIZE-1)] + x[Chunk::index(0, CHUNK_SIZE-2)]);
    x[Chunk::index(CHUNK_SIZE-1, 0)] = 0.5 * (x[Chunk::index(CHUNK_SIZE-2, 0)] + x[Chunk::index(CHUNK_SIZE-1, 1)]);
    x[Chunk::index(CHUNK_SIZE-1, CHUNK_SIZE-1)] = 0.5 * (x[Chunk::index(CHUNK_SIZE-2, CHUNK_SIZE-1)] + x[Chunk::index(CHUNK_SIZE-1, CHUNK_SIZE-2)]);
}
