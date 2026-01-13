//! DEM-FLIP Bridge: Two-way Coupling
//!
//! Handles momentum transfer between DEM particles and FLIP fluid.
//! Applies drag forces from fluid to particles and vice versa.

@group(0) @binding(0) var<storage, read_write> dem_positions: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read_write> dem_velocities: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read> dem_masses: array<f32>>;
@group(0) @binding(3) var<storage, read> dem_radii: array<f32>>;
@group(0) @binding(4) var<storage, read> dem_template_ids: array<u32>>;
@group(0) @binding(5) var<storage, read> dem_flags: array<u32>>;

// FLIP grid data (MAC grid)
@group(0) @binding(6) var<storage, read> grid_u: array<array<f32>>; // X-velocity faces
@group(0) @binding(7) var<storage, read> grid_v: array<array<f32>>; // Y-velocity faces
@group(0) @binding(8) var<storage, read> grid_w: array<array<f32>>; // Z-velocity faces
@group(0) @binding(9) var<storage, read_write> grid_u_next: array<array<f32>>;
@group(0) @binding(10) var<storage, read_write> grid_v_next: array<array<f32>>;
@group(0) @binding(11) var<storage, read_write> grid_w_next: array<array<f32>>;

@group(0) @binding(12) var<uniform> flip_params: FlipParams;
@group(0) @binding(13) var<uniform> dem_params: DemParams;

struct FlipParams {
    width: u32,
    height: u32,
    depth: u32,
    cell_size: f32,
    dt: f32,
}

struct DemParams {
    drag_coefficient: f32,
    density_water: f32,
    gravity: vec3<f32>,
    dt: f32,
    dem_particle_count: u32,
    dem_particle_start: u32, // Offset in particle arrays
}

const PARTICLE_ACTIVE = 1u;
const WORKGROUP_SIZE = 64u;

// Cubic B-spline kernel (same as FLIP)
fn bspline_weight(x: f32) -> f32 {
    let ax = abs(x);
    if ax >= 2.0 {
        return 0.0;
    } else if ax >= 1.0 {
        let w = 2.0 - ax;
        return w * w * w * 0.5;
    } else {
        let w = 1.0 - ax;
        return (1.5 * w * w - 0.75) * w + 0.25;
    }
}

// Sample grid velocity at particle position (trilinear interpolation)
fn sample_grid_velocity(pos: vec3<f32>) -> vec3<f32> {
    let cell_pos = pos / flip_params.cell_size - vec3<f32>(0.5, 0.5, 0.5);
    let cell = vec3<i32>(floor(cell_pos));
    let frac = cell_pos - vec3<f32>(f32(cell.x), f32(cell.y), f32(cell.z));
    
    // Clamp to grid bounds
    let clamped_cell = clamp(cell, vec3<i32>(0, 0, 0), 
                                     vec3<i32>(i32(flip_params.width) - 1, 
                                                 i32(flip_params.height) - 1, 
                                                 i32(flip_params.depth) - 1));
    
    // Compute weights
    let wx0 = bspline_weight(frac.x);
    let wx1 = bspline_weight(1.0 - frac.x);
    let wy0 = bspline_weight(frac.y);
    let wy1 = bspline_weight(1.0 - frac.y);
    let wz0 = bspline_weight(frac.z);
    let wz1 = bspline_weight(1.0 - frac.z);
    
    var vel = vec3<f32>(0.0, 0.0, 0.0);
    
    // Trilinear interpolation
    for dz in 0u..=2u {
        for dy in 0u..=2u {
            for dx in 0u..=2u {
                let sample_cell = clamped_cell + vec3<i32>(i32(dx) - 1, i32(dy) - 1, i32(dz) - 1);
                
                if sample_cell.x >= 0 && sample_cell.x < i32(flip_params.width) &&
                   sample_cell.y >= 0 && sample_cell.y < i32(flip_params.height) &&
                   sample_cell.z >= 0 && sample_cell.z < i32(flip_params.depth) {
                    
                    let idx = u32(sample_cell.z) * flip_params.width * flip_params.height +
                              u32(sample_cell.y) * flip_params.width +
                              u32(sample_cell.x);
                    
                    let weight = wx0 * wx1 * wy0 * wy1 * wz0 * wz1;
                    vel.x += grid_u[sample_cell.z][sample_cell.y][sample_cell.x] * weight;
                    vel.y += grid_v[sample_cell.z][sample_cell.y][sample_cell.x + 1] * weight; // Offset for v faces
                    vel.z += grid_w[sample_cell.z][sample_cell.y + 1][sample_cell.x] * weight; // Offset for w faces
                }
            }
        }
    }
    
    return vel;
}

// Scatter force to grid (same as P2G)
fn scatter_force(pos: vec3<f32>, force: vec3<f32>, mass: f32) {
    let cell_pos = pos / flip_params.cell_size - vec3<f32>(0.5, 0.5, 0.5);
    let cell = vec3<i32>(floor(cell_pos));
    let frac = cell_pos - vec3<f32>(f32(cell.x), f32(cell.y), f32(cell.z));
    
    let wx0 = bspline_weight(frac.x);
    let wx1 = bspline_weight(1.0 - frac.x);
    let wy0 = bspline_weight(frac.y);
    let wy1 = bspline_weight(1.0 - frac.y);
    let wz0 = bspline_weight(frac.z);
    let wz1 = bspline_weight(1.0 - frac.z);
    
    for dz in 0u..=2u {
        for dy in 0u..=2u {
            for dx in 0u..=2u {
                let sample_cell = cell + vec3<i32>(i32(dx) - 1, i32(dy) - 1, i32(dz) - 1);
                
                if sample_cell.x >= 0 && sample_cell.x < i32(flip_params.width) &&
                   sample_cell.y >= 0 && sample_cell.y < i32(flip_params.height) &&
                   sample_cell.z >= 0 && sample_cell.z < i32(flip_params.depth) {
                    
                    let idx = u32(sample_cell.z) * flip_params.width * flip_params.height +
                              u32(sample_cell.y) * flip_params.width +
                              u32(sample_cell.x);
                    
                    let weight = wx0 * wx1 * wy0 * wy1 * wz0 * wz1;
                    
                    // Add momentum contribution (force * dt = momentum)
                    let momentum = force * flip_params.dt * mass;
                    
                    // Atomic operations for grid update
                    atomicAdd(&grid_u_next[sample_cell.z][sample_cell.y][sample_cell.x], momentum.x * weight);
                    atomicAdd(&grid_v_next[sample_cell.z][sample_cell.y][sample_cell.x + 1], momentum.y * weight);
                    atomicAdd(&grid_w_next[sample_cell.z][sample_cell.y + 1][sample_cell.x], momentum.z * weight);
                }
            }
        }
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= dem_params.dem_particle_count { return; }
    
    let particle_idx = dem_params.dem_particle_start + idx;
    let flags = dem_flags[particle_idx];
    
    if (flags & PARTICLE_ACTIVE) == 0u { return; }
    
    let pos = dem_positions[particle_idx];
    let vel = dem_velocities[particle_idx];
    let mass = dem_masses[particle_idx];
    let radius = dem_radii[particle_idx];
    let template_id = dem_template_ids[particle_idx];
    
    // 1. Apply FLIP forces to DEM particle
    let water_vel = sample_grid_velocity(pos);
    let relative_vel = water_vel - vel;
    
    // Drag force (heavier particles affected less)
    let density_ratio = mass / (4.0/3.0 * 3.14159 * radius * radius * radius); // Approx density
    let drag_factor = dem_params.drag_coefficient / density_ratio;
    let drag_force = relative_vel * drag_factor;
    
    // Buoyancy (reduced gravity)
    let gravity_force = dem_params.gravity * mass * (1.0 - dem_params.density_water / density_ratio);
    
    // Update DEM particle velocity
    let total_force = drag_force + gravity_force;
    let accel = total_force / mass;
    dem_velocities[particle_idx] = vel + accel * dem_params.dt;
    
    // 2. Apply DEM reaction forces to FLIP grid
    // For single spheres, use sphere radius
    // For multi-sphere clumps, we'd need to track contacts from collision shader
    // For now, just apply equal and opposite reaction force
    let reaction_force = -drag_force * mass; // Reaction on fluid
    
    // Scatter reaction force to grid
    scatter_force(pos, reaction_force, mass);
}