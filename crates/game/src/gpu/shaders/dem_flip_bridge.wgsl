//! DEM-FLIP Bridge: Two-way Coupling
//!
//! Handles momentum transfer between DEM particles and FLIP fluid.
//! Applies drag forces from fluid to particles and vice versa.
//!
//! Grid layout (MAC staggered):
//! - U velocities: stored at left YZ faces, (width+1) x height x depth
//! - V velocities: stored at bottom XZ faces, width x (height+1) x depth
//! - W velocities: stored at back XY faces, width x height x (depth+1)

// DEM particle data (SoA layout matching dem_3d.rs)
@group(0) @binding(0) var<storage, read> dem_positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dem_velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> dem_flags: array<u32>;
@group(0) @binding(3) var<storage, read> dem_template_ids: array<u32>;
@group(0) @binding(4) var<storage, read> templates: array<GpuClumpTemplate>;

// FLIP grid data (MAC grid - flat arrays)
@group(0) @binding(5) var<storage, read> grid_u: array<f32>;  // (width+1) x height x depth
@group(0) @binding(6) var<storage, read> grid_v: array<f32>;  // width x (height+1) x depth
@group(0) @binding(7) var<storage, read> grid_w: array<f32>;  // width x height x (depth+1)

// Force accumulation buffer (written to DEM force buffer for integration)
@group(0) @binding(8) var<storage, read_write> dem_forces: array<vec4<f32>>;

// Parameters
@group(0) @binding(9) var<uniform> bridge_params: BridgeParams;

struct GpuClumpTemplate {
    sphere_count: u32,
    mass: f32,
    radius: f32,
    _pad0: f32,
    inertia_inv: mat3x3<f32>,
}

struct BridgeParams {
    // FLIP grid dimensions
    width: u32,
    height: u32,
    depth: u32,
    cell_size: f32,
    // Physics parameters
    dt: f32,
    drag_coefficient: f32,
    density_water: f32,
    _pad0: f32,
    gravity: vec4<f32>,
    // DEM particle range
    dem_particle_count: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

const PARTICLE_ACTIVE = 1u;
const WORKGROUP_SIZE = 64u;
const MAX_TEMPLATES = 100u;
const PI: f32 = 3.14159265359;

// MAC grid index functions
fn u_index(i: u32, j: u32, k: u32) -> u32 {
    // U grid: (width+1) x height x depth
    return k * (bridge_params.width + 1u) * bridge_params.height + j * (bridge_params.width + 1u) + i;
}

fn v_index(i: u32, j: u32, k: u32) -> u32 {
    // V grid: width x (height+1) x depth
    return k * bridge_params.width * (bridge_params.height + 1u) + j * bridge_params.width + i;
}

fn w_index(i: u32, j: u32, k: u32) -> u32 {
    // W grid: width x height x (depth+1)
    return k * bridge_params.width * bridge_params.height + j * bridge_params.width + i;
}

fn sample_u_value(i: i32, j: i32, k: i32) -> f32 {
    if i < 0 || i > i32(bridge_params.width) { return 0.0; }
    if j < 0 || j >= i32(bridge_params.height) { return 0.0; }
    if k < 0 || k >= i32(bridge_params.depth) { return 0.0; }
    return grid_u[u_index(u32(i), u32(j), u32(k))];
}

fn sample_v_value(i: i32, j: i32, k: i32) -> f32 {
    if i < 0 || i >= i32(bridge_params.width) { return 0.0; }
    if j < 0 || j > i32(bridge_params.height) { return 0.0; }
    if k < 0 || k >= i32(bridge_params.depth) { return 0.0; }
    return grid_v[v_index(u32(i), u32(j), u32(k))];
}

fn sample_w_value(i: i32, j: i32, k: i32) -> f32 {
    if i < 0 || i >= i32(bridge_params.width) { return 0.0; }
    if j < 0 || j >= i32(bridge_params.height) { return 0.0; }
    if k < 0 || k > i32(bridge_params.depth) { return 0.0; }
    return grid_w[w_index(u32(i), u32(j), u32(k))];
}

// Sample U velocity at position (U is at left YZ faces: i, j+0.5, k+0.5)
fn sample_u(pos: vec3<f32>) -> f32 {
    let sample_pos = max(vec3<f32>(0.0), pos / bridge_params.cell_size - vec3<f32>(0.0, 0.5, 0.5));

    let base_x = i32(floor(sample_pos.x));
    let base_y = i32(floor(sample_pos.y));
    let base_z = i32(floor(sample_pos.z));

    let fx = sample_pos.x - f32(base_x);
    let fy = sample_pos.y - f32(base_y);
    let fz = sample_pos.z - f32(base_z);

    let v000 = sample_u_value(base_x, base_y, base_z);
    let v100 = sample_u_value(base_x + 1, base_y, base_z);
    let v010 = sample_u_value(base_x, base_y + 1, base_z);
    let v110 = sample_u_value(base_x + 1, base_y + 1, base_z);
    let v001 = sample_u_value(base_x, base_y, base_z + 1);
    let v101 = sample_u_value(base_x + 1, base_y, base_z + 1);
    let v011 = sample_u_value(base_x, base_y + 1, base_z + 1);
    let v111 = sample_u_value(base_x + 1, base_y + 1, base_z + 1);

    let v00 = mix(v000, v100, fx);
    let v10 = mix(v010, v110, fx);
    let v01 = mix(v001, v101, fx);
    let v11 = mix(v011, v111, fx);

    let v0 = mix(v00, v10, fy);
    let v1 = mix(v01, v11, fy);

    return mix(v0, v1, fz);
}

// Sample V velocity at position (V is at bottom XZ faces: i+0.5, j, k+0.5)
fn sample_v(pos: vec3<f32>) -> f32 {
    let sample_pos = max(vec3<f32>(0.0), pos / bridge_params.cell_size - vec3<f32>(0.5, 0.0, 0.5));

    let base_x = i32(floor(sample_pos.x));
    let base_y = i32(floor(sample_pos.y));
    let base_z = i32(floor(sample_pos.z));

    let fx = sample_pos.x - f32(base_x);
    let fy = sample_pos.y - f32(base_y);
    let fz = sample_pos.z - f32(base_z);

    let v000 = sample_v_value(base_x, base_y, base_z);
    let v100 = sample_v_value(base_x + 1, base_y, base_z);
    let v010 = sample_v_value(base_x, base_y + 1, base_z);
    let v110 = sample_v_value(base_x + 1, base_y + 1, base_z);
    let v001 = sample_v_value(base_x, base_y, base_z + 1);
    let v101 = sample_v_value(base_x + 1, base_y, base_z + 1);
    let v011 = sample_v_value(base_x, base_y + 1, base_z + 1);
    let v111 = sample_v_value(base_x + 1, base_y + 1, base_z + 1);

    let v00 = mix(v000, v100, fx);
    let v10 = mix(v010, v110, fx);
    let v01 = mix(v001, v101, fx);
    let v11 = mix(v011, v111, fx);

    let v0 = mix(v00, v10, fy);
    let v1 = mix(v01, v11, fy);

    return mix(v0, v1, fz);
}

// Sample W velocity at position (W is at back XY faces: i+0.5, j+0.5, k)
fn sample_w(pos: vec3<f32>) -> f32 {
    let sample_pos = max(vec3<f32>(0.0), pos / bridge_params.cell_size - vec3<f32>(0.5, 0.5, 0.0));

    let base_x = i32(floor(sample_pos.x));
    let base_y = i32(floor(sample_pos.y));
    let base_z = i32(floor(sample_pos.z));

    let fx = sample_pos.x - f32(base_x);
    let fy = sample_pos.y - f32(base_y);
    let fz = sample_pos.z - f32(base_z);

    let v000 = sample_w_value(base_x, base_y, base_z);
    let v100 = sample_w_value(base_x + 1, base_y, base_z);
    let v010 = sample_w_value(base_x, base_y + 1, base_z);
    let v110 = sample_w_value(base_x + 1, base_y + 1, base_z);
    let v001 = sample_w_value(base_x, base_y, base_z + 1);
    let v101 = sample_w_value(base_x + 1, base_y, base_z + 1);
    let v011 = sample_w_value(base_x, base_y + 1, base_z + 1);
    let v111 = sample_w_value(base_x + 1, base_y + 1, base_z + 1);

    let v00 = mix(v000, v100, fx);
    let v10 = mix(v010, v110, fx);
    let v01 = mix(v001, v101, fx);
    let v11 = mix(v011, v111, fx);

    let v0 = mix(v00, v10, fy);
    let v1 = mix(v01, v11, fy);

    return mix(v0, v1, fz);
}

// Sample full velocity vector at position
fn sample_grid_velocity(pos: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(sample_u(pos), sample_v(pos), sample_w(pos));
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= bridge_params.dem_particle_count { return; }

    let flags = dem_flags[idx];
    if (flags & PARTICLE_ACTIVE) == 0u { return; }

    let template_idx = dem_template_ids[idx];
    if template_idx >= MAX_TEMPLATES { return; }

    let mass = templates[template_idx].mass;
    let radius = templates[template_idx].radius;
    if mass <= 0.0 || radius <= 0.0 { return; }

    let pos = dem_positions[idx].xyz;
    let vel = dem_velocities[idx].xyz;

    // Check if particle is within FLIP grid bounds
    let grid_max = vec3<f32>(
        f32(bridge_params.width) * bridge_params.cell_size,
        f32(bridge_params.height) * bridge_params.cell_size,
        f32(bridge_params.depth) * bridge_params.cell_size
    );
    if pos.x < 0.0 || pos.x >= grid_max.x ||
       pos.y < 0.0 || pos.y >= grid_max.y ||
       pos.z < 0.0 || pos.z >= grid_max.z {
        return;  // Outside fluid domain
    }

    // 1. Sample fluid velocity at particle position
    let water_vel = sample_grid_velocity(pos);
    let relative_vel = water_vel - vel;

    // 2. Compute particle density from mass and radius
    let volume = (4.0 / 3.0) * PI * radius * radius * radius;
    let particle_density = mass / volume;

    // 3. Drag force: F_drag = C_d * (rho_water / rho_particle) * (v_water - v_particle)
    // Higher drag coefficient = particle follows water more closely
    // Divide by particle density so heavier particles are less affected
    let drag_factor = bridge_params.drag_coefficient * bridge_params.density_water / particle_density;
    let drag_force = relative_vel * drag_factor * mass;  // Force = factor * mass * relative_vel

    // 4. Buoyancy force (separate from gravity per issue go-oqa)
    // F_buoyancy = rho_water * V * g (upward when submerged)
    let buoyancy_force = bridge_params.density_water * volume * (-bridge_params.gravity.xyz);

    // 5. Accumulate forces to DEM force buffer
    // Note: Gravity is applied in dem_integration.wgsl, so we only add drag + buoyancy here
    let total_force = drag_force + buoyancy_force;

    // Add to existing forces (may have contact forces from DEM collision)
    let existing_force = dem_forces[idx].xyz;
    dem_forces[idx] = vec4<f32>(existing_force + total_force, 0.0);

    // Note: Two-way coupling (reaction force on fluid) would require atomic writes
    // to the FLIP grid, which is handled separately in a dedicated scatter pass.
    // This shader focuses on the DEM side of the coupling.
}
