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

// Quadratic B-spline kernel (same as FLIP P2G)
fn quadratic_bspline_1d(x: f32) -> f32 {
    let ax = abs(x);
    if ax < 0.5 {
        return 0.75 - ax * ax;
    } else if ax < 1.5 {
        let t = 1.5 - ax;
        return 0.5 * t * t;
    }
    return 0.0;
}

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

// Sample U velocity at position (U is at left YZ faces: i, j+0.5, k+0.5)
fn sample_u(pos: vec3<f32>) -> f32 {
    let cell_size = bridge_params.cell_size;
    let width = bridge_params.width;
    let height = bridge_params.height;
    let depth = bridge_params.depth;

    // U sample point offset: (0, 0.5, 0.5) in cell coords
    let u_pos = pos / cell_size - vec3<f32>(0.0, 0.5, 0.5);
    let base = vec3<i32>(floor(u_pos));
    let frac = u_pos - vec3<f32>(base);

    // Precompute 1D weights for -1, 0, +1 offsets
    let wx = array<f32, 3>(
        quadratic_bspline_1d(frac.x + 1.0),
        quadratic_bspline_1d(frac.x),
        quadratic_bspline_1d(frac.x - 1.0)
    );
    let wy = array<f32, 3>(
        quadratic_bspline_1d(frac.y + 1.0),
        quadratic_bspline_1d(frac.y),
        quadratic_bspline_1d(frac.y - 1.0)
    );
    let wz = array<f32, 3>(
        quadratic_bspline_1d(frac.z + 1.0),
        quadratic_bspline_1d(frac.z),
        quadratic_bspline_1d(frac.z - 1.0)
    );

    var result = 0.0;
    for (var dk: i32 = -1; dk <= 1; dk++) {
        let nk = base.z + dk;
        if nk < 0 || nk >= i32(depth) { continue; }
        let w_z = wz[dk + 1];

        for (var dj: i32 = -1; dj <= 1; dj++) {
            let nj = base.y + dj;
            if nj < 0 || nj >= i32(height) { continue; }
            let w_yz = w_z * wy[dj + 1];

            for (var di: i32 = -1; di <= 1; di++) {
                let ni = base.x + di;
                if ni < 0 || ni > i32(width) { continue; }  // U grid has width+1
                let weight = w_yz * wx[di + 1];

                let idx = u_index(u32(ni), u32(nj), u32(nk));
                result += grid_u[idx] * weight;
            }
        }
    }
    return result;
}

// Sample V velocity at position (V is at bottom XZ faces: i+0.5, j, k+0.5)
fn sample_v(pos: vec3<f32>) -> f32 {
    let cell_size = bridge_params.cell_size;
    let width = bridge_params.width;
    let height = bridge_params.height;
    let depth = bridge_params.depth;

    // V sample point offset: (0.5, 0, 0.5) in cell coords
    let v_pos = pos / cell_size - vec3<f32>(0.5, 0.0, 0.5);
    let base = vec3<i32>(floor(v_pos));
    let frac = v_pos - vec3<f32>(base);

    let wx = array<f32, 3>(
        quadratic_bspline_1d(frac.x + 1.0),
        quadratic_bspline_1d(frac.x),
        quadratic_bspline_1d(frac.x - 1.0)
    );
    let wy = array<f32, 3>(
        quadratic_bspline_1d(frac.y + 1.0),
        quadratic_bspline_1d(frac.y),
        quadratic_bspline_1d(frac.y - 1.0)
    );
    let wz = array<f32, 3>(
        quadratic_bspline_1d(frac.z + 1.0),
        quadratic_bspline_1d(frac.z),
        quadratic_bspline_1d(frac.z - 1.0)
    );

    var result = 0.0;
    for (var dk: i32 = -1; dk <= 1; dk++) {
        let nk = base.z + dk;
        if nk < 0 || nk >= i32(depth) { continue; }
        let w_z = wz[dk + 1];

        for (var dj: i32 = -1; dj <= 1; dj++) {
            let nj = base.y + dj;
            if nj < 0 || nj > i32(height) { continue; }  // V grid has height+1
            let w_yz = w_z * wy[dj + 1];

            for (var di: i32 = -1; di <= 1; di++) {
                let ni = base.x + di;
                if ni < 0 || ni >= i32(width) { continue; }
                let weight = w_yz * wx[di + 1];

                let idx = v_index(u32(ni), u32(nj), u32(nk));
                result += grid_v[idx] * weight;
            }
        }
    }
    return result;
}

// Sample W velocity at position (W is at back XY faces: i+0.5, j+0.5, k)
fn sample_w(pos: vec3<f32>) -> f32 {
    let cell_size = bridge_params.cell_size;
    let width = bridge_params.width;
    let height = bridge_params.height;
    let depth = bridge_params.depth;

    // W sample point offset: (0.5, 0.5, 0) in cell coords
    let w_pos = pos / cell_size - vec3<f32>(0.5, 0.5, 0.0);
    let base = vec3<i32>(floor(w_pos));
    let frac = w_pos - vec3<f32>(base);

    let wx = array<f32, 3>(
        quadratic_bspline_1d(frac.x + 1.0),
        quadratic_bspline_1d(frac.x),
        quadratic_bspline_1d(frac.x - 1.0)
    );
    let wy = array<f32, 3>(
        quadratic_bspline_1d(frac.y + 1.0),
        quadratic_bspline_1d(frac.y),
        quadratic_bspline_1d(frac.y - 1.0)
    );
    let wz = array<f32, 3>(
        quadratic_bspline_1d(frac.z + 1.0),
        quadratic_bspline_1d(frac.z),
        quadratic_bspline_1d(frac.z - 1.0)
    );

    var result = 0.0;
    for (var dk: i32 = -1; dk <= 1; dk++) {
        let nk = base.z + dk;
        if nk < 0 || nk > i32(depth) { continue; }  // W grid has depth+1
        let w_z = wz[dk + 1];

        for (var dj: i32 = -1; dj <= 1; dj++) {
            let nj = base.y + dj;
            if nj < 0 || nj >= i32(height) { continue; }
            let w_yz = w_z * wy[dj + 1];

            for (var di: i32 = -1; di <= 1; di++) {
                let ni = base.x + di;
                if ni < 0 || ni >= i32(width) { continue; }
                let weight = w_yz * wx[di + 1];

                let idx = w_index(u32(ni), u32(nj), u32(nk));
                result += grid_w[idx] * weight;
            }
        }
    }
    return result;
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