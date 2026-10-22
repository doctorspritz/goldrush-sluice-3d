// Pressure solver (3D) using Checkerboard SOR (Successive Over-Relaxation)
//
// Red-black ordering allows parallel updates without data races.
// In 3D, red cells have (i + j + k) % 2 == 0, black cells have (i + j + k) % 2 == 1.
//
// Uses 6-neighbor Laplacian stencil for pressure Poisson equation:
//   ∇²p = -∇·u (divergence)
//
// Boundary conditions:
// - Solid cells: Neumann BC (dp/dn = 0), pressure copied from fluid
// - Air cells: Dirichlet BC (p = 0)

struct Params {
    width: u32,
    height: u32,
    depth: u32,
    omega: f32,  // SOR relaxation factor (typically 1.5-1.9 for 3D)
    h_sq: f32,   // cell_size^2, needed to scale divergence in Poisson equation
    open_boundaries: u32,  // Bitmask: 1=-X, 2=+X, 4=-Y, 8=+Y, 16=-Z, 32=+Z
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(1) var<storage, read> divergence: array<f32>;
@group(0) @binding(2) var<storage, read> cell_type: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(5) var<storage,read> mobility_u:array<f32>;
@group(0) @binding(6) var<storage,read> mobility_v:array<f32>;
@group(0) @binding(7) var<storage,read> mobility_w:array<f32>;
// Cell type constants (matching sluice_3d_visual convention)
const CELL_AIR: u32 = 0u;
const CELL_FLUID: u32 = 1u;
const CELL_SOLID: u32 = 2u;

fn get_index(i: u32, j: u32, k: u32) -> u32 {
    return k * params.width * params.height + j * params.width + i;
}

fn is_valid(i: u32, j: u32, k: u32) -> bool {
    return i < params.width && j < params.height && k < params.depth;
}

fn get_cell_type(i: u32, j: u32, k: u32) -> u32 {
    if (!is_valid(i, j, k)) {
        return CELL_SOLID;  // Treat out-of-bounds as solid
    }
    return cell_type[get_index(i, j, k)];
}

fn get_pressure(i: u32, j: u32, k: u32) -> f32 {
    if (!is_valid(i, j, k)) {
        return 0.0;
    }
    let ct = cell_type[get_index(i, j, k)];
    if (ct == CELL_SOLID || ct == CELL_AIR) {
        // Solid/Air cells: use zero pressure (Dirichlet for air, handled by neighbor count for solid)
        return 0.0;
    }
    return pressure[get_index(i, j, k)];
}

// Get neighbor pressure with proper boundary conditions
// - Solid: Neumann BC (dp/dn = 0) - use current cell's pressure
// - Air: Dirichlet BC (p = 0) - atmospheric pressure
// - Fluid: use actual pressure
fn get_neighbor_pressure(ni: u32, nj: u32, nk: u32, current_p: f32) -> f32 {
    if (!is_valid(ni, nj, nk)) {
        return current_p;  // Treat out-of-bounds as Neumann
    }
    let ct = cell_type[get_index(ni, nj, nk)];
    if (ct == CELL_SOLID) {
        return current_p;  // Neumann BC: use current cell pressure
    }
    if (ct == CELL_AIR) {
        return 0.0;  // Dirichlet BC: atmospheric pressure
    }
    return pressure[get_index(ni, nj, nk)];  // Fluid: actual pressure
}

// Process "red" cells: (i + j + k) % 2 == 0
fn solve_red(id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;

    if (!is_valid(i, j, k)) {
        return;
    }

    // Skip non-red cells (black cells have (i+j+k) % 2 == 1)
    if (((i + j + k) % 2u) != 0u) {
        return;
    }

    let idx = get_index(i, j, k);

    // Skip non-fluid cells
    if (cell_type[idx] != CELL_FLUID) {
        return;
    }

    if variable_mobility() {solve_weighted(id,neighbour_mask(id));return;}

    let p = pressure[idx];

    // Gather neighbor pressures with proper boundary conditions
    // Always count 6 neighbors (proper Neumann/Dirichlet handling)
    var sum_neighbors = 0.0;

    // -X neighbor
    if (i > 0u) {
        sum_neighbors += get_neighbor_pressure(i - 1u, j, k, p);
    } else {
        if ((params.open_boundaries & 1u) != 0u) {
            sum_neighbors += 0.0;
        } else {
            sum_neighbors += p;
        }
    }

    // +X neighbor
    if (i < params.width - 1u) {
        sum_neighbors += get_neighbor_pressure(i + 1u, j, k, p);
    } else {
        if ((params.open_boundaries & 2u) != 0u) {
            sum_neighbors += 0.0;
        } else {
            sum_neighbors += p;
        }
    }

    // -Y neighbor
    if (j > 0u) {
        sum_neighbors += get_neighbor_pressure(i, j - 1u, k, p);
    } else {
        if ((params.open_boundaries & 4u) != 0u) {
            sum_neighbors += 0.0;
        } else {
            sum_neighbors += p;
        }
    }

    // +Y neighbor
    if (j < params.height - 1u) {
        sum_neighbors += get_neighbor_pressure(i, j + 1u, k, p);
    } else {
        if ((params.open_boundaries & 8u) != 0u) {
            sum_neighbors += 0.0;
        } else {
            sum_neighbors += p;
        }
    }

    // -Z neighbor
    if (k > 0u) {
        sum_neighbors += get_neighbor_pressure(i, j, k - 1u, p);
    } else {
        if ((params.open_boundaries & 16u) != 0u) {
            sum_neighbors += 0.0;
        } else {
            sum_neighbors += p;
        }
    }

    // +Z neighbor
    if (k < params.depth - 1u) {
        sum_neighbors += get_neighbor_pressure(i, j, k + 1u, p);
    } else {
        if ((params.open_boundaries & 32u) != 0u) {
            sum_neighbors += 0.0;
        } else {
            sum_neighbors += p;
        }
    }

    // Gauss-Seidel update with SOR
    // Poisson equation: (p_neighbors - 6p) / h² = -div
    // Rearranged: p = (sum_neighbors - h² * div) / 6
    let new_p = (sum_neighbors - divergence[idx] * params.h_sq) / 6.0;
    // SOR: weighted average of old and new values
    pressure[idx] = mix(pressure[idx], new_p, params.omega);
}

// Process "black" cells: (i + j + k) % 2 == 1
fn solve_black(id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;

    if (!is_valid(i, j, k)) {
        return;
    }

    // Skip non-black cells (red cells have (i+j+k) % 2 == 0)
    if (((i + j + k) % 2u) != 1u) {
        return;
    }

    let idx = get_index(i, j, k);

    if (cell_type[idx] != CELL_FLUID) {
        return;
    }

    if variable_mobility() {solve_weighted(id,neighbour_mask(id));return;}

    let p = pressure[idx];

    // Gather neighbor pressures with proper boundary conditions
    var sum_neighbors = 0.0;

    // -X neighbor
    if (i > 0u) {
        sum_neighbors += get_neighbor_pressure(i - 1u, j, k, p);
    } else {
        if ((params.open_boundaries & 1u) != 0u) {
            sum_neighbors += 0.0;
        } else {
            sum_neighbors += p;
        }
    }

    // +X neighbor
    if (i < params.width - 1u) {
        sum_neighbors += get_neighbor_pressure(i + 1u, j, k, p);
    } else {
        if ((params.open_boundaries & 2u) != 0u) {
            sum_neighbors += 0.0;
        } else {
            sum_neighbors += p;
        }
    }

    // -Y neighbor
    if (j > 0u) {
        sum_neighbors += get_neighbor_pressure(i, j - 1u, k, p);
    } else {
        if ((params.open_boundaries & 4u) != 0u) {
            sum_neighbors += 0.0;
        } else {
            sum_neighbors += p;
        }
    }

    // +Y neighbor
    if (j < params.height - 1u) {
        sum_neighbors += get_neighbor_pressure(i, j + 1u, k, p);
    } else {
        if ((params.open_boundaries & 8u) != 0u) {
            sum_neighbors += 0.0;
        } else {
            sum_neighbors += p;
        }
    }

    // -Z neighbor
    if (k > 0u) {
        sum_neighbors += get_neighbor_pressure(i, j, k - 1u, p);
    } else {
        if ((params.open_boundaries & 16u) != 0u) {
            sum_neighbors += 0.0;
        } else {
            sum_neighbors += p;
        }
    }

    // +Z neighbor
    if (k < params.depth - 1u) {
        sum_neighbors += get_neighbor_pressure(i, j, k + 1u, p);
    } else {
        if ((params.open_boundaries & 32u) != 0u) {
            sum_neighbors += 0.0;
        } else {
            sum_neighbors += p;
        }
    }

    // Poisson equation: p = (sum_neighbors - h² * div) / 6
    let new_p = (sum_neighbors - divergence[idx] * params.h_sq) / 6.0;
    pressure[idx] = mix(pressure[idx], new_p, params.omega);
}

// Compute divergence of velocity field
// This is typically called once before the pressure iterations
@compute @workgroup_size(8, 8, 4)
fn compute_divergence(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    let k = id.z;

    if (!is_valid(i, j, k)) {
        return;
    }

    let idx = get_index(i, j, k);

    if (cell_type[idx] != CELL_FLUID) {
        // Note: divergence array must be read_write for this entry point
        // but we're writing to it - caller should bind appropriately
        return;
    }

    // Divergence computed from staggered velocity grid would go here
    // But typically divergence is computed by a separate shader that has
    // access to the velocity grids. This entry point is a placeholder.
}

// Apply pressure gradient to velocity (subtract gradient from velocity)
// This is called after pressure solve to make velocity divergence-free
//
// Note: This requires separate entry points for U, V, W grids since they
// have different staggered layouts. See apply_pressure_gradient_*.wgsl

@compute @workgroup_size(8,8,4)
fn pressure_red(@builtin(global_invocation_id) id:vec3<u32>) {solve_red(id);}
@compute @workgroup_size(8,8,4)
fn pressure_black(@builtin(global_invocation_id) id:vec3<u32>) {solve_black(id);}
// Compact the checkerboard's X coordinate without an index buffer or indirect
// dispatch. Every original cell is visited in the same red/black sweep.
@compute @workgroup_size(32,2,2)
fn pressure_red_packed(@builtin(global_invocation_id) id:vec3<u32>) {
 solve_red(vec3(id.x*2u+((id.y+id.z)&1u),id.y,id.z));
}
@compute @workgroup_size(32,2,2)
fn pressure_black_packed(@builtin(global_invocation_id) id:vec3<u32>) {
 solve_black(vec3(id.x*2u+((id.y+id.z+1u)&1u),id.y,id.z));
}

// Neighbour types are constant during all sweeps of a pressure solve. Cache
// their boundary interpretation once, while preserving the six-term sum order.
@group(0) @binding(4) var<storage,read_write> stencil: array<u32>;
fn neighbour_code(q: vec3<i32>, boundary: u32) -> u32 {
    if any(q < vec3(0)) || any(q >= vec3<i32>(i32(params.width),i32(params.height),i32(params.depth))) {
        return select(2u,0u,(params.open_boundaries & boundary)!=0u);
    }
    let ct=cell_type[get_index(u32(q.x),u32(q.y),u32(q.z))];
    return select(select(0u,1u,ct==CELL_FLUID),2u,ct==CELL_SOLID);
}
@compute @workgroup_size(8,8,4)
fn prepare_stencil(@builtin(global_invocation_id) id:vec3<u32>) {
    if !is_valid(id.x,id.y,id.z) {return;}
    let idx=get_index(id.x,id.y,id.z);
    if cell_type[idx]!=CELL_FLUID {stencil[idx]=0u;return;}
    stencil[idx]=neighbour_mask(id);
}
fn stencil_pressure(code:u32,idx:u32,p:f32)->f32 {
    if code==1u {return pressure[idx];}
    if code==2u {return p;}
    return 0.;
}
fn solve_stencil(id:vec3<u32>) {
    if !is_valid(id.x,id.y,id.z) {return;}
    let idx=get_index(id.x,id.y,id.z);
    let mask=stencil[idx];
    if (mask&0x80000000u)==0u {return;}
    if variable_mobility() {solve_weighted(id,mask);return;}
    let p=pressure[idx];
    var sum=0.;
    sum+=stencil_pressure(mask&3u,idx-1u,p);
    sum+=stencil_pressure((mask>>2u)&3u,idx+1u,p);
    sum+=stencil_pressure((mask>>4u)&3u,idx-params.width,p);
    sum+=stencil_pressure((mask>>6u)&3u,idx+params.width,p);
    sum+=stencil_pressure((mask>>8u)&3u,idx-params.width*params.height,p);
    sum+=stencil_pressure((mask>>10u)&3u,idx+params.width*params.height,p);
    let new_p=(sum-divergence[idx]*params.h_sq)/6.;
    pressure[idx]=mix(pressure[idx],new_p,params.omega);
}
@compute @workgroup_size(32,2,2)
fn pressure_red_stencil(@builtin(global_invocation_id) id:vec3<u32>) {
    solve_stencil(vec3(id.x*2u+((id.y+id.z)&1u),id.y,id.z));
}
@compute @workgroup_size(32,2,2)
fn pressure_black_stencil(@builtin(global_invocation_id) id:vec3<u32>) {
    solve_stencil(vec3(id.x*2u+((id.y+id.z+1u)&1u),id.y,id.z));
}

fn neighbour_mask(id:vec3<u32>)->u32 {
    let p=vec3<i32>(id);
    return 0x80000000u
        | neighbour_code(p+vec3(-1,0,0),1u)
        | (neighbour_code(p+vec3(1,0,0),2u)<<2u)
        | (neighbour_code(p+vec3(0,-1,0),4u)<<4u)
        | (neighbour_code(p+vec3(0,1,0),8u)<<6u)
        | (neighbour_code(p+vec3(0,0,-1),16u)<<8u)
        | (neighbour_code(p+vec3(0,0,1),32u)<<10u);
}
fn solve_weighted(id:vec3<u32>,mask:u32) {
    let idx=get_index(id.x,id.y,id.z);
    let nu=(params.width+1u)*params.height;
    let ui=id.z*nu+id.y*(params.width+1u)+id.x;
    let vi=id.z*params.width*(params.height+1u)+id.y*params.width+id.x;
    let wi=idx;
    // The same fixed neighbour summation order is used by every dispatch.
    var terms=weighted_term(mask&3u,idx-1u,face_mobility(0u,ui));
    terms+=weighted_term((mask>>2u)&3u,idx+1u,face_mobility(0u,ui+1u));
    terms+=weighted_term((mask>>4u)&3u,idx-params.width,face_mobility(1u,vi));
    terms+=weighted_term((mask>>6u)&3u,idx+params.width,face_mobility(1u,vi+params.width));
    terms+=weighted_term((mask>>8u)&3u,idx-params.width*params.height,face_mobility(2u,wi));
    terms+=weighted_term((mask>>10u)&3u,idx+params.width*params.height,face_mobility(2u,wi+params.width*params.height));
    let diagonal=terms.x;let sum=terms.y;
    if diagonal<=0. {pressure[idx]=0.;return;}
    let new_p=(sum-divergence[idx]*params.h_sq)/diagonal;
    pressure[idx]=mix(pressure[idx],new_p,params.omega);
}

fn weighted_term(code:u32,neighbour:u32,coefficient:f32)->vec2<f32> {
    if code==2u {return vec2(0.);}
    if code==1u {return vec2(coefficient,coefficient*pressure[neighbour]);}
    return vec2(coefficient,0.);
}
