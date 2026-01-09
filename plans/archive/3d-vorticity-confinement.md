# Plan: 3D Vorticity Confinement for GPU FLIP Simulation

## Problem Statement

The current 3D GPU FLIP simulation produces laminar flow without visible vortex structures. Water flows smoothly over riffles without the turbulent eddies expected in real sluice boxes. This is because:

1. **Numerical dissipation** - Grid-based methods lose rotational energy at each timestep
2. **No vorticity confinement** - The 2D code has `apply_vorticity_confinement()` but 3D has none
3. **Coarse grid resolution** - 160×40×32 at 0.03m cells can't resolve small vortices naturally

## Solution: Fedkiw Vorticity Confinement (3D)

Add a GPU compute shader that counteracts numerical dissipation by injecting energy back into rotational structures. Based on Fedkiw et al. 2001 "Visual Simulation of Smoke".

### Mathematical Foundation

**3D Vorticity (Curl of Velocity):**
```
ω = ∇ × v

ωx = ∂w/∂y - ∂v/∂z
ωy = ∂u/∂z - ∂w/∂x
ωz = ∂v/∂x - ∂u/∂y
```

**Vorticity Confinement Force:**
```
1. Compute vorticity magnitude: |ω| = sqrt(ωx² + ωy² + ωz²)
2. Compute normalized location vector: N = ∇|ω| / |∇|ω||
3. Apply force: F = ε × h × (N × ω)

Where:
- ε = confinement strength (0.05-0.125, start with 0.05)
- h = cell size (grid spacing for scale independence)
- N × ω = cross product (force perpendicular to both)
```

**Why This Works:**
- Vortices have high |ω| at their cores
- Gradient ∇|ω| points toward vortex centers
- Cross product N × ω creates a tangential force that spins up the vortex
- This counters the numerical viscosity that damps rotation

## Implementation Plan

### Phase 1: Vorticity Storage Buffers

**File:** `crates/game/src/gpu/flip_3d.rs`

Add GPU buffers to store vorticity components:

```rust
// Vorticity buffers (cell-centered, same size as pressure)
vorticity_x_buffer: wgpu::Buffer,  // ωx component
vorticity_y_buffer: wgpu::Buffer,  // ωy component
vorticity_z_buffer: wgpu::Buffer,  // ωz component
vorticity_mag_buffer: wgpu::Buffer, // |ω| magnitude
```

Buffer size: `width × height × depth × 4 bytes` each (same as cell_type buffer).

### Phase 2: Vorticity Computation Shader

**File:** `crates/game/src/gpu/shaders/vorticity_3d.wgsl`

```wgsl
// Compute 3D vorticity (curl) from staggered velocity grid
//
// Input: u, v, w grids (staggered MAC grid)
// Output: vorticity_x, vorticity_y, vorticity_z, vorticity_mag (cell-centered)

struct VorticityParams {
    width: u32,
    height: u32,
    depth: u32,
    cell_size: f32,
}

@group(0) @binding(0) var<uniform> params: VorticityParams;
@group(0) @binding(1) var<storage, read> u: array<f32>;      // (width+1) × height × depth
@group(0) @binding(2) var<storage, read> v: array<f32>;      // width × (height+1) × depth
@group(0) @binding(3) var<storage, read> w: array<f32>;      // width × height × (depth+1)
@group(0) @binding(4) var<storage, read> cell_type: array<u32>;
@group(0) @binding(5) var<storage, read_write> vort_x: array<f32>;
@group(0) @binding(6) var<storage, read_write> vort_y: array<f32>;
@group(0) @binding(7) var<storage, read_write> vort_z: array<f32>;
@group(0) @binding(8) var<storage, read_write> vort_mag: array<f32>;

// Index helpers for staggered grids
fn u_index(i: u32, j: u32, k: u32) -> u32 {
    return i + j * (params.width + 1u) + k * (params.width + 1u) * params.height;
}

fn v_index(i: u32, j: u32, k: u32) -> u32 {
    return i + j * params.width + k * params.width * (params.height + 1u);
}

fn w_index(i: u32, j: u32, k: u32) -> u32 {
    return i + j * params.width + k * params.width * params.height;
}

fn cell_index(i: u32, j: u32, k: u32) -> u32 {
    return i + j * params.width + k * params.width * params.height;
}

@compute @workgroup_size(8, 8, 4)
fn compute_vorticity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    let k = gid.z;

    // Bounds check (skip boundary cells)
    if (i < 1u || i >= params.width - 1u ||
        j < 1u || j >= params.height - 1u ||
        k < 1u || k >= params.depth - 1u) {
        return;
    }

    let idx = cell_index(i, j, k);

    // Only compute for fluid cells
    if (cell_type[idx] != 1u) { // 1 = Fluid
        vort_x[idx] = 0.0;
        vort_y[idx] = 0.0;
        vort_z[idx] = 0.0;
        vort_mag[idx] = 0.0;
        return;
    }

    let inv_2h = 0.5 / params.cell_size;

    // Sample velocities at cell center using averaging
    // For U: average u[i,j,k] and u[i+1,j,k] etc.

    // ωx = ∂w/∂y - ∂v/∂z
    let dw_dy = (w[w_index(i, j+1u, k)] - w[w_index(i, j-1u, k)]) * inv_2h;
    let dv_dz = (v[v_index(i, j, k+1u)] - v[v_index(i, j, k-1u)]) * inv_2h;
    let omega_x = dw_dy - dv_dz;

    // ωy = ∂u/∂z - ∂w/∂x
    let du_dz = (u[u_index(i, j, k+1u)] - u[u_index(i, j, k-1u)]) * inv_2h;
    let dw_dx = (w[w_index(i+1u, j, k)] - w[w_index(i-1u, j, k)]) * inv_2h;
    let omega_y = du_dz - dw_dx;

    // ωz = ∂v/∂x - ∂u/∂y (this is the 2D curl component)
    let dv_dx = (v[v_index(i+1u, j, k)] - v[v_index(i-1u, j, k)]) * inv_2h;
    let du_dy = (u[u_index(i, j+1u, k)] - u[u_index(i, j-1u, k)]) * inv_2h;
    let omega_z = dv_dx - du_dy;

    // Store components
    vort_x[idx] = omega_x;
    vort_y[idx] = omega_y;
    vort_z[idx] = omega_z;
    vort_mag[idx] = sqrt(omega_x * omega_x + omega_y * omega_y + omega_z * omega_z);
}
```

### Phase 3: Vorticity Confinement Force Shader

**File:** `crates/game/src/gpu/shaders/vorticity_confine_3d.wgsl`

```wgsl
// Apply vorticity confinement force to velocity grid
//
// F = ε × h × (N × ω)
// Where N = ∇|ω| / |∇|ω|| (normalized gradient of vorticity magnitude)

struct ConfineParams {
    width: u32,
    height: u32,
    depth: u32,
    epsilon_h_dt: f32,  // ε × h × dt (pre-multiplied)
}

@group(0) @binding(0) var<uniform> params: ConfineParams;
@group(0) @binding(1) var<storage, read> vort_x: array<f32>;
@group(0) @binding(2) var<storage, read> vort_y: array<f32>;
@group(0) @binding(3) var<storage, read> vort_z: array<f32>;
@group(0) @binding(4) var<storage, read> vort_mag: array<f32>;
@group(0) @binding(5) var<storage, read> cell_type: array<u32>;
@group(0) @binding(6) var<storage, read_write> u: array<f32>;
@group(0) @binding(7) var<storage, read_write> v: array<f32>;
@group(0) @binding(8) var<storage, read_write> w: array<f32>;

fn cell_index(i: u32, j: u32, k: u32) -> u32 {
    return i + j * params.width + k * params.width * params.height;
}

fn u_index(i: u32, j: u32, k: u32) -> u32 {
    return i + j * (params.width + 1u) + k * (params.width + 1u) * params.height;
}

fn v_index(i: u32, j: u32, k: u32) -> u32 {
    return i + j * params.width + k * params.width * (params.height + 1u);
}

fn w_index(i: u32, j: u32, k: u32) -> u32 {
    return i + j * params.width + k * params.width * params.height;
}

@compute @workgroup_size(8, 8, 4)
fn apply_confinement(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    let k = gid.z;

    // Skip boundary cells (need neighbors for gradient)
    if (i < 2u || i >= params.width - 2u ||
        j < 2u || j >= params.height - 2u ||
        k < 2u || k >= params.depth - 2u) {
        return;
    }

    let idx = cell_index(i, j, k);

    // Only apply to fluid cells
    if (cell_type[idx] != 1u) {
        return;
    }

    // Check for air neighbors (skip free surface - like 2D code)
    let has_air = cell_type[cell_index(i-1u, j, k)] == 0u ||
                  cell_type[cell_index(i+1u, j, k)] == 0u ||
                  cell_type[cell_index(i, j-1u, k)] == 0u ||
                  cell_type[cell_index(i, j+1u, k)] == 0u ||
                  cell_type[cell_index(i, j, k-1u)] == 0u ||
                  cell_type[cell_index(i, j, k+1u)] == 0u;
    if (has_air) {
        return;
    }

    // Compute gradient of vorticity magnitude (central differences)
    let grad_x = (vort_mag[cell_index(i+1u, j, k)] - vort_mag[cell_index(i-1u, j, k)]) * 0.5;
    let grad_y = (vort_mag[cell_index(i, j+1u, k)] - vort_mag[cell_index(i, j-1u, k)]) * 0.5;
    let grad_z = (vort_mag[cell_index(i, j, k+1u)] - vort_mag[cell_index(i, j, k-1u)]) * 0.5;

    // Normalize gradient to get N
    let grad_len = sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z) + 1e-5;
    let nx = grad_x / grad_len;
    let ny = grad_y / grad_len;
    let nz = grad_z / grad_len;

    // Get vorticity at this cell
    let wx = vort_x[idx];
    let wy = vort_y[idx];
    let wz = vort_z[idx];

    // Compute force: F = ε × h × (N × ω)
    // Cross product: N × ω
    let fx = (ny * wz - nz * wy) * params.epsilon_h_dt;
    let fy = (nz * wx - nx * wz) * params.epsilon_h_dt;
    let fz = (nx * wy - ny * wx) * params.epsilon_h_dt;

    // Apply force to velocity grid (distribute to staggered faces)
    // Each cell center affects its surrounding faces
    atomicAdd(&u[u_index(i, j, k)], fx * 0.5);
    atomicAdd(&u[u_index(i+1u, j, k)], fx * 0.5);
    atomicAdd(&v[v_index(i, j, k)], fy * 0.5);
    atomicAdd(&v[v_index(i, j+1u, k)], fy * 0.5);
    atomicAdd(&w[w_index(i, j, k)], fz * 0.5);
    atomicAdd(&w[w_index(i, j, k+1u)], fz * 0.5);
}
```

**Note:** The atomic adds may need adjustment - alternatively, apply to cell-centered velocity first, then scatter to faces in a second pass to avoid race conditions.

### Phase 4: Rust Integration

**File:** `crates/game/src/gpu/flip_3d.rs`

#### 4.1 Add Parameters Struct

```rust
/// Vorticity computation parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct VorticityParams3D {
    width: u32,
    height: u32,
    depth: u32,
    cell_size: f32,
}

/// Vorticity confinement parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct VortConfineParams3D {
    width: u32,
    height: u32,
    depth: u32,
    epsilon_h_dt: f32,  // ε × h × dt pre-multiplied
}
```

#### 4.2 Add Buffers and Pipelines to GpuFlip3D

```rust
// In GpuFlip3D struct:

// Vorticity buffers
vorticity_x_buffer: wgpu::Buffer,
vorticity_y_buffer: wgpu::Buffer,
vorticity_z_buffer: wgpu::Buffer,
vorticity_mag_buffer: wgpu::Buffer,

// Vorticity pipelines
vorticity_compute_pipeline: wgpu::ComputePipeline,
vorticity_compute_bind_group: wgpu::BindGroup,
vorticity_confine_pipeline: wgpu::ComputePipeline,
vorticity_confine_bind_group: wgpu::BindGroup,

// Parameters
vorticity_params_buffer: wgpu::Buffer,
vort_confine_params_buffer: wgpu::Buffer,

/// Vorticity confinement strength (default 0.05, range 0.0-0.25)
pub vorticity_epsilon: f32,
```

#### 4.3 Update step_internal()

Insert vorticity confinement between gravity and pressure solve:

```rust
fn step_internal(&mut self, ...) {
    // ... existing P2G, BC, store old velocities ...

    // 4. Apply gravity
    // ... existing gravity code ...

    // 5. Apply flow acceleration
    // ... existing flow code ...

    // NEW: 5.5. Vorticity Confinement
    if self.vorticity_epsilon > 0.0 {
        // Compute vorticity
        let vort_params = VorticityParams3D {
            width: self.width,
            height: self.height,
            depth: self.depth,
            cell_size: self.cell_size,
        };
        queue.write_buffer(&self.vorticity_params_buffer, 0, bytemuck::bytes_of(&vort_params));

        {
            let mut pass = encoder.begin_compute_pass(...);
            pass.set_pipeline(&self.vorticity_compute_pipeline);
            pass.set_bind_group(0, &self.vorticity_compute_bind_group, &[]);
            let wg_x = (self.width + 7) / 8;
            let wg_y = (self.height + 7) / 8;
            let wg_z = (self.depth + 3) / 4;
            pass.dispatch_workgroups(wg_x, wg_y, wg_z);
        }

        // Apply confinement force
        let confine_params = VortConfineParams3D {
            width: self.width,
            height: self.height,
            depth: self.depth,
            epsilon_h_dt: self.vorticity_epsilon * self.cell_size * dt,
        };
        queue.write_buffer(&self.vort_confine_params_buffer, 0, bytemuck::bytes_of(&confine_params));

        {
            let mut pass = encoder.begin_compute_pass(...);
            pass.set_pipeline(&self.vorticity_confine_pipeline);
            pass.set_bind_group(0, &self.vorticity_confine_bind_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, wg_z);
        }
    }

    // 6. Pressure solve
    // ... existing pressure code ...
}
```

### Phase 5: Testing & Tuning

#### 5.1 Visual Verification

Run `industrial_sluice` and look for:
- Visible rotation behind riffles
- Eddies that persist for multiple frames
- Flow that curls back upstream in recirculation zones

#### 5.2 Parameter Tuning

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `epsilon` | 0.05 | 0.0-0.25 | Higher = stronger vortices, risk of instability |
| Run every N frames | 1 | 1-4 | Skip frames for performance (2D uses every 2) |

Start conservative (ε=0.05), increase if vortices still dissipate too fast.

#### 5.3 Diagnostic Output

Add optional vorticity magnitude output to flow particles:

```rust
// In build_flow_particles():
let vort_mag = sample_vorticity_magnitude(p.position);
let color = if vort_mag > threshold {
    [1.0, 0.0, 0.0, 0.8]  // Red for high vorticity
} else {
    // Existing velocity color
};
```

### Phase 6: Optimizations

1. **Skip every 2 frames** - Like 2D code, can run confinement every other frame
2. **Fused kernel** - Combine compute + confine into single pass if atomics work
3. **Avoid atomics** - Use separate force buffer, add to velocity in final pass
4. **Adaptive ε** - Higher near obstacles, lower in open flow

## Files to Create/Modify

| File | Action |
|------|--------|
| `crates/game/src/gpu/shaders/vorticity_3d.wgsl` | CREATE |
| `crates/game/src/gpu/shaders/vorticity_confine_3d.wgsl` | CREATE |
| `crates/game/src/gpu/flip_3d.rs` | MODIFY - add buffers, pipelines, step integration |
| `crates/game/examples/industrial_sluice.rs` | MODIFY - expose epsilon parameter |

## Expected Results

Before (current):
- Laminar flow over riffles
- Water slides smoothly downstream
- No visible turbulence

After (with vorticity confinement):
- Visible eddies behind each riffle
- Recirculation zones where flow reverses
- Persistent vortex structures that evolve over time
- More realistic sluice box behavior

## References

1. Fedkiw, Stam, Jensen (2001) "Visual Simulation of Smoke" - Original vorticity confinement
2. Zhang & Bridson (2015) "Restoring the Missing Vorticity" - IVOCK improvements
3. `crates/sim/src/grid/vorticity.rs` - Working 2D implementation
4. `docs/research/vortex-formation-best-practices.md` - Project research

## Risks

1. **Instability** - Too high ε can cause energy blow-up. Start low.
2. **Boundary artifacts** - May need to skip cells near solids (like 2D does)
3. **Performance** - Two extra compute passes per frame. ~5-10% overhead expected.
4. **Atomics** - May need non-atomic version if race conditions cause issues.

## Implementation Order

1. [ ] Create vorticity storage buffers
2. [ ] Implement vorticity compute shader
3. [ ] Implement confinement force shader
4. [ ] Integrate into step_internal()
5. [ ] Test with ε=0.05
6. [ ] Tune parameters
7. [ ] Add diagnostic visualization
8. [ ] Performance optimization
