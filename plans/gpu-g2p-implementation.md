# GPU G2P Implementation Plan

## Overview

Grid-to-Particle (G2P) transfer is the next optimization target after GPU P2G. Unlike P2G which requires atomic operations, G2P is **embarrassingly parallel** - each particle independently reads from the grid.

| Particles | CPU G2P | % of Frame | Target |
|-----------|---------|------------|--------|
| 150K      | 1ms     | 3%         | <0.5ms |
| 300K      | 3ms     | 5%         | <0.8ms |
| 600K      | 7ms     | 7%         | <1.5ms |
| 1M        | 11ms    | 18%        | <2ms   |

**Expected speedup**: 5-10x (no atomics, pure gather)

## Current CPU Implementation Analysis

The CPU G2P (`transfer.rs:334-612`) has two distinct paths:

### 1. Water Particles (APIC)
- Sample grid velocity with quadratic B-spline (3×3 stencil)
- Reconstruct affine velocity matrix C
- FLIP/PIC blend (97% FLIP, 3% PIC)
- Clamp velocity delta for stability

### 2. Sediment Particles (Lagrangian)
- Sample grid velocity for drag calculation
- Apply vorticity-based suspension (lift + swirl)
- Material-specific settling velocity (Ferguson-Church)
- Separate PIC ratio (70% PIC, 30% FLIP for sand)

### Key Data Flow
```
Inputs:
  - particle.position
  - particle.velocity (for FLIP)
  - particle.old_grid_velocity (stored from store_old_velocities)
  - particle.material
  - particle.state (Suspended/Bedload)
  - particle.effective_diameter()
  - grid.u, grid.v
  - grid.u_old, grid.v_old (for FLIP delta)

Outputs:
  - particle.velocity (updated)
  - particle.affine_velocity (C matrix, for water only)
  - particle.old_grid_velocity (updated)
```

## Design Decision: Split vs Unified Kernel

### Option A: Single Kernel (Recommended for MVP)
- Handle water and sediment with material-based branching
- Simpler buffer management
- WGSL branch divergence is acceptable for 80/20 water/sediment ratio

### Option B: Two Kernels
- `g2p_water.wgsl` - APIC with C matrix
- `g2p_sediment.wgsl` - Lagrangian with settling
- Better occupancy if sediment ratio varies wildly
- More complex orchestration

**Recommendation**: Start with Option A, split later if profiling shows divergence issues.

## Implementation Design

### Particle Buffer Layout

Reuse P2G buffer structure, add required fields:

```rust
// Per-particle data (SoA for coalescing)
struct GpuParticles {
    positions: wgpu::Buffer,       // vec2<f32>
    velocities: wgpu::Buffer,      // vec2<f32> (read + write)
    c_matrices: wgpu::Buffer,      // mat2x2<f32> (write for water)
    materials: wgpu::Buffer,       // u32 (material + state flags)
    old_grid_vel: wgpu::Buffer,    // vec2<f32> (for FLIP delta)
    diameters: wgpu::Buffer,       // f32 (for settling)
}
```

### Grid Buffers (Read-only for G2P)

```rust
struct GridBuffers {
    u: wgpu::Buffer,        // Current U velocities
    v: wgpu::Buffer,        // Current V velocities
    u_old: wgpu::Buffer,    // Old U (before forces)
    v_old: wgpu::Buffer,    // Old V (before forces)
    cell_type: wgpu::Buffer, // For in-fluid detection (sediment)
    vorticity: wgpu::Buffer, // For suspension (optional Phase 2)
}
```

### Shader Design

```wgsl
// g2p.wgsl

struct Params {
    cell_size: f32,
    width: u32,
    height: u32,
    particle_count: u32,
    dt: f32,
    gravity: f32,
    flip_ratio_water: f32,  // 0.97
    pic_ratio_sand: f32,    // 0.70
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> c_matrices: array<mat2x2<f32>>;
@group(0) @binding(4) var<storage, read> materials: array<u32>;
@group(0) @binding(5) var<storage, read_write> old_grid_vel: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read> diameters: array<f32>;
@group(0) @binding(7) var<storage, read> grid_u: array<f32>;
@group(0) @binding(8) var<storage, read> grid_v: array<f32>;
@group(0) @binding(9) var<storage, read> grid_u_old: array<f32>;
@group(0) @binding(10) var<storage, read> grid_v_old: array<f32>;
@group(0) @binding(11) var<storage, read> cell_type: array<u32>;

// Material flags (match Rust enum)
const MATERIAL_WATER: u32 = 0u;
const MATERIAL_SAND: u32 = 1u;
const MATERIAL_MAGNETITE: u32 = 2u;
const MATERIAL_GOLD: u32 = 3u;

// State flags (packed in upper bits)
const STATE_SUSPENDED: u32 = 0u;
const STATE_BEDLOAD: u32 = 1u;

fn quadratic_bspline_1d(x: f32) -> f32 {
    let ax = abs(x);
    if (ax < 0.5) {
        return 0.75 - ax * ax;
    } else if (ax < 1.5) {
        let t = 1.5 - ax;
        return 0.5 * t * t;
    }
    return 0.0;
}

fn quadratic_bspline(delta: vec2<f32>) -> f32 {
    return quadratic_bspline_1d(delta.x) * quadratic_bspline_1d(delta.y);
}

fn sample_grid_velocity(pos: vec2<f32>) -> vec2<f32> {
    let cell_size = params.cell_size;
    let width = params.width;
    let height = params.height;

    var vel = vec2(0.0);
    var u_weight = 0.0;
    var v_weight = 0.0;

    // Sample U (staggered left)
    let u_pos = pos / cell_size - vec2(0.0, 0.5);
    let base_i_u = i32(floor(u_pos.x));
    let base_j_u = i32(floor(u_pos.y));
    let fx_u = u_pos.x - f32(base_i_u);
    let fy_u = u_pos.y - f32(base_j_u);

    for (var dj: i32 = -1; dj <= 1; dj++) {
        for (var di: i32 = -1; di <= 1; di++) {
            let ni = base_i_u + di;
            let nj = base_j_u + dj;
            if (ni < 0 || ni > i32(width) || nj < 0 || nj >= i32(height)) { continue; }

            let delta = vec2(fx_u - f32(di), fy_u - f32(dj));
            let w = quadratic_bspline(delta);
            if (w <= 0.0) { continue; }

            let idx = u32(nj) * (width + 1u) + u32(ni);
            vel.x += w * grid_u[idx];
            u_weight += w;
        }
    }

    // Sample V (staggered bottom)
    let v_pos = pos / cell_size - vec2(0.5, 0.0);
    let base_i_v = i32(floor(v_pos.x));
    let base_j_v = i32(floor(v_pos.y));
    let fx_v = v_pos.x - f32(base_i_v);
    let fy_v = v_pos.y - f32(base_j_v);

    for (var dj: i32 = -1; dj <= 1; dj++) {
        for (var di: i32 = -1; di <= 1; di++) {
            let ni = base_i_v + di;
            let nj = base_j_v + dj;
            if (ni < 0 || ni >= i32(width) || nj < 0 || nj > i32(height)) { continue; }

            let delta = vec2(fx_v - f32(di), fy_v - f32(dj));
            let w = quadratic_bspline(delta);
            if (w <= 0.0) { continue; }

            let idx = u32(nj) * width + u32(ni);
            vel.y += w * grid_v[idx];
            v_weight += w;
        }
    }

    // Normalize
    if (u_weight > 0.0) { vel.x /= u_weight; }
    if (v_weight > 0.0) { vel.y /= v_weight; }

    return vel;
}

fn sample_old_grid_velocity(pos: vec2<f32>) -> vec2<f32> {
    // Same as sample_grid_velocity but using grid_u_old/grid_v_old
    // ... (duplicated or factored out with buffer param)
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.particle_count) { return; }

    let pos = positions[id.x];
    let old_vel = velocities[id.x];
    let material = materials[id.x] & 0xFFu;
    let state = (materials[id.x] >> 8u) & 0xFFu;

    let v_grid = sample_grid_velocity(pos);
    let v_grid_old = sample_old_grid_velocity(pos);
    let grid_delta = v_grid - v_grid_old;

    // ========== WATER (APIC) ==========
    if (material == MATERIAL_WATER) {
        // Reconstruct C matrix
        let c_mat = compute_c_matrix(pos, id.x);
        c_matrices[id.x] = c_mat;

        // FLIP/PIC blend
        let flip_vel = old_vel + grid_delta;
        let pic_vel = v_grid;
        let flip_ratio = params.flip_ratio_water;

        // Clamp delta for stability
        let max_dv = 5.0 * params.cell_size / params.dt;
        var clamped_delta = grid_delta;
        if (length(grid_delta) > max_dv) {
            clamped_delta = normalize(grid_delta) * max_dv;
        }
        let flip_vel_clamped = old_vel + clamped_delta;

        var new_vel = flip_ratio * flip_vel_clamped + (1.0 - flip_ratio) * pic_vel;

        // Safety clamp
        let speed = length(new_vel);
        if (speed > 2000.0) {
            new_vel = new_vel * (2000.0 / speed);
        }

        velocities[id.x] = new_vel;
        old_grid_vel[id.x] = v_grid;
        return;
    }

    // ========== SEDIMENT (Lagrangian) ==========
    // Check if in fluid cell
    let ci = u32(pos.x / params.cell_size);
    let cj = u32(pos.y / params.cell_size);
    let cell_idx = cj * params.width + ci;
    let in_fluid = cell_type[cell_idx] == 1u; // CellType::Fluid

    var new_vel = old_vel;

    if (in_fluid) {
        // FLIP/PIC blend for sand
        let pic_vel = v_grid;
        let old_grid = old_grid_vel[id.x];
        let flip_vel = old_vel + (v_grid - old_grid);

        let pic_ratio = params.pic_ratio_sand;
        new_vel = pic_ratio * pic_vel + (1.0 - pic_ratio) * flip_vel;
        old_grid_vel[id.x] = v_grid;
    } else {
        // In air - reset old grid velocity
        old_grid_vel[id.x] = vec2(0.0);
    }

    // Apply settling (material-specific)
    let settling_vel = get_settling_velocity(material, diameters[id.x]);
    let settling_factor = 0.62 * (settling_vel / 28.0); // Normalized to sand

    // Skip bedload settling (handled by advection/DEM)
    if (state != STATE_BEDLOAD) {
        new_vel.y += params.gravity * settling_factor * params.dt;
    }

    velocities[id.x] = new_vel;
}

fn compute_c_matrix(pos: vec2<f32>, particle_idx: u32) -> mat2x2<f32> {
    let cell_size = params.cell_size;
    let d_inv = 4.0 / (cell_size * cell_size); // APIC D^-1

    var c = mat2x2(0.0, 0.0, 0.0, 0.0);

    // Sample U for C.x_axis
    let u_pos = pos / cell_size - vec2(0.0, 0.5);
    let base_i = i32(floor(u_pos.x));
    let base_j = i32(floor(u_pos.y));
    let fx = u_pos.x - f32(base_i);
    let fy = u_pos.y - f32(base_j);

    var u_weight_sum = 0.0;
    for (var dj: i32 = -1; dj <= 1; dj++) {
        for (var di: i32 = -1; di <= 1; di++) {
            let ni = base_i + di;
            let nj = base_j + dj;
            if (ni < 0 || ni > i32(params.width) || nj < 0 || nj >= i32(params.height)) { continue; }

            let delta = vec2(fx - f32(di), fy - f32(dj));
            let w = quadratic_bspline(delta);
            if (w <= 0.0) { continue; }

            let idx = u32(nj) * (params.width + 1u) + u32(ni);
            let u_val = grid_u[idx];

            let offset = vec2(
                f32(ni) * cell_size - pos.x,
                (f32(nj) + 0.5) * cell_size - pos.y
            );
            c[0] += offset * (w * u_val * d_inv);
            u_weight_sum += w;
        }
    }

    // Sample V for C.y_axis
    let v_pos = pos / cell_size - vec2(0.5, 0.0);
    let base_i_v = i32(floor(v_pos.x));
    let base_j_v = i32(floor(v_pos.y));
    let fx_v = v_pos.x - f32(base_i_v);
    let fy_v = v_pos.y - f32(base_j_v);

    var v_weight_sum = 0.0;
    for (var dj: i32 = -1; dj <= 1; dj++) {
        for (var di: i32 = -1; di <= 1; di++) {
            let ni = base_i_v + di;
            let nj = base_j_v + dj;
            if (ni < 0 || ni >= i32(params.width) || nj < 0 || nj > i32(params.height)) { continue; }

            let delta = vec2(fx_v - f32(di), fy_v - f32(dj));
            let w = quadratic_bspline(delta);
            if (w <= 0.0) { continue; }

            let idx = u32(nj) * params.width + u32(ni);
            let v_val = grid_v[idx];

            let offset = vec2(
                (f32(ni) + 0.5) * cell_size - pos.x,
                f32(nj) * cell_size - pos.y
            );
            c[1] += offset * (w * v_val * d_inv);
            v_weight_sum += w;
        }
    }

    // Normalize
    if (u_weight_sum > 0.0) { c[0] /= u_weight_sum; }
    if (v_weight_sum > 0.0) { c[1] /= v_weight_sum; }

    return c;
}

fn get_settling_velocity(material: u32, diameter: f32) -> f32 {
    // Ferguson-Church settling (simplified)
    // v_s = R * g * d^2 / (C1 * nu + sqrt(0.75 * C2 * R * g * d^3))

    let density_ratio = select(
        1.65,  // sand (2650/1000 - 1)
        select(
            4.2,   // magnetite (5200/1000 - 1)
            18.3,  // gold (19300/1000 - 1)
            material == MATERIAL_GOLD
        ),
        material == MATERIAL_MAGNETITE
    );

    let nu = 1.0e-6; // kinematic viscosity
    let g = 9.81;
    let C1 = 18.0;
    let C2 = 1.0;

    let R = density_ratio;
    let d = diameter * 0.001; // mm to m
    let d3 = d * d * d;

    let v_s = R * g * d * d / (C1 * nu + sqrt(0.75 * C2 * R * g * d3));

    // Convert m/s to px/s (1m = 100px approx)
    return v_s * 100.0;
}
```

## Implementation Phases

### Phase 1: Basic GPU G2P (Water Only)
1. Create `GpuG2pSolver` struct modeled on `GpuP2gSolver`
2. Implement `g2p.wgsl` with water-only path (APIC + FLIP blend)
3. Add grid_u_old/grid_v_old buffers
4. Validate against CPU output (single-particle test)
5. Benchmark: expect 5-10x speedup

### Phase 2: Sediment Support
1. Add material flags, state, diameter to particle buffer
2. Implement sediment branch in shader
3. Add cell_type buffer for in-fluid detection
4. Validate sediment settling behavior

### Phase 3: Integration
1. Add `store_old_velocities` to GPU (simple copy kernel)
2. Modify `finalize_after_pressure()` to use GPU G2P
3. Add 'G' key toggle for GPU/CPU G2P
4. Remove CPU-GPU sync points

### Phase 4: Optimization (Optional)
1. Profile workgroup sizes (64, 128, 256, 512)
2. Consider texture sampling for grid (hardware interpolation)
3. Add vorticity sampling for Phase 2 suspension

## Files to Create

| File | Purpose |
|------|---------|
| `crates/game/src/gpu/g2p.rs` | GPU G2P orchestration |
| `crates/game/src/gpu/shaders/g2p.wgsl` | G2P compute shader |
| `crates/game/examples/gpu_g2p_benchmark.rs` | CPU vs GPU benchmark |

## Files to Modify

| File | Changes |
|------|---------|
| `crates/game/src/gpu/mod.rs` | Add `g2p` module export |
| `crates/game/src/main.rs` | Add GPU G2P path + toggle |
| `crates/sim/src/flip/mod.rs` | Add helper methods for GPU integration |

## Expected Performance

| Particles | CPU (ms) | GPU Target (ms) | Speedup |
|-----------|----------|-----------------|---------|
| 50K       | 0.6      | 0.3             | 2x      |
| 100K      | 1.1      | 0.4             | 2.8x    |
| 200K      | 2.3      | 0.6             | 3.8x    |
| 500K      | 5.5      | 1.2             | 4.6x    |
| 1M        | 11.0     | 2.0             | 5.5x    |

Note: G2P is memory-bound (read-heavy), so speedup is lower than P2G.
GPU benefit comes from parallelism, not from avoiding atomics.

## Validation Strategy

### Unit Tests
1. **Single particle in center**: Compare CPU vs GPU velocity output
2. **C matrix reconstruction**: Verify affine velocity matches
3. **Sediment settling**: Check settling velocity calculation

### Integration Tests
1. Run 100 frames with GPU G2P, compare final particle positions to CPU
2. Visual inspection in game (toggle with 'G' key)

### Regression Test
- `p2g_g2p_round_trip.rs`: Particles with known velocity → P2G → G2P → same velocity

## Success Criteria

1. **Correctness**: GPU output matches CPU within floating-point tolerance
2. **Performance**: <2ms for 1M particles
3. **Stability**: No velocity explosions or NaN propagation
4. **Integration**: Drop-in replacement in simulation loop

---

## DEM Integration Considerations

The simulation pipeline after G2P is:
```
G2P → advect_particles → apply_dem_settling → remove_out_of_bounds
```

DEM (`dem.rs`) handles sediment particle-particle collisions with:
- Spatial hash neighbor queries (O(n) with hash, O(n²) without)
- Spring-damper contact model
- Sleep system for stable particles
- 4 collision resolution iterations per frame

### Current Timing

DEM is hidden in the "rest" phase of `update_profiled()`:
```
Phase 7 "rest": advect + DEM + bounds check
```

At high sediment counts, DEM can become significant (~5-15ms for 100k sediment particles).

### GPU G2P + DEM Options

**Option A: Hybrid (Recommended for MVP)**
```
GPU P2G → GPU G2P → readback → CPU advect → CPU DEM
```
- Simpler: only G2P on GPU, rest stays CPU
- Sync point after G2P (readback particle velocities)
- DEM uses existing spatial hash infrastructure

**Option B: Full GPU Pipeline (Future)**
```
GPU P2G → GPU G2P → GPU advect → GPU DEM → (no readback until render)
```
- Eliminates sync stalls
- Requires GPU spatial hash (parallel construction)
- GPU DEM is complex (atomic updates for contacts, sleep state)

### GPU DEM Complexity

If GPU DEM is needed later:

1. **Spatial Hash on GPU**
   - Parallel cell assignment (counting sort or radix sort)
   - Build cell head/next linked lists with atomics
   - Reference: [NVIDIA particle simulation](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda)

2. **Contact Resolution**
   - Each particle checks neighbors in parallel
   - Atomic force accumulation (similar to P2G)
   - Multiple iterations for stability

3. **Sleep System**
   - Track per-particle sleep state
   - Atomic wake flags when impacted
   - Skip sleeping particles in shader

**Estimated effort**: GPU DEM is 2-3x more complex than GPU G2P.

### Recommendation

For the GPU G2P implementation:
1. **Phase 1-3**: Implement GPU G2P with CPU DEM (Option A)
2. **Profile**: Measure if DEM becomes the new bottleneck
3. **Phase 5 (optional)**: GPU DEM if DEM > 10ms at target particle counts

The current DEM has a sleep system that reduces work for settled particles, so GPU acceleration may not be critical unless you have many actively colliding sediment particles.

---

## References

- Existing GPU P2G: `crates/game/src/gpu/p2g.rs`
- CPU G2P: `crates/sim/src/flip/transfer.rs:334-612`
- CPU DEM: `crates/sim/src/dem.rs`
- [blub G2P shader](https://github.com/Wumpf/blub/blob/main/shaders/simulation/g2p.wgsl)
- [APIC paper](https://www.math.ucla.edu/~jteran/papers/JSSTS15.pdf) - C matrix formula
- [NVIDIA GPU Particle Sim](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda) - GPU spatial hash
