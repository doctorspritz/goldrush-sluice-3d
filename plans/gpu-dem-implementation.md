# GPU DEM Implementation Plan

## Research Summary

### Published GPU DEM Approaches

Based on research from multiple sources:

1. **[NVIDIA GPU Gems Chapter 29](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-29-real-time-rigid-body-simulation-gpus)** - 5-stage pipeline:
   - Particle value computation (forces)
   - Grid generation (spatial hash)
   - Collision detection
   - Momentum computation
   - Position/quaternion integration

2. **[Taichi DEM](https://docs.taichi-lang.org/blog/acclerate-collision-detection-with-taichi)** - Grid-based O(n) approach:
   - 3x3 grid cell neighborhood search
   - Parallel prefix sum for cell offsets
   - Atomic operations for particle insertion
   - <200 lines of code for minimal 2D DEM

3. **[lisyarus WebGPU Particle Life](https://lisyarus.github.io/blog/posts/particle-life-simulation-in-browser-using-webgpu.html)** - WGSL implementation:
   - Linearized bin storage (particles in same bin are contiguous)
   - 3-phase binning: count → prefix sum → sort
   - `atomicAdd` for concurrent bin counting
   - 3x3 neighbor iteration in force shader

4. **[PhasicFlow](https://github.com/PhasicFlow/phasicFlow)** - Industrial DEM (80M particles)
5. **[Chrono::GPU](https://www.mdpi.com/2227-9717/9/10/1813)** - 130M particles with Hertzian contacts

### Current CPU DEM Analysis

**File:** `crates/sim/src/dem.rs` (950 lines)

**Key bottleneck:** `integrate_and_collide_coupled()` at lines 452-748:
- Uses spatial hash (cell_head/particle_next linked lists)
- 3x3 neighborhood iteration
- 4 solver iterations
- Spring-damper contact model with friction
- Sleep system for stable particles

**Performance profile (from earlier benchmarks):**
- DEM collision detection: ~30-40ms at 100k sediment particles
- Scales as O(n) due to spatial hash, but constant factor is large

## GPU DEM Design

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  CPU: Upload particles, dispatch compute, download results │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  GPU Pipeline (5 dispatches per frame)                      │
│                                                             │
│  1. GRID_CLEAR    - Reset bin counts to zero                │
│  2. BIN_COUNT     - atomicAdd to count particles per cell   │
│  3. PREFIX_SUM    - Compute bin offsets (parallel scan)     │
│  4. BIN_INSERT    - Sort particles into bins                │
│  5. DEM_FORCES    - Compute contacts + integrate (N iters)  │
└─────────────────────────────────────────────────────────────┘
```

### Buffer Layout

| Buffer | Type | Size | Usage |
|--------|------|------|-------|
| `positions` | Storage RW | N * 8 bytes | vec2<f32> positions |
| `velocities` | Storage RW | N * 8 bytes | vec2<f32> velocities |
| `materials` | Storage R | N * 4 bytes | u32 material type |
| `radii` | Storage R | N * 4 bytes | f32 particle radius |
| `bin_counts` | Storage RW | GRID_W * GRID_H * 4 | u32 per cell |
| `bin_offsets` | Storage RW | GRID_W * GRID_H * 4 | u32 prefix sum |
| `sorted_indices` | Storage RW | N * 4 bytes | u32 particle indices sorted by bin |
| `params` | Uniform | 64 bytes | DemParams struct |
| `staging` | MAP_READ | N * 16 bytes | Download positions + velocities |

### Shader Design

#### 1. `dem_bin_count.wgsl`

```wgsl
@group(0) @binding(0) var<uniform> params: DemParams;
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> bin_counts: array<atomic<u32>>;

@compute @workgroup_size(256)
fn bin_count(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.particle_count) { return; }

    let pos = positions[id.x];
    let gi = u32(pos.x / params.cell_size);
    let gj = u32(pos.y / params.cell_size);
    let bin_idx = gj * params.grid_width + gi;

    atomicAdd(&bin_counts[bin_idx], 1u);
}
```

#### 2. `dem_prefix_sum.wgsl`

Parallel prefix sum using work-efficient algorithm (Blelloch scan):
- Reduce phase: O(n) operations, O(log n) depth
- Downsweep phase: propagate exclusive prefix sums

#### 3. `dem_bin_insert.wgsl`

```wgsl
@compute @workgroup_size(256)
fn bin_insert(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.particle_count) { return; }

    let pos = positions[id.x];
    let bin_idx = compute_bin_index(pos);

    // Get unique index within bin
    let local_idx = atomicAdd(&bin_counts[bin_idx], 1u);
    let global_idx = bin_offsets[bin_idx] + local_idx;

    sorted_indices[global_idx] = id.x;
}
```

#### 4. `dem_forces.wgsl`

```wgsl
struct DemParams {
    cell_size: f32,
    grid_width: u32,
    grid_height: u32,
    particle_count: u32,
    dt: f32,
    gravity: f32,
    contact_stiffness: f32,
    damping_ratio: f32,
    friction_coeff: f32,
    solver_iterations: u32,
}

@compute @workgroup_size(256)
fn dem_forces(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.particle_count) { return; }

    let idx = id.x;
    var pos = positions[idx];
    var vel = velocities[idx];
    let radius = radii[idx];
    let mass = compute_mass(materials[idx], radius);

    // Apply gravity
    vel.y += params.gravity * params.dt;

    // Move
    pos += vel * params.dt;

    // Find my bin
    let gi = i32(pos.x / params.cell_size);
    let gj = i32(pos.y / params.cell_size);

    // Iterate 3x3 neighborhood
    for (var dj = -1; dj <= 1; dj++) {
        for (var di = -1; di <= 1; di++) {
            let ni = gi + di;
            let nj = gj + dj;

            if (ni < 0 || ni >= i32(params.grid_width) ||
                nj < 0 || nj >= i32(params.grid_height)) { continue; }

            let bin_idx = u32(nj) * params.grid_width + u32(ni);
            let bin_start = bin_offsets[bin_idx];
            let bin_end = bin_offsets[bin_idx + 1];

            for (var k = bin_start; k < bin_end; k++) {
                let j_idx = sorted_indices[k];
                if (j_idx <= idx) { continue; }  // Only process each pair once

                let pos_j = positions[j_idx];
                let diff = pos - pos_j;
                let dist_sq = dot(diff, diff);
                let radius_j = radii[j_idx];
                let contact_dist = radius + radius_j;

                if (dist_sq >= contact_dist * contact_dist) { continue; }

                let dist = sqrt(dist_sq);
                let overlap = contact_dist - dist;

                if (overlap > 0.0) {
                    let normal = diff / dist;

                    // Spring-damper force
                    let vel_j = velocities[j_idx];
                    let rel_vel = vel - vel_j;
                    let v_n = dot(rel_vel, normal);

                    let mass_j = compute_mass(materials[j_idx], radius_j);
                    let m_eff = (mass * mass_j) / (mass + mass_j);
                    let c_n = params.damping_ratio * 2.0 * sqrt(params.contact_stiffness * m_eff);
                    let f_n = max(params.contact_stiffness * overlap - c_n * v_n, 0.0);

                    // Position correction
                    let push = overlap * 0.5;
                    let ratio_i = mass_j / (mass + mass_j);
                    pos += normal * push * ratio_i;

                    // Velocity impulse
                    let impulse = normal * f_n * params.dt;
                    vel += impulse / mass;
                }
            }
        }
    }

    // Write back
    positions[idx] = pos;
    velocities[idx] = vel;
}
```

### Integration with Existing Code

**File:** `crates/game/src/gpu/dem.rs` (NEW)

```rust
pub struct GpuDemSolver {
    // Pipelines
    bin_count_pipeline: wgpu::ComputePipeline,
    prefix_sum_pipeline: wgpu::ComputePipeline,
    bin_insert_pipeline: wgpu::ComputePipeline,
    dem_forces_pipeline: wgpu::ComputePipeline,

    // Buffers
    positions_buffer: wgpu::Buffer,
    velocities_buffer: wgpu::Buffer,
    materials_buffer: wgpu::Buffer,
    radii_buffer: wgpu::Buffer,
    bin_counts_buffer: wgpu::Buffer,
    bin_offsets_buffer: wgpu::Buffer,
    sorted_indices_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,

    // Bind groups
    bin_count_bind_group: wgpu::BindGroup,
    prefix_sum_bind_group: wgpu::BindGroup,
    bin_insert_bind_group: wgpu::BindGroup,
    dem_forces_bind_group: wgpu::BindGroup,

    max_particles: usize,
    grid_width: u32,
    grid_height: u32,
}

impl GpuDemSolver {
    pub fn new(gpu: &Gpu, width: u32, height: u32, max_particles: usize) -> Self {
        // Create shaders, pipelines, buffers, bind groups
        todo!()
    }

    pub fn execute(
        &mut self,
        gpu: &Gpu,
        particles: &mut Particles,
        cell_size: f32,
        dt: f32,
        gravity: f32,
        params: &DemParams,
    ) {
        // 1. Upload sediment particle data
        self.upload_particles(gpu, particles);

        // 2. Clear bin counts
        self.clear_bins(gpu);

        // 3. Count particles per bin
        self.bin_count(gpu);

        // 4. Prefix sum for bin offsets
        self.prefix_sum(gpu);

        // 5. Insert particles into sorted array
        self.bin_insert(gpu);

        // 6. Run DEM forces (multiple iterations)
        for _ in 0..params.solver_iterations {
            self.dem_forces(gpu);
        }

        // 7. Download results
        self.download_particles(gpu, particles);
    }
}
```

**Modify:** `crates/game/src/main.rs`

```rust
// In App struct
dem_solver: Option<GpuDemSolver>,

// In resume()
self.dem_solver = Some(GpuDemSolver::new(&gpu, SIM_WIDTH, SIM_HEIGHT, 500_000));

// In update() - replace CPU DEM with GPU DEM
if let (Some(gpu), Some(solver)) = (&self.gpu, &mut self.dem_solver) {
    // Skip CPU DEM, use GPU instead
    solver.execute(
        gpu,
        &mut self.sim.particles,
        CELL_SIZE,
        dt,
        GRAVITY,
        &self.sim.dem.params,
    );
}
```

### Implementation Steps

1. **Create shader files:**
   - `crates/game/src/gpu/shaders/dem_bin_count.wgsl`
   - `crates/game/src/gpu/shaders/dem_prefix_sum.wgsl`
   - `crates/game/src/gpu/shaders/dem_bin_insert.wgsl`
   - `crates/game/src/gpu/shaders/dem_forces.wgsl`

2. **Create Rust solver:**
   - `crates/game/src/gpu/dem.rs`
   - Follow pattern from `advection.rs`

3. **Add to GPU module:**
   - `crates/game/src/gpu/mod.rs` - add `pub mod dem;`

4. **Integrate in main.rs:**
   - Add solver to App struct
   - Initialize in resume()
   - Call in update() instead of CPU DEM

5. **Handle edge cases:**
   - Floor/SDF collision (sample SDF texture like advection shader)
   - Sleep system (can be GPU or CPU post-process)
   - Water coupling (sample FLIP velocity grid if needed)

### Expected Performance

| Metric | CPU (current) | GPU (expected) |
|--------|---------------|----------------|
| 100k sediment particles | ~35ms | ~5-8ms |
| Collision detection | O(n) with high constant | O(n) with GPU parallelism |
| Bottleneck | Sequential iteration | Memory bandwidth |

### Risks and Mitigations

1. **Prefix sum complexity:** Use wgpu-based parallel scan or simple sequential fallback
2. **Atomic contention:** If too many particles in one bin, atomic operations slow down. Mitigation: larger grid (more cells)
3. **Download latency:** Position readback adds ~2-3ms. Can be hidden with async if needed

### References

- [lisyarus WebGPU Particle Life](https://lisyarus.github.io/blog/posts/particle-life-simulation-in-browser-using-webgpu.html) - WGSL spatial hash
- [Taichi DEM Optimization](https://docs.taichi-lang.org/blog/acclerate-collision-detection-with-taichi) - Grid-based algorithm
- [par-particle-life](https://crates.io/crates/par-particle-life) - Rust/wgpu implementation
- [PhasicFlow](https://github.com/PhasicFlow/phasicFlow) - Industrial GPU DEM
- [Chrono::GPU](https://www.mdpi.com/2227-9717/9/10/1813) - 130M particle benchmarks
