# GPU Integration Plan

## Target: 30 FPS at 1 Million Particles

### Current Performance (CPU-heavy)

| Particles | Total (ms) | Pressure | P2G (CPU) | G2P | FPS |
|-----------|------------|----------|-----------|-----|-----|
| 150K      | 38ms       | 23ms     | 8ms       | 1ms | ~26 |
| 300K      | 57ms       | 23ms     | 20ms      | 3ms | ~18 |
| 600K      | 99ms       | 24ms     | 44ms      | 7ms | ~10 |
| 1M        | 138ms      | 24ms     | 81ms      | 11ms| ~7  |
| 1.5M      | 206ms      | 26ms     | 119ms     | 15ms| ~5  |
| 2M        | 280ms      | 27ms     | 146ms     | 23ms| ~4  |

**Critical Finding**: At 2M particles, simulation becomes unstable (pressure exploding to 1216, divergence not converging). Root cause: 30 SOR iterations insufficient for large domains.

### Target Budget (33ms frame)

| Component | Budget | Current (1M) | Strategy |
|-----------|--------|--------------|----------|
| P2G       | 5ms    | 81ms         | GPU compute |
| Pressure  | 8ms    | 24ms         | GPU MGPCG |
| G2P       | 3ms    | 11ms         | GPU compute |
| Advect    | 2ms    | 2ms          | GPU compute |
| Render    | 8ms    | 8ms          | Already GPU |
| Overhead  | 7ms    | -            | Buffer |
| **Total** | 33ms   | 138ms        | **4.2x faster** |

---

## Research Summary: Pressure Solver Best Practices

### Key Findings from Literature

1. **Jacobi vs Multigrid** ([vassvik gist](https://gist.github.com/vassvik/f06a453c18eae03a9ad4dc8cc011d2dc))
   - 1 V-cycle ≈ 1000 Jacobi iterations at ~10-iteration cost
   - 160 Jacobi iterations still show smoke smearing
   - Multigrid is "the holy grail" - fast AND accurate

2. **GPU Multigrid Challenges** ([NVIDIA Blog](https://developer.nvidia.com/blog/high-performance-geometric-multi-grid-gpu-acceleration/))
   - Fine levels run efficiently on GPU
   - Coarse levels become latency-limited
   - **Hybrid approach**: GPU for fine levels, CPU for coarse → 30% speedup

3. **MGPCG (Multigrid-Preconditioned Conjugate Gradient)** ([UC Davis](https://www.math.ucdavis.edu/~jteran/papers/MST10.pdf), [ACM](https://dl.acm.org/doi/10.1145/3432261.3432273))
   - Uses lightweight multigrid as CG preconditioner
   - **~20 iterations for convergence** (vs hundreds for plain CG)
   - Reduces iterations to <15% of plain PCG → 3-6x speedup
   - More robust on irregular domains
   - Main kernels achieve >90% roofline performance

4. **PCG with Incomplete Poisson Preconditioner** ([Bitterli](https://benedikt-bitterli.me/gpu-fluid.html), [blub](https://github.com/wumpf/blub))
   - GPU-friendly alternative to Modified Incomplete Cholesky
   - Used in production wgpu implementations
   - Key bottleneck: dot product reduction

5. **Red-Black Gauss-Seidel** ([cuda-multigrid](https://github.com/ooreilly/cuda-multigrid))
   - Required for GPU parallelism (vs sequential G-S)
   - Converges in 10-11 iterations with multigrid
   - Trade-off: 2 passes to DRAM, 50% thread utilization per pass

### Recommended Architecture: MGPCG

```
Pressure Solve (GPU):
┌─────────────────────────────────────────┐
│ PCG Outer Loop (~15-20 iterations)      │
│ ┌─────────────────────────────────────┐ │
│ │ Multigrid V-Cycle Preconditioner    │ │
│ │ ├─ Level 0: Red-Black G-S (GPU)     │ │
│ │ ├─ Level 1: Red-Black G-S (GPU)     │ │
│ │ ├─ Level 2: Red-Black G-S (GPU)     │ │
│ │ └─ Coarse: Direct solve (GPU/CPU)   │ │
│ └─────────────────────────────────────┘ │
│ + Dot products (parallel reduction)     │
│ + Vector updates (trivially parallel)   │
└─────────────────────────────────────────┘
```

---

## Phase 1: Stable GPU Pressure (MGPCG) ⬅️ PRIORITY

### Problem Statement
Current 30-iteration SOR diverges at high particle counts because:
- SOR only eliminates high-frequency errors efficiently
- Low-frequency errors (spanning domain) converge very slowly
- CPU multigrid uses 8 V-cycles × ~70 smoothing ≈ 560 effective iterations

### Solution: GPU Multigrid-Preconditioned CG

**Why MGPCG over pure Multigrid:**
- More robust on irregular fluid domains
- Convergence guaranteed (CG property)
- Lightweight MG sufficient as preconditioner
- Easier to implement than full multigrid

### Implementation Steps

#### 1.1 GPU Multigrid V-Cycle Preconditioner
```wgsl
// mg_vcycle.wgsl - One V-cycle as PCG preconditioner
// Uses 3-4 levels, Red-Black G-S smoother

@compute @workgroup_size(8, 8)
fn smooth_red(@builtin(global_invocation_id) id: vec3<u32>) {
    // Red points: (i+j) % 2 == 0
    // Gauss-Seidel update using 4 neighbors
}

@compute @workgroup_size(8, 8)
fn smooth_black(@builtin(global_invocation_id) id: vec3<u32>) {
    // Black points: (i+j) % 2 == 1
}

@compute @workgroup_size(8, 8)
fn restrict(@builtin(global_invocation_id) id: vec3<u32>) {
    // Full-weighting restriction to coarse grid
}

@compute @workgroup_size(8, 8)
fn prolongate(@builtin(global_invocation_id) id: vec3<u32>) {
    // Bilinear interpolation to fine grid
}
```

#### 1.2 GPU PCG Driver
```wgsl
// pcg.wgsl - Conjugate gradient operations

@compute @workgroup_size(256)
fn compute_residual(...) { /* r = b - Ax */ }

@compute @workgroup_size(256)
fn apply_laplacian(...) { /* Ap = A * p */ }

@compute @workgroup_size(256)
fn vector_update(...) { /* x += alpha * p */ }

@compute @workgroup_size(256)
fn dot_product_partial(...) { /* Parallel reduction */ }
```

#### 1.3 Multi-Level Grid Buffers
```rust
struct MgLevels {
    // Level 0: Full resolution (e.g., 512x512)
    pressure_0: wgpu::Buffer,
    residual_0: wgpu::Buffer,

    // Level 1: Half resolution (256x256)
    pressure_1: wgpu::Buffer,
    residual_1: wgpu::Buffer,

    // Level 2: Quarter resolution (128x128)
    pressure_2: wgpu::Buffer,
    residual_2: wgpu::Buffer,

    // Level 3: Coarsest (64x64) - direct solve OK
    pressure_3: wgpu::Buffer,
}
```

### Expected Performance
- **Iterations**: 15-20 PCG iterations (vs 30+ SOR that still diverges)
- **Time per iteration**: ~0.3ms (V-cycle) + ~0.1ms (CG ops) = ~0.4ms
- **Total**: ~6-8ms for fully converged pressure
- **Stability**: Guaranteed convergence at any particle count

---

## Phase 2: GPU P2G Transfer

### Current State
- P2G is 60-80% of frame time at high particle counts
- Linear O(n) scaling with particle count
- Single-threaded on CPU

### GPU Implementation

#### 2.1 Atomic Scatter Approach
```wgsl
// p2g.wgsl
struct Particle {
    position: vec2<f32>,
    velocity: vec2<f32>,
    c_matrix: mat2x2<f32>,
    mass: f32,
    _pad: f32,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> u_sum: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> u_weight: array<atomic<i32>>;
// ... v_sum, v_weight

@compute @workgroup_size(256)
fn p2g_scatter(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = particles[id.x];
    let base_cell = vec2<i32>(floor(p.position));

    // 3x3 neighborhood
    for (var di = -1; di <= 1; di++) {
        for (var dj = -1; dj <= 1; dj++) {
            let cell = base_cell + vec2(di, dj);
            let weight = quadratic_bspline(p.position - vec2<f32>(cell));

            // APIC momentum transfer
            let offset = vec2<f32>(cell) - p.position;
            let v_apic = p.velocity + p.c_matrix * offset;

            // Atomic accumulation (fixed-point for atomicAdd)
            let idx = cell_index(cell);
            atomicAdd(&u_sum[idx], i32(v_apic.x * weight * 1000000.0));
            atomicAdd(&u_weight[idx], i32(weight * 1000000.0));
        }
    }
}

@compute @workgroup_size(8, 8)
fn p2g_divide(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.y * grid_width + id.x;
    let w = f32(atomicLoad(&u_weight[idx])) / 1000000.0;
    if (w > 0.0) {
        grid_u[idx] = f32(atomicLoad(&u_sum[idx])) / 1000000.0 / w;
    }
}
```

#### 2.2 Alternative: Sorting + Binning
For even better performance with 1M+ particles:
1. Sort particles by cell (GPU radix sort)
2. Compute cell boundaries (prefix sum)
3. Each cell processes its particles (no atomics)

### Expected Performance
- **1M particles**: 81ms (CPU) → ~3-5ms (GPU)
- **Bottleneck shifts** to pressure solve or memory bandwidth

---

## Phase 3: GPU G2P Transfer

### Implementation
```wgsl
// g2p.wgsl - Embarrassingly parallel (no atomics)
@compute @workgroup_size(256)
fn g2p(@builtin(global_invocation_id) id: vec3<u32>) {
    var p = particles[id.x];
    let base_cell = vec2<i32>(floor(p.position));

    var v_pic = vec2(0.0);
    var v_flip = vec2(0.0);
    var new_c = mat2x2(0.0);

    // Gather from 3x3 neighborhood
    for (var di = -1; di <= 1; di++) {
        for (var dj = -1; dj <= 1; dj++) {
            let cell = base_cell + vec2(di, dj);
            let weight = quadratic_bspline(p.position - vec2<f32>(cell));
            let grad = quadratic_bspline_grad(p.position - vec2<f32>(cell));

            let idx = cell_index(cell);
            let v_new = vec2(grid_u[idx], grid_v[idx]);
            let v_old = vec2(grid_u_old[idx], grid_v_old[idx]);

            v_pic += weight * v_new;
            v_flip += weight * (v_new - v_old);

            // APIC C matrix
            new_c += outer_product(v_new, grad) * weight;
        }
    }

    // FLIP blend
    let flip_ratio = material_flip_ratio(p.material);
    p.velocity = p.velocity + flip_ratio * v_flip + (1.0 - flip_ratio) * (v_pic - p.velocity);
    p.c_matrix = new_c;

    particles[id.x] = p;
}
```

### Expected Performance
- **1M particles**: 11ms (CPU) → ~1-2ms (GPU)

---

## Phase 4: GPU Advection + SDF

```wgsl
// advect.wgsl
@compute @workgroup_size(256)
fn advect(@builtin(global_invocation_id) id: vec3<u32>) {
    var p = particles[id.x];

    // Euler integration
    p.position += p.velocity * dt;

    // SDF collision
    let sdf = sample_sdf_texture(p.position);
    if (sdf < 0.0) {
        let grad = sdf_gradient(p.position);
        p.position -= grad * sdf;
        // Friction/bounce on velocity
    }

    particles[id.x] = p;
}
```

---

## Full GPU Pipeline Architecture

```
Frame N:
┌─────────────────────────────────────────────────────────┐
│                        GPU                               │
├─────────────────────────────────────────────────────────┤
│ 1. P2G Scatter       [compute, atomics]         ~3ms    │
│ 2. P2G Divide        [compute]                  ~0.5ms  │
│ 3. Apply Gravity     [compute]                  ~0.2ms  │
│ 4. Compute Divergence[compute]                  ~0.3ms  │
│ 5. MGPCG Pressure    [compute, ~15 iterations]  ~6ms    │
│ 6. Apply Pressure    [compute]                  ~0.3ms  │
│ 7. G2P Gather        [compute]                  ~2ms    │
│ 8. Advection + SDF   [compute]                  ~1ms    │
│ 9. Render            [graphics]                 ~8ms    │
├─────────────────────────────────────────────────────────┤
│ Total estimate: ~21ms (48 FPS) at 1M particles          │
└─────────────────────────────────────────────────────────┘
         ↑                                    ↓
     Upload: dt, spawn                   Download: diagnostics (async)
```

### Data Residency

**Stays on GPU (persistent buffers):**
- Particle buffer (position, velocity, C, material)
- Grid buffers (u, v, u_old, v_old, pressure, divergence)
- Multigrid level buffers
- SDF texture
- Cell type buffer

**Uploaded per frame:**
- Uniforms: dt, gravity, frame_number
- New spawned particles (append)

**Downloaded (async, optional):**
- Divergence max (for diagnostics)
- Velocity max
- Particle positions (for CPU collision detection if needed)

---

## Implementation Priority

### Phase 1: MGPCG Pressure (Critical Path)
1. Implement Red-Black G-S smoother shader
2. Implement restriction/prolongation shaders
3. Implement V-cycle orchestration
4. Implement PCG with MG preconditioner
5. Validate convergence matches CPU multigrid
6. Test stability at 2M particles

### Phase 2: GPU P2G (Biggest Win)
1. Implement atomic scatter shader
2. Implement divide pass
3. Validate against CPU P2G output
4. Profile atomic contention

### Phase 3: GPU G2P
1. Implement gather shader
2. Validate FLIP blend
3. Test visual quality

### Phase 4: Full Pipeline
1. GPU advection + SDF
2. Remove CPU-GPU transfers
3. Async diagnostics download
4. Profile end-to-end

---

## Files to Create

| File | Purpose |
|------|---------|
| `gpu/mgpcg.rs` | MGPCG pressure solver orchestration |
| `gpu/shaders/mg_smooth.wgsl` | Red-Black G-S smoother |
| `gpu/shaders/mg_restrict.wgsl` | Restriction operator |
| `gpu/shaders/mg_prolongate.wgsl` | Prolongation operator |
| `gpu/shaders/pcg_ops.wgsl` | CG vector operations |
| `gpu/p2g.rs` | P2G orchestration |
| `gpu/shaders/p2g.wgsl` | P2G compute shader |
| `gpu/g2p.rs` | G2P orchestration |
| `gpu/shaders/g2p.wgsl` | G2P compute shader |
| `gpu/advect.rs` | Advection orchestration |
| `gpu/shaders/advect.wgsl` | Advection shader |
| `gpu/simulation.rs` | Full GPU pipeline coordinator |

---

## Success Metrics

1. **Stability**: No divergence/explosion at 2M particles
2. **30 FPS at 1M particles** (33ms frame budget)
3. **No CPU-GPU sync stalls** in main loop
4. **Visual quality** matches CPU reference

---

## References

- [GPU Gems: Fast Fluid Dynamics](https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu)
- [GPU Gems 3: Real-Time 3D Fluids](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-30-real-time-simulation-and-rendering-3d-fluids)
- [Benedikt Bitterli: GPU Fluid](https://benedikt-bitterli.me/gpu-fluid.html)
- [blub: wgpu fluid simulation](https://github.com/wumpf/blub)
- [vassvik: Projection Comparison](https://gist.github.com/vassvik/f06a453c18eae03a9ad4dc8cc011d2dc)
- [NVIDIA: GPU Geometric Multigrid](https://developer.nvidia.com/blog/high-performance-geometric-multi-grid-gpu-acceleration/)
- [cuda-multigrid: GPU Poisson](https://github.com/ooreilly/cuda-multigrid)
- [UC Davis: Parallel MG Poisson](https://www.math.ucdavis.edu/~jteran/papers/MST10.pdf)
- [ACM: MG-CG on Block-Structured Grid](https://dl.acm.org/doi/10.1145/3432261.3432273)
