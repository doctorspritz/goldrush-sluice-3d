# Phase 1: GPU MGPCG Pressure Solver - Detailed Implementation Plan

## Overview

Replace the current 30-iteration SOR solver with a Multigrid-Preconditioned Conjugate Gradient (MGPCG) solver that converges reliably at any particle count.

**Goal**: Stable pressure solve in ~8ms at 1M particles

## Current State Analysis

### Existing GPU Infrastructure (`crates/game/src/gpu/pressure.rs`)
- wgpu buffers: pressure, divergence, cell_type, params
- Staging buffers for CPU↔GPU transfer
- Red-Black SOR pipelines (8x8 workgroups)
- `upload()`, `upload_warm()`, `solve()`, `download()` API

### Existing CPU Multigrid (`crates/sim/src/grid/mod.rs`)
- 4-6 levels: 512→256→128→64→32→16 (halving until <16)
- V-cycle: pre_smooth=10, post_smooth=10, coarse_solve=50
- Red-Black Gauss-Seidel smoother
- Full-weighting restriction (2x2 averaging)
- Bilinear prolongation
- 8 V-cycles per solve

### Key Insight
CPU multigrid works well. GPU MGPCG wraps a lightweight V-cycle as a preconditioner inside CG, getting guaranteed convergence in ~15-20 iterations.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GpuMgpcgSolver                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PCG Outer Loop (15-20 iterations):                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. r = b - Ax         (compute residual)            │   │
│  │ 2. z = M⁻¹r           (V-cycle preconditioner)      │   │
│  │ 3. p = z (or β update)(direction update)            │   │
│  │ 4. Ap = A*p           (matrix-vector multiply)      │   │
│  │ 5. α = rᵀz / pᵀAp     (step size via dot products)  │   │
│  │ 6. x += αp, r -= αAp  (solution/residual update)    │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                 │
│  V-Cycle Preconditioner (z = M⁻¹r):                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Level 0 (512²): Smooth Red→Black (3 iters)          │   │
│  │     ↓ restrict                                       │   │
│  │ Level 1 (256²): Smooth Red→Black (3 iters)          │   │
│  │     ↓ restrict                                       │   │
│  │ Level 2 (128²): Smooth Red→Black (3 iters)          │   │
│  │     ↓ restrict                                       │   │
│  │ Level 3 (64²):  Direct solve (20 iters)             │   │
│  │     ↑ prolongate                                     │   │
│  │ Level 2: Post-smooth (3 iters)                       │   │
│  │     ↑ prolongate                                     │   │
│  │ Level 1: Post-smooth (3 iters)                       │   │
│  │     ↑ prolongate                                     │   │
│  │ Level 0: Post-smooth (3 iters) → output z           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Buffer Layout

### Per-Level Buffers (GPU)
For a 512x512 grid with 4 levels:

| Level | Size | pressure | residual | divergence | cell_type | Total |
|-------|------|----------|----------|------------|-----------|-------|
| 0 | 512² | 1MB | 1MB | 1MB | 1MB | 4MB |
| 1 | 256² | 256KB | 256KB | 256KB | 256KB | 1MB |
| 2 | 128² | 64KB | 64KB | 64KB | 64KB | 256KB |
| 3 | 64² | 16KB | 16KB | 16KB | 16KB | 64KB |
| **Total** | | | | | | **~5.3MB** |

### PCG Vectors (Level 0 only)
| Buffer | Size | Purpose |
|--------|------|---------|
| x | 1MB | Solution (pressure) |
| r | 1MB | Residual |
| z | 1MB | Preconditioned residual |
| p | 1MB | Search direction |
| Ap | 1MB | A * p result |
| **Total** | **5MB** | |

### Reduction Buffers (for dot products)
| Buffer | Size | Purpose |
|--------|------|---------|
| partial_sums | 4KB | Workgroup partial sums |
| final_sum | 16B | Final scalar result |

**Total GPU Memory**: ~10MB (negligible)

---

## Shader Design

### 1. Red-Black Smoother (`mg_smooth.wgsl`)
Reuses existing `pressure.wgsl` logic but operates on any level:

```wgsl
struct LevelParams {
    width: u32,
    height: u32,
    level: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(1) var<storage, read> divergence: array<f32>;
@group(0) @binding(2) var<storage, read> cell_type: array<u32>;
@group(0) @binding(3) var<uniform> params: LevelParams;

@compute @workgroup_size(8, 8)
fn smooth_red(@builtin(global_invocation_id) id: vec3<u32>) {
    // Same logic as current pressure_red, using params.width/height
}

@compute @workgroup_size(8, 8)
fn smooth_black(@builtin(global_invocation_id) id: vec3<u32>) {
    // Same logic as current pressure_black
}
```

### 2. Restriction (`mg_restrict.wgsl`)
Full-weighting 2x2 → 1:

```wgsl
@group(0) @binding(0) var<storage, read> fine_residual: array<f32>;
@group(0) @binding(1) var<storage, read> fine_cell_type: array<u32>;
@group(0) @binding(2) var<storage, read_write> coarse_divergence: array<f32>;
@group(0) @binding(3) var<storage, read_write> coarse_cell_type: array<u32>;

struct RestrictParams {
    fine_width: u32,
    fine_height: u32,
    coarse_width: u32,
    coarse_height: u32,
}
@group(0) @binding(4) var<uniform> params: RestrictParams;

@compute @workgroup_size(8, 8)
fn restrict(@builtin(global_invocation_id) id: vec3<u32>) {
    let ci = id.x;
    let cj = id.y;
    if (ci >= params.coarse_width || cj >= params.coarse_height) { return; }

    let fi = ci * 2u;
    let fj = cj * 2u;

    // Average 2x2 block of fluid cells
    var sum = 0.0;
    var count = 0u;
    var any_fluid = false;

    for (var dj = 0u; dj < 2u; dj++) {
        for (var di = 0u; di < 2u; di++) {
            let fii = fi + di;
            let fjj = fj + dj;
            if (fii < params.fine_width && fjj < params.fine_height) {
                let f_idx = fjj * params.fine_width + fii;
                if (fine_cell_type[f_idx] == CELL_FLUID) {
                    sum += fine_residual[f_idx];
                    count += 1u;
                    any_fluid = true;
                }
            }
        }
    }

    let c_idx = cj * params.coarse_width + ci;
    coarse_divergence[c_idx] = select(0.0, sum / f32(count), count > 0u);
    coarse_cell_type[c_idx] = select(CELL_AIR, CELL_FLUID, any_fluid);
}
```

### 3. Prolongation (`mg_prolongate.wgsl`)
Bilinear interpolation + correction:

```wgsl
@group(0) @binding(0) var<storage, read> coarse_pressure: array<f32>;
@group(0) @binding(1) var<storage, read_write> fine_pressure: array<f32>;
@group(0) @binding(2) var<storage, read> fine_cell_type: array<u32>;

@compute @workgroup_size(8, 8)
fn prolongate(@builtin(global_invocation_id) id: vec3<u32>) {
    let fi = id.x;
    let fj = id.y;
    if (fi >= params.fine_width || fj >= params.fine_height) { return; }

    let f_idx = fj * params.fine_width + fi;
    if (fine_cell_type[f_idx] != CELL_FLUID) { return; }

    // Bilinear interpolation from coarse grid
    let cx = f32(fi) / 2.0;
    let cy = f32(fj) / 2.0;

    let ci0 = u32(floor(cx));
    let cj0 = u32(floor(cy));
    let ci1 = min(ci0 + 1u, params.coarse_width - 1u);
    let cj1 = min(cj0 + 1u, params.coarse_height - 1u);

    let tx = cx - f32(ci0);
    let ty = cy - f32(cj0);

    let p00 = coarse_pressure[cj0 * params.coarse_width + ci0];
    let p10 = coarse_pressure[cj0 * params.coarse_width + ci1];
    let p01 = coarse_pressure[cj1 * params.coarse_width + ci0];
    let p11 = coarse_pressure[cj1 * params.coarse_width + ci1];

    let correction = (1.0 - tx) * (1.0 - ty) * p00
                   + tx * (1.0 - ty) * p10
                   + (1.0 - tx) * ty * p01
                   + tx * ty * p11;

    fine_pressure[f_idx] += correction;
}
```

### 4. PCG Operations (`pcg_ops.wgsl`)

```wgsl
// Compute residual: r = b - Ax (Laplacian)
@compute @workgroup_size(8, 8)
fn compute_residual(...) {
    // r[idx] = divergence[idx] - laplacian(pressure, idx)
}

// Apply Laplacian: Ap = A * p
@compute @workgroup_size(8, 8)
fn apply_laplacian(...) {
    // Ap[idx] = laplacian(p, idx)
}

// Vector operations: x += alpha * p, r -= alpha * Ap
@compute @workgroup_size(256)
fn axpy(...) {
    // x[i] += alpha * y[i]
}

// Dot product with parallel reduction
@compute @workgroup_size(256)
fn dot_product_partial(...) {
    // Reduce to partial sums per workgroup
    var shared: array<f32, 256>;
    // ... reduction logic
}
```

---

## Rust Orchestration

### New File: `crates/game/src/gpu/mgpcg.rs`

```rust
pub struct GpuMgpcgSolver {
    // Grid dimensions
    width: u32,
    height: u32,
    num_levels: usize,

    // Per-level buffers
    levels: Vec<MgLevel>,

    // PCG vectors (level 0 size)
    x: wgpu::Buffer,  // Solution (alias for levels[0].pressure)
    r: wgpu::Buffer,  // Residual
    z: wgpu::Buffer,  // Preconditioned residual
    p: wgpu::Buffer,  // Search direction
    ap: wgpu::Buffer, // A * p

    // Reduction buffers
    partial_sums: wgpu::Buffer,
    final_sum: wgpu::Buffer,
    sum_staging: wgpu::Buffer,

    // Pipelines
    smooth_red: wgpu::ComputePipeline,
    smooth_black: wgpu::ComputePipeline,
    restrict: wgpu::ComputePipeline,
    prolongate: wgpu::ComputePipeline,
    compute_residual: wgpu::ComputePipeline,
    apply_laplacian: wgpu::ComputePipeline,
    axpy: wgpu::ComputePipeline,
    dot_partial: wgpu::ComputePipeline,
    dot_finalize: wgpu::ComputePipeline,

    // Bind groups per level
    smooth_bind_groups: Vec<wgpu::BindGroup>,
    restrict_bind_groups: Vec<wgpu::BindGroup>,
    prolongate_bind_groups: Vec<wgpu::BindGroup>,
}

struct MgLevel {
    width: u32,
    height: u32,
    pressure: wgpu::Buffer,
    residual: wgpu::Buffer,
    divergence: wgpu::Buffer,
    cell_type: wgpu::Buffer,
}

impl GpuMgpcgSolver {
    pub fn new(gpu: &GpuContext, width: u32, height: u32) -> Self {
        // Build level hierarchy (same logic as CPU)
        let mut levels = Vec::new();
        let (mut w, mut h) = (width, height);
        while w >= 16 && h >= 16 {
            levels.push(MgLevel::new(gpu, w, h));
            w /= 2;
            h /= 2;
        }
        if levels.is_empty() || (w >= 4 && h >= 4) {
            levels.push(MgLevel::new(gpu, w, h));
        }

        // Create PCG vectors, pipelines, bind groups...
        // ...
    }

    /// Upload divergence and cell_type from CPU
    pub fn upload(&self, gpu: &GpuContext, divergence: &[f32], cell_type: &[u32]) {
        // Upload to level 0
        gpu.queue.write_buffer(&self.levels[0].divergence, 0, bytemuck::cast_slice(divergence));
        gpu.queue.write_buffer(&self.levels[0].cell_type, 0, bytemuck::cast_slice(cell_type));
    }

    /// Upload with warm start
    pub fn upload_warm(&self, gpu: &GpuContext, divergence: &[f32], cell_type: &[u32], pressure: &[f32]) {
        self.upload(gpu, divergence, cell_type);
        gpu.queue.write_buffer(&self.levels[0].pressure, 0, bytemuck::cast_slice(pressure));
    }

    /// Run MGPCG solve
    pub fn solve(&self, gpu: &GpuContext, max_iterations: u32, tolerance: f32) {
        let mut encoder = gpu.device.create_command_encoder(&Default::default());

        // Initial residual: r = b - Ax
        self.dispatch_residual(&mut encoder);

        // PCG iteration
        for iter in 0..max_iterations {
            // z = M⁻¹r (V-cycle preconditioner)
            self.dispatch_vcycle(&mut encoder);

            // rz = rᵀz
            let rz = self.dispatch_dot(&mut encoder, &self.r, &self.z);

            if iter == 0 {
                // p = z
                self.dispatch_copy(&mut encoder, &self.z, &self.p);
            } else {
                // β = rz / rz_old
                // p = z + β*p
                self.dispatch_axpy(&mut encoder, beta, &self.p, &self.z, &self.p);
            }

            // Ap = A*p
            self.dispatch_laplacian(&mut encoder, &self.p, &self.ap);

            // pAp = pᵀAp
            let pAp = self.dispatch_dot(&mut encoder, &self.p, &self.ap);

            // α = rz / pAp
            // x += αp
            // r -= αAp
            self.dispatch_pcg_update(&mut encoder, alpha);

            // Check convergence (optional, can skip for fixed iterations)
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
    }

    fn dispatch_vcycle(&self, encoder: &mut wgpu::CommandEncoder) {
        let max_level = self.levels.len() - 1;
        self.dispatch_vcycle_recursive(encoder, 0, max_level);
    }

    fn dispatch_vcycle_recursive(&self, encoder: &mut wgpu::CommandEncoder, level: usize, max_level: usize) {
        let pre_smooth = 3;
        let post_smooth = 3;
        let coarse_solve = 20;

        // Pre-smooth
        for _ in 0..pre_smooth {
            self.dispatch_smooth(encoder, level);
        }

        if level == max_level {
            // Coarse solve
            for _ in 0..coarse_solve {
                self.dispatch_smooth(encoder, level);
            }
        } else {
            // Compute residual
            self.dispatch_level_residual(encoder, level);

            // Restrict to coarse
            self.dispatch_restrict(encoder, level, level + 1);

            // Clear coarse pressure
            self.dispatch_clear(encoder, level + 1);

            // Recurse
            self.dispatch_vcycle_recursive(encoder, level + 1, max_level);

            // Prolongate
            self.dispatch_prolongate(encoder, level + 1, level);

            // Post-smooth
            for _ in 0..post_smooth {
                self.dispatch_smooth(encoder, level);
            }
        }
    }

    /// Download pressure to CPU
    pub fn download(&self, gpu: &GpuContext, pressure: &mut [f32]) {
        // Same as existing implementation
    }
}
```

---

## Implementation Steps

### Step 1: Multi-Level Buffer Infrastructure
- [ ] Create `MgLevel` struct with GPU buffers
- [ ] Build level hierarchy in `GpuMgpcgSolver::new()`
- [ ] Create bind group layouts for each operation
- [ ] Test: verify buffer sizes and allocation

### Step 2: Port Smoother to Parametric Level
- [ ] Modify `pressure.wgsl` → `mg_smooth.wgsl` with level params
- [ ] Create `smooth_red` and `smooth_black` pipelines
- [ ] Create bind groups for each level
- [ ] Test: run smoother on level 0, compare with existing solver

### Step 3: Implement Restriction
- [ ] Create `mg_restrict.wgsl`
- [ ] Create restrict pipeline and bind groups
- [ ] Test: upload test data, restrict, download, compare with CPU

### Step 4: Implement Prolongation
- [ ] Create `mg_prolongate.wgsl`
- [ ] Create prolongate pipeline and bind groups
- [ ] Test: upload coarse data, prolongate, compare with CPU bilinear

### Step 5: Implement V-Cycle
- [ ] Create recursive `dispatch_vcycle` method
- [ ] Implement level residual computation
- [ ] Test: run single V-cycle, compare divergence reduction with CPU

### Step 6: Implement PCG Operations
- [ ] Create `pcg_ops.wgsl` (residual, laplacian, axpy, dot)
- [ ] Implement parallel reduction for dot products
- [ ] Create PCG vector buffers
- [ ] Test: verify dot product accuracy

### Step 7: Integrate PCG + V-Cycle
- [ ] Implement full `solve()` method
- [ ] Add iteration count and tolerance parameters
- [ ] Test: compare final pressure with CPU multigrid

### Step 8: Validation & Benchmarking
- [ ] Run at 1M particles, verify no divergence
- [ ] Run at 2M particles, verify stability
- [ ] Measure timing: target <8ms
- [ ] Compare visual quality with CPU reference

---

## Risk Mitigation

### Risk 1: Dot Product Accuracy
**Problem**: Parallel reduction can accumulate floating-point errors.
**Mitigation**: Use Kahan summation or double precision for partial sums.

### Risk 2: Coarse Level GPU Inefficiency
**Problem**: Small grids (64x64, 32x32) don't utilize GPU well.
**Mitigation**: Accept the overhead (~0.1ms) or run coarsest level on CPU.

### Risk 3: Synchronization Overhead
**Problem**: Multiple dispatches per V-cycle may cause GPU stalls.
**Mitigation**: Batch all dispatches into single command encoder.

### Risk 4: Memory Bandwidth
**Problem**: Many buffer reads/writes per iteration.
**Mitigation**: Use 16-bit floats for intermediate levels if needed.

---

## Success Criteria

1. **Correctness**: GPU output matches CPU multigrid within 1e-4
2. **Stability**: No divergence at 2M particles over 5 minutes
3. **Performance**: <8ms at 1M particles (512x512 grid, 15-20 iterations)
4. **Visual Quality**: No visible artifacts compared to CPU reference

---

## Files to Create

| File | Lines (est) | Purpose |
|------|-------------|---------|
| `gpu/mgpcg.rs` | 500 | Main orchestration |
| `gpu/shaders/mg_smooth.wgsl` | 100 | Red-Black smoother |
| `gpu/shaders/mg_restrict.wgsl` | 60 | Restriction operator |
| `gpu/shaders/mg_prolongate.wgsl` | 50 | Prolongation operator |
| `gpu/shaders/pcg_ops.wgsl` | 150 | CG vector operations |
| `gpu/mod.rs` (update) | +10 | Export new module |

**Total**: ~870 lines of new code
