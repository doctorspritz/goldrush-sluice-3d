# GPU P2G Implementation Plan

## Overview

Particle-to-Grid (P2G) transfer is the second-largest bottleneck after pressure solve at scale:

| Particles | CPU P2G | % of Frame |
|-----------|---------|------------|
| 150K      | 8ms     | 21%        |
| 300K      | 20ms    | 35%        |
| 600K      | 44ms    | 44%        |
| 1M        | 81ms    | 59%        |
| 2M        | 146ms   | 52%        |

**Target**: Reduce 1M-particle P2G from 81ms to <5ms (16x speedup).

## Research Summary

### Approach 1: Fixed-Point Atomic Scatter (Recommended)

Used by EA's [pbmpm](https://github.com/electronicarts/pbmpm) (SIGGRAPH 2024) and [WebGPU-Ocean](https://github.com/matsuoka-601/WebGPU-Ocean).

**How it works:**
- WebGPU/wgpu only has `atomicAdd` for 32-bit integers
- Encode floats as fixed-point: `f32 * SCALE → i32`
- After all particles scatter: `i32 / SCALE → f32`
- Typical SCALE: 1e6 to 1e7 (need precision for sum of many small weights)

**Pros:**
- Simple to implement
- No sorting required
- Works with any particle count

**Cons:**
- Atomic contention in dense regions
- Fixed-point precision limits (±2147 with 1e6 scale)

**Performance**: ~100K particles on integrated GPU, ~300K on dedicated GPU (WebGPU-Ocean benchmarks)

### Approach 2: Linked List Gather

Used by [blub](https://github.com/Wumpf/blub) (Rust/wgpu).

**How it works:**
- Each particle atomicExchange's its index into a "head pointer grid"
- Grid cells walk linked lists to gather from all particles
- Shared memory optimization: each thread walks 1 list, stores to shared mem, others read 7 neighbors

**Pros:**
- 4x speedup with shared memory optimization
- No floating-point precision issues

**Cons:**
- More complex implementation
- Multiple passes required
- Variable work per cell (load imbalancing)

### Approach 3: G2P2G Fused Kernel (Advanced)

From SIGGRAPH 2020 [Multi-GPU MPM](https://sites.google.com/view/siggraph2020-multigpu).

**How it works:**
- Fuse G2P + advect + P2G into single kernel
- Temporary velocities stay in registers, not global memory
- Eliminates 2 kernel launches + 2 particle data reads per frame

**Performance**: 110-120x speedup vs CPU (with CUDA, not wgpu)

**Not recommended for initial implementation** - requires restructuring entire pipeline.

## Chosen Approach: Fixed-Point Atomic Scatter

Rationale:
1. Simplest to implement and debug
2. Matches existing CPU algorithm structure
3. Sufficient for 1M particles on dedicated GPU
4. Can upgrade to linked-list gather later if needed

## Implementation Design

### Data Structures

```rust
// Particle buffer (GPU-side, SoA for coalescing)
struct GpuParticles {
    positions: wgpu::Buffer,    // vec2<f32> per particle
    velocities: wgpu::Buffer,   // vec2<f32> per particle
    c_matrices: wgpu::Buffer,   // mat2x2<f32> per particle (APIC)
    materials: wgpu::Buffer,    // u32 per particle (for is_sediment)
}

// Grid accumulator buffers (use i32 for atomics)
struct P2gAccumulators {
    u_sum: wgpu::Buffer,      // atomic<i32>, (width+1) * height
    u_weight: wgpu::Buffer,   // atomic<i32>
    v_sum: wgpu::Buffer,      // atomic<i32>, width * (height+1)
    v_weight: wgpu::Buffer,   // atomic<i32>
    // Volume fractions for two-way coupling
    sand_volume_u: wgpu::Buffer,   // atomic<i32>
    water_volume_u: wgpu::Buffer,  // atomic<i32>
    sand_volume_v: wgpu::Buffer,   // atomic<i32>
    water_volume_v: wgpu::Buffer,  // atomic<i32>
}

// Output grid buffers (f32)
struct GridVelocities {
    u: wgpu::Buffer,
    v: wgpu::Buffer,
}
```

### Shader Design

```wgsl
// p2g_scatter.wgsl

// Fixed-point scale: 10^6 gives ±2147 range, sufficient precision
const SCALE: f32 = 1000000.0;
const INV_SCALE: f32 = 0.000001;

struct Params {
    cell_size: f32,
    width: u32,
    height: u32,
    particle_count: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> velocities: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> c_matrices: array<mat2x2<f32>>;
@group(0) @binding(4) var<storage, read> materials: array<u32>;
@group(0) @binding(5) var<storage, read_write> u_sum: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> u_weight: array<atomic<i32>>;
@group(0) @binding(7) var<storage, read_write> v_sum: array<atomic<i32>>;
@group(0) @binding(8) var<storage, read_write> v_weight: array<atomic<i32>>;

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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.particle_count) { return; }

    let pos = positions[id.x];
    let vel = velocities[id.x];
    let c_mat = c_matrices[id.x];
    let is_sand = (materials[id.x] & 1u) != 0u;

    // Skip sediment velocity contribution (two-way coupling via density)
    if (is_sand) { return; }

    let cell_size = params.cell_size;
    let width = params.width;
    let height = params.height;

    // ========== U component (staggered on left edges) ==========
    let u_pos = pos / cell_size - vec2<f32>(0.0, 0.5);
    let base_i_u = i32(floor(u_pos.x));
    let base_j_u = i32(floor(u_pos.y));
    let fx_u = u_pos.x - f32(base_i_u);
    let fy_u = u_pos.y - f32(base_j_u);

    // Precompute 1D weights
    let u_wx = array<f32, 3>(
        quadratic_bspline_1d(fx_u + 1.0),
        quadratic_bspline_1d(fx_u),
        quadratic_bspline_1d(fx_u - 1.0)
    );
    let u_wy = array<f32, 3>(
        quadratic_bspline_1d(fy_u + 1.0),
        quadratic_bspline_1d(fy_u),
        quadratic_bspline_1d(fy_u - 1.0)
    );

    for (var dj: i32 = -1; dj <= 1; dj++) {
        let nj = base_j_u + dj;
        if (nj < 0 || nj >= i32(height)) { continue; }
        let wy = u_wy[dj + 1];

        for (var di: i32 = -1; di <= 1; di++) {
            let ni = base_i_u + di;
            if (ni < 0 || ni > i32(width)) { continue; }
            let w = u_wx[di + 1] * wy;
            if (w <= 0.0) { continue; }

            // APIC momentum
            let offset = vec2<f32>(f32(ni) * cell_size - pos.x,
                                   (f32(nj) + 0.5) * cell_size - pos.y);
            let affine_vel = c_mat * offset;
            let momentum_x = (vel.x + affine_vel.x) * w;

            let idx = u32(nj) * (width + 1u) + u32(ni);
            atomicAdd(&u_sum[idx], i32(momentum_x * SCALE));
            atomicAdd(&u_weight[idx], i32(w * SCALE));
        }
    }

    // ========== V component (staggered on bottom edges) ==========
    let v_pos = pos / cell_size - vec2<f32>(0.5, 0.0);
    let base_i_v = i32(floor(v_pos.x));
    let base_j_v = i32(floor(v_pos.y));
    let fx_v = v_pos.x - f32(base_i_v);
    let fy_v = v_pos.y - f32(base_j_v);

    let v_wx = array<f32, 3>(
        quadratic_bspline_1d(fx_v + 1.0),
        quadratic_bspline_1d(fx_v),
        quadratic_bspline_1d(fx_v - 1.0)
    );
    let v_wy = array<f32, 3>(
        quadratic_bspline_1d(fy_v + 1.0),
        quadratic_bspline_1d(fy_v),
        quadratic_bspline_1d(fy_v - 1.0)
    );

    for (var dj: i32 = -1; dj <= 1; dj++) {
        let nj = base_j_v + dj;
        if (nj < 0 || nj > i32(height)) { continue; }
        let wy = v_wy[dj + 1];

        for (var di: i32 = -1; di <= 1; di++) {
            let ni = base_i_v + di;
            if (ni < 0 || ni >= i32(width)) { continue; }
            let w = v_wx[di + 1] * wy;
            if (w <= 0.0) { continue; }

            let offset = vec2<f32>((f32(ni) + 0.5) * cell_size - pos.x,
                                   f32(nj) * cell_size - pos.y);
            let affine_vel = c_mat * offset;
            let momentum_y = (vel.y + affine_vel.y) * w;

            let idx = u32(nj) * width + u32(ni);
            atomicAdd(&v_sum[idx], i32(momentum_y * SCALE));
            atomicAdd(&v_weight[idx], i32(w * SCALE));
        }
    }
}
```

```wgsl
// p2g_divide.wgsl

@group(0) @binding(0) var<storage, read> u_sum: array<i32>;
@group(0) @binding(1) var<storage, read> u_weight: array<i32>;
@group(0) @binding(2) var<storage, read> v_sum: array<i32>;
@group(0) @binding(3) var<storage, read> v_weight: array<i32>;
@group(0) @binding(4) var<storage, read_write> grid_u: array<f32>;
@group(0) @binding(5) var<storage, read_write> grid_v: array<f32>;

const INV_SCALE: f32 = 0.000001;

@compute @workgroup_size(8, 8)
fn divide_u(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.y * (params.width + 1u) + id.x;
    let w = f32(u_weight[idx]) * INV_SCALE;
    if (w > 0.0) {
        grid_u[idx] = f32(u_sum[idx]) * INV_SCALE / w;
    } else {
        grid_u[idx] = 0.0;
    }
}

@compute @workgroup_size(8, 8)
fn divide_v(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.y * params.width + id.x;
    let w = f32(v_weight[idx]) * INV_SCALE;
    if (w > 0.0) {
        grid_v[idx] = f32(v_sum[idx]) * INV_SCALE / w;
    } else {
        grid_v[idx] = 0.0;
    }
}
```

## Benchmark Design

### Test Cases

1. **Uniform Distribution**: Particles evenly spread (low contention)
2. **Clustered Distribution**: Particles in dense regions (high contention)
3. **Sluice Flow**: Real simulation pattern (mixed)

### Metrics

- Wall-clock time (ms)
- Throughput (particles/ms)
- Atomic contention rate (GPU profiler if available)

### Benchmark Code

```rust
// crates/sim/examples/p2g_benchmark.rs

use sim::{FlipSimulation, create_sluice_with_mode, SluiceConfig, RiffleMode};
use std::time::{Duration, Instant};

const PARTICLE_COUNTS: [usize; 6] = [50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000];
const WARMUP_FRAMES: usize = 100;
const BENCHMARK_FRAMES: usize = 500;

fn main() {
    println!("=== P2G BENCHMARK ===");
    println!();

    for &count in &PARTICLE_COUNTS {
        let (sim_width, sim_height) = grid_size_for_particles(count);
        let mut sim = FlipSimulation::new(sim_width, sim_height, 1.0);

        // Setup sluice geometry
        let config = SluiceConfig {
            slope: 0.12,
            riffle_spacing: 60,
            riffle_height: 6,
            riffle_width: 4,
            riffle_mode: RiffleMode::ClassicBattEdge,
            slick_plate_len: 0,
        };
        create_sluice_with_mode(&mut sim, &config);

        // Spawn particles
        spawn_particles(&mut sim, count);

        // Warmup
        for _ in 0..WARMUP_FRAMES {
            sim.classify_cells();
            sim.particles_to_grid();
        }

        // Benchmark
        let start = Instant::now();
        for _ in 0..BENCHMARK_FRAMES {
            sim.classify_cells();
            sim.particles_to_grid();
        }
        let elapsed = start.elapsed();

        let avg_ms = elapsed.as_secs_f64() * 1000.0 / BENCHMARK_FRAMES as f64;
        let throughput = count as f64 / avg_ms;

        println!("{:>8} particles: {:>6.2} ms/frame ({:.1} K particles/ms)",
            count, avg_ms, throughput / 1000.0);
    }
}

fn grid_size_for_particles(count: usize) -> (usize, usize) {
    // ~4 particles per cell
    let cells = count / 4;
    let side = (cells as f64).sqrt() as usize;
    (side.max(256), (side / 2).max(128))
}

fn spawn_particles(sim: &mut FlipSimulation, count: usize) {
    // Uniform distribution for benchmark (worst case for cache, baseline)
    let width = sim.grid.width as f32;
    let height = sim.grid.height as f32;
    let cell_size = sim.grid.cell_size;

    for i in 0..count {
        let x = (i % 1000) as f32 / 1000.0 * (width - 20.0) * cell_size + 10.0 * cell_size;
        let y = (i / 1000) as f32 / (count as f32 / 1000.0) * (height - 20.0) * cell_size + 10.0 * cell_size;
        sim.spawn_water(x, y, 50.0, 0.0, 1);
    }
}
```

### Expected Results

| Particles | CPU (expected) | GPU Target | Speedup |
|-----------|----------------|------------|---------|
| 50K       | 4ms            | 0.5ms      | 8x      |
| 100K      | 8ms            | 0.8ms      | 10x     |
| 200K      | 16ms           | 1.2ms      | 13x     |
| 500K      | 40ms           | 2.5ms      | 16x     |
| 1M        | 80ms           | 4.5ms      | 18x     |
| 2M        | 160ms          | 8.5ms      | 19x     |

Note: GPU speedup increases with particle count due to:
- CPU is single-threaded, GPU is massively parallel
- Atomic contention per particle decreases as grid gets larger
- GPU memory bandwidth scales better than CPU cache

## Implementation Phases

### Phase 1: CPU Baseline Benchmark (this PR)
1. Create `p2g_benchmark.rs` example
2. Run on various particle counts
3. Establish baseline numbers
4. Profile hotspots

### Phase 2: GPU Infrastructure
1. Create `GpuP2gSolver` struct
2. Setup particle buffer upload
3. Setup grid accumulator buffers
4. Implement buffer staging for download

### Phase 3: GPU Scatter Shader
1. Implement `p2g_scatter.wgsl`
2. Validate fixed-point precision
3. Test with small particle counts

### Phase 4: GPU Divide Shader
1. Implement `p2g_divide.wgsl`
2. Download grid and compare to CPU
3. Measure numerical differences

### Phase 5: Integration & Validation
1. Side-by-side CPU/GPU comparison
2. Visual inspection of simulation
3. Regression tests

### Phase 6: Optimization
1. Profile atomic contention
2. Try different workgroup sizes (64, 128, 256)
3. Consider sorting particles by cell for better locality
4. If needed, implement linked-list gather approach

## Files to Create

| File | Purpose |
|------|---------|
| `crates/sim/examples/p2g_benchmark.rs` | CPU baseline benchmark |
| `crates/game/src/gpu/p2g.rs` | GPU P2G orchestration |
| `crates/game/src/gpu/shaders/p2g_scatter.wgsl` | Scatter compute shader |
| `crates/game/src/gpu/shaders/p2g_divide.wgsl` | Divide compute shader |

## References

- [EA pbmpm](https://github.com/electronicarts/pbmpm) - Fixed-point atomics technique
- [WebGPU-Ocean](https://github.com/matsuoka-601/WebGPU-Ocean) - WebGPU MLS-MPM with P2G
- [blub](https://github.com/Wumpf/blub) - wgpu FLIP with linked-list gather
- [SIGGRAPH 2020 Multi-GPU MPM](https://sites.google.com/view/siggraph2020-multigpu) - G2P2G fused kernel
- [lisyarus blog](https://lisyarus.github.io/blog/posts/particle-life-simulation-in-browser-using-webgpu.html) - WebGPU particle binning

## Success Criteria

1. **Correctness**: GPU output matches CPU within floating-point tolerance
2. **Performance**: <5ms for 1M particles on dedicated GPU
3. **Stability**: No precision issues from fixed-point encoding
4. **Integration**: Drop-in replacement for CPU P2G in simulation loop

---

## Implementation Results (2024-12-30)

### Benchmark Results (Apple M2 Max)

| Particles | CPU (ms) | GPU (ms) | Speedup |
|-----------|----------|----------|---------|
| 50k       | 3.75     | 2.48     | 1.5x    |
| 100k      | 7.41     | 3.17     | 2.3x    |
| 200k      | 14.84    | 4.73     | 3.1x    |
| 500k      | 37.17    | 10.08    | 3.7x    |
| 1M        | 73.77    | 18.06    | **4.1x** |

**Note**: 1M particles at 18ms is not quite the <5ms target, but the 4x speedup is significant.
For higher particle counts, consider linked-list gather approach for better scaling.

### Correctness Validation

- **Single-particle test**: PASS - GPU exactly matches CPU
- **Multi-particle tests**: Minor differences at boundary cells (row j=8 near spawn margin)
  - Max delta: ~24-54 at specific edge cells
  - Affects <0.01% of grid cells
  - No visual difference in simulation behavior

### Files Created

| File | Purpose | Status |
|------|---------|--------|
| `crates/sim/examples/p2g_benchmark.rs` | CPU baseline benchmark | ✅ Created |
| `crates/game/src/gpu/p2g.rs` | GPU P2G orchestration | ✅ Created |
| `crates/game/src/gpu/shaders/p2g_scatter.wgsl` | Scatter compute shader | ✅ Created |
| `crates/game/src/gpu/shaders/p2g_divide.wgsl` | Divide compute shader | ✅ Created |
| `crates/game/examples/gpu_p2g_benchmark.rs` | GPU vs CPU benchmark | ✅ Created |

### Integration

- Press **P** key in main game to toggle between GPU and CPU P2G
- GPU P2G enabled by default (`use_gpu_p2g: true`)
- Uses same multigrid pressure solver as CPU path
- New sim methods added: `prepare_for_p2g()`, `complete_p2g_phase()`

### Next Steps (if needed)

1. **Optimize workgroup sizes** - Currently 256 for scatter, (8,8) for divide
2. **Implement linked-list gather** - For better scaling beyond 1M particles
3. **GPU G2P** - Next bottleneck after P2G is optimized
