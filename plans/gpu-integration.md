# GPU Integration Plan

## Current State

### What's Working
- **wgpu v23 infrastructure**: Device, queue, surface management
- **Particle rendering**: Instanced quads with per-particle color/size
- **Terrain rendering**: Solid cells rendered as colored blocks

### What's Working (GPU Compute)
- **GPU Pressure Solver** (`crates/game/src/gpu/pressure.rs`) - **VALIDATED & ENABLED**
  - Checkerboard SOR (Red/Black passes) with compute shaders
  - Upload/download with staging buffers
  - Warm start support
  - **Status**: Enabled in main.rs - produces identical divergence reduction to CPU
  - **Findings**: Pressure fields differ slightly due to SOR vs Gauss-Seidel, but visual quality is equivalent

### Performance Benchmarks (GPU Pressure Enabled)

| Particles | Total (ms) | Pressure GPU | P2G (CPU) | FPS |
|-----------|------------|--------------|-----------|-----|
| 150K      | 20         | 5ms          | 8ms       | ~50 |
| 300K      | 40         | 5ms          | 20ms      | ~25 |
| 600K      | 78         | 5ms          | 44ms      | ~13 |
| 1M        | 138        | 5ms          | 81ms      | ~7  |
| 1.5M      | 202        | 5ms          | 121ms     | ~5  |
| 2M        | 243        | 5ms          | 146ms     | ~4  |

**Bottleneck Analysis:**
- GPU pressure: Constant ~5ms (excellent scaling)
- P2G: 60-80% of frame time, linear scaling (NEXT TARGET)
- G2P: ~10% of frame time, linear scaling
- SDF: Constant ~3ms (grid-based)

---

## Phase 1: Fix GPU Pressure Solver - COMPLETED

### Resolution
- GPU pressure solver **validated and enabled**
- Produces identical divergence reduction to CPU multigrid
- Pressure field shapes differ slightly but visual quality is equivalent
- Uses 30 iterations of checkerboard SOR (omega=1.9)

---

## Phase 2: GPU P2G Transfer

### Goal
Move particle-to-grid transfer to GPU. This is the largest single-threaded bottleneck.

### Design
```wgsl
// p2g.wgsl
struct Particle {
    position: vec2<f32>,
    velocity: vec2<f32>,
    c_matrix: mat2x2<f32>,  // APIC affine matrix
    material: u32,
    _padding: u32,
}

@compute @workgroup_size(256)
fn p2g(@builtin(global_invocation_id) id: vec3<u32>) {
    let p_idx = id.x;
    // Each particle scatters to 3x3 grid neighborhood
    // Use atomicAdd for accumulation
}
```

### Challenges
1. **Atomic operations**: Multiple particles write to same grid cell
2. **Memory layout**: SoA vs AoS for coalesced access
3. **Weight computation**: Quadratic B-splines per particle

### Implementation Steps
1. Create `p2g.wgsl` compute shader
2. Add particle buffer (storage, read)
3. Add grid buffers (u_sum, u_weight, v_sum, v_weight - storage, read_write)
4. Use `atomicAdd` for accumulation
5. Add division pass to compute final velocities

---

## Phase 3: GPU G2P Transfer

### Goal
Move grid-to-particle transfer to GPU. Pairs naturally with P2G.

### Design
```wgsl
// g2p.wgsl
@compute @workgroup_size(256)
fn g2p(@builtin(global_invocation_id) id: vec3<u32>) {
    let p_idx = id.x;
    // Each particle gathers from 3x3 grid neighborhood
    // No atomics needed - each particle writes only to itself
}
```

### Benefits
- Embarrassingly parallel (no atomics)
- Natural pairing with P2G
- FLIP blend: `v_new = v_pic + flip_ratio * (v_grid_new - v_grid_old)`

### Implementation Steps
1. Create `g2p.wgsl` compute shader
2. Store `v_grid_old` on GPU between P2G and G2P
3. Output updated particle velocities and C matrices
4. Handle different PIC/FLIP ratios per material

---

## Phase 4: GPU Advection

### Goal
Move particle position update to GPU.

### Design
```wgsl
// advect.wgsl
@compute @workgroup_size(256)
fn advect(@builtin(global_invocation_id) id: vec3<u32>) {
    let p_idx = id.x;
    var p = particles[p_idx];

    // Simple Euler: p.position += p.velocity * dt
    // SDF collision: if sdf < 0, project onto surface
    p.position += p.velocity * dt;

    // SDF collision
    let sdf = sample_sdf(p.position);
    if sdf < 0.0 {
        let grad = sdf_gradient(p.position);
        p.position -= grad * sdf;
    }

    particles[p_idx] = p;
}
```

### Dependencies
- Requires SDF on GPU (upload each frame or compute on GPU)
- Simple enough to implement early

---

## Phase 5: Full GPU Pipeline

### Target Architecture
```
Frame N:
┌─────────────────────────────────────────────────┐
│                    GPU                          │
├─────────────────────────────────────────────────┤
│ 1. P2G Transfer (compute)                       │
│ 2. Gravity + Forces (compute)                   │
│ 3. Pressure Solve (compute, multi-pass)         │
│ 4. G2P Transfer (compute)                       │
│ 5. Advection (compute)                          │
│ 6. Render (graphics)                            │
└─────────────────────────────────────────────────┘
        ↑                              ↓
    Upload once              Download if needed
   (parameters)               (diagnostics)
```

### Data That Stays on GPU
- Particle positions, velocities, C matrices
- Grid velocities (u, v, u_old, v_old)
- Pressure, divergence, cell_type
- SDF

### Data Uploaded Per Frame
- Simulation parameters (dt, gravity, etc.)
- New spawned particles (append to buffer)

### Data Downloaded (Optional)
- Particle positions for debug visualization
- Diagnostics (divergence, velocity max)

---

## Implementation Priority

### Short Term (1-2 sessions)
1. **Debug GPU pressure solver** - Low risk, high value
2. **Add GPU SDF upload** - Required for advection

### Medium Term (3-5 sessions)
3. **GPU P2G transfer** - Biggest CPU bottleneck
4. **GPU G2P transfer** - Natural pair with P2G

### Long Term (5+ sessions)
5. **GPU advection** - Enables full GPU pipeline
6. **GPU spatial hash** - For DEM/neighbor queries
7. **GPU DEM settling** - For granular physics

---

## Technical Notes

### Buffer Layout (Current)
```rust
// Particle instance for rendering (32 bytes)
struct ParticleInstance {
    position: [f32; 2],
    color: [f32; 4],
    size: f32,
    _padding: f32,
}
```

### Recommended Compute Layout
```rust
// Particle for simulation (64 bytes, aligned)
#[repr(C)]
struct GpuParticle {
    position: [f32; 2],     // 8
    velocity: [f32; 2],     // 8
    c_matrix: [f32; 4],     // 16 (Mat2 as 4 floats)
    v_old: [f32; 2],        // 8 (for FLIP blend)
    material: u32,          // 4
    state: u32,             // 4
    diameter: f32,          // 4
    jam_time: f32,          // 4
    _padding: [f32; 2],     // 8 (align to 64)
}
```

### Workgroup Size Recommendations
- **Per-particle ops** (P2G, G2P, advect): 256 threads
- **Per-cell ops** (pressure): 8x8 = 64 threads
- **Reductions** (diagnostics): 256 threads with shared memory

---

## Files to Create

| File | Purpose |
|------|---------|
| `crates/game/src/gpu/p2g.rs` | P2G compute orchestration |
| `crates/game/src/gpu/g2p.rs` | G2P compute orchestration |
| `crates/game/src/gpu/advect.rs` | Advection compute |
| `crates/game/src/gpu/shaders/p2g.wgsl` | P2G compute shader |
| `crates/game/src/gpu/shaders/g2p.wgsl` | G2P compute shader |
| `crates/game/src/gpu/shaders/advect.wgsl` | Advection shader |
| `crates/game/src/gpu/simulation.rs` | Full GPU sim coordinator |

---

## Success Metrics

1. **GPU pressure solver matches CPU** - Divergence AND visual quality
2. **60 FPS with 100k+ particles** - Currently ~30-40 FPS at 50k
3. **<5ms total frame time** - Currently ~12-16ms at 50k particles
4. **No CPU-GPU sync stalls** - Use async readback for diagnostics
