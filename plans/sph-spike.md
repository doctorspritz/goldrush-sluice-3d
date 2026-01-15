# SPH Spike: IISPH friction_sluice on wgpu

**Goal:** Validate IISPH for multi-phase fluid (water + gold/sand) targeting 1M particles.

**Why IISPH over basic SPH:**
- Basic SPH uses EOS (stiffness-based pressure) → compressible, unstable at high density ratios
- IISPH iteratively solves for pressure → enforces incompressibility like FLIP's Poisson solve
- 3-5 pressure iterations vs 120 grid iterations = potentially faster

**Success Criteria:**
- [ ] 60 FPS @ 100k particles (spike target)
- [ ] Compression ratio < 1.02 (IISPH should nail this)
- [ ] Gold settles without explosion
- [ ] Clear path to 1M particles

---

## IISPH Algorithm Overview

```
Per Frame:
┌─────────────────────────────────────────────────────────┐
│ 1. PREDICT POSITIONS                                    │
│    v_pred = v + F_external * dt                         │
│    x_pred = x + v_pred * dt                             │
├─────────────────────────────────────────────────────────┤
│ 2. BUILD SPATIAL HASH (on predicted positions)          │
│    - Compute cell indices                               │
│    - Bitonic sort by cell                               │
│    - Build offset table                                 │
├─────────────────────────────────────────────────────────┤
│ 3. COMPUTE DENSITY & DIAGONAL (d_ii)                    │
│    - ρ_pred = Σ_j m_j W(x_pred_i - x_pred_j)           │
│    - d_ii = -dt² Σ_j m_j/ρ_j² |∇W_ij|²                 │
│    (How much does my density change if I push myself?)  │
├─────────────────────────────────────────────────────────┤
│ 4. IISPH PRESSURE LOOP (3-5 iterations)                 │
│    ┌─────────────────────────────────────────────────┐  │
│    │ 4a. Compute sum_d_ij * p_j for all neighbors    │  │
│    │     (Off-diagonal contribution)                 │  │
│    ├─────────────────────────────────────────────────┤  │
│    │ 4b. Compute density error:                      │  │
│    │     ρ_err = ρ_pred - ρ_0                        │  │
│    ├─────────────────────────────────────────────────┤  │
│    │ 4c. Update pressure (Jacobi relaxation):        │  │
│    │     p_new = (1-ω)p + ω(ρ_err - sum_d_ij·p_j)/d_ii│  │
│    └─────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────┤
│ 5. APPLY PRESSURE FORCES                                │
│    F_pressure = -Σ_j m_j(p_i/ρ_i² + p_j/ρ_j²)∇W_ij    │
│    v = v_pred + F_pressure/m * dt                       │
│    x = x_pred + (v - v_pred) * dt                       │
├─────────────────────────────────────────────────────────┤
│ 6. BOUNDARY COLLISION (SDF)                             │
│    - Push out of walls                                  │
│    - Apply friction                                     │
└─────────────────────────────────────────────────────────┘
```

---

## GPU Kernel Pipeline (wgpu Compute Shaders)

### Dispatch Order Per Frame

```
Pass 1: predict_and_hash      [N particles]
Pass 2: bitonic_sort          [log²(N) dispatches]
Pass 3: build_offsets         [num_cells]
Pass 4: compute_density_dii   [N particles]

// IISPH Loop (3-5x)
Pass 5a: compute_sum_dij_pj   [N particles]
Pass 5b: update_pressure      [N particles]
// end loop

Pass 6: apply_pressure_force  [N particles]
Pass 7: boundary_collision    [N particles]
```

### Buffer Layout (SoA)

```rust
// Per-particle buffers (N × size)
positions:      Buffer<vec3<f32>>     // 12 bytes
velocities:     Buffer<vec3<f32>>     // 12 bytes
positions_pred: Buffer<vec3<f32>>     // 12 bytes (predicted)
densities:      Buffer<f32>           // 4 bytes
pressures:      Buffer<f32>           // 4 bytes
pressures_old:  Buffer<f32>           // 4 bytes (for Jacobi)
d_ii:           Buffer<f32>           // 4 bytes (diagonal)
sum_dij_pj:     Buffer<f32>           // 4 bytes (off-diagonal sum)
phases:         Buffer<u32>           // 4 bytes (water/sand/gold)
masses:         Buffer<f32>           // 4 bytes (phase-dependent)

// Spatial hash buffers
cell_indices:   Buffer<u32>           // N × 4 bytes
particle_order: Buffer<u32>           // N × 4 bytes (sorted order)
cell_offsets:   Buffer<u32>           // num_cells × 4 bytes

// Total per particle: ~64 bytes
// 1M particles = 64 MB (fits in GPU memory easily)
```

---

## Implementation Plan

### Phase 1: Spatial Hash Infrastructure

#### 1.1 `sph_predict_hash.wgsl`
```wgsl
@compute @workgroup_size(256)
fn predict_and_hash(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    // Predict position
    var v = velocities[i];
    v += vec3(0.0, params.gravity, 0.0) * params.dt;
    let p = positions[i] + v * params.dt;

    positions_pred[i] = p;
    velocities[i] = v;  // Store predicted velocity

    // Compute cell index
    let cell = vec3<i32>(floor(p / params.cell_size));
    let hash = cell_hash(cell);
    cell_indices[i] = hash;
    particle_order[i] = i;  // Will be sorted
}
```

#### 1.2 Bitonic Sort (reuse existing `particle_sort_*.wgsl`)
- Sort `particle_order` by `cell_indices`
- Already implemented in codebase

#### 1.3 `sph_build_offsets.wgsl`
```wgsl
@compute @workgroup_size(256)
fn build_offsets(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    let my_cell = cell_indices[particle_order[i]];

    if (i == 0u || cell_indices[particle_order[i - 1u]] != my_cell) {
        cell_offsets[my_cell] = i;
    }
}
```

### Phase 2: IISPH Core Kernels

#### 2.1 `sph_density_dii.wgsl`
```wgsl
// SPH Kernels
fn poly6(r2: f32, h2: f32) -> f32 {
    if (r2 >= h2) { return 0.0; }
    let diff = h2 - r2;
    return params.poly6_coef * diff * diff * diff;
}

fn spiky_grad(r: vec3<f32>, dist: f32) -> vec3<f32> {
    if (dist >= params.h || dist < 0.0001) { return vec3(0.0); }
    let diff = params.h - dist;
    return params.spiky_grad_coef * diff * diff * (r / dist);
}

@compute @workgroup_size(256)
fn compute_density_dii(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    let pi = positions_pred[particle_order[i]];
    var rho = 0.0;
    var dii = 0.0;
    let h2 = params.h * params.h;

    // Iterate 3x3x3 neighbor cells
    let cell = vec3<i32>(floor(pi / params.cell_size));
    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let neighbor_cell = cell + vec3(dx, dy, dz);
                let hash = cell_hash(neighbor_cell);
                let start = cell_offsets[hash];
                let end = cell_offsets[hash + 1u];

                for (var k = start; k < end; k++) {
                    let j = particle_order[k];
                    let pj = positions_pred[j];
                    let r = pi - pj;
                    let r2 = dot(r, r);

                    if (r2 < h2) {
                        let mj = masses[j];
                        rho += mj * poly6(r2, h2);

                        // d_ii contribution (diagonal)
                        let dist = sqrt(r2);
                        let grad = spiky_grad(r, dist);
                        dii -= params.dt2 * mj * dot(grad, grad);
                    }
                }
            }
        }
    }

    densities[particle_order[i]] = rho;
    d_ii_buffer[particle_order[i]] = dii / (rho * rho);
}
```

#### 2.2 `sph_sum_dij.wgsl` (IISPH iteration part A)
```wgsl
@compute @workgroup_size(256)
fn compute_sum_dij_pj(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    let idx = particle_order[i];
    let pi = positions_pred[idx];
    let rhoi = densities[idx];
    var sum = 0.0;

    // Neighbor loop (same structure as density)
    // sum += d_ij * p_j where d_ij = dt² * m_j/ρ_j² * ∇W_ij · ∇W_ji
    for (/* neighbor cells */) {
        for (/* particles in cell */) {
            let j = particle_order[k];
            if (j == idx) { continue; }

            let pj_pos = positions_pred[j];
            let r = pi - pj_pos;
            let dist = length(r);

            if (dist < params.h) {
                let mj = masses[j];
                let rhoj = densities[j];
                let pj = pressures[j];
                let grad = spiky_grad(r, dist);

                // d_ij = -dt² * m_j/ρ_j² * |∇W|²
                let d_ij = -params.dt2 * mj / (rhoj * rhoj) * dot(grad, grad);
                sum += d_ij * pj;
            }
        }
    }

    sum_dij_pj[idx] = sum;
}
```

#### 2.3 `sph_update_pressure.wgsl` (IISPH iteration part B)
```wgsl
@compute @workgroup_size(256)
fn update_pressure(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    let idx = particle_order[i];
    let rho = densities[idx];
    let rho_err = rho - params.rest_density;
    let dii = d_ii_buffer[idx];
    let sum = sum_dij_pj[idx];
    let p_old = pressures[idx];

    // Jacobi relaxation
    var p_new: f32;
    if (abs(dii) > 0.0001) {
        p_new = (1.0 - params.omega) * p_old + params.omega * (rho_err - sum) / dii;
    } else {
        p_new = 0.0;
    }

    // Clamp negative pressure (prevents tension instability)
    p_new = max(p_new, 0.0);

    pressures[idx] = p_new;
}
```

#### 2.4 `sph_apply_pressure.wgsl`
```wgsl
@compute @workgroup_size(256)
fn apply_pressure_force(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    let idx = particle_order[i];
    let pi = positions_pred[idx];
    let rhoi = densities[idx];
    let pressi = pressures[idx];
    let mi = masses[idx];
    var f_pressure = vec3(0.0);

    // Neighbor loop
    for (/* neighbor cells */) {
        for (/* particles in cell */) {
            let j = particle_order[k];
            if (j == idx) { continue; }

            let pj = positions_pred[j];
            let r = pi - pj;
            let dist = length(r);

            if (dist < params.h && dist > 0.0001) {
                let mj = masses[j];
                let rhoj = densities[j];
                let pressj = pressures[j];
                let grad = spiky_grad(r, dist);

                // Symmetric pressure force
                f_pressure -= mj * (pressi / (rhoi * rhoi) + pressj / (rhoj * rhoj)) * grad;
            }
        }
    }

    // Update velocity and position
    let v_pred = velocities[idx];
    let v_new = v_pred + f_pressure / mi * params.dt;
    let x_new = positions[idx] + v_new * params.dt;

    velocities[idx] = v_new;
    positions[idx] = x_new;
}
```

### Phase 3: Rust Orchestrator

#### 3.1 `crates/game/src/gpu/sph_3d.rs`

```rust
pub struct GpuSph3D {
    // Pipelines
    predict_hash_pipeline: wgpu::ComputePipeline,
    sort_pipelines: Vec<wgpu::ComputePipeline>,  // Bitonic sort stages
    build_offsets_pipeline: wgpu::ComputePipeline,
    density_dii_pipeline: wgpu::ComputePipeline,
    sum_dij_pipeline: wgpu::ComputePipeline,
    update_pressure_pipeline: wgpu::ComputePipeline,
    apply_pressure_pipeline: wgpu::ComputePipeline,
    boundary_pipeline: wgpu::ComputePipeline,

    // Particle buffers
    positions: wgpu::Buffer,
    velocities: wgpu::Buffer,
    positions_pred: wgpu::Buffer,
    densities: wgpu::Buffer,
    pressures: wgpu::Buffer,
    d_ii: wgpu::Buffer,
    sum_dij_pj: wgpu::Buffer,
    phases: wgpu::Buffer,
    masses: wgpu::Buffer,

    // Spatial hash buffers
    cell_indices: wgpu::Buffer,
    particle_order: wgpu::Buffer,
    cell_offsets: wgpu::Buffer,

    // Parameters
    num_particles: u32,
    max_particles: u32,
    h: f32,  // Kernel radius
    cell_size: f32,
    rest_density: f32,
    pressure_iterations: u32,
}

impl GpuSph3D {
    pub fn step(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let workgroups = (self.num_particles + 255) / 256;

        // 1. Predict + Hash
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.predict_hash_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // 2. Bitonic Sort
        self.bitonic_sort(encoder);

        // 3. Build Offsets
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.build_offsets_pipeline);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // 4. Density + d_ii
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.density_dii_pipeline);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // 5. IISPH Pressure Loop
        for _ in 0..self.pressure_iterations {
            // 5a. Sum d_ij * p_j
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.sum_dij_pipeline);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            // 5b. Update pressure
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.update_pressure_pipeline);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
        }

        // 6. Apply pressure + integrate
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.apply_pressure_pipeline);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // 7. Boundary collision
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.boundary_pipeline);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }
}
```

### Phase 4: Example Integration

#### 4.1 `sph_friction_sluice.rs`
- Copy `friction_sluice.rs`
- Replace `GpuFlip3D` → `GpuSph3D`
- Remove APIC buffers (C matrices)
- Add phase/mass initialization for water/gold/sand
- Keep: DEM, SDF, visualization

---

## Key Parameters

```rust
// IISPH constants
const H: f32 = 0.02;                    // Kernel radius (2× particle spacing)
const CELL_SIZE: f32 = H;               // Hash cell = kernel radius
const REST_DENSITY: f32 = 1000.0;       // kg/m³ (water)
const PRESSURE_ITERATIONS: u32 = 4;     // IISPH iterations
const OMEGA: f32 = 0.5;                 // Relaxation factor

// SPH kernel coefficients (precomputed)
const POLY6_COEF: f32 = 315.0 / (64.0 * PI * H^9);
const SPIKY_GRAD_COEF: f32 = -45.0 / (PI * H^6);

// Phase masses
const WATER_MASS: f32 = 1.0;
const SAND_MASS: f32 = 2.6;
const GOLD_MASS: f32 = 19.3;
```

---

## File Checklist

### New Files
- [ ] `crates/game/src/gpu/sph_3d.rs`
- [ ] `crates/game/src/gpu/shaders/sph_predict_hash.wgsl`
- [ ] `crates/game/src/gpu/shaders/sph_build_offsets.wgsl`
- [ ] `crates/game/src/gpu/shaders/sph_density_dii.wgsl`
- [ ] `crates/game/src/gpu/shaders/sph_sum_dij.wgsl`
- [ ] `crates/game/src/gpu/shaders/sph_update_pressure.wgsl`
- [ ] `crates/game/src/gpu/shaders/sph_apply_pressure.wgsl`
- [ ] `crates/game/examples/sph_friction_sluice.rs`

### Modified Files
- [ ] `crates/game/src/gpu/mod.rs` - Export sph_3d

### Reused (no changes needed)
- `particle_sort_*.wgsl` - Bitonic sort
- `sdf_collision_3d.wgsl` - Boundary collision
- `ScreenSpaceFluidRenderer` - Visualization

---

## Validation Metrics

### 1. Compression Ratio
```rust
// GPU reduction: avg(density) / rest_density
// Target: 1.00 - 1.02 (IISPH should nail this)
```

### 2. Pressure Convergence
```rust
// Track max |ρ_err| per IISPH iteration
// Should decrease each iteration
// < 1% error after 4 iterations = good
```

### 3. Gold Settling Time
```rust
// Spawn 1000 gold particles at top
// Measure frames until 95% reach bottom 10% of domain
// Should be ~2-3× faster than sand
```

### 4. Performance
```rust
// Target: 60 FPS @ 100k particles (spike)
// Path to: 60 FPS @ 1M particles (production)
// Kernel timing breakdown for optimization
```

---

## Optimizations

### Sleep Mechanic (Kinematic Sleep)
Particles stuck in riffles stop moving → skip them in compute.

```wgsl
// In predict_and_hash kernel
const SLEEP_VELOCITY_THRESHOLD: f32 = 0.001;
const SLEEP_FRAMES_REQUIRED: u32 = 10u;

if (length(velocities[i]) < SLEEP_VELOCITY_THRESHOLD) {
    sleep_counter[i] += 1u;
} else {
    sleep_counter[i] = 0u;
}

let is_sleeping = sleep_counter[i] >= SLEEP_FRAMES_REQUIRED;
sleep_flags[i] = select(0u, 1u, is_sleeping);
```

**Implementation:**
- Add `sleep_counter: Buffer<u32>` and `sleep_flags: Buffer<u32>`
- In density/pressure kernels: `if (sleep_flags[i] == 1u) { return; }`
- Wake particles if neighbor has high velocity or pressure
- **Expected gain:** 30-40% speedup at steady state

### Screen-Space Fluid Rendering
Already implemented: `ScreenSpaceFluidRenderer` in `crates/game/src/gpu/fluid_renderer.rs`
- Renders particle depth to texture
- Bilateral filter smoothing
- Normal reconstruction from depth
- Fresnel + refraction for water surface

No changes needed - just wire it up to SPH particle positions.

---

## Risk Mitigation

**Risk 1: IISPH doesn't converge for gold**
- Try: More iterations (4 → 8)
- Try: Lower omega (0.5 → 0.3)
- Try: Separate pressure solve for phases
- Fallback: Density-weighted mass averaging

**Risk 2: Neighbor search too slow**
- Optimize: Reduce H (fewer neighbors)
- Optimize: Use morton codes for better cache locality
- Optimize: Shared memory for neighbor data

**Risk 3: Numerical instability**
- Clamp negative pressures
- Limit velocity changes per step
- Add artificial viscosity for stability
