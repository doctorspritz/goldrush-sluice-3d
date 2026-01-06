# Plan: Drucker-Prager Yield for Sediment (Granular Clogging Physics)

## Problem Statement

Current sediment model uses settling velocity + drag. This works for suspended particles but fails to capture:

1. **Angle of repose** - piles form without realistic slopes
2. **Clogging mechanics** - packed sediment should resist flow, then yield in chunks
3. **Underwater granular behavior** - buoyancy affects effective stress and yield

The sediment either flows (suspended) or is static (bed). There's no "jammed but can yield" state.

## Goal

Add Drucker-Prager yield criterion to sediment so that:
- Packed sediment resists deformation until stress exceeds yield
- Clogged sluices break in chunks when flow increases (not just surface entrainment)
- Piles form at realistic underwater angle of repose (~35°)
- Vorticity preservation for water remains intact (don't touch water path)

## Physics Background

### Drucker-Prager Yield Criterion

```
yield_stress = cohesion + pressure × tan(friction_angle)

if shear_stress > yield_stress:
    material YIELDS (flows)
else:
    material is JAMMED (moves with neighbors)
```

For sand: friction_angle ≈ 30-35°, cohesion ≈ 0

### Underwater Effective Stress

Buoyancy reduces effective pressure:
```
effective_pressure = total_pressure × (1 - ρ_water / ρ_sediment)
```

For sand (ρ=2650) in water (ρ=1000): buoyancy_factor ≈ 0.62

This means underwater sand yields MORE easily → steeper angle of repose.

## Architecture Decision

**Option A**: Compute stress per-particle from velocity gradient (true MPM)
- Pro: Physically correct
- Con: Need to track deformation gradient F, more complex

**Option B**: Estimate local pressure/shear from neighbor density + grid velocity
- Pro: Fits current pipeline, no F matrix needed
- Con: Approximate

**Chosen: Option B** - pragmatic approximation that fits current architecture.

## Implementation Plan

### Phase 1: Sediment Pressure Buffer

**Goal**: Compute local pressure on each sediment particle from weight of sediment above.

#### New shader: `sediment_pressure_3d.wgsl`

Per-cell pass that computes pressure from sediment column weight:

```wgsl
// For each cell, accumulate sediment mass above
// pressure[i,j,k] = Σ (sediment_mass[i,j',k] × g) for j' > j

@compute @workgroup_size(8, 1, 8)
fn compute_sediment_pressure(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let k = id.z;

    // Scan from top to bottom, accumulating pressure
    var accumulated_mass: f32 = 0.0;

    for (var j: i32 = i32(params.height) - 1; j >= 0; j--) {
        let cell_idx = cell_index(i, u32(j), k);
        let sediment_count = f32(sediment_counts[cell_idx]);
        let cell_mass = sediment_count * params.particle_mass;

        // Pressure from sediment above (buoyancy-corrected)
        let effective_weight = cell_mass * params.gravity * params.buoyancy_factor;
        accumulated_mass += effective_weight;

        sediment_pressure[cell_idx] = accumulated_mass / params.cell_area;
    }
}
```

#### New buffer

```rust
// In GpuFlip3D
sediment_pressure_buffer: wgpu::Buffer,  // f32 per cell
```

#### Pipeline integration

Insert after `sediment_fraction` pass, before G2P:

```
P2G scatter
    ↓
sediment_fraction (existing)
    ↓
sediment_pressure (NEW)  ← Phase 1
    ↓
boundary conditions
    ↓
pressure solve
    ↓
G2P (modified in Phase 2)
```

**Files:**
- `crates/game/src/gpu/shaders/sediment_pressure_3d.wgsl` (new)
- `crates/game/src/gpu/flip_3d.rs` (add buffer, pipeline, dispatch)

---

### Phase 2: Shear Rate Estimation

**Goal**: Estimate shear rate at each sediment particle from grid velocity gradient.

Rather than a separate pass, compute inline in G2P using the velocity samples we already gather.

#### Modify `g2p_3d.wgsl`

After sampling grid velocities for the particle, compute velocity gradient:

```wgsl
// Already have: new_velocity sampled from grid
// Add: estimate velocity gradient from stencil

// Sample velocities at offset positions to estimate gradient
let vel_px = sample_grid_velocity(pos + vec3(h, 0, 0));
let vel_mx = sample_grid_velocity(pos - vec3(h, 0, 0));
let vel_py = sample_grid_velocity(pos + vec3(0, h, 0));
let vel_my = sample_grid_velocity(pos - vec3(0, h, 0));
let vel_pz = sample_grid_velocity(pos + vec3(0, 0, h));
let vel_mz = sample_grid_velocity(pos - vec3(0, 0, h));

let dudx = (vel_px.x - vel_mx.x) / (2.0 * h);
let dudy = (vel_py.x - vel_my.x) / (2.0 * h);
let dudz = (vel_pz.x - vel_mz.x) / (2.0 * h);
let dvdx = (vel_px.y - vel_mx.y) / (2.0 * h);
let dvdy = (vel_py.y - vel_my.y) / (2.0 * h);
let dvdz = (vel_pz.y - vel_mz.y) / (2.0 * h);
let dwdx = (vel_px.z - vel_mx.z) / (2.0 * h);
let dwdy = (vel_py.z - vel_my.z) / (2.0 * h);
let dwdz = (vel_pz.z - vel_mz.z) / (2.0 * h);

// Strain rate tensor (symmetric part of velocity gradient)
// D = 0.5 * (∇v + ∇v^T)
let D_xx = dudx;
let D_yy = dvdy;
let D_zz = dwdz;
let D_xy = 0.5 * (dudy + dvdx);
let D_xz = 0.5 * (dudz + dwdx);
let D_yz = 0.5 * (dvdz + dwdy);

// Shear rate magnitude (second invariant of strain rate)
// |D| = sqrt(2 * D:D) = sqrt(2 * (D_xx² + D_yy² + D_zz² + 2*D_xy² + 2*D_xz² + 2*D_yz²))
let shear_rate = sqrt(2.0 * (
    D_xx*D_xx + D_yy*D_yy + D_zz*D_zz +
    2.0*(D_xy*D_xy + D_xz*D_xz + D_yz*D_yz)
));
```

**Note**: This reuses the grid velocity sampling we already do. The extra samples (±h offsets) add cost but avoid a separate pass.

**Alternative**: Use the C matrix (affine velocity field) to estimate gradient. The C matrix from APIC already encodes local velocity gradient! For sediment we zero it currently, but we could compute it for gradient estimation only.

**Files:**
- `crates/game/src/gpu/shaders/g2p_3d.wgsl` (modify sediment branch)

---

### Phase 3: Drucker-Prager Yield in G2P

**Goal**: Apply yield criterion to determine if sediment flows or jams.

#### Modify sediment branch in `g2p_3d.wgsl`

```wgsl
struct DruckerPragerParams {
    friction_angle: f32,      // radians, ~0.55 for 32°
    cohesion: f32,            // Pa, ~0 for sand
    buoyancy_factor: f32,     // 1 - ρ_w/ρ_s ≈ 0.62
    viscosity: f32,           // Pa·s, for yielded flow
    jammed_drag: f32,         // high drag when not yielding
    min_pressure: f32,        // prevent division issues
    _pad: vec2<f32>,
}

@group(0) @binding(15) var<uniform> dp_params: DruckerPragerParams;
@group(0) @binding(16) var<storage, read> sediment_pressure: array<f32>;

// In sediment branch (density > 1.0):
if (density > 1.0) {
    // Sample local sediment pressure
    let cell_i = clamp(i32(pos.x / cell_size), 0, width - 1);
    let cell_j = clamp(i32(pos.y / cell_size), 0, height - 1);
    let cell_k = clamp(i32(pos.z / cell_size), 0, depth - 1);
    let pressure = max(sediment_pressure[cell_index(cell_i, cell_j, cell_k)], dp_params.min_pressure);

    // Compute shear rate (from Phase 2)
    let shear_rate = compute_shear_rate(pos, ...);

    // Drucker-Prager yield criterion
    let yield_stress = dp_params.cohesion + pressure * tan(dp_params.friction_angle);
    let shear_stress = dp_params.viscosity * shear_rate;  // τ = η * γ̇

    var final_velocity: vec3<f32>;

    if (shear_stress > yield_stress) {
        // YIELDING - granular flow (Bingham-like viscoplastic)
        // Effective viscosity increases as we approach yield
        let excess_stress = shear_stress - yield_stress;
        let flow_rate = excess_stress / dp_params.viscosity;

        // Blend toward grid velocity based on how much we exceed yield
        let yield_ratio = clamp(excess_stress / (shear_stress + 0.001), 0.0, 1.0);
        let drag = yield_ratio * sediment_params.drag_rate * params.dt;
        final_velocity = mix(old_particle_vel, new_velocity, drag);

        // Still apply settling (reduced when yielding/flowing)
        final_velocity.y -= sediment_params.settling_velocity * params.dt * (1.0 - yield_ratio * 0.5);

    } else {
        // JAMMED - move with the pack (high drag, near-rigid)
        // Strong coupling to grid velocity
        let jam_drag = 1.0 - exp(-dp_params.jammed_drag * params.dt);
        final_velocity = mix(old_particle_vel, new_velocity, jam_drag);

        // Settling still applies but reduced in jammed state
        final_velocity.y -= sediment_params.settling_velocity * params.dt * 0.3;
    }

    // Vorticity lift (existing)
    let vort = vorticity_mag[cell_index(cell_i, cell_j, cell_k)];
    let vort_excess = max(vort - sediment_params.vorticity_threshold, 0.0);
    let lift_factor = clamp(sediment_params.vorticity_lift * vort_excess, 0.0, 0.9);
    final_velocity.y += sediment_params.settling_velocity * params.dt * lift_factor;

    // Clamp
    let speed = length(final_velocity);
    if (speed > params.max_velocity) {
        final_velocity *= params.max_velocity / speed;
    }

    velocities[id.x] = final_velocity;
    // Keep C matrix zeroed for sediment (no APIC affine)
    c_col0[id.x] = vec3<f32>(0.0);
    c_col1[id.x] = vec3<f32>(0.0);
    c_col2[id.x] = vec3<f32>(0.0);
    return;
}
```

**Files:**
- `crates/game/src/gpu/shaders/g2p_3d.wgsl`
- `crates/game/src/gpu/g2p_3d.rs` (add DP params buffer + binding)

---

### Phase 4: Parameter Tuning & Diagnostics

**Goal**: Expose parameters for real-time tuning, add debug visualization.

#### New parameters in `industrial_sluice.rs`

```rust
struct DruckerPragerConfig {
    friction_angle_deg: f32,  // UI: 25-40°, default 32°
    cohesion: f32,            // UI: 0-100 Pa, default 0
    viscosity: f32,           // UI: 0.1-10 Pa·s, default 1.0
    jammed_drag: f32,         // UI: 10-100, default 50
}
```

#### Debug visualization

Add yield state to particle color output:
- **Blue**: water (unchanged)
- **Yellow**: sediment yielding (flowing)
- **Orange**: sediment jammed (packed)
- **Red**: sediment at yield boundary

This requires passing yield state out of G2P or computing in render shader.

**Files:**
- `crates/game/examples/industrial_sluice.rs`
- `crates/game/src/gpu/shaders/particle_render.wgsl` (optional debug color)

---

### Phase 5: Angle of Repose Validation

**Goal**: Verify piles form at correct angle.

#### Test scenario

1. Spawn sediment column above flat bed
2. Let it collapse
3. Measure final pile angle
4. Compare to expected: ~35° underwater (32° × 1/buoyancy_factor correction)

#### Automated test

```rust
#[test]
fn test_angle_of_repose() {
    // Setup: column of sediment
    // Run: 1000 steps
    // Measure: slope of final pile
    // Assert: within 5° of expected
}
```

**Files:**
- `crates/game/tests/angle_of_repose.rs` (new)

---

### Phase 6: Clogging Behavior Validation

**Goal**: Verify clog forms and breaks realistically.

#### Test scenario

1. Reduce flow to let sediment accumulate behind riffle
2. Observe clog formation (should stabilize at angle of repose)
3. Increase flow
4. Observe clog breaking (should see chunks yield, not just surface erosion)

#### Metrics

- Time to clog formation
- Clog height at equilibrium
- Flow rate required to break clog
- Whether breakage is gradual (entrainment) or sudden (yield failure)

**Files:**
- `crates/game/examples/industrial_sluice.rs` (add clog test scenario)

---

## Pipeline Summary

```
P2G scatter (water momentum only, sediment counted)
    ↓
sediment_fraction (existing)
    ↓
sediment_pressure (NEW - Phase 1)
    ↓
boundary conditions
    ↓
pressure solve (water only)
    ↓
porosity_drag (existing - slows water in packed sediment)
    ↓
G2P with Drucker-Prager (NEW - Phase 3)
    ├─ Water: FLIP/APIC (unchanged, vorticity preserved)
    └─ Sediment: yield check → jammed or flowing
    ↓
advect + collision
    ↓
bed update (existing)
```

## New Buffers

| Buffer | Type | Size | Purpose |
|--------|------|------|---------|
| `sediment_pressure` | f32 | width×height×depth | Pressure from sediment column |
| `dp_params` | uniform | ~32 bytes | Drucker-Prager constants |

## New Shaders

| Shader | Workgroups | Purpose |
|--------|------------|---------|
| `sediment_pressure_3d.wgsl` | (width, 1, depth) | Column pressure scan |

## Parameters

| Parameter | Default | Range | Physical Meaning |
|-----------|---------|-------|------------------|
| `friction_angle` | 32° | 25-40° | Internal friction (sand ~30-35°) |
| `cohesion` | 0 Pa | 0-1000 | Binding force (0 for dry sand) |
| `buoyancy_factor` | 0.62 | 0.5-0.7 | 1 - ρ_water/ρ_sediment |
| `viscosity` | 1.0 Pa·s | 0.1-10 | Resistance when yielding |
| `jammed_drag` | 50 | 10-100 | Coupling when not yielding |

## Performance Impact

- **sediment_pressure pass**: O(width × depth × height) - one column scan per XZ
- **G2P shear calculation**: ~6 extra velocity samples per sediment particle
- **Estimated overhead**: 5-15% on sediment-heavy scenes

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Shear estimation noisy near boundaries | Clamp minimum pressure, smooth shear |
| Jammed particles oscillate | High jammed_drag, velocity damping |
| Performance regression | Profile, consider computing shear in separate pass if needed |
| Yield threshold too sharp | Add smoothing region around yield surface |

## Acceptance Criteria

1. ✅ Pile of sediment forms at ~35° underwater slope (±5°)
2. ✅ Clog behind riffle stabilizes (doesn't grow forever)
3. ✅ Increased flow causes chunk failure, not just surface erosion
4. ✅ Water vorticity unchanged (run existing vorticity tests)
5. ✅ Performance within 20% of current at 200k particles

## Follow-ups (Deferred)

- True MPM with deformation gradient F (more accurate but complex)
- Per-material friction angles (sand vs gravel vs mud)
- Cohesion for wet mud (non-zero cohesion)
- GPU-based angle of repose measurement for adaptive parameters
