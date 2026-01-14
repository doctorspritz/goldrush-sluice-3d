# IISPH / DEM Spike Implementation Plan

**Goal:** Water fills a bucket test - validate IISPH incompressibility with measurable diagnostics.

**Status:** Previous attempts failed due to unclear bugs. This plan takes a methodical, gated approach with explicit debug outputs.

---

## Previous Attempts and Failures

### Attempt 1: Simple Position-Based Repulsion (Failed)
- Added spring forces `f_pressure += repel_dir * overlap * 200.0`
- **Result:** Particles clumped or exploded depending on constants
- **Lesson:** Position-based methods fight against IISPH pressure solve

### Attempt 2: Inline Boundary Collision (Partial Success)
- Added boundary handling directly in `bf_apply_pressure` kernel
- **Result:** Floor collision worked, walls did not
- **Lesson:** Order of operations matters; boundary was being applied twice

### Attempt 3: Aggressive Damping + Velocity Clamping (Failed)
- Added 2% per-frame damping and max velocity clamp
- **Result:** Particles still tunneled through walls
- **Lesson:** Damping hides problems rather than fixing root cause

### Root Cause Analysis (From Research)

**Identified Bugs:**

| Bug | Location | Impact |
|-----|----------|--------|
| d_ij calculation incorrect | `sph_bruteforce.wgsl:151-152` | Pressure solve doesn't converge |
| Position repulsion mixed with IISPH | `sph_bruteforce.wgsl:226-232` | Forces fight each other |
| Double boundary collision | Two kernels apply boundaries | Conflicting bounce vs kill velocity |
| Only 4 pressure iterations | `sph_3d.rs:76` | Insufficient for convergence |
| No convergence tracking | N/A | Can't tell if solver is working |

**Mathematical Reference (Correct IISPH):**

```
d_ii = -dt² * ||sum_j(m_j * ∇W_ij)||² / ρ_i²

(Ap)_i = sum_j( m_j * (d_ii * p_i - d_jj * p_j) · ∇W_ij )

p_i := p_i + (ω / a_ii) * (ρ_0 - ρ_i* - (Ap)_i)

a_i^pressure = -sum_j( m_j * (p_i/ρ_i² + p_j/ρ_j²) * ∇W_ij )
```

---

## Implementation Plan (Gated Phases)

### Phase 0: Debug Infrastructure (MUST DO FIRST)

**Gate:** Can measure density, pressure, and convergence from GPU

#### 0.1 Add Debug Readback Buffers

Create staging buffers to read GPU values back to CPU.

**File:** `crates/game/src/gpu/sph_3d.rs`

```rust
// Add to GpuSph3D struct:
debug_density_staging: wgpu::Buffer,
debug_pressure_staging: wgpu::Buffer,
debug_convergence: wgpu::Buffer,  // Single f32 for avg density error
```

**Method:** `read_debug_density(device, queue) -> Vec<f32>`
**Method:** `read_debug_pressure(device, queue) -> Vec<f32>`

#### 0.2 Add Convergence Tracking

**File:** `crates/game/src/gpu/shaders/sph_debug.wgsl`

New kernel to compute average density error:

```wgsl
@compute @workgroup_size(256)
fn compute_density_error(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) { return; }

    let rho = densities[i];
    let error = abs(rho - params.rest_density) / params.rest_density;

    // Atomic add to shared accumulator
    atomicAdd(&total_error, u32(error * 10000.0));  // Fixed-point
}
```

#### 0.3 Bucket Test Logging

**File:** `crates/game/examples/bucket_test.rs`

Add per-frame logging:

```rust
struct FrameMetrics {
    frame: u32,
    particle_count: u32,
    avg_density_error: f32,  // As percentage of rest_density
    max_density: f32,
    min_density: f32,
    avg_pressure: f32,
    max_pressure: f32,
    iterations_used: u32,
}

// Log every N frames to console
if frame % 60 == 0 {
    let metrics = sph.read_metrics(&device, &queue);
    println!(
        "Frame {}: {} particles, density_err={:.2}%, rho=[{:.0},{:.0}], p_max={:.1}",
        metrics.frame, metrics.particle_count,
        metrics.avg_density_error * 100.0,
        metrics.min_density, metrics.max_density,
        metrics.max_pressure
    );
}
```

#### 0.4 Visual Debug Mode

Add density-colored rendering to bucket_test:

```rust
// In render pass, color particles by density deviation
// Blue = rho < rest_density (expansion)
// White = rho ≈ rest_density (good)
// Red = rho > rest_density (compression)

fn density_to_color(rho: f32, rest_density: f32) -> [f32; 3] {
    let deviation = (rho - rest_density) / rest_density;
    let clamped = deviation.clamp(-0.1, 0.1);  // ±10% range
    if clamped < 0.0 {
        [0.0, 0.0, 1.0 + clamped * 10.0]  // Blue to white
    } else {
        [1.0, 1.0 - clamped * 10.0, 1.0 - clamped * 10.0]  // White to red
    }
}
```

**Gate Criteria:**
- [x] Can print density statistics every frame
- [x] Can see compression ratio in logs (expected < 1.05)
- [ ] Visual feedback shows density distribution (deferred to Phase 0.4)

---

### Phase 1: Fix IISPH Pressure Solver

**Gate:** Hydrostatic test passes (still water column, uniform pressure)

#### 1.1 Remove Conflicting Code

**File:** `crates/game/src/gpu/shaders/sph_bruteforce.wgsl`

DELETE lines 226-232 (position-based repulsion):
```wgsl
// DELETE THIS ENTIRE BLOCK:
// Simple position-based repulsion (replaces IISPH for now)
let target_dist = params.h * 0.6;
if (dist < target_dist) {
    ...
}
```

DELETE duplicate boundary in `bf_boundary` kernel OR `bf_apply_pressure`. Keep ONE boundary handling location.

#### 1.2 Fix d_ij Calculation

Current (INCORRECT):
```wgsl
let d_ij = -params.dt2 / max(rhoj * rhoj, 0.0001) * dot(grad, grad);
sum += d_ij * pj_pressure;
```

Correct IISPH off-diagonal:
```wgsl
// d_ij contribution to sum
// The off-diagonal term is: sum_j( d_ij · d_ji * p_j )
// where d_ij = dt² / ρ_i² * ∇W_ij and d_ji = dt² / ρ_j² * ∇W_ji

// Since ∇W_ji = -∇W_ij, the product simplifies to:
// (dt² / ρ_i²) * (dt² / ρ_j²) * |∇W_ij|² * p_j

let grad_mag2 = dot(grad, grad);
let d_ij_dji = params.dt2 * params.dt2 / (rhoi * rhoi) / max(rhoj * rhoj, 0.0001) * grad_mag2;
sum += d_ij_dji * pj_pressure;
```

Wait - this needs more careful analysis. Let me reference the original IISPH formulation...

Actually, the simpler symmetric Jacobi form is:
```wgsl
// Compute contribution of neighbor j to (Ap)_i
// Using symmetric form for stability
let term = params.dt2 * (pressi / (rhoi * rhoi) + pj_pressure / (rhoj * rhoj)) * grad_mag2;
```

**Action:** Create clean IISPH shader from scratch based on reference implementation.

#### 1.3 Increase Pressure Iterations

**File:** `crates/game/src/gpu/sph_3d.rs`

Change line 76:
```rust
pressure_iters: 4,  // CHANGE TO:
pressure_iters: 20, // 20 minimum for convergence
```

Add adaptive iteration with early exit:
```rust
// In step_bruteforce():
for iter in 0..self.params.pressure_iters {
    // Run sum_dij + update_pressure

    // Check convergence every 5 iterations
    if iter % 5 == 4 {
        let error = self.read_density_error(queue);
        if error < 0.01 {  // 1% threshold
            break;
        }
    }
}
```

#### 1.4 Calibrate Rest Density

The kernel coefficients depend on particle spacing. For mass = 1.0:

```rust
// Expected density when particles are at equilibrium spacing
// With h = 0.04, particles should be ~h/2 = 0.02 apart
// Number of neighbors in 3D: ~50
// rest_density should be calibrated to match actual kernel sum

// Test: spawn static grid of particles, measure density
// Adjust rest_density to match measured value
```

**Gate Criteria:**
- [ ] Still water column test: particles don't move (pressure balances gravity)
- [ ] Density deviation < 5% throughout fluid
- [ ] Console logs show pressure solver converging (error decreasing per iteration)

---

### Phase 2: Hydrostatic Validation Test

**Gate:** Particles settle into stable configuration

#### 2.1 Create Hydrostatic Test

**File:** `crates/game/examples/hydrostatic_test.rs`

```rust
// Fill bottom half of domain with particles in grid
// Apply gravity
// Run simulation
// Measure: particles should NOT move after settling
// Pressure should increase linearly with depth

fn spawn_hydrostatic_grid(sph: &mut GpuSph3D, queue: &wgpu::Queue) {
    let mut positions = Vec::new();
    let spacing = H * 0.5;  // Particle spacing

    for z in 0..20 {
        for y in 0..10 {  // Bottom half
            for x in 0..20 {
                positions.push(Vec3::new(
                    x as f32 * spacing + 0.1,
                    y as f32 * spacing + 0.1,
                    z as f32 * spacing + 0.1,
                ));
            }
        }
    }

    let velocities = vec![Vec3::ZERO; positions.len()];
    sph.upload_particles(queue, &positions, &velocities);
}

// Validation criteria:
// - After 1000 frames, max velocity < 0.01 m/s
// - Density at bottom > density at top (hydrostatic pressure)
// - No particles escaped domain
```

#### 2.2 Measure Settling Time

Log velocity statistics:
```rust
let velocities = sph.read_velocities(&device, &queue);
let max_vel = velocities.iter().map(|v| v.length()).max();
let avg_vel = velocities.iter().map(|v| v.length()).sum::<f32>() / velocities.len() as f32;

if frame % 100 == 0 {
    println!("Frame {}: max_vel={:.4}, avg_vel={:.4}", frame, max_vel, avg_vel);
}

// Success: max_vel < 0.01 after settling
```

**Gate Criteria:**
- [ ] Hydrostatic test runs without explosion
- [ ] Particles settle within 500 frames
- [ ] Final max velocity < 0.01 m/s
- [ ] Pressure profile matches expected (linear with depth)

---

### Phase 3: Dynamic Validation (Dam Break)

**Gate:** Dam break behaves physically

#### 3.1 Dam Break Test

**File:** `crates/game/examples/dam_break_test.rs`

```rust
// Classic SPH validation: column of water released
// Should spread horizontally, splash at far wall

fn spawn_dam(sph: &mut GpuSph3D, queue: &wgpu::Queue) {
    let mut positions = Vec::new();
    let spacing = H * 0.5;

    // Water column on left side
    for z in 0..10 {
        for y in 0..20 {
            for x in 0..10 {
                positions.push(Vec3::new(
                    x as f32 * spacing + 0.05,
                    y as f32 * spacing + 0.05,
                    z as f32 * spacing + 0.2,
                ));
            }
        }
    }

    let velocities = vec![Vec3::ZERO; positions.len()];
    sph.upload_particles(queue, &positions, &velocities);
}

// Validation:
// - Wave front should reach ~2x initial width by t=0.5s
// - No particles should tunnel through floor
// - Density should stay within ±10% of rest density
```

#### 3.2 Compression Ratio Tracking

```rust
// Track max compression over time
let densities = sph.read_densities(&device, &queue);
let max_rho = densities.iter().copied().fold(0.0, f32::max);
let compression_ratio = max_rho / REST_DENSITY;

if compression_ratio > 1.05 {
    eprintln!("WARNING: Compression ratio {:.2} exceeds 5%", compression_ratio);
}
```

**Gate Criteria:**
- [ ] Dam break visually correct (spreads, splashes)
- [ ] Compression ratio stays < 1.10 throughout
- [ ] No particles tunnel through boundaries
- [ ] Simulation stable for 10+ seconds

---

### Phase 4: Bucket Fill Test (Original Goal)

**Gate:** Water fills bucket and settles

#### 4.1 Fix Bucket Test

**File:** `crates/game/examples/bucket_test.rs`

Changes:
1. Use `append_particles()` not `upload_particles()` for continuous spawn
2. Lower spawn rate (10 particles/frame)
3. Add metrics logging
4. Fix spawn position to be inside domain

```rust
const SPAWN_RATE: usize = 10;
const SPAWN_HEIGHT: f32 = 1.0;  // Must be < GRID_SIZE_Y * CELL_SIZE

// Spawn narrow stream
if particle_count < MAX_PARTICLES {
    let mut new_positions = Vec::new();
    let mut new_velocities = Vec::new();

    for i in 0..SPAWN_RATE {
        new_positions.push(Vec3::new(
            0.3 + 0.01 * (i as f32),  // Narrow X spread
            SPAWN_HEIGHT,
            0.3,  // Center Z
        ));
        new_velocities.push(Vec3::new(0.0, -1.0, 0.0));  // Downward
    }

    sph.append_particles(queue, &new_positions, &new_velocities);
}
```

#### 4.2 Success Criteria

After filling:
- [ ] Water level rises smoothly
- [ ] Density stays within ±5% of rest density
- [ ] No particles escape through walls
- [ ] Settled water is visually flat on top
- [ ] Console shows converged pressure solve

---

### Phase 5: DEM Fallback (If IISPH Insufficient)

Only implement if IISPH cannot achieve stability.

#### 5.1 Simple DEM Contact

For particle-particle collisions when SPH isn't enough:

```wgsl
// Linear spring-damper contact
let contact_radius = params.h * 0.3;  // Smaller than kernel radius
let delta = contact_radius * 2.0 - dist;

if delta > 0.0 {
    let k = 10000.0;  // Spring stiffness
    let c = 100.0;    // Damping

    let rel_vel = vi - vj;
    let normal = normalize(r);
    let v_normal = dot(rel_vel, normal);

    let f_contact = normal * (k * delta - c * v_normal);
    f_total += f_contact;
}
```

#### 5.2 Hybrid SPH-DEM

Run DEM contact forces AFTER SPH pressure solve:
1. SPH: density → pressure → pressure acceleration
2. DEM: contact detection → contact forces
3. Integrate: v += dt * (a_pressure + a_contact)

**Gate Criteria:**
- [ ] Particles don't interpenetrate
- [ ] Stable stacking behavior
- [ ] Performance impact < 20% (due to O(n²) contacts)

---

## Debug Checklist

Use these diagnostics to identify problems:

### Symptom: Particles Explode
- [ ] Check pressure values (should be positive, finite)
- [ ] Check d_ii values (should be negative, finite)
- [ ] Reduce timestep by 2x
- [ ] Reduce omega to 0.3

### Symptom: Particles Clump
- [ ] Check density at clump (should be >> rest_density)
- [ ] Verify spiky kernel used for pressure (not poly6)
- [ ] Check pressure gradient direction (should be repulsive)

### Symptom: Particles Tunnel Through Walls
- [ ] Check boundary collision location (should be AFTER integration)
- [ ] Verify boundary applies to ALL particles (not just some)
- [ ] Reduce max velocity or timestep

### Symptom: Solver Doesn't Converge
- [ ] Log density error per iteration (should decrease)
- [ ] Check a_ii values (should be non-zero)
- [ ] Increase iterations to 50+
- [ ] Verify rest_density matches actual particle density

### Symptom: Particles Jitter at Surface
- [ ] This is normal for SPH (kernel truncation at free surface)
- [ ] Add surface tension (optional)
- [ ] Use XSPH viscosity for smoothing

---

## Files to Modify

| File | Changes |
|------|---------|
| `sph_3d.rs` | Debug buffers, metrics readback, iteration count |
| `sph_bruteforce.wgsl` | Remove repulsion hack, fix d_ij, single boundary |
| `bucket_test.rs` | Logging, spawn rate, validation checks |
| NEW: `sph_debug.wgsl` | Convergence tracking kernel |
| NEW: `hydrostatic_test.rs` | Static validation test |
| NEW: `dam_break_test.rs` | Dynamic validation test |

---

## Expected Behavior (Success Criteria)

### Hydrostatic Test
- Particles in grid formation
- After settling: max velocity < 0.01 m/s
- Pressure increases linearly with depth

### Dam Break Test
- Smooth wave propagation
- Splash at far wall
- Density deviation < 10%

### Bucket Fill Test
- Stream enters bucket smoothly
- Water level rises
- Surface settles flat
- No escaping particles
- Compression ratio < 1.05

---

## References

- [IISPH Paper (Ihmsen et al., 2014)](https://cg.informatik.uni-freiburg.de/publications/2013_TVCG_IISPH.pdf)
- [Interactive IISPH Demo](https://interactivecomputergraphics.github.io/physics-simulation/examples/iisph.html)
- [SPlisHSPlasH Reference Implementation](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH)
- [SPH Tutorial (physics-simulation.org)](https://sph-tutorial.physics-simulation.org/)
