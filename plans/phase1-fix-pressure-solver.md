# Phase 1: Fix IISPH Pressure Solver

## Problem Analysis

From diagnostics:
- `rest_density = 1000.0` (physical kg/m³)
- Measured density = **1,000,000 - 77,000,000** (kernel units)
- Pressure = **0.0** (always!)

### Root Cause

The density is computed in "kernel units" not physical units:
```
poly6_coef = 315 / (64 * π * h⁹)
           = 315 / (64 * 3.14159 * 0.04⁹)
           = 315 / (64 * 3.14159 * 2.62e-13)
           ≈ 5.98e12  (huge!)
```

With particles at spacing ~0.02m and h=0.04m:
- Each neighbor contributes: `poly6_coef * (h² - r²)³ ≈ 24,000`
- With ~50 neighbors: `density ≈ 1,200,000`

But `rest_density = 1000`, so:
- `rho_err = 1,200,000 - 1000 = 1,199,000` (massive positive error)
- `dii_val = -dt² * |sum|² / rho² ≈ -tiny` (because rho² is huge)
- `p_new = omega * rho_err / dii_val = huge_negative`
- After clamp: `p_new = 0.0` ← **ALWAYS ZERO!**

## Solution

**Calibrate `rest_density` to match actual kernel sum at equilibrium.**

For particles in a cubic lattice with spacing `s`:
```
rest_density = sum over neighbors of poly6(r²)
```

With h = 0.04m and target spacing s = h * 0.5 = 0.02m:
- Theoretical: ~30-50 neighbors within kernel radius
- Each contributes ~poly6(r²) based on distance

We'll compute this empirically by:
1. Spawning a static grid of particles
2. Running ONE density pass
3. Reading back average density
4. Using that as rest_density

## Implementation Tasks

### Task 1: Add `calibrate_rest_density()` method

**File:** `crates/game/src/gpu/sph_3d.rs`

Add method that:
1. Creates a temporary grid of particles at spacing h*0.5
2. Runs bf_density_dii kernel once
3. Reads back densities
4. Returns median density (robust to boundary effects)

```rust
pub fn calibrate_rest_density(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> f32 {
    // Spawn 10x10x10 grid at spacing h*0.5
    let spacing = self.params.h * 0.5;
    let mut positions = Vec::new();
    for z in 0..10 {
        for y in 0..10 {
            for x in 0..10 {
                positions.push(Vec3::new(
                    0.2 + x as f32 * spacing,
                    0.2 + y as f32 * spacing,
                    0.2 + z as f32 * spacing,
                ));
            }
        }
    }
    let velocities = vec![Vec3::ZERO; positions.len()];

    // Upload and run density kernel
    self.upload_particles(queue, &positions, &velocities);
    // Run predict (to copy to positions_pred)
    // Run density_dii
    // Read back densities
    // Return median
}
```

### Task 2: Call calibration at startup

**File:** `crates/game/examples/bucket_test.rs`

After creating GpuSph3D, call calibration:

```rust
let calibrated_density = sph.calibrate_rest_density(&device, &queue);
println!("Calibrated rest_density: {}", calibrated_density);
sph.set_rest_density(queue, calibrated_density);
```

### Task 3: Add `set_rest_density()` method

**File:** `crates/game/src/gpu/sph_3d.rs`

```rust
pub fn set_rest_density(&mut self, queue: &wgpu::Queue, rest_density: f32) {
    self.params.rest_density = rest_density;
    queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));
}
```

### Task 4: Increase pressure iterations

**File:** `crates/game/src/gpu/sph_3d.rs`

Change line 88:
```rust
pressure_iters: 4,  // Change to:
pressure_iters: 20,
```

### Task 5: Remove position-based repulsion hack

**File:** `crates/game/src/gpu/shaders/sph_bruteforce.wgsl`

DELETE lines 226-232 (the spring repulsion code):
```wgsl
// DELETE THIS ENTIRE BLOCK:
// Simple position-based repulsion (replaces IISPH for now)
let target_dist = params.h * 0.6;  // Target particle spacing
if (dist < target_dist) {
    let overlap = target_dist - dist;
    let repel_dir = normalize(r);
    f_pressure += repel_dir * overlap * 200.0;  // Gentler spring
}
```

### Task 6: Remove duplicate boundary handling

**File:** `crates/game/src/gpu/shaders/sph_bruteforce.wgsl`

The boundary collision is applied TWICE:
1. In `bf_apply_pressure()` lines 254-275
2. In `bf_boundary()` lines 301-326

Keep the one in `bf_apply_pressure()` (kills velocity on contact).
DELETE the entire `bf_boundary()` kernel body OR make it a no-op.

Actually, simpler: just remove the `bf_boundary` dispatch from the Rust code.

**File:** `crates/game/src/gpu/sph_3d.rs`

In `step_bruteforce()`, comment out or remove the bf_boundary dispatch.

## Acceptance Criteria

After these changes:
- [ ] `calibrate_rest_density()` returns a value ~1,000,000 (matching actual kernel sum)
- [ ] Pressure values are non-zero (check via diagnostics)
- [ ] Density error decreases over frames (pressure solver working)
- [ ] Particles don't pile up in a single point

## Verification

```bash
cargo run --example bucket_test --release
```

Expected output changes:
- `density_err` should be < 10% (was 150000%)
- `p_max` should be > 0 (was 0.0)
- `compression` should be < 2.0 (was 77000)
