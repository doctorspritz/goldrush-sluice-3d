<!-- TODO: Review -->

# Plan: Vorticity-Driven Sand Suspension

## Goal
Make sand particles swirl within the water flow instead of just dragging along the bottom.

## Key Insight
Vorticity (curl of velocity field) measures local rotation. High vorticity regions have eddies that can carry particles against gravity. We already compute vorticity for confinement - reuse it for sand.

---

## Current State

### Update Loop Order (flip.rs:150-230)
1. P2G transfer
2. Store old velocities
3. Apply gravity
4. **Vorticity confinement** ← vorticity computed here
5. Pressure projection
6. Extrapolate velocities
7. **G2P transfer** ← sand gets velocity here (lines 767-826)
8. Build spatial hash
9. Advect particles

### Current Sand G2P (flip.rs:767-826)
```rust
if particle.is_sediment() {
    let v_grid = grid.sample_velocity_bspline(pos);

    if cell_type == CellType::Fluid {
        // PIC/FLIP blend to follow water
        let pic_vel = v_grid;
        let flip_vel = particle.velocity + grid_delta;
        particle.velocity = sand_pic_ratio * pic_vel + (1.0 - sand_pic_ratio) * flip_vel;
    }

    // PROBLEM: Always applies constant settling
    const SETTLING_FACTOR: f32 = 0.62;
    particle.velocity.y += GRAVITY * SETTLING_FACTOR * dt;
    return;
}
```

### Vorticity Infrastructure
- `grid.vorticity: Vec<f32>` - stored at cell centers
- `grid.compute_vorticity()` - computes ω = ∂v/∂x - ∂u/∂y
- Called inside `apply_vorticity_confinement_with_piles()` before G2P

---

## Implementation Plan

### Step 1: Add Vorticity Sampling Method

**File**: `crates/sim/src/grid.rs`

Add bilinear interpolation for vorticity at arbitrary position:

```rust
/// Sample vorticity at position using bilinear interpolation
/// Vorticity is stored at cell centers
pub fn sample_vorticity(&self, pos: Vec2) -> f32 {
    // Cell center coordinates
    let x = pos.x / self.cell_size - 0.5;
    let y = pos.y / self.cell_size - 0.5;

    let i0 = (x.floor() as i32).clamp(0, self.width as i32 - 2) as usize;
    let j0 = (y.floor() as i32).clamp(0, self.height as i32 - 2) as usize;
    let i1 = i0 + 1;
    let j1 = j0 + 1;

    let tx = (x - i0 as f32).clamp(0.0, 1.0);
    let ty = (y - j0 as f32).clamp(0.0, 1.0);

    // Bilinear interpolation
    let v00 = self.vorticity[j0 * self.width + i0];
    let v10 = self.vorticity[j0 * self.width + i1];
    let v01 = self.vorticity[j1 * self.width + i0];
    let v11 = self.vorticity[j1 * self.width + i1];

    let v0 = v00 * (1.0 - tx) + v10 * tx;
    let v1 = v01 * (1.0 - tx) + v11 * tx;

    v0 * (1.0 - ty) + v1 * ty
}
```

**Test**: Unit test that vorticity sampling returns expected values at cell centers.

---

### Step 2: Modify Sand G2P with Vorticity-Based Suspension

**File**: `crates/sim/src/flip.rs` (around line 820)

Replace constant settling with vorticity-modulated settling:

```rust
// Sample vorticity at particle position
let vorticity = grid.sample_vorticity(pos);
let vort_magnitude = vorticity.abs();

// Vorticity creates lift force perpendicular to flow
// In 2D: positive ω (CCW rotation) → force perpendicular to velocity gradient
// Simplified: use vorticity magnitude to reduce settling

// Tunable parameters
const SETTLING_FACTOR: f32 = 0.62;      // Base settling (unchanged)
const VORT_LIFT_SCALE: f32 = 0.5;       // How much vorticity counters settling
const VORT_SWIRL_SCALE: f32 = 0.1;      // How much vorticity adds tangential motion

// 1. Compute settling reduction from vorticity
// High vorticity → less settling (particle stays suspended)
let lift_factor = (vort_magnitude * VORT_LIFT_SCALE).min(1.0);
let effective_settling = SETTLING_FACTOR * (1.0 - lift_factor);

// 2. Add swirl motion from vorticity
// Vorticity ω creates velocity perpendicular to the flow direction
// For simplicity: add velocity perpendicular to current particle velocity
let v_normalized = if particle.velocity.length() > 0.1 {
    particle.velocity.normalize()
} else {
    Vec2::new(1.0, 0.0)
};
let v_perp = Vec2::new(-v_normalized.y, v_normalized.x);
let swirl_velocity = v_perp * vorticity * VORT_SWIRL_SCALE;

// Apply modified settling + swirl
particle.velocity.y += GRAVITY * effective_settling * dt;
particle.velocity += swirl_velocity * dt;
```

---

### Step 3: Add Tuning Parameters to FlipSimulation

**File**: `crates/sim/src/flip.rs` (struct definition around line 55)

```rust
pub struct FlipSimulation {
    // ... existing fields ...

    /// Scale factor for vorticity lift effect on sand
    /// Higher = sand stays suspended longer in rotating flow
    pub vorticity_lift_scale: f32,

    /// Scale factor for vorticity swirl effect on sand
    /// Higher = sand follows rotational motion more
    pub vorticity_swirl_scale: f32,
}

// In new() or Default:
vorticity_lift_scale: 0.5,
vorticity_swirl_scale: 0.1,
```

---

### Step 4: Visual Testing

Create test scenario with clear vortex:
1. Strong inlet flow
2. Obstacle creating wake vortices
3. Sand particles spawned upstream

Expected behavior:
- Sand in calm water: sinks to bottom (bedload)
- Sand in fast flow: moves horizontally, settles slowly
- Sand in vortex: swirls, stays suspended, follows rotation

---

## Physics Justification

### Why Vorticity = Suspension?

In real sediment transport:
- Turbulent eddies carry particles upward against gravity
- Eddy turnover time vs settling time determines suspension
- Vorticity magnitude is a proxy for eddy intensity

The Rouse number relationship:
```
P = ws / (κ * u*)

where u* ≈ √(τ/ρ) and τ relates to velocity gradients
```

High vorticity → high velocity gradients → high shear stress → low effective Rouse number → suspension.

### Why Add Swirl Velocity?

In a vortex, particles don't just resist settling - they **rotate with the eddy**.

Angular velocity of fluid parcel = ω/2 (half the vorticity).

Adding perpendicular velocity proportional to ω approximates this rotation.

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Energy blow-up from lift | Cap lift_factor at 1.0, test with high vorticity |
| Sand never settles | Tune VORT_LIFT_SCALE, ensure calm regions have low ω |
| Swirl looks artificial | Start with low VORT_SWIRL_SCALE, increase gradually |
| Performance (sampling) | Vorticity sampling is O(1), similar to velocity sampling |

---

## Testing Strategy

1. **Unit test**: `sample_vorticity()` returns correct values at known positions
2. **Diagnostic**: Print/log vorticity magnitude at sand positions
3. **Visual test**: Sand in circular flow should orbit, not sink
4. **Regression**: Existing settling tests should still pass with ω=0

---

## Implementation Order

1. [x] Add `sample_vorticity()` to Grid - `grid.rs:512-538`
2. [x] Add unit test for vorticity sampling - `vortex_tests.rs:513-585`
3. [x] Modify sand G2P with lift - `flip.rs:820-861`
4. [x] Add swirl component - implemented with lift
5. [ ] Visual test: verify sand rotates in vortices
6. [ ] Tune parameters for realistic look
7. [ ] Add parameters to FlipSimulation struct for runtime tuning (if needed)

## Current Parameters (flip.rs:831-833)

```rust
const SETTLING_FACTOR: f32 = 0.62;     // Base settling (unchanged)
const VORT_LIFT_SCALE: f32 = 0.3;      // How much vorticity counters settling
const VORT_SWIRL_SCALE: f32 = 0.05;    // How much vorticity adds tangential motion
```

## Tuning Guide

- **Sand settles too fast**: Increase `VORT_LIFT_SCALE` (try 0.5-1.0)
- **Sand never settles**: Decrease `VORT_LIFT_SCALE` (try 0.1-0.2)
- **Sand doesn't swirl enough**: Increase `VORT_SWIRL_SCALE` (try 0.1-0.2)
- **Sand swirls too chaotically**: Decrease `VORT_SWIRL_SCALE` (try 0.02-0.03)
