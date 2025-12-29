# CFL Violation - Particles Travel 2 Cells Per Step

**Priority:** P3 - Low Priority (splashes not important)
**Status:** pending
**Tags:** physics, performance, damping, water
**Dependencies:** none

## Problem Statement

The simulation has a CFL number of ~2.0, meaning particles travel 2 cell widths per timestep. Literature recommends CFL < 1.0 for accurate water simulation. This is the **primary cause** of honey-like behavior due to excessive interpolation smoothing.

**Current values:**
- `dt = 1/60` (16.67ms timestep)
- `CELL_SIZE = 1.0`
- `max_velocity ≈ 80-120 pixels/sec`
- `CFL = velocity * dt / cell_size = 120 * 0.0167 / 1.0 = 2.0`

## Findings

### Why This Causes Honey Behavior

1. **Interpolation Diffusion**: Each P2G and G2P cycle uses quadratic B-splines over a 3-cell support. When particles jump 2 cells, the interpolation samples completely different grid cells each frame, causing severe numerical smoothing.

2. **Gradient Loss**: Sharp velocity gradients are smeared out because particles skip over them. Vortices cannot form properly.

3. **Angular Momentum Loss**: APIC's affine velocity matrix becomes inaccurate when the particle moves too far from its local grid neighborhood.

### Literature References

From Blender FLIP Fluids:
> "A finer grid requires a smaller time step to satisfy the CFL condition"

From research papers:
> "Explicit methods: CFL ≤ 1.0 (required for stability)"
> "General structured grids: CFL = 1.3 is safe default"

## Proposed Solutions

### Option A: Reduce Timestep (Recommended)
```rust
// In main.rs, change from:
let dt = 1.0 / 60.0;
// To:
let dt = 1.0 / 120.0;  // CFL = 1.0
```

**Pros:**
- Simple change
- CFL becomes 1.0 (recommended)
- Immediate visual improvement

**Cons:**
- Doubles simulation steps per rendered frame
- ~2x performance cost

**Effort:** Small
**Risk:** Low

### Option B: Sub-stepping
```rust
// Run 2 simulation steps per frame
for _ in 0..2 {
    sim.update(dt / 2.0);
}
```

**Pros:**
- Same visual frame rate
- Better physics accuracy

**Cons:**
- 2x performance cost
- May need to adjust other parameters

**Effort:** Small
**Risk:** Low

### Option C: Adaptive Timestep
```rust
fn adaptive_dt(&self) -> f32 {
    let max_v = self.particles.iter()
        .map(|p| p.velocity.length())
        .fold(0.0f32, f32::max);
    let cfl_target = 0.5;
    let dt_max = cfl_target * self.grid.cell_size / max_v;
    dt_max.clamp(1.0/240.0, 1.0/60.0)
}
```

**Pros:**
- Optimal timestep for current velocities
- Best quality/performance ratio

**Cons:**
- More complex implementation
- Variable simulation rate

**Effort:** Medium
**Risk:** Medium

## Recommended Action

**Option A: Reduce timestep to dt = 1/120**

This is the simplest fix with the highest impact. Combined with increased pressure iterations, this should eliminate honey-like behavior.

## Technical Details

### Affected Files
- `crates/game/src/main.rs:389` - `let dt = 1.0 / 60.0;`

### Components Affected
- Particle advection quality
- P2G/G2P transfer accuracy
- Pressure solve convergence

### Database Changes
None

## Acceptance Criteria

- [ ] CFL number is ≤ 1.0 for typical velocities
- [ ] Water flows like water, not honey
- [ ] No visible interpolation artifacts
- [ ] Simulation remains stable
- [ ] Performance is acceptable (≥30 FPS)

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2025-12-27 | Diagnosis | CFL=2.0 identified as primary cause of damping |

## Resources

- [CFL Condition (SimScale)](https://www.simscale.com/blog/cfl-condition/)
- [FLIP Tutorial (Ten Minute Physics)](https://matthias-research.github.io/pages/tenMinutePhysics/18-flip.pdf)
- Performance Oracle agent analysis
