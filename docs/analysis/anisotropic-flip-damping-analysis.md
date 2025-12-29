# Analysis: Anisotropic FLIP Damping at Free Surface

**Date:** 2025-12-27
**Status:** PENDING USER REVIEW

---

## Summary

The G2P (grid-to-particle) transfer uses **anisotropic FLIP/PIC blending** that applies **30% PIC damping to vertical velocity at free surface**. This is an intentional damping to "kill vertical fizz" but also kills vertical momentum.

---

## Code Location

**File:** `crates/sim/src/flip.rs:555-582`

```rust
// Check if particle is near surface (cell adjacent to air)
let gi = (pos.x / cell_size) as i32;
let gj = (pos.y / cell_size) as i32;
let near_surface = if gi >= 1 && gi < width as i32 - 1 && gj >= 1 && gj < height as i32 - 1 {
    let idx_up = (gj - 1) as usize * width + gi as usize;
    let idx_down = (gj + 1) as usize * width + gi as usize;
    let idx_left = gj as usize * width + (gi - 1) as usize;
    let idx_right = gj as usize * width + (gi + 1) as usize;

    grid.cell_type[idx_up] == CellType::Air ||
    grid.cell_type[idx_down] == CellType::Air ||
    grid.cell_type[idx_left] == CellType::Air ||
    grid.cell_type[idx_right] == CellType::Air
} else {
    true // Treat boundary as surface
};

const FLIP_X: f32 = 0.98;   // High FLIP for horizontal (preserve transport)
const FLIP_Y_BULK: f32 = 0.95;   // High FLIP for vertical in bulk
const FLIP_Y_SURFACE: f32 = 0.70; // More PIC-ish for vertical at surface

let flip_y = if near_surface { FLIP_Y_SURFACE } else { FLIP_Y_BULK };

let vx = FLIP_X * flip_velocity.x + (1.0 - FLIP_X) * pic_velocity.x;
let vy = flip_y * flip_velocity.y + (1.0 - flip_y) * pic_velocity.y;
particle.velocity = Vec2::new(vx, vy);
```

---

## What The Code Does

### 1. Near-Surface Detection

A particle is classified as "near surface" if:
- Any of its 4 adjacent cells (up/down/left/right) is `CellType::Air`, OR
- It's in a boundary cell (gi < 1, gi >= width-1, gj < 1, gj >= height-1)

### 2. Anisotropic FLIP/PIC Blend

| Component | Condition | FLIP Ratio | PIC Blend | Damping/frame |
|-----------|-----------|------------|-----------|---------------|
| Horizontal (X) | Always | 0.98 | 2% | Low |
| Vertical (Y) | In bulk | 0.95 | 5% | Moderate |
| Vertical (Y) | Near surface | 0.70 | **30%** | **HIGH** |

### 3. What PIC Damping Does

In the FLIP/PIC blend:
```
v_new = FLIP_RATIO * flip_velocity + (1 - FLIP_RATIO) * pic_velocity
```

Where:
- `flip_velocity = old_particle_velocity + (new_grid_velocity - old_grid_velocity)`
- `pic_velocity = new_grid_velocity` (directly from grid)

**PIC component** pulls particle velocity toward the grid-averaged velocity each frame. The grid velocity is **smoother/more diffusive** than individual particle velocities because P2G transfer averages nearby particles.

With 30% PIC at surface:
- Each frame, 30% of vertical velocity is replaced by grid-averaged velocity
- This causes **numerical diffusion** that smooths out velocity variations
- Over multiple frames, this compounds into significant energy loss

---

## Impact Analysis

### Affected Particles

In an open-channel sluice flow:
- **Top layer of water** (cells adjacent to air above) → near_surface = true
- **Edge cells** (gi=0, gi=511, gj=0, gj=383) → near_surface = true

For a water layer 20 cells deep:
- ~5-10% of particles are adjacent to air (surface layer)
- These particles get 30% vertical PIC damping per frame

### Mathematical Impact

For surface particles with 30% PIC (FLIP_RATIO = 0.70):

If particle has vertical velocity v and grid has smoothed velocity v_grid:
```
v_new = 0.70 * v + 0.30 * v_grid
```

If v_grid < v (grid is smoother), the particle loses velocity toward the grid average.

Worst case (v_grid = 0, particle has v):
- Frame 1: v → 0.70v
- Frame 2: 0.70v → 0.49v
- Frame 5: 0.70^5 * v = 0.168v (83% lost)
- Frame 10: 0.70^10 * v = 0.028v (97% lost)

This is **extremely aggressive damping** for surface particles.

---

## Why This Was Added

Comment says:
```rust
// Near surface: damp y more aggressively (more PIC-ish) to kill vertical fizz
```

**Intent:** Reduce vertical oscillations/noise at the free surface.

**Problem:** Also kills ALL vertical momentum at the surface, causing:
- Water to "flatten" instead of splash
- No vertical turbulence
- Honey-like behavior for anything trying to rise or fall near surface

---

## Comparison with Literature

Standard FLIP/PIC blend ratios:
- **Houdini FLIP:** 0.95-0.97 (uniform)
- **Blender FLIP Fluids:** 0.95 (uniform)
- **Academic papers:** 0.95-1.0 (pure FLIP to slight PIC)

**No reference implementation uses anisotropic blending with 0.70 for any component.**

---

## Proposed Fix Options

### Option A: Uniform FLIP Ratio (Recommended)

Remove anisotropic blending, use single ratio:

```rust
// Replace lines 574-582 with:
const FLIP_RATIO: f32 = 0.97; // Industry standard
particle.velocity = FLIP_RATIO * flip_velocity + (1.0 - FLIP_RATIO) * pic_velocity;
```

**Pros:**
- Simple
- Matches industry practice
- Preserves momentum everywhere

**Cons:**
- May have "vertical fizz" at surface (the original problem this was trying to solve)
- May need to address surface noise another way

**Lines to change:** 555-582 (replace ~28 lines with ~3 lines)

### Option B: Less Aggressive Surface Damping

Keep anisotropic, but reduce surface damping:

```rust
const FLIP_Y_SURFACE: f32 = 0.90; // Was 0.70, now 10% PIC instead of 30%
```

**Pros:**
- Still addresses surface noise somewhat
- Less momentum loss

**Cons:**
- Still anisotropic (non-standard)
- Still some extra damping at surface

**Lines to change:** 576 only

### Option C: Remove Surface Detection Entirely

Delete the near_surface logic, use uniform bulk values:

```rust
const FLIP_X: f32 = 0.98;
const FLIP_Y: f32 = 0.95;

let vx = FLIP_X * flip_velocity.x + (1.0 - FLIP_X) * pic_velocity.x;
let vy = FLIP_Y * flip_velocity.y + (1.0 - FLIP_Y) * pic_velocity.y;
particle.velocity = Vec2::new(vx, vy);
```

**Pros:**
- Simpler than Option A (keeps component-wise but removes surface check)
- No surface penalty

**Cons:**
- Still has slight anisotropy (0.98 vs 0.95)

**Lines to change:** 555-582 (remove near_surface check, keep component-wise)

---

## Recommendation

**Option A (Uniform FLIP Ratio)** is recommended because:
1. Simplest implementation
2. Matches industry standard
3. Removes all sources of excessive damping
4. If surface noise is a problem, address it with proper surface tension or other methods, not PIC damping

---

## Verification Plan

After applying fix:
1. Run 60 seconds
2. Check momentum diagnostic (should show < 10% loss/sec instead of 70%)
3. Visually verify water flows naturally
4. Check for surface noise/fizz issues (may need follow-up)

---

## Risk

**Medium risk.** If surface fizz was a real problem before, it may return. But that's a separate issue from momentum loss, and should be fixed properly rather than with aggressive PIC damping.
