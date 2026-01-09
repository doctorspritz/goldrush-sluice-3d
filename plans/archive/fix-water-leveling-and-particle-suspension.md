# Fix Water Leveling and Particle Suspension

## Problem Statement

Water forms "hills" instead of leveling flat, and mud/sediment doesn't get carried in water flow. The current implementation has two competing systems (Navier-Stokes + CA) that fight each other instead of working together.

## Root Cause Analysis

Based on research into Noita, Powder Toy, and other falling sand implementations:

1. **Water Hills**: CA systems move water one cell per frame. Without multi-step horizontal spread, water piles up faster than it can spread.

2. **Missing Suspension**: Current code has separate movement logic for each material type. Particles should ALL read from the velocity field and move accordingly - density only affects *settling*, not *transport*.

3. **Wrong Approach**: Column height comparison is computationally expensive and doesn't match how real fluids work. The velocity field already encodes pressure information via the projection step.

## Research Findings

### From Powder Toy (proven approach)
- Uses `rt = 30` parameter - water checks up to 30 horizontal positions per frame
- Continues checking until space found OR max positions checked
- This allows water to "skip" over obstacles and spread much faster

### From Noita GDC Talk
- All particles updated bottom-to-top
- Density-based displacement (heavier sinks through lighter)
- Same velocity field drives all fluid particles

### From Sandspiel (hybrid approach)
- Navier-Stokes on GPU creates velocity field
- ALL particles read velocity and move accordingly
- Bidirectional coupling between fluid sim and particles

### Suspension Physics (simplified Rouse number)
- When `flow_speed > settling_velocity`: particle stays suspended
- When `flow_speed < settling_velocity`: particle settles
- Settling velocity depends on density: heavier = faster settling

## Proposed Solution

### Phase 1: Multi-Step Horizontal Spread

Replace single-step horizontal spread with Powder Toy's approach:

```rust
// world.rs - update_liquid()
const HORIZONTAL_SEARCH_RANGE: i32 = 20; // Like Powder Toy's rt=30

// Instead of checking just left/right neighbor:
for distance in 1..=HORIZONTAL_SEARCH_RANGE {
    for dx in [dx1 * distance, dx2 * distance] {
        let nx = wx + dx;
        if self.get_material(nx, wy) == Material::Air {
            // Check if path is clear (no obstacles between)
            if self.path_clear(wx, wy, nx, wy) {
                self.swap_with_velocity(wx, wy, nx, wy);
                return true;
            }
        }
        // Stop searching this direction if blocked
        if self.get_material(nx, wy).is_solid() {
            break;
        }
    }
}
```

### Phase 2: Unified Velocity-Based Movement

All fluid/suspendable particles use the SAME movement logic:

```rust
// world.rs or update.rs

fn update_particle(&mut self, wx: i32, wy: i32) -> bool {
    let material = self.get_material(wx, wy);
    let density = material.density();

    // 1. Read velocity from grid (same for ALL particles)
    let (vx, vy) = self.get_velocity(wx, wy);
    let speed = vx.hypot(vy);

    // 2. Settling velocity based on density (heavier settles faster)
    let settling_velocity = (density as f32 - 10.0) * 0.02; // Water density = 10

    // 3. Suspension check: is flow strong enough to carry this particle?
    let is_suspended = speed > settling_velocity;

    // 4. Movement
    if is_suspended {
        // Move with flow - ALL particles move the same way when suspended
        let move_x = if vx > 0.1 { 1 } else if vx < -0.1 { -1 } else { 0 };
        let move_y = if vy > 0.1 { 1 } else if vy < -0.1 { -1 } else { 0 };
        // ... try to move in velocity direction
    } else {
        // Settling behavior - density determines what happens
        // Add gravity, try to fall/sink
        let vy = vy + GRAVITY;
        // ... standard falling logic
    }
}
```

### Phase 3: Remove Column-Based Leveling

Delete the complex column height comparison code. The multi-step spread + velocity field handles leveling naturally.

```rust
// DELETE from world.rs:
// - find_water_body_surface()
// - flatten_water_body()
// - auto_level_water()
// - Column height scanning in update_liquid()
```

### Phase 4: Tune Constants

Based on research:

```rust
// fluid.rs
const VISCOSITY: f32 = 0.0;      // Zero for water (inviscid)
const ITERATIONS: usize = 4;     // Back to 4 for better pressure solve

// update logic
const HORIZONTAL_SEARCH_RANGE: i32 = 20;  // Multi-step spread distance
const GRAVITY: f32 = 0.3;                  // Slightly stronger
const BASE_SETTLING_VELOCITY: f32 = 0.5;   // Water's settling threshold
```

## Acceptance Criteria

- [ ] Water spreads horizontally up to 20 cells per frame (no hills)
- [ ] Mud placed in flowing water gets carried downstream
- [ ] Soil particles can be suspended in fast-moving water
- [ ] Water finds a flat level within ~30 frames of being poured
- [ ] No regression in FPS (stay above 30 FPS at 256x256)

## Files to Modify

| File | Changes |
|------|---------|
| `crates/sim/src/world.rs` | Replace column-based spread with multi-step, unify particle movement |
| `crates/sim/src/update.rs` | Align with unified velocity-based movement |
| `crates/sim/src/fluid.rs` | Tune viscosity to 0, iterations to 4 |

## Key Insight

> "All particles should have similar characteristics - mud in water will flow along with it, albeit with more density"

The velocity field is the source of truth. ALL particles:
1. Read velocity from grid
2. Move in velocity direction if `speed > settling_velocity`
3. Settle (fall) if `speed < settling_velocity`

Density only affects the settling threshold, not the transport mechanics.

## References

- [Powder Toy source - rt parameter](https://github.com/The-Powder-Toy/The-Powder-Toy)
- [Noita GDC Talk](https://www.gdcvault.com/play/1025695/Exploring-the-Tech-and-Design/)
- [Making Sandspiel - hybrid approach](https://maxbittker.com/making-sandspiel/)
- [W-Shadow water simulation](https://w-shadow.com/blog/2009/09/29/falling-sand-style-water-simulation/)
- [Rouse number for suspension](https://en.wikipedia.org/wiki/Rouse_number)
\n\n---\n## ARCHIVED: 2026-01-09\n\n**Superseded by:** 3D FLIP approach\n\nThese CA-based water leveling fixes were from the older cellular automata system. The chunk-local issues described here don't apply to the 3D FLIP solver which handles water leveling through proper pressure projection.
