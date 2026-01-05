---
status: pending
priority: p1
issue_id: "021"
tags: [code-review, physics, geometry, critical]
dependencies: []
---

# Water Spawns at Exact Riffle Top Height

## Problem Statement

Water particles spawn at y=0.44m, which is exactly the height of the first riffle top (y=0.44m). This makes it geometrically impossible for water to overflow riffles - it starts at the barrier level, not above it.

**Why it matters:** For water to flow over a riffle, it must be ABOVE the riffle top. Currently water spawns AT the riffle level.

## Findings

### Calculation (from Pattern Recognition agent):

For gold_sluice_3d:
- `GRID_WIDTH = 48`, `CELL_SIZE = 0.04`, `slope = 0.20`
- `inlet_floor_y = (48 - 1) * 0.20 = 9.4` → truncates to 9 cells
- `spawn_y_base = (9 + 2.0) * 0.04 = 0.44m`

For first riffle at x=6:
- `riffle_floor_y = (48 - 1 - 6) * 0.20 = 8.2` → truncates to 8 cells
- `riffle_top_y = 8 + 3 = 11` (with riffle_height=3)
- `riffle_top_world = 11 * 0.04 = 0.44m`

**Water spawn height == Riffle top height == 0.44m**

### Location:
`crates/sim3d/src/sluice.rs` lines 154-158:
```rust
let inlet_floor_y = ((width - 1) as f32 * config.slope) as usize;
let spawn_y_base = (inlet_floor_y as f32 + 2.0) * dx;  // +2 cells is not enough!
```

The "+2 cells" is hardcoded and does not account for riffle height (3 cells).

## Proposed Solutions

### Option A: Calculate spawn height based on first riffle top
**Pros:** Ensures water is always above riffles
**Cons:** Couples spawn logic to riffle geometry
**Effort:** Small
**Risk:** Low

```rust
// Calculate spawn height based on first riffle top
let first_riffle_x = config.slick_plate_len;
let first_riffle_floor = ((width - 1 - first_riffle_x) as f32 * config.slope) as usize;
let first_riffle_top = first_riffle_floor + config.riffle_height;

// Spawn water 2 cells ABOVE the first riffle top
let spawn_y_base = (first_riffle_top as f32 + 2.0) * dx;
```

### Option B: Reduce riffle height from 3 to 2 cells
**Pros:** No code changes to spawn logic
**Cons:** Reduces physical realism of riffles
**Effort:** Trivial
**Risk:** Low - but changes simulation character

```rust
let sluice_config = SluiceConfig {
    riffle_height: 2,  // Reduce from 3 to 2
    // ...
};
```

### Option C: Increase spawn depth (more cells above floor)
**Pros:** Simple
**Cons:** Changes inlet dynamics
**Effort:** Trivial
**Risk:** Medium - may affect inlet flow behavior

```rust
let spawn_y_base = (inlet_floor_y as f32 + 4.0) * dx;  // +4 instead of +2
```

## Recommended Action

Option A - spawn height calculated from riffle geometry

## Technical Details

**Affected files:**
- `crates/sim3d/src/sluice.rs` - spawn_inlet_water function

## Acceptance Criteria

- [ ] Spawn height is at least 2 cells above first riffle top
- [ ] Water visibly flows OVER riffles, not into them
- [ ] Add warning if spawn height <= riffle top

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-03 | Identified by code review | Geometric impossibility |

## Resources

- riffle_diagnostic.rs output shows pooling at x=5
