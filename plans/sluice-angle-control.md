# Plan: Sluice Angle Control

## Goal
Allow user to adjust sluice slope angle during gameplay via keyboard controls.

## Current State
- `SluiceConfig.slope` exists and is configurable (currently hardcoded to 0.12 in game.rs)
- Other sluice params (riffle spacing/height/width) already have keyboard controls (Q/A, W/S, E/D)
- `riffle_dirty` flag triggers terrain rebuild - already works
- Slope is "rise per cell" where 0.25 ≈ 14 degrees

## Implementation

### Step 1: Add slope keyboard controls
**File:** `crates/game/src/main.rs`

Add controls following existing pattern (near line 230-260):
```rust
// Slope control: Z/X keys (next to other sluice controls)
if is_key_pressed(KeyCode::Z) {
    sluice_config.slope = (sluice_config.slope - 0.02).max(0.0);
    riffle_dirty = true;
}
if is_key_pressed(KeyCode::X) {
    sluice_config.slope = (sluice_config.slope + 0.02).min(0.5);
    riffle_dirty = true;
}
```

**Slope limits:**
- Min: 0.0 (flat)
- Max: 0.5 (~27 degrees, steeper gets unrealistic)
- Step: 0.02 (about 1 degree increments)

### Step 2: Display current slope in UI
**File:** `crates/game/src/main.rs`

Add to the debug text display (near line 460):
```rust
&format!("Slope: {:.0}° (Z/X)", sluice_config.slope.atan().to_degrees()),
```

### Step 3: Adjust inlet Y position with slope
When slope changes significantly, the floor height at inlet changes. The inlet should stay above the floor.

**File:** `crates/game/src/main.rs`

Make inlet_y dynamic based on floor height at inlet_x:
```rust
let floor_at_inlet = (SIM_HEIGHT / 4) as f32; // base floor height
let inlet_y = floor_at_inlet - 10.0;          // 10 cells above floor
```

This is already roughly correct but should be verified.

### Step 4: Test and tune
- Verify shallow angles (0.05) still flow
- Verify steep angles (0.4) don't break physics
- Check gold settling behavior at different angles
- Ensure riffles still render correctly at extremes

## Files Changed
1. `crates/game/src/main.rs` - keyboard handling + UI display

## Acceptance Criteria
- [ ] Z key decreases slope, X key increases slope
- [ ] Slope displayed in UI with degree value
- [ ] Terrain rebuilds smoothly when slope changes
- [ ] Water flows correctly at min/max slopes
- [ ] Riffles render correctly at all angles

## Future: Visual Angle Indicator
Later enhancement: draw a small angle arc or slope line in corner of screen showing current angle visually.

---

## Next Step After This: Emitter Placement

Once angle control works, emitter placement is next. Current state:
- Inlet position hardcoded: `inlet_x = 5.0`, `inlet_y = floor - 10`
- Will need: mouse click to set position, or arrow key adjustment
- Consider: spawn rate tied to position? Multiple emitters?
