# Plan: Emitter Placement

## Goal
Allow user to position the water/sediment emitter during gameplay.

## Current State
- `inlet_x = 5.0` hardcoded inside update loop (line 370)
- `inlet_y = (SIM_HEIGHT / 4 - 10)` hardcoded (line 372)
- Mouse left-click already spawns water at cursor (for testing)
- No visual indicator for emitter position

## Design Decision: Right-click to place

**Why right-click:**
- Left-click already used for manual water spawning
- Intuitive "place object" interaction
- No keyboard modifier needed
- Works naturally with existing controls

## Implementation

### Step 1: Move inlet position to mutable variables
Move `inlet_x` and `inlet_y` outside the main loop (near other tunables ~line 145):
```rust
// Inlet position (adjustable via right-click)
let mut inlet_x: f32 = 5.0;
let mut inlet_y: f32 = (SIM_HEIGHT / 4 - 10) as f32;
```

### Step 2: Add right-click to set position
Add input handling (near line 355):
```rust
// Right-click to set emitter position
if is_mouse_button_pressed(MouseButton::Right) {
    let (mx, my) = mouse_position();
    inlet_x = mx / SCALE;
    inlet_y = my / SCALE;
}
```

### Step 3: Clamp inlet position to valid area
Emitter should stay:
- Above the floor (based on slope at that x position)
- Within grid bounds
- Away from right edge (outlet)

```rust
// Clamp to valid bounds
inlet_x = inlet_x.clamp(2.0, (SIM_WIDTH - 50) as f32);
// Calculate floor at inlet_x
let base_floor = (SIM_HEIGHT / 4) as f32;
let floor_at_x = base_floor + (inlet_x - sluice_config.slick_plate_len as f32).max(0.0) * sluice_config.slope;
inlet_y = inlet_y.clamp(5.0, floor_at_x - 5.0);
```

### Step 4: Draw emitter indicator
Add visual marker for emitter position (in render section):
```rust
// Draw emitter position indicator
let emit_screen_x = inlet_x * SCALE;
let emit_screen_y = inlet_y * SCALE;
draw_circle(emit_screen_x, emit_screen_y, 6.0, Color::from_rgba(100, 200, 255, 200));
draw_circle_lines(emit_screen_x, emit_screen_y, 8.0, 2.0, Color::from_rgba(255, 255, 255, 150));
```

### Step 5: Update UI display
Show emitter position in status:
```rust
&format!("EMITTER: ({:.0}, {:.0}) | vx={:.0} vy={:.0} rate={} x{}",
    inlet_x, inlet_y, inlet_vx, inlet_vy, spawn_rate, flow_multiplier),
```

Update help text:
```
"Right-click = Set Emitter | ←→ vx | Shift+←→ vy | ..."
```

## Files Changed
1. `crates/game/src/main.rs`

## Acceptance Criteria
- [ ] Right-click sets emitter position
- [ ] Emitter position clamped to valid area (above floor, within bounds)
- [ ] Visual indicator shows current emitter position
- [ ] UI displays emitter coordinates
- [ ] Water and sediments spawn from new position
- [ ] Position persists across reset (R key)
