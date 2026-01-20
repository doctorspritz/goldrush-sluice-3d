# UI Overlay Implementation Plan

## Summary
Add a Full Debug UI overlay using **egui** to the goldrush sluice 3D simulation.

## Dependencies
Add to `crates/game/Cargo.toml`:
```toml
egui = "0.29"
egui-wgpu = "0.29"
egui-winit = "0.29"
```

## New Files

### `crates/game/src/debug_ui/mod.rs`
```rust
pub mod state;
pub mod panels;
pub mod integration;

pub use state::DebugUiState;
pub use integration::EguiIntegration;
```

### `crates/game/src/debug_ui/state.rs`
- `DebugUiState` struct with all UI-bound values
- Panel open/closed toggles
- Stats (FPS, particle counts)
- Controls (paused, emit rates)
- Physics params (FLIP, sediment, gold, DEM)
- Diagnostics (divergence, pressure, cell counts)

### `crates/game/src/debug_ui/panels.rs`
Panel drawing functions:
- `draw_stats_panel()` - FPS, frame#, particle counts
- `draw_controls_panel()` - Pause/Reset buttons, rate sliders
- `draw_toggles_panel()` - DEM, Sorted P2G, Async Readback checkboxes
- `draw_flow_panel()` - Velocity, depth, flow rate metrics
- `draw_physics_panel()` - Collapsible: FLIP, Sediment, Gold, DEM params
- `draw_diagnostics_panel()` - Divergence, pressure, cell type stats

### `crates/game/src/debug_ui/integration.rs`
`EguiIntegration` struct:
- `new()` - Initialize egui context, winit state, wgpu renderer
- `on_window_event()` - Process winit events, return if consumed
- `begin_frame()` - Start egui frame
- `end_frame_and_render()` - Tessellate and render UI overlay

## Modifications to Existing Files

### `crates/game/src/lib.rs`
```rust
pub mod debug_ui;
```

### `crates/game/src/main.rs`

1. **Add to App struct:**
   - `egui: Option<EguiIntegration>`
   - `debug_ui_state: DebugUiState`

2. **In `init_gpu()`:** Initialize EguiIntegration

3. **In `window_event()`:**
   - Call `egui.on_window_event()` first
   - If consumed, skip normal event handling (except RedrawRequested)
   - Add F1 key to toggle `debug_ui_state.ui_visible`

4. **Add methods:**
   - `sync_ui_state()` - Copy simulation state to UI state
   - `apply_ui_state()` - Apply UI changes back to simulation

5. **In `render()`:**
   - After 3D render pass, if UI visible:
   - Call `sync_ui_state()`
   - `egui.begin_frame()`
   - Draw all panels
   - `egui.end_frame_and_render()` (separate pass, LoadOp::Load, no depth)
   - Call `apply_ui_state()`

## Implementation Order

1. Add dependencies, create module skeleton, verify compile
2. Implement EguiIntegration, add to App, init in init_gpu()
3. Wire up event handling with egui consumption
4. Implement DebugUiState + stats/controls panels
5. Add toggles, flow metrics, physics params panels
6. Integrate diagnostics display
7. Test all interactions

## Key Design Decisions

- **Overlay rendering**: UI renders after 3D scene with `LoadOp::Load` (no clear)
- **No depth buffer**: UI pass has no depth attachment
- **Event interception**: egui processes events first, passes through if not consumed
- **Bi-directional sync**: State flows simulation→UI for display, UI→simulation for changes

## Verification

1. F1 toggles UI visibility
2. Mouse drag rotates camera when not over UI panels
3. UI panels block mouse interaction with 3D scene
4. Sliders modify simulation in real-time
5. Pause/Reset buttons work
6. FPS remains smooth with UI visible
