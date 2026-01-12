# Visual Test Harness - INTEGRATED INTO WASHPLANT_EDITOR

## How to Use

```bash
cargo run --example washplant_editor --release
```

Then:
1. Press `T` to enter TEST MODE
2. Press `1-9` or `0` to select a test
3. Watch particles and verify behavior matches PASS criteria
4. Press `T` or `ESC` to exit test mode

## Test Definitions

All tests run on the SAME simulation infrastructure as normal editor mode.

| Key | Test Name | What You Should See | PASS | FAIL |
|-----|-----------|---------------------|------|------|
| 1 | DEM: Floor Collision | Particles fall, bounce, settle | All rest ON floor (y > 0) | Clip through floor or hover |
| 2 | DEM: Wall Collision | Particles hit walls, bounce back | Reflect off walls, stay inside | Pass through walls |
| 3 | DEM: Density Separation | Gold (yellow) + sand (gray) in water | Yellow sinks below gray | Same height or gold on top |
| 4 | DEM: Settling Time | 50 particles dropped | Motion stops in 5s | Still bouncing after 5s |
| 5 | Fluid: Flow Downhill | Water released on tilted gutter | Flows from high to low end | Stuck or wrong direction |
| 6 | Fluid: Pool Equilibrium | Flat pool of water | Surface flat, no motion | Ripples or spontaneous energy |
| 7 | Fluid: Wall Containment | Water in gutter | Stays between walls | Leaks through walls/floor |
| 8 | Sediment: Settle in Still | Drop sediment into still pool | Particles descend, rest on floor | Float, stuck, or teleport |
| 9 | Sediment: Transport by Flow | Sediment in flowing water | Particles move with flow | Stuck or move against flow |
| 0 | Integration: Sluice Capture | Gold+sand over sluice riffles | Gold behind riffles, sand washes | All wash through or all stuck |

## Implementation

Tests are integrated directly into `crates/game/examples/washplant_editor.rs`:

- `VisualTest` struct defines each test with key, name, expect, watch, category
- `toggle_test_mode()` enters/exits test mode
- `run_test(idx)` sets up and runs selected test
- `setup_test_scenario(idx)` creates appropriate layout and starts simulation

Window title shows: `TEST X: Name | What to Watch`
Console shows: Detailed PASS/FAIL criteria

## Why This Approach

Tests run on the EXACT same simulation infrastructure as normal editor mode:
- Same `FlipSimulation3D` for fluid
- Same `GpuFlip3D` for GPU acceleration
- Same `ClusterSimulation3D` for DEM particles
- Same piece geometry (gutters, sluices)
- Same collision detection (SDF)

If a test passes here, it works in the real simulation.
If a test fails here, fix the simulation code - don't write workaround tests.
