# Physics Visual Demo Plan

**Principle:** Tests that pass mean nothing if the visual behavior is wrong. Each physics layer needs visual validation in washplant_editor.

## Current Test Coverage Summary

### Layer 1: SWE (Shallow Water Equations) - `sim3d/tests/swe_physics.rs`
| Test | Status | Visual Demo Needed |
|------|--------|-------------------|
| flat_pool_remains_still | PASS | See: still water in any piece |
| water_flows_downhill | PASS | See: flow down gutters/sluices |
| steady_state_velocity_matches_manning | PASS | Needs measurement overlay |
| mass_conservation_closed_system | PASS | Needs volume counter |
| wave_speed_matches_theory | PASS | Drop test to visualize |
| no_spontaneous_energy_creation | PASS | Implicit in stability |

### Layer 2: Terrain Collapse/Erosion - `sim3d/tests/terrain_physics.rs`, `collapse_test.rs`
| Test | Status | Visual Demo Needed |
|------|--------|-------------------|
| oversteep_slope_collapses | PASS | **Demo 1: Sediment Pile Collapse** |
| sediment_settles_in_still_water | PASS | **Demo 2: Underwater Settling** |
| sediment_advection_conserves_mass | PASS | Implicit |
| erosion_above_velocity_threshold | PASS | **Demo 3: Flow Erosion** |
| flat_terrain_stays_flat | PASS | Implicit |
| bedrock_immune_to_erosion | PASS | Implicit |
| layer_erosion_sequence | PASS | **Demo 4: Layered Terrain** |

### Layer 3: DEM (Discrete Element) - `game/tests/dem_*.rs`
| Test | Status | Visual Demo Needed |
|------|--------|-------------------|
| floor_collision | PASS | **Demo 5: Drop Test** |
| wall_collision | PASS | **Demo 6: Wall Bounce** |
| clump_collision | PASS | **Demo 7: Clump Pool** |
| no_penetration | PASS | Visual - no clumps through floor |
| static_friction | PASS | **Demo 8: Rest on Slope** |
| kinetic_friction | PASS | **Demo 9: Sliding Deceleration** |
| wet_vs_dry_friction | PASS | **Demo 10: Wet/Dry Comparison** |
| settling_time | PASS | Visual in any settling |
| density_separation | PASS | **Demo 11: Gold Separation** |
| angle_of_repose | PASS | Visual pile shape |

### Layer 4: FLIP 3D - `sim3d/tests/transfer_test.rs`, etc.
| Test | Status | Visual Demo Needed |
|------|--------|-------------------|
| momentum_conservation | PASS | Implicit in stable flow |
| pressure_solver | PASS | Implicit in incompressibility |
| g2p/p2g transfer | PASS | Implicit |

---

## Visual Demo Integration Plan

### Phase 1: Basic Washplant Editor Validation

**Goal:** Verify current washplant_editor shows all physics layers working together.

1. Run `cargo run --example washplant_editor --release`
2. Press Space to start simulation
3. Verify:
   - [ ] Water flows down gutters
   - [ ] Water flows into sluices
   - [ ] DEM particles fall and settle
   - [ ] Particles collide with piece geometry
   - [ ] No particles fall through floors

### Phase 2: Add Demo Modes to Washplant Editor

Add keyboard shortcuts to switch demo scenarios:

| Key | Demo | Physics Being Validated |
|-----|------|------------------------|
| F1 | **Sediment Pile** | Angle of repose collapse |
| F2 | **Drop Test** | DEM floor collision + bounce |
| F3 | **Density Separation** | Heavy vs light settling |
| F4 | **Wet vs Dry Friction** | Side-by-side comparison |
| F5 | **Flow Erosion** | Erosion above threshold |
| F6 | **Wall Bounce** | DEM wall collision |

### Phase 3: Individual Demo Descriptions

#### Demo F1: Sediment Pile Collapse
- Start with tall sediment pile at center
- Watch collapse to ~32° angle of repose
- **Expected:** Pile spreads radially until slopes < 32°
- **Test validated:** `test_sediment_collapse_angle`

#### Demo F2: Drop Test (DEM)
- Drop 100 clumps from 1m height
- Watch bounce and settle
- **Expected:** Single bounce ~4cm (e²=0.04), then settle
- **Test validated:** `test_dem_floor_collision`

#### Demo F3: Density Separation
- Mix gold (heavy, yellow) and gangue (light, gray) clumps
- Let settle in container
- **Expected:** Gold sinks to bottom, gangue on top
- **Test validated:** `test_dem_density_separation`

#### Demo F4: Wet vs Dry Friction
- Two identical slopes side-by-side
- Left: dry friction (μ=0.5)
- Right: wet friction (μ=0.08)
- Release clumps simultaneously
- **Expected:** Wet clumps slide 2x+ farther
- **Test validated:** `test_dem_wet_vs_dry_friction`

#### Demo F5: Flow Erosion
- Sloped terrain with sediment
- Water flows over
- **Expected:** No erosion below threshold, erosion above
- **Test validated:** `test_erosion_above_velocity_threshold`

#### Demo F6: Wall Bounce
- Launch clumps at wall
- **Expected:** Bounce back with v_x damped by restitution
- **Test validated:** `test_dem_wall_collision`

---

## Implementation Priority

1. **Verify existing washplant_editor works** (Phase 1)
2. **Add F1: Sediment Pile** - simplest, validates terrain collapse
3. **Add F2: Drop Test** - validates DEM collision
4. **Add F3: Density Separation** - core gold mining physics
5. Remaining demos as needed

---

## Debug Checklist

When visual behavior doesn't match test:

1. **Print SDF values** in `collision_response_only` - are they negative inside geometry?
2. **Check coordinate transforms** - world space vs grid space
3. **Verify grid_offset** - is it being applied correctly?
4. **Check piece SDFs** - does each piece have its own grid?
5. **Run cargo test** - does the unit test still pass?

The visual is the truth. If it doesn't look right, the test is missing something.
