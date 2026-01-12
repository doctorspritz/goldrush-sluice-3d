# Physics Visual Validation - Complete Test Coverage

**Rule:** No test is accepted until you can SEE it working.

---

## Test Categories & Visual Examples

### Category 1: SWE Physics (6 tests) → `swe_visual.rs`

| Test | What to See |
|------|-------------|
| `flat_pool_remains_still` | Still water, no ripples, no drift |
| `water_flows_downhill` | Water moves from high to low |
| `steady_state_velocity_matches_manning` | Consistent flow speed on slope |
| `mass_conservation_closed_system` | Water volume counter stays constant |
| `wave_speed_matches_theory` | Drop creates wave, measure travel time |
| `no_spontaneous_energy_creation` | Still water stays still forever |

### Category 2: Terrain Collapse (6 tests) → `collapse_visual.rs`

| Test | What to See |
|------|-------------|
| `test_sediment_collapse_angle` | Pile spreads to ~32° slopes |
| `test_sediment_uses_sand_angle_regardless_of_substrate` | Same angle on different bases |
| `test_flat_surface_stable` | Flat sediment doesn't move |
| `test_slope_threshold` | Steep collapses, gentle stays |
| `test_multiple_material_angles` | Different materials, different angles |
| `test_collapse_mass_conservation` | Volume counter stays constant |

### Category 3: Terrain Erosion (11 tests) → `erosion_visual.rs`

| Test | What to See |
|------|-------------|
| `flat_terrain_stays_flat` | Flat ground with no flow doesn't erode |
| `oversteep_slope_collapses_to_angle_of_repose` | Cliff collapses to stable angle |
| `collapse_conserves_mass_all_materials` | Total volume constant during collapse |
| `no_erosion_below_critical_velocity` | Slow water doesn't erode |
| `erosion_above_velocity_threshold` | Fast water erodes |
| `material_specific_erosion_rates` | Hard rock erodes slower than dirt |
| `bedrock_immune_to_erosion` | Bedrock never erodes |
| `soft_layers_erode_first` | Dirt erodes before gravel |
| `hard_layers_protect_beneath` | Gravel cap protects dirt below |
| `layer_erosion_sequence` | Layers erode top-to-bottom |
| `total_solid_volume_constant_during_collapse` | No mass created/destroyed |

### Category 4: Sediment Transport (5 tests) → `sediment_visual.rs`

| Test | What to See |
|------|-------------|
| `sediment_settles_in_still_water` | Particles sink in still water |
| `settling_rate_matches_parameters` | Heavy settles faster than light |
| `suspended_sediment_advects_with_flow` | Sediment moves with water |
| `sediment_transport_capacity_velocity_dependent` | Fast water carries more |
| `sediment_advection_conserves_mass` | Total sediment constant |
| `mass_conservation_through_erosion_deposition_cycle` | Pick up and drop same amount |

### Category 5: DEM Collision (5 tests) → `dem_collision_visual.rs`

| Test | What to See |
|------|-------------|
| `test_dem_floor_collision` | Clump drops, bounces, settles |
| `test_dem_wall_collision` | Clump hits wall, bounces back |
| `test_dem_clump_collision` | Two clumps collide, both bounce |
| `test_dem_collision_no_penetration` | 100 clumps, none through floor |
| `test_dem_collides_with_gutter_sdf` | Clumps stay in gutter shape |

### Category 6: DEM Friction (4 tests) → `dem_friction_visual.rs`

| Test | What to See |
|------|-------------|
| `test_dem_static_friction` | Clump at rest on slope stays still |
| `test_dem_kinetic_friction` | Moving clump slows down |
| `test_dem_wet_vs_dry_friction` | Wet clump slides farther |
| `test_dem_friction_finite` | Fast clump doesn't instant-stop |

### Category 7: DEM Settling (3 tests) → `dem_settling_visual.rs`

| Test | What to See |
|------|-------------|
| `test_dem_settling_time` | Dropped clumps settle within 5s |
| `test_dem_density_separation` | Gold sinks below gangue |
| `test_dem_angle_of_repose` | Clump pile has ~30° slopes |

### Category 8: DEM-Water Coupling (3 tests) → `dem_water_visual.rs`

| Test | What to See |
|------|-------------|
| `test_gold_settles_faster_than_sand_in_water` | Gold sinks faster in water |
| `test_water_drag_slows_clumps` | Clumps slow down in water |
| `test_flip_and_dem_together` | Water + clumps both work |

### Category 9: FLIP 3D Transfer (6 tests) → Internal, no visual needed

These test internal grid math - validated by Categories 1-8 working.

### Category 10: Water Flow (7 tests) → `water_flow_visual.rs`

| Test | What to See |
|------|-------------|
| `sediment_moves_downstream_with_water` | Sediment goes with flow |
| `test_excavation_no_water_creation` | Digging doesn't create water |
| `test_excavation_under_water_no_phantom_mass` | Underwater dig doesn't duplicate |
| `test_water_surface_clamped_to_ground` | No water below ground |
| `test_water_volume_calculation` | Volume counter accurate |
| `test_cpu_water_flows_downhill` | Water goes downhill |
| `test_cpu_water_mass_conservation` | Water volume constant |
| `test_cpu_water_spreads_on_flat_terrain` | Water spreads evenly |

---

## Visual Example Files to Create

```
crates/game/examples/
├── visual_swe.rs              # SWE physics demos
├── visual_collapse.rs         # Terrain collapse demos
├── visual_erosion.rs          # Erosion demos
├── visual_sediment.rs         # Sediment transport demos
├── visual_dem_collision.rs    # DEM collision demos
├── visual_dem_friction.rs     # DEM friction demos
├── visual_dem_settling.rs     # DEM settling demos
├── visual_dem_water.rs        # DEM-water coupling demos
└── visual_water_flow.rs       # Water flow demos
```

Each file should:
1. Have keyboard controls to switch between tests (1-9 keys)
2. Display which test is being shown
3. Show relevant metrics on screen (volume, velocity, etc.)
4. Reset with R key
5. Be runnable standalone: `cargo run --example visual_swe --release`

---

## Implementation Order

1. **visual_dem_collision.rs** - Most fundamental, clumps + SDF
2. **visual_dem_friction.rs** - Friction is core to sluice behavior
3. **visual_dem_settling.rs** - Gold separation is the goal
4. **visual_swe.rs** - Water flow basics
5. **visual_sediment.rs** - Sediment in water
6. **visual_erosion.rs** - Terrain modification
7. **visual_collapse.rs** - Angle of repose
8. **visual_dem_water.rs** - Full coupling
9. **visual_water_flow.rs** - Edge cases

---

## Acceptance Criteria

For each visual example:

- [ ] I can run it
- [ ] I can see the physics happening
- [ ] The behavior matches what the test claims
- [ ] It works in washplant_editor context too
