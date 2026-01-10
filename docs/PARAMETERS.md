# Parameters Reference

**Date:** 2026-01-10
**Status:** Initial documentation

---

## Legend

| Status | Meaning |
|--------|---------|
| **LOCKED** | Known-good value. DO NOT CHANGE without re-running full test suite. |
| **TUNABLE** | Can be adjusted for testing/tuning. Document changes. |
| **UNKNOWN** | Purpose unclear or value unvalidated. Needs investigation. |

---

## Grid Parameters (friction_sluice)

| Parameter | Value | File:Line | Purpose | Status |
|-----------|-------|-----------|---------|--------|
| `CELL_SIZE` | 0.01 | friction_sluice.rs:26 | Size of each grid cell in meters | **LOCKED** |
| `GRID_WIDTH` | 162 | friction_sluice.rs:29 | Grid X dimension (cells) | TUNABLE |
| `GRID_HEIGHT` | 52 | friction_sluice.rs:31 | Grid Y dimension (cells) | TUNABLE |
| `GRID_DEPTH` | 40 | friction_sluice.rs:30 | Grid Z dimension (cells) | TUNABLE |
| `MAX_PARTICLES` | 300,000 | friction_sluice.rs:32 | Maximum particle count | TUNABLE |

---

## Simulation Parameters (friction_sluice)

| Parameter | Value | File:Line | Purpose | Status |
|-----------|-------|-----------|---------|--------|
| `GRAVITY` | -9.8 | friction_sluice.rs:35 | Gravitational acceleration (m/s²) | **LOCKED** |
| `PRESSURE_ITERS` | 120 | friction_sluice.rs:36 | MGPCG pressure solver iterations | **LOCKED** |
| `SUBSTEPS` | 2 | friction_sluice.rs:37 | Physics substeps per frame | **LOCKED** |
| `GPU_SYNC_STRIDE` | 4 | friction_sluice.rs:44 | Frames between GPU readbacks | TUNABLE |

---

## Emission Parameters (friction_sluice)

| Parameter | Value | File:Line | Purpose | Status |
|-----------|-------|-----------|---------|--------|
| `WATER_EMIT_RATE` | 200 | friction_sluice.rs:42 | Water particles emitted per frame | TUNABLE |
| `SEDIMENT_EMIT_RATE` | 2 | friction_sluice.rs:43 | Sediment particles emitted per frame | TUNABLE |
| `TRACER_INTERVAL_FRAMES` | 300 | friction_sluice.rs:38 | Frames between tracer spawns | TUNABLE |
| `TRACER_COUNT` | 3 | friction_sluice.rs:39 | Tracers spawned per interval | TUNABLE |

---

## Sediment Material Parameters (friction_sluice)

| Parameter | Value | File:Line | Purpose | Status |
|-----------|-------|-----------|---------|--------|
| `GANGUE_RADIUS_CELLS` | 0.12 | friction_sluice.rs:47 | Gangue grain radius (× cell_size) | TUNABLE |
| `GOLD_RADIUS_CELLS` | 0.02 | friction_sluice.rs:48 | Gold grain radius (× cell_size) | TUNABLE |
| `GANGUE_DENSITY` | 2.7 | friction_sluice.rs:49 | Gangue density (g/cm³) | **LOCKED** |
| `GOLD_DENSITY` | 19.3 | friction_sluice.rs:50 | Gold density (g/cm³) | **LOCKED** |
| `GOLD_FRACTION` | 0.05 | friction_sluice.rs:51 | Fraction of sediment that is gold | TUNABLE |

---

## GPU FLIP Physics Parameters (flip_3d.rs)

### Vorticity Confinement

| Parameter | Default | File:Line | Purpose | Status |
|-----------|---------|-----------|---------|--------|
| `vorticity_epsilon` | 0.05 | flip_3d.rs:2374 | Vorticity confinement strength | TUNABLE |

### Sediment Physics

| Parameter | Default | File:Line | Purpose | Status |
|-----------|---------|-----------|---------|--------|
| `sediment_rest_particles` | 8.0 | flip_3d.rs:2375 | Target particles per cell for density projection | **LOCKED** |
| `sediment_friction_threshold` | 0.1 | flip_3d.rs:2376 | Flow velocity threshold for friction activation | TUNABLE |
| `sediment_friction_strength` | 0.5 | flip_3d.rs:2377 | Friction damping coefficient | TUNABLE |
| `sediment_settling_velocity` | 0.05 | flip_3d.rs:2378 | Base settling velocity (m/s) | TUNABLE |
| `sediment_vorticity_lift` | 1.5 | flip_3d.rs:2379 | Lift coefficient from vorticity | TUNABLE |
| `sediment_vorticity_threshold` | 3.0 | flip_3d.rs:2380 | Vorticity threshold for lift activation | TUNABLE |
| `sediment_drag_coefficient` | 6.0 | flip_3d.rs:2381 | Water-sediment drag coefficient | TUNABLE |
| `sediment_porosity_drag` | 3.0 | flip_3d.rs:2386 | Porosity-based flow reduction | TUNABLE |

### Gold-Specific Physics

| Parameter | Default | File:Line | Purpose | Status |
|-----------|---------|-----------|---------|--------|
| `gold_density_threshold` | 10.0 | flip_3d.rs:2382 | Density above which gold physics applies | TUNABLE |
| `gold_drag_multiplier` | 1.0 | flip_3d.rs:2383 | Drag multiplier for gold particles | TUNABLE |
| `gold_settling_velocity` | 0.02 | flip_3d.rs:2384 | Gold-specific settling velocity (m/s) | TUNABLE |
| `gold_flake_lift` | 0.0 | flip_3d.rs:2385 | Additional lift for gold flakes | TUNABLE |

---

## Density Projection Parameters (flip_3d.rs)

| Parameter | Value | File:Line | Purpose | Status |
|-----------|-------|-----------|---------|--------|
| `rest_density` | 8.0 | flip_3d.rs:3219 | Target particles per cell | **LOCKED** |
| `density_iterations` | 40 | flip_3d.rs:3256 | Jacobi iterations for density projection | TUNABLE |

---

## Sluice Geometry Parameters (sluice_geometry.rs)

| Parameter | Default | File:Line | Purpose | Status |
|-----------|---------|-----------|---------|--------|
| `grid_width` | 160 | sluice_geometry.rs:136 | Sluice length in cells | TUNABLE |
| `grid_height` | 24 | sluice_geometry.rs:137 | Sluice height in cells | TUNABLE |
| `grid_depth` | 24 | sluice_geometry.rs:138 | Sluice width in cells | TUNABLE |
| `cell_size` | 0.25 | sluice_geometry.rs:139 | Cell size for geometry (note: different from sim) | UNKNOWN |
| `floor_height_left` | 10 | sluice_geometry.rs:140 | Upstream floor height (cells) | TUNABLE |
| `floor_height_right` | 3 | sluice_geometry.rs:141 | Downstream floor height (cells) | TUNABLE |
| `riffle_spacing` | varies | sluice_geometry.rs | Distance between riffles | TUNABLE |
| `riffle_height` | varies | sluice_geometry.rs | Riffle height above floor | TUNABLE |
| `riffle_thickness` | varies | sluice_geometry.rs | Riffle X thickness | TUNABLE |

---

## Physical Constants

| Constant | Value | Source | Notes |
|----------|-------|--------|-------|
| Shields critical | 0.045 | Literature | Onset of sediment motion |
| Water density | 1000 kg/m³ | Physical | Reference for buoyancy |
| Gangue density | 2700 kg/m³ | Physical | Typical rock density |
| Gold density | 19300 kg/m³ | Physical | Pure gold density |

---

## Known Interactions

### Cell Size ↔ Pressure Iterations
- Smaller cells require more pressure iterations
- CELL_SIZE=0.01 + PRESSURE_ITERS=120 is validated stable
- DO NOT reduce iterations without testing divergence

### Rest Density ↔ Volume Conservation
- `sediment_rest_particles` and `rest_density` should match (both 8.0)
- Changing affects water level and pile height
- Too low: particles spread too thin
- Too high: explosive compression

### Vorticity Lift ↔ Settling
- `sediment_vorticity_lift` counteracts `sediment_settling_velocity`
- Balance determines suspension vs. settling behavior
- Currently set for moderate suspension in flow

---

## Parameters NOT Centralized (TODO)

The following parameters exist in multiple places with potentially different values:

1. **dt (timestep)** - computed from SUBSTEPS and frame rate, not a single constant
2. **FLIP ratio** - hardcoded in g2p_3d.wgsl, not exposed to Rust
3. **Pressure solver omega** - inside MGPCG shaders
4. **Collision SDF padding** - in multiple shader files

These should be consolidated in Phase 2 cleanup.
