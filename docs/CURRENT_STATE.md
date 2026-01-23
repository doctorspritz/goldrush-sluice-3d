# Current System State

**Date:** 2026-01-10
**Primary Example:** `washplant_editor` (canonical)
**Branch:** `master`

---

**Note:** Metrics and parameter snapshots below were captured from the now-archived
`friction_sluice` example. Treat them as historical reference only.

## Working Components

### 1. GPU FLIP 3D Simulation
- **Status:** WORKING
- **Files:** `crates/game/src/gpu/flip_3d.rs`, `crates/sim3d/`
- **Metrics:** 22-24 FPS at ~11k particles, scales to 300k max
- **Notes:** MGPCG pressure solver with 80 iterations (default), 2 substeps

### 2. Water Physics
- **Status:** WORKING
- **Evidence:** Divergence stable at 0.3-1.4, flow over riffles works
- **Notes:** Only water particles mark cells as Fluid (sediment is purely Lagrangian)

### 3. Sluice Geometry / SDF
- **Status:** WORKING
- **Files:** `crates/game/src/sluice_geometry.rs`
- **Evidence:** `Sluice: Marked 109670 solid cells`, SDF min/max correct
- **Notes:** Riffle geometry, walls, exit zones all working

### 4. Flow Measurement System
- **Status:** WORKING
- **Evidence:** Reports velocity (0.36-0.46 m/s), depth (p50=0.01m), width, Q
- **Notes:** Window-based sampling in [0.06m, 0.36m] range

### 5. Screen-Space Fluid Rendering
- **Status:** WORKING
- **Files:** `crates/game/src/gpu/fluid_renderer.rs`
- **Notes:** Recently implemented, replaces older point-sprite rendering

### 6. DEM Sediment Transport
- **Status:** PARTIALLY WORKING
- **Evidence:** Particles spawn, move with water via drag
- **Files:** `ClusterSimulation3D` in `crates/sim3d/`
- **Notes:** One-way coupling only - sediment feels water, water ignores sediment

### 7. Tracer Particles
- **Status:** WORKING
- **Evidence:** `Tracer spawned at frame 0 (idx 202, 203, 204)`
- **Notes:** Spawn at configurable intervals for flow visualization

---

## Broken Components

### 1. Sediment Collision/Stacking
- **Status:** UNSOLVED
- **Documented:** `docs/solutions/simulation-issues/sediment-collision-not-working.md`
- **Root Cause:** Sediment particles float above grid cells (avg_y 0.06 units above bed), never captured by P2G transfer
- **Effect:** Particles form piles without stacking, no particle-particle collision
- **Infrastructure:** Voxel-based jamming system is IMPLEMENTED but cannot activate (zero sediment in grid cells)
- **Possible Directions:**
  - Debug P2G sediment counting
  - Check particle density values (P2G uses `density > 1.0` to identify sediment)
  - Investigate grid cell sizing relative to particle size
  - Dual-grid approach for sediment
  - Direct particle-particle collision (skip grid)

### 2. Two-Way Fluid-Sediment Coupling
- **Status:** NOT IMPLEMENTED
- **Current:** One-way only - sediment experiences drag from water, water doesn't feel reaction
- **Effect:** Violates Newton's third law, momentum disappears from system
- **Notes:** Intentional design choice to keep pressure solver stable

---

## Reverted/Failed Attempts

### 1. GPU-Owned Particles Optimization
- **Status:** REVERTED
- **Documented:** `docs/solutions/gpu-owned-particles-failure.md`
- **What Happened:** Attempted to keep particles GPU-resident, broke FLIP velocity continuity
- **Result:** Water became static jello
- **Lesson:** FLIP requires CPU-GPU velocity sync between frames

### 2. Velocity-Based Sediment Corrections
- **Status:** REJECTED BY USER
- **What Happened:** Added velocity corrections to sediment density shader
- **Result:** "Sand behaving like fluid"

### 3. Settling Investigation (Vorticity Lift Changes)
- **Status:** REVERTED
- **What Happened:** Disabled vorticity lift, modified g2p settling
- **Result:** Particles disappeared entirely

---

## Unknown / Needs Investigation

| Component | Status | Notes |
|-----------|--------|-------|
| Bed heightfield system | UNKNOWN | May exist in worktree, not in master |
| Erosion mechanics | NOT IMPLEMENTED | Referenced in protocol but not found |
| Water volume compensation | NOT IMPLEMENTED | Deleted particles not replaced |
| Shaker deck FLIP coupling | IN PROGRESS | Has uncommitted changes |

---

## Key Parameters (friction_sluice, archived)

| Parameter | Value | Status |
|-----------|-------|--------|
| CELL_SIZE | 0.01 | LOCKED |
| PRESSURE_ITERS | 80 | TUNABLE (quality) |
| SUBSTEPS | 2 | LOCKED |
| GRAVITY | -9.8 | LOCKED |
| WATER_EMIT_RATE | 200 | TUNABLE |
| SEDIMENT_EMIT_RATE | 2 | TUNABLE |
| MAX_PARTICLES | 300,000 | TUNABLE |
| GPU_SYNC_STRIDE | 4 | TUNABLE |

---

## Project Structure

```
crates/
  game/               # GPU visual simulation, wgpu rendering
    src/gpu/
      flip_3d.rs      # Main GPU FLIP implementation
      fluid_renderer.rs  # Screen-space water rendering
      mgpcg.rs        # Multigrid pressure solver
      shaders/        # WGSL compute shaders
    examples/
      washplant_editor.rs  # Canonical example (sluice editor)
      world_sim.rs         # Canonical example (world)
  sim3d/              # 3D simulation library
    src/
      lib.rs          # Entry point
      world.rs        # 2.5D World Stack

docs/
  solutions/          # Documented problems and solutions
  research/           # Technical research
  compound/           # Failure post-mortems
```

---

## Runtime Observations (friction_sluice, archived; 8 second run)

```
Frame 1   | FPS: 1.0  | Particles: 205 (water: 203, sediment: 2)
Frame 21  | FPS: 19.4 | Particles: 2225 (water: 2203, sediment: 22)
Frame 44  | FPS: 22.6 | Particles: 4245 (water: 4203, sediment: 42)
Frame 68  | FPS: 23.3 | Particles: 6669 (water: 6603, sediment: 66)
Frame 92  | FPS: 24.0 | Particles: 9093 (water: 9003, sediment: 90)
Frame 116 | FPS: 24.0 | Particles: 11517 (water: 11403, sediment: 114)

Flow metrics: v=0.36 m/s | depth p50=0.010 m | width=0.38 m | Q=0.001 m3/s
Gravel velocity: FLIP=0.47 m/s | DEM=0.45 m/s
```

**No crashes, no NaN, simulation stable.**

---

## Next Priority

**Debug why sediment particles don't register in grid cells during P2G.**

This is blocking:
- Sediment collision/stacking
- Bed formation
- Proper gold/gangue separation

The voxel-based jamming infrastructure is already in place - it just needs particles to actually appear in the grid.
