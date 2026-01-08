# Sediment Collision Not Working (PR-2)

## Status: UNSOLVED - Known Issue

## Problem

Sediment particles don't stack or have collision. They form massive piles that appear unaffected by water and don't interact with each other properly.

## Root Cause Analysis

Diagnostics reveal that **sediment particles float above grid cells** and are never captured by the P2G (particle-to-grid) transfer:

```
Riffle area diagnostics:
- Bed height: 0.280-0.287 units
- Sediment avg_y: 0.343 units (0.06 units ABOVE bed)
- All grid cells show: sed=0 (ZERO sediment particles in cells)
```

The voxel-based jamming system implemented in this PR requires particles to be INSIDE grid cells to count them and mark cells as SOLID. Since sediment floats in continuous space between cells, the collision system cannot activate.

## What Was Tried

### 1. Velocity-Based Corrections (REJECTED)
- Added velocity corrections to `sediment_density_correct_3d.wgsl`
- Result: "sand behaving like fluid" - made problem worse
- User explicitly rejected this approach

### 2. Voxel-Based Collision System (IMPLEMENTED BUT INACTIVE)
Successfully implemented but cannot activate due to root cause:

**Files created/modified:**
- `sediment_cell_type_3d.wgsl` - Marks cells SOLID when sediment dominates
- `sdf_collision_3d.wgsl` - Prevents particles entering SOLID cells
- `flip_3d.rs` - 5-iteration bottom-up jamming propagation
- Added `has_support_below()` to prevent mid-air jamming
- Added sediment-to-water dominance check for mixed cells

**Why it doesn't work:** Zero sediment particles in grid cells means jamming criteria `sed_count >= 3` never triggers.

### 3. Settling Investigation (REVERTED - BROKE PHYSICS)
Attempted to investigate why sediment doesn't settle:

- Disabled vorticity lift (`sediment_vorticity_lift: 0.0`)
- Removed yield-based settling reduction in g2p_3d.wgsl
- Increased settling velocity

**Result:** Completely broke sediment physics - particles disappeared entirely. All changes reverted.

## Current State

- Simulation runs without crashes
- Voxel-based collision infrastructure is in place
- Diagnostics print every 30 frames showing cell types and particle counts
- Original sediment physics restored (`sediment_vorticity_lift: 0.08`)
- **Problem remains unsolved**

## Diagnostics Added

The `print_jamming_diagnostics()` method in `flip_3d.rs` outputs:
1. Cell type counts (SOLID/FLUID/AIR)
2. Vorticity statistics and lift cancellation percentage
3. Sample cells in riffle area showing sed/wat counts
4. Riffle column support chain with vorticity per cell

## Possible Directions (Not Attempted)

1. **Debug P2G sediment counting** - Why doesn't `atomicAdd(&sediment_count[home_idx], 1)` capture particles?
2. **Check particle density values** - P2G uses `if (density > 1.0)` to identify sediment
3. **Investigate grid cell sizing** - Are particles too small relative to cell size?
4. **Dual-grid approach** - Separate finer grid for sediment collision
5. **Direct particle-particle collision** - Skip grid entirely for sediment

## Key Files

- `crates/game/src/gpu/shaders/sediment_cell_type_3d.wgsl` - Jamming logic
- `crates/game/src/gpu/shaders/sdf_collision_3d.wgsl` - Collision enforcement
- `crates/game/src/gpu/shaders/p2g_scatter_3d.wgsl` - P2G sediment counting (line 108-110)
- `crates/game/src/gpu/shaders/g2p_3d.wgsl` - Drucker-Prager physics
- `crates/game/src/gpu/flip_3d.rs` - Main simulation, diagnostics

## Buffer Flags Fixed

These buffers needed COPY_SRC for diagnostic readback:
- `pressure_3d.rs`: cell_type_buffer
- `p2g_3d.rs`: sediment_count_buffer, particle_count_buffer
- `flip_3d.rs`: vorticity_mag_buffer
