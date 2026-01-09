# Heightfield LOD + Streaming Prototype (GPU-First)

This folder is intentionally separate from the main workspace. It documents and prototypes a tiled, multi-LOD 2.5D heightfield sim suitable for a 2km x 2km world at 0.2m base resolution on an M2 Max (32GB unified).

## Goals
- Keep near-field at 0.2m square cells where activity is high.
- Allow the rest of the world to remain live, but at lower cadence/LOD.
- Run the sim on GPU compute and stream tiles on demand.
- Preserve mass continuity (water + sediment) across LOD boundaries.

## Non-Goals
- Integrating with the current game/sim crates.
- Rendering polish or final shaders.
- Full physics correctness beyond the heightfield model.

## Constraints
- 2km x 2km at 0.2m would be 100M cells. A single grid is too large.
- M2 Max 32GB unified memory: assume 3-6GB sim budget.

## Proposed Architecture
- Multi-LOD tiled heightfield. Each LOD uses a fixed tile resolution (e.g., 256x256 cells).
- Activity-driven tile promotion/demotion, plus a small camera ring pinned at L0.
- Slow-update coarse baseline for the full world (L3/L4), faster updates in active zones.
- GPU compute pipeline per tile (surface, flux, depth, erosion) plus cross-LOD upsample/downsample.

## Data Layout (Per Tile)
Per-cell buffers (current GPU heightfield uses 11 f32 arrays):
- bedrock, paydirt, overburden, sediment
- water_depth, vel_x, vel_z, suspended_sediment
- water_surface, flux_x, flux_z

Suggested packing options:
- RGBA32F textures: group 4 fields per texture for cache-friendly access.
- Use 16F where safe (velocities/suspended) to reduce memory.
- Double-buffer only the fields that need ping-pong (water/flux).

## LOD Stack (Example)
- L0: 0.2m cell
- L1: 0.4m cell
- L2: 0.8m cell
- L3: 1.6m cell (full world baseline)

With 256x256 tiles:
- L0 tile size: 51.2m
- L3 tile size: 409.6m (approx 5x5 tiles for 2km)

## Activity-Driven LOD
Compute per-tile activity on GPU, then reduce to a small buffer for CPU decisions:
- activity = max( |v|, d(height)/dt, d(water)/dt, erosion_rate, slope_exceed )
- promote if activity >= threshold for N frames
- demote if activity < threshold for M frames (hysteresis)
- always keep a camera ring of L0 tiles

## Update Scheduling
Per LOD update stride (example):
- L0: every frame
- L1: every 2 frames
- L2: every 4 frames
- L3: every 8 frames

Use dt accumulation with per-LOD max dt to keep CFL stability.

## Cross-LOD Coupling
Avoid seams and chunk boundary artifacts:
- Upsample: pull coarse boundary state into fine tiles before fine update.
- Downsample: push fine results into coarse tiles after update.
- Ensure mass conservation for water + sediment when aggregating.
- Keep ghost-cell borders for each tile to stabilize flux across edges.

## Streaming + Eviction
- Fixed tile pools per LOD to cap memory.
- Evict tiles that are inactive for M frames (except camera ring).
- Optionally serialize evicted tiles to disk, or keep a small CPU cache.

## Rendering Integration (Future)
- Sample highest available LOD at each pixel.
- Use skirts or blend zones to hide LOD transitions.
- Only upload GPU terrain meshes for visible tiles.

## GPU Passes (Per Tile)
- update_surface (water surface = ground + depth)
- update_flux (shallow water flux)
- update_depth (depth and velocities)
- update_erosion (sediment exchange)
- activity_reduce (write per-tile activity metric)
- upsample/downsample (LOD coupling)

## Prototype Layout
- config: prototypes/heightfield_lod/config.toml
- code: prototypes/heightfield_lod/src/main.rs

The prototype runs a CPU-only scheduler that:
- Computes activity maps
- Selects tiles per LOD with pool limits
- Enforces a global memory budget with LRU-style eviction
- Reports memory and dispatch counts

## Prototype Output (Optional)
- CSV log for plotting tile counts, memory, and dispatch counts.
- ASCII map for visualizing tile residency and activity on a chosen LOD.
  - `#` = selected tile, `C` = camera tile, other glyphs reflect activity intensity.
- PPM image output for quick visual inspection (tile grid heatmap).
- Composite PPM image (all LODs in a single mosaic) for side-by-side comparison.
- Thrash metric (moving window churn) logged to CSV and console.

## Next Steps (When Integrating)
- Introduce a GPU tile atlas + bindless indices
- Implement upsample/downsample kernels with conservation
- Replace the scheduler outputs with real GPU dispatches
- Add streaming to disk (optional)
