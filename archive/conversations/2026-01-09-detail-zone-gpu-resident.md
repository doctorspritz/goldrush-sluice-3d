# 2026-01-09: Detail Zone GPU-Resident FLIP + Bed Height LOD

## Session Goal
Unblock detail_zone by fixing wgpu validation errors and finish the GPU-resident path (bed height -> cell types -> FLIP -> render), then document the combined 2.5D/3D system.

## What User Said
- "move to phase 2"
- "continue"
- posted wgpu validation errors for bed height resample pipeline and uniform buffer size mismatch
- "it loads! congratulations, that was a massive slog. can you record a log of this in conversations. can you update an architecture.md"

## Starting State
- detail_zone had GPU bed height resample and a GPU-resident path stubbed in.
- WGPU validation errors:
  - storage binding mismatch for bed height resample pipeline
  - uniform buffer size mismatch (32 vs 48) in bed-height cell type compute

## What We Did

### 1. Fix bed height resample pipeline layout
- Set bed height storage bindings to read_write to match the pipeline layout.

### 2. Fix bed height cell type uniform size
- Adjusted WGSL padding to 32 bytes to match the host struct size.

### 3. GPU-resident integration
- Added GPU cell type build from bed height (new compute shader + pipeline).
- Added GPU emitter bindings for C buffers (avoid undefined data).
- Switched particle rendering to read positions from GPU storage buffer.
- Added origin_y in bed height resample params to align world/local frames.
- Added `step_in_place` path for fully GPU-resident simulation; CPU fallback retained.

### 4. Documentation
- Updated `plans/ARCHITECTURE.md` with the combined 2.5D + 3D GPU system and residency map.

## Files Changed
- `crates/game/examples/detail_zone.rs`
- `crates/game/src/gpu/flip_3d.rs`
- `crates/game/src/gpu/bridge_3d.rs`
- `crates/game/src/gpu/heightfield.rs`
- `crates/game/src/gpu/shaders/heightfield_bed_height.wgsl`
- `crates/game/src/gpu/shaders/cell_type_from_bed_height_3d.wgsl`
- `crates/game/src/gpu/shaders/particle_emitter_3d.wgsl`
- `crates/game/src/gpu/shaders/sdf_collision_3d.wgsl`
- `plans/ARCHITECTURE.md`

## Tests
- `cargo test -p game --examples`

## Regressions
None observed.

## Current State
detail_zone loads with the GPU-resident path available (bed height -> cell types -> FLIP -> render). CPU fallback path still works for debugging.

## Next Steps
1. Re-run `cargo run --example detail_zone --release` after any shader changes to confirm no validation errors.
2. Profile GPU path vs CPU fallback to validate residency performance benefits.
