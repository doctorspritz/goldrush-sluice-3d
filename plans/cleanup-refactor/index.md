# Cleanup Refactor Review Log

Process: One file per review entry. Each entry captures purpose, links to other
files, and cleanup signals (dead code, conflicts, outdated content, old plans,
old tests, unused declarations).

Archive location for outdated plans: `plans/archive/`

## `CLAUDE.md`
- Purpose: Project safety/workflow instructions and quick run/test commands.
- Links/interactions:
  - `crates/sim/src/flip.rs` (stale path; actual module is `crates/sim/src/flip/mod.rs`)
  - `crates/sim/src/grid.rs` (stale path; actual module is `crates/sim/src/grid/mod.rs`)
  - `crates/sim/src/particle.rs` (particle types)
  - `crates/game/src/main.rs` (visual demo entrypoint)
  - `docs/solutions/` (problem/solution docs)
- Potential issues:
  - Outdated file paths for flip/grid modules; should be corrected or generalized to module dirs.

## `plans/archive/gpu-integration.md`
- Purpose: GPU integration performance plan and phased roadmap (MGPCG, P2G, G2P, advection).
- Links/interactions:
  - `crates/game/src/gpu/mgpcg.rs` (MGPCG orchestration)
  - `crates/game/src/gpu/p2g.rs`, `crates/game/src/gpu/g2p.rs`
  - `crates/game/src/gpu/shaders/mg_smooth.wgsl`, `crates/game/src/gpu/shaders/mg_restrict.wgsl`, `crates/game/src/gpu/shaders/mg_prolongate.wgsl`
  - `crates/game/src/gpu/shaders/pcg_ops.wgsl`
  - Mentions `gpu/shaders/advect.wgsl`, `gpu/shaders/p2g.wgsl` (not present; repo has `advect_3d.wgsl` and split `p2g_*` shaders)
- Potential issues:
  - Outdated file names/paths in “Files to Create” (e.g., `advect.wgsl`, `p2g.wgsl`) vs current shader set.
  - Examples use 2D vectors and 2D grids; may conflict with current 3D implementation focus.
  - Performance numbers and stability notes appear historical; no date or validation against current code.

## `plans/hybrid-lod-architecture.md`
- Purpose: High-level plan for clipmap LOD + 2.5D SWE heightfield coupled to 3D PIC/FLIP/DEM zones.
- Links/interactions:
  - `crates/game/src/gpu/shaders/heightfield_render.wgsl` (mentioned for stitching/morphing)
  - `crates/game/src/gpu/heightfield.rs` (loads `heightfield_render.wgsl`)
  - `crates/game/src/gpu/shaders/heightfield_water.wgsl` (SWE solver)
  - `crates/game/src/gpu/shaders/heightfield_erosion.wgsl` (mentions SWE flux)
- Potential issues:
  - Mentions `HeightfieldRender` and `ParticleSimulationZone` types but no matching symbols found in code; likely unimplemented/outdated naming.
  - Roadmap phases may be superseded; no dates or links to current implementation status.
- Action taken:
  - Updated `plans/hybrid-lod-architecture.md` to remove placeholder type names, fix SWE typo, and add a status/anchors section.

## `plans/archive/flip-performance-optimization.md`
- Purpose: Performance optimization plan for PIC/FLIP hot paths (allocations, dead code).
- Links/interactions:
  - `crates/sim/src/flip/transfer.rs` (`particles_to_grid_impl` uses pre-allocated buffers)
  - `crates/sim/src/grid/vorticity.rs` (`compute_vorticity`/`apply_vorticity_confinement_with_piles`)
- Potential issues:
  - References `crates/sim/src/flip.rs` and `crates/sim/src/grid.rs`, which no longer exist (modules split under `flip/` and `grid/`).
  - “Pre-allocate P2G buffers” and “pre-allocate curl buffer” are already implemented (`u_sum/u_weight/v_sum/v_weight` and `self.vorticity`).
  - “resolve_solid_collision” no longer present; dead-code item already addressed.
  - Performance target uses 5000 particles, conflicting with current performance-testing guidance (100k+).

## `plans/archive/dfsph_integration_handoff.md`
- Purpose: Handoff notes for integrating a DFSPH solver into the game demo.
- Links/interactions:
  - `crates/dfsph/src/simulation.rs` (mentions alpha sign fix)
  - `crates/game/src/main.rs` (claimed to be rewritten for DFSPH)
  - `crates/game/src/render.rs` (claimed corrupted; file not present)
- Potential issues:
  - References `crates/game/src/render.rs`, which does not exist.
  - No `dfsph` references found in `crates/game/src/` (plan likely obsolete).

## `plans/archive/sediment-sizes.md`
- Purpose: Plan for variable sediment sizes and rendering changes.
- Links/interactions:
  - `crates/sim/src/particle.rs` (diameter, sphericity, roughness, effective_diameter)
  - `crates/sim/src/flip/spawning.rs` (spawn_with_variation, use_variable_diameter)
  - `crates/game/src/gpu/renderer.rs` (particle size for rendering)
- Potential issues:
  - References `crates/game/src/render.rs`, which does not exist; rendering is in `crates/game/src/gpu/renderer.rs`.
  - Most items appear already implemented (diameter, sphericity/roughness, spawn variation, rendering uses diameter).
  - Toggle/UI step not present in current game code (plan suggests a key toggle that is missing).

## `plans/archive/async-gpu-pipeline.md`
- Purpose: Plan for async/double-buffered GPU readback to avoid stalls.
- Links/interactions:
  - `crates/game/src/gpu/flip_3d.rs` (`step_async`, readback slots, async readback)
  - `crates/game/examples/industrial_sluice.rs` (calls `step_async`)
- Potential issues:
  - Plan appears fully implemented; likely historical and can be archived.

## `plans/archive/gpu-g2p-implementation.md`
- Purpose: Plan for GPU G2P implementation (APIC-FLIP gather).
- Links/interactions:
  - `crates/game/src/gpu/g2p.rs` (2D GPU G2P implementation)
  - `crates/game/src/gpu/g2p_3d.rs` + `crates/game/src/gpu/shaders/g2p_3d.wgsl` (3D GPU G2P implementation)
  - `crates/sim/src/flip/transfer.rs` (CPU G2P reference)
- Potential issues:
  - Plan appears implemented (GPU G2P modules exist; used by `crates/game/src/gpu/flip_3d.rs`).
  - References 2D-only buffers and CPU transfer.rs line numbers that no longer match.
