# Scale Constants TODO

> **Created:** 2026-01-09
> **Priority:** Medium

## Problem

Currently FLIP `cell_size` and clump `particle_radius` are defined independently in examples. This makes it hard to:
- Ensure consistent physics coupling
- Change resolution without breaking scale relationships

## Current State

| Location | Parameter | Value |
|----------|-----------|-------|
| `industrial_sluice.rs:29` | `CELL_SIZE` | `0.03` (30mm) |
| `flip/mod.rs:171` | `particle_spacing` | `cell_size * 0.4` (2D only) |
| `sim3d/clump.rs:42` | `spacing` | `radius * 2.2` |

## Solution

Add linked scale constants:

```rust
// Shared simulation scale (in meters)
const CELL_SIZE: f32 = 0.005; // 5mm for riffle vortex (future target)
const PARTICLE_RADIUS: f32 = CELL_SIZE * 0.4; // 2mm
const PARTICLES_PER_CELL: f32 = 8.0; // Standard FLIP density
```

## Tasks

- [ ] Create `crates/sim/src/constants.rs` or similar
- [ ] Define `CELL_SIZE`, `PARTICLE_RADIUS` relationship
- [ ] Update `industrial_sluice.rs` to use shared constants
- [ ] Update `sim3d` clump generation to derive from `CELL_SIZE`
- [ ] Document scale relationship in plan
