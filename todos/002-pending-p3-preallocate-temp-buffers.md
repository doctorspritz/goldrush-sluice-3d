# Pre-Allocate Temporary Buffers in Hot Paths

**Priority:** P3 - Nice-to-have
**Status:** pending (deferred)
**Tags:** performance, memory, allocation
**Issue ID:** 017

## Profiling Results (2025-12-28)

**Tested:** 5K-9.5K particles in 200×100 grid (0.5 cell size)
- 5K particles: 6.1ms/frame (163 FPS)
- 9.5K particles: 7.5ms/frame (133 FPS)
- Actual allocations: **~0.5MB/frame** (not 8-9MB as originally estimated)
- p99 latency spikes (15-21ms) are OS/GC, not allocation pressure

**Conclusion:** At current particle counts, allocator overhead is not the bottleneck.
The simulation already exceeds 60 FPS by 2x. Defer optimization until higher particle
counts (100K+) become necessary.

## Problem Statement

Several functions allocate temporary buffers every frame, causing unnecessary heap allocations in hot paths. At 100k particles, this creates significant allocator pressure.

## Findings

### Allocation Sites

#### 1. `compute_neighbor_counts()` - flip.rs:1796-1797
```rust
let positions: Vec<Vec2> = self.particles.list.iter().map(|p| p.position).collect();
let materials: Vec<ParticleMaterial> = self.particles.list.iter().map(|p| p.material).collect();
```
**Cost:** ~1.2MB per frame (12 bytes × 100k particles)

#### 2. `extrapolate_u_layer()` - grid.rs:1079-1082
```rust
let mut new_known = vec![false; u_known.len()];      // ~393k bools
let mut new_values = vec![0.0f32; self.u.len()];     // ~393k floats
```
**Cost:** ~1.9MB per frame (per layer, 2 layers = ~3.8MB)

#### 3. `extrapolate_v_layer()` - grid.rs:1158-1159
```rust
let mut new_known = vec![false; v_known.len()];      // ~393k bools
let mut new_values = vec![0.0f32; self.v.len()];     // ~393k floats
```
**Cost:** ~1.9MB per frame (per layer, 2 layers = ~3.8MB)

### Total Per-Frame Allocation
~8-9MB of heap allocations per frame, all immediately freed.

## Proposed Solution

### Add Pre-Allocated Buffers to Structs

#### For FlipSimulation
```rust
pub struct FlipSimulation {
    // Existing fields...

    // Pre-allocated scratch buffers for compute_neighbor_counts
    scratch_positions: Vec<Vec2>,
    scratch_materials: Vec<ParticleMaterial>,
}

fn compute_neighbor_counts(&mut self) {
    // Resize if needed (rarely happens)
    let len = self.particles.len();
    self.scratch_positions.resize(len, Vec2::ZERO);
    self.scratch_materials.resize(len, ParticleMaterial::Water);

    // Copy without allocation
    for (i, p) in self.particles.iter().enumerate() {
        self.scratch_positions[i] = p.position;
        self.scratch_materials[i] = p.material;
    }
    // ... use scratch buffers
}
```

#### For Grid (Velocity Extrapolation)
```rust
pub struct Grid {
    // Existing fields...

    // Pre-allocated buffers for extrapolation
    extrapolate_u_known_temp: Vec<bool>,
    extrapolate_u_values_temp: Vec<f32>,
    extrapolate_v_known_temp: Vec<bool>,
    extrapolate_v_values_temp: Vec<f32>,
}
```

**Effort:** Medium (1-2 hours)
**Risk:** Low
**Expected gain:** 5-10% frame time improvement from reduced allocator pressure

## Acceptance Criteria

- [ ] No heap allocations in `compute_neighbor_counts()`
- [ ] No heap allocations in `extrapolate_velocities()`
- [ ] Buffers sized correctly at startup or resize on demand
- [ ] Same simulation results

## Files to Modify

- `crates/sim/src/flip.rs` - Add scratch buffers, modify `compute_neighbor_counts()`
- `crates/sim/src/grid.rs` - Add extrapolation buffers to Grid struct

## Related

- 002-pending-p1-memory-allocation-in-solver-loop.md (similar issue in pressure solver)
