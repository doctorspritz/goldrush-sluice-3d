# P2G Shared Memory Aggregation

## Problem

Current P2G scatter does 81 atomic operations per particle (27 each for U, V, W grids).
At 1M particles = 81M atomics competing for ~300k grid cells = massive contention.

## Attempt 1: Tile-Based Shared Memory Atomics

### Design
- Workgroup: 256 threads, Tile: 4×4×4 cells
- Accumulate to shared memory arrays, flush to global at end

### Result: FAILED - No Performance Benefit

The approach used `atomicAdd` on shared memory, which just moves contention from global to shared memory. Within a workgroup, multiple particles still compete for the same shared memory cells.

**Benchmark:**
- SORTED+Tiled at 61k: ~15.8 FPS
- Unsorted at 74k: ~17.9-18.5 FPS

The tiled approach was actually SLOWER due to:
1. Shared memory initialization overhead (zeroing 608 i32s per workgroup)
2. Shared memory atomics still cause contention within workgroup
3. Flush phase adds extra global atomics
4. Tile boundary handling complexity

## Alternative Approaches to Explore

### Option A: Cell-Centric Dispatch (Promising)
Instead of one thread per particle, one thread per GRID CELL:
1. Use cell_offsets from counting sort to find particle range for each cell
2. Thread iterates over all particles in its cell + neighbors
3. Accumulates WITHOUT atomics (single writer per cell)
4. Writes final value with single store

**Pros:** Zero atomics in accumulation phase
**Cons:** Variable work per thread (load imbalance), complex neighbor iteration

### Option B: Subgroup Operations
Use subgroup (warp/wavefront) operations to reduce within groups:
1. Each thread computes contributions to its 27 target cells
2. Use `subgroupAdd` to combine contributions targeting same cell
3. One thread per target cell does single atomic

**Pros:** Leverages hardware shuffle operations
**Cons:** Complex mapping, WGSL subgroup support varies by platform

### Option C: Two-Pass Binning
1. Pass 1: Each particle writes (cell_idx, contribution) to append buffer
2. Sort by cell_idx (or use counting sort bins)
3. Pass 2: Each cell accumulates its bin without atomics

**Pros:** Clean separation, no atomics in accumulation
**Cons:** Extra memory, extra passes, complex data movement

## Attempt 2: Cell-Centric Dispatch

### Design
- One thread per grid node (instead of per particle)
- Each thread iterates particles in 27 neighboring cells
- Uses cell_offsets from counting sort for particle ranges
- Zero atomics in accumulation (single writer per grid node)
- Four entry points: scatter_u, scatter_v, scatter_w, count_particles

### Result: FAILED - 15× SLOWER

**Benchmark:**
- Cell-centric at 80k particles: ~0.8-1.0 FPS
- Particle-centric (sorted) at 115k particles: ~14.8 FPS

**Root Cause: Sparse Grid + Cache-Unfriendly Reads**

1. **Sparse occupancy**: 80k particles / 125k cells = 0.64 particles/cell average
   - Most cells are empty → threads do no useful work
   - Particle-centric only processes existing particles

2. **Scattered memory reads**: Each grid node reads particles from 27 cells
   - Random access pattern has terrible cache performance
   - Particles are sorted by cell, but grid nodes access overlapping regions
   - Different threads in workgroup access non-contiguous data

3. **Atomics are fast**: Modern GPUs optimize atomic operations heavily
   - Particle-centric atomics with sorted particles have good L2 hit rates
   - Scattered reads in cell-centric are slower than sorted atomics

4. **Too many threads**: 375k grid nodes (U+V+W) vs 80k particles
   - Each thread does O(27 * particles_per_cell) sequential work
   - High thread count doesn't help with sparse data

**When Cell-Centric WOULD Win:**
- Dense grids (many particles per cell)
- Smaller grids (fewer wasted threads)
- Higher particle counts relative to grid size

### Files Created (Can Be Removed)
- `shaders/p2g_cell_centric_3d.wgsl`
- `p2g_cell_centric_3d.rs`

## Current Status

Both optimization attempts failed:
- Tiled shared memory: Just moves atomics to shared memory, no benefit
- Cell-centric dispatch: Sparse grid makes scattered reads slower than atomics

Sorting provides ~5-10% cache coherence benefit on its own.

## Decision

**Park P2G optimizations.** The current sorted particle-centric approach with atomics
is actually well-suited for our sparse fluid simulation. GPUs handle atomics efficiently,
and particle sorting provides modest cache benefits.

For future optimization, consider:
- Adaptive dispatch based on particle density
- Hybrid approaches (cell-centric for dense regions, particle-centric for sparse)
- Focus optimization efforts on pressure solver instead (currently 8ms/frame)
