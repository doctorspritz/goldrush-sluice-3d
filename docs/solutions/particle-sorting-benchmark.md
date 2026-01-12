# GPU Particle Sorting for P2G Cache Coherence

**Date:** 2026-01-12
**Status:** Fully functional (bugs fixed)
**Branch:** particle-sorting worktree

## Problem

P2G (Particle-to-Grid) scatter operations have random memory access patterns because particles are stored in spawn order, not spatial order. At high particle counts (100k+), this can cause GPU cache thrashing.

## Hypothesis

Sorting particles by cell index before P2G scatter should improve cache coherence, reducing memory bandwidth and improving FPS.

## Implementation

6-pass GPU counting sort:
1. **compute_keys** - Calculate cell index for each particle
2. **count** - Histogram of particles per cell (atomic increments)
3. **local_prefix_sum** - Blelloch scan within workgroups (512 elements each)
4. **scan_block_sums** - Sequential scan of block totals for multi-block support
5. **add_block_offsets** - Add scanned block totals back to local results
6. **scatter** - Reorder particles to sorted output buffers

Files:
- `crates/game/src/gpu/shaders/particle_sort_*.wgsl` (4 shaders)
- `crates/game/src/gpu/particle_sort.rs` (Rust orchestrator)
- Toggle: Press '5' key during simulation

## Benchmark Results

### At ~71k particles
| Mode | FPS Range |
|------|-----------|
| unsorted | 17.3-17.8 |
| SORTED (overhead only) | 16.4-17.2 |
| **Overhead** | ~4-6% |

### At ~148k particles
| Mode | FPS Range | Notes |
|------|-----------|-------|
| unsorted | 16.3-16.8 | Steady state baseline |
| SORTED | 16.1-16.3 | Initial comparison |
| **Overhead** | ~2-3% | Proportionally smaller |

Note: Long-running tests showed FPS degradation over time (thermal throttling suspected), making precise comparison difficult.

## Conclusions

1. **Sorting overhead is real but small** (~2-6% depending on particle count)
2. **Overhead scales sub-linearly** - smaller % at higher counts
3. **Sorted P2G now works** - Full path including sorted scatter is functional

## Bugs Fixed (2026-01-12)

### Bug 1: Multi-block prefix sum broken
- **Symptom**: Buffer overflow at high particle counts; physics broken (low velocities)
- **Root cause**: Blelloch scan only worked for ≤512 cells; `add_block_offsets` added wrong values
- **Fix**: Added `scan_block_sums` pass to compute exclusive prefix sum of block totals

### Bug 2: Grid buffers not propagated
- **Symptom**: Sorted P2G prevented water flow (velocities ~0.30 m/s vs ~0.75 m/s)
- **Root cause**: `sorted_p2g` writes to its own grid buffers, but rest of pipeline uses `p2g` buffers
- **Fix**: Added `copy_buffer_to_buffer` to copy grid results from sorted_p2g to p2g

## Next Steps (if pursuing further)

1. Profile at 500k-1M particles (project target)
2. Compare FPS with sorting enabled vs disabled at high particle counts
3. Measure actual cache hit rate improvement (if possible)

## Decision

**Fully functional.** Sorting infrastructure is complete with bug fixes. Default is now sorted P2G. Toggle with '5' key to compare performance. Ready for high-particle benchmarking when project reaches 1M particle target.
