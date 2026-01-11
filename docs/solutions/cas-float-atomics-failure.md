# CAS-Based Float Atomics: Why Fixed-Point Wins

**Date:** 2026-01-11
**Status:** ABANDONED - Fixed-point approach is superior

## The Problem

WebGPU/WGSL only supports `atomic<i32>` and `atomic<u32>`, not `atomic<f32>`. The P2G scatter shader needs to atomically accumulate floating-point momentum values from ~500k particles to grid nodes.

## What We Tried

### Approach 1: Fixed-Point (Current)
```wgsl
const SCALE: f32 = 1000000.0;
let scaled = i32(momentum * SCALE);
atomicAdd(&sum[idx], scaled);
// Later: velocity = f32(sum) / SCALE / f32(weight)
```

### Approach 2: CAS-Based Float Atomics
```wgsl
var old = atomicLoad(&sum[idx]);
loop {
    let old_f = bitcast<f32>(old);
    let new_f = old_f + momentum;
    let result = atomicCompareExchangeWeak(&sum[idx], old, bitcast<u32>(new_f));
    if result.exchanged { break; }
    old = result.old_value;
}
```

## Benchmark Results

| Approach | Steady-State FPS | Stability |
|----------|------------------|-----------|
| Fixed-point | 58-60 FPS | Rock solid |
| CAS atomics | 55-60 FPS typical | Severe spikes to 2-20 FPS |

## Why CAS Failed

### Contention Cascade
When many particles target the same grid cell (common at water surfaces):

1. Thread A loads old value, computes new
2. Thread B loads same old value, computes new
3. Thread A's CAS succeeds
4. Thread B's CAS fails → must retry
5. More threads pile up → exponential retry growth

Fixed-point `atomicAdd` avoids this because the hardware guarantees single-operation success—serialization happens in silicon, not in shader retry loops.

### WGSL Limitations Made It Worse
- Cannot pass storage pointers to functions in WGSL
- Had to inline CAS loops at every accumulation site (6 places per velocity component)
- Code bloat with no benefit

## What About Native Float Atomics?

Investigated `wgpu::Features::SHADER_FLOAT32_ATOMIC`:
- Only works for SPIR-V shaders, not WGSL
- WGSL float atomics are still an open proposal (as of 2026-01)
- Would require rewriting shaders in SPIR-V or waiting for WGSL spec update

## Lessons Learned

1. **Hardware atomics beat software emulation** - CAS loops can't compete with native atomicAdd
2. **Fixed-point precision is fine** - 1/1,000,000 scale gives ~6 decimal places, more than enough for fluid sim
3. **Contention is the killer** - P2G scatter has inherently high contention; any retry-based approach suffers
4. **Benchmark stability, not just peak FPS** - CAS looked similar at peak but had catastrophic spikes

## Files Created (Now Removed)

- `.worktrees/cas-atomics/crates/game/src/gpu/shaders/p2g_scatter_3d.wgsl` - CAS version
- `.worktrees/cas-atomics/crates/game/src/gpu/shaders/p2g_divide_3d.wgsl` - u32→f32 bitcast version

Worktree removed without merging.
