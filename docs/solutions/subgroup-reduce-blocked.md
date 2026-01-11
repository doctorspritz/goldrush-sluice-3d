# Subgroup Reduce for P2G Scatter: Blocked by naga

**Date:** 2026-01-11
**Status:** BLOCKED - Requires naga/wgpu update

## The Idea

Use WGSL subgroup operations to reduce atomic contention in P2G scatter:

```wgsl
// Instead of: each thread does atomicAdd
atomicAdd(&grid[idx], contribution);

// Do: subgroup threads targeting same cell combine first
let combined = subgroupAdd(contribution);  // Only threads with same idx sum together
if (am_first_in_group) {
    atomicAdd(&grid[idx], combined);  // One atomic per unique target
}
```

Expected benefit: Up to 32x fewer atomics if threads in a subgroup target the same cells.

## Why It's Blocked

### wgpu Feature vs WGSL Syntax

wgpu 23 has `Features::SUBGROUP` which is supported on Metal/Vulkan/DX12. However:

1. **WGSL requires `enable subgroups;`** to use subgroup built-ins
2. **naga marks this as "unimplemented"** - shader parsing fails
3. **Without the enable directive**, built-ins like `subgroup_invocation_id` are unknown

Error message:
```
Shader 'P2G 3D Scatter Shader' parsing error:
no definition in scope for identifier: 'subgroup_invocation_id'
```

### GitHub Issues

- [gfx-rs/wgpu#7471](https://github.com/gfx-rs/wgpu/issues/7471) - `enable subgroups;` unknown
- [gfx-rs/wgpu#8180](https://github.com/gfx-rs/wgpu/issues/8180) - Missing enable declaration
- [gfx-rs/wgpu#5555](https://github.com/gfx-rs/wgpu/issues/5555) - Subgroup tracking issue

The underlying issue: subgroups were added for SPIR-V passthrough, but WGSL frontend support is incomplete.

## Workarounds

### Option 1: SPIR-V Shader

Write the P2G scatter shader in GLSL or raw SPIR-V, bypassing naga's WGSL parser.
Downsides: lose WGSL ergonomics, harder to maintain.

### Option 2: Wait for naga Update

Monitor wgpu releases for WGSL subgroup enable support.
The subgroup spec was added to WebGPU CR in January 2025.

### Option 3: Alternative Optimizations

- **Particle sorting** - Pre-sort by cell for spatial coherence
- **Tile-based scatter** - Workgroup-level accumulation in shared memory
- **Gather instead of scatter** - Grid nodes pull from spatial hash

## When to Revisit

Check when upgrading wgpu beyond version 23. Test with:

```wgsl
enable subgroups;

@compute @workgroup_size(256)
fn test() {
    let lane = subgroup_invocation_id;
    let size = subgroup_size;
}
```

If this compiles, subgroups are ready.
