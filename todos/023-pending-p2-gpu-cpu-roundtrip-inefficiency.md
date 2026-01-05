---
status: pending
priority: p2
issue_id: "023"
tags: [code-review, performance, gpu]
dependencies: []
---

# GPU-CPU Round Trip Inefficiency in FLIP Step

## Problem Statement

The simulation downloads grid velocities from GPU to CPU, then re-uploads them to a separate G2P module. This wastes ~290KB of PCIe bandwidth per frame.

## Findings

### From Performance Oracle agent:

```
P2G buffers (GPU) --> Download to CPU --> Upload to G2P buffers (GPU)
                   ^^^^^^^^^^^^^^^^^^^
                   UNNECESSARY ROUND TRIP
```

The `GpuG2p3D` has its own separate grid buffers instead of sharing with `GpuP2g3D`.

### Location:
`crates/game/src/gpu/flip_3d.rs` lines 609-650

### Impact:
- U grid: 47KB, V grid: 48KB, W grid: 50KB
- Total: ~145KB download + 145KB upload = 290KB wasted
- At 60 FPS: 17.4 MB/s unnecessary PCIe traffic

## Proposed Solutions

### Option A: Share buffers between P2G and G2P
**Pros:** Eliminates round-trip entirely
**Cons:** Requires refactoring G2P module
**Effort:** Medium
**Risk:** Low

```rust
let g2p = GpuG2p3D::new_with_shared_buffers(
    device,
    &p2g.grid_u_buffer,  // Share, don't copy
    &p2g.grid_v_buffer,
    &p2g.grid_w_buffer,
);
```

### Option B: Keep separate buffers, use GPU-to-GPU copy
**Pros:** Minimal code change
**Cons:** Still has copy overhead (but much faster than CPU)
**Effort:** Small
**Risk:** Low

## Recommended Action

Option A for proper fix, Option B as quick improvement

## Technical Details

**Affected files:**
- `crates/game/src/gpu/flip_3d.rs`
- `crates/game/src/gpu/g2p_3d.rs`

## Acceptance Criteria

- [ ] No grid velocity round-trips through CPU
- [ ] Frame time reduced

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-03 | Identified by performance review | Optimization opportunity |
