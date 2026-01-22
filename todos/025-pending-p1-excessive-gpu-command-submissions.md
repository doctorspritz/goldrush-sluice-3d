---
status: pending
priority: p1
issue_id: "025"
tags: [code-review, performance, gpu, critical]
dependencies: []
---

# Excessive GPU Command Submissions Causing Pipeline Stalls

## Problem Statement

The `run_gpu_passes()` function submits **30+ separate command buffers** per simulation step. Each `queue.submit()` creates a potential GPU pipeline stall where the CPU waits for GPU acknowledgment.

**Why it matters:** This is the primary bottleneck preventing 1M particles at 60fps. Batching commands could yield 20-40% improvement.

## Findings

### Current Submit Count (per frame):
- Line 3799: P2G submit
- Line 3841: Sediment fraction submit
- Line 3874: Sediment pressure submit
- Line 3924: BC enforcement submit
- Line 3951: Grid copy submit
- Lines 3986-4008: Cell type (5 iterations = 5 submits + 1 expand)
- Line 4159: Vorticity submit
- Line 4167: Pressure submit (120 iterations internally)
- Line 4227: Porosity drag submit
- Lines 4285-4359: Velocity extrapolation (4 passes x 4 = 16 submits)
- Line 4386: G2P submit
- Lines 4428-4512: Density projection (4 submits)

**Total: ~35+ queue.submit() calls per frame**

**Location:** `crates/game/src/gpu/flip_3d.rs`, `run_gpu_passes()` function (lines 3685-4680)

## Proposed Solutions

### Option A: Mega-batch into 3 Submissions (Recommended)
**Pros:** Maximizes GPU utilization, minimal API changes
**Cons:** Requires understanding pass dependencies
**Effort:** Medium (4-6 hours)
**Risk:** Low

```rust
// Submit 1: P2G + Sediment + BC + Cell Type
// Submit 2: Gravity + Vorticity + Pressure + Extrapolation
// Submit 3: G2P + Density Projection + Advection
```

### Option B: Single Command Buffer
**Pros:** Maximum batching
**Cons:** Harder to debug, memory pressure
**Effort:** Large
**Risk:** Medium

## Recommended Action

Option A - Batch related passes together.

## Technical Details

**Affected files:**
- `crates/game/src/gpu/flip_3d.rs` - `run_gpu_passes()` method

**Key insight:** Most passes have no true data dependency that requires CPU intervention. They can be encoded into a single command buffer.

## Acceptance Criteria

- [x] Reduce queue.submit() calls from 35+ to <10 per frame
- [ ] Profile showing reduced CPU-GPU sync overhead
- [ ] No visual regressions
- [ ] Frame time improvement measurable

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-19 | Created | Found by performance-oracle agent |
| 2026-01-22 | Batched run_gpu_passes into 3 submits | Pre-pressure, pressure/BC/drag, and G2P/SDF grouped |

## Resources

- Performance Oracle Agent report
- `crates/game/src/gpu/flip_3d.rs` lines 3685-4680
