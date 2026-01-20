---
status: pending
priority: p2
issue_id: "031"
tags: [code-review, performance, gpu]
dependencies: []
---

# Velocity Extrapolation Over-Submission (16 submits)

## Problem Statement

Velocity extrapolation runs 4 passes with **4 separate submits per pass** (U, V, W, finalize), creating **16 command submissions** just for this feature.

## Findings

**Location:** `crates/game/src/gpu/flip_3d.rs`, lines 4288-4362

```rust
for _pass in 0..4 {
    queue.submit(/*U*/);
    queue.submit(/*V*/);
    queue.submit(/*W*/);
    queue.submit(/*finalize*/);
}
```

## Proposed Solutions

### Option A: Batch and reduce iterations (Recommended)
**Pros:** 8x reduction in submits
**Cons:** None
**Effort:** Small (1-2 hours)
**Risk:** Low

```rust
for _pass in 0..2 {  // 2 passes sufficient for FLIP stability
    let mut encoder = device.create_command_encoder(...);
    self.encode_velocity_extrap_uvw(&mut encoder);
    self.encode_velocity_extrap_finalize(&mut encoder);
    queue.submit(encoder.finish()); // 1 submit instead of 4
}
```

## Acceptance Criteria

- [ ] Reduce from 16 to 2-4 submits
- [ ] Visual quality maintained
- [ ] Consider reducing passes from 4 to 2

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-19 | Created | Found by performance-oracle agent |
