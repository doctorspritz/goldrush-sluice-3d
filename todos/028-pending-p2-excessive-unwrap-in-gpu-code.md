---
status: pending
priority: p2
issue_id: "028"
tags: [code-review, error-handling, gpu]
dependencies: []
---

# Excessive unwrap() in GPU Code

## Problem Statement

GPU buffer mapping uses `.unwrap()` extensively (~50+ instances). Buffer mapping can fail due to device loss, out-of-memory, etc. Currently, failures cause panics.

## Findings

### Examples:
```rust
// flip_3d.rs line 4777
rx.recv().unwrap().unwrap();

// sph_3d.rs line 954
slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());
rx.recv().unwrap().unwrap();
```

**Locations:** Multiple files in `crates/game/src/gpu/`

## Proposed Solutions

### Option A: Replace with expect() (Quick Fix)
**Pros:** Immediate improvement, descriptive errors
**Cons:** Still panics
**Effort:** Small (1-2 hours)
**Risk:** Low

### Option B: Introduce GpuError type (Proper Fix)
**Pros:** Graceful error handling
**Cons:** API changes, more work
**Effort:** Medium (4-6 hours)
**Risk:** Low

```rust
#[derive(Debug)]
pub enum GpuError {
    BufferMapFailed(wgpu::BufferAsyncError),
    DeviceLost,
    OutOfMemory,
}
```

## Acceptance Criteria

- [ ] No bare `.unwrap()` on buffer operations
- [ ] All unwraps have descriptive expect() messages OR proper error handling

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-19 | Created | Found by security-sentinel and architecture-strategist agents |
