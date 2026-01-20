---
status: pending
priority: p2
issue_id: "030"
tags: [code-review, performance, gpu]
dependencies: []
---

# CPU-Side Buffer Clearing Wastes Memory/Bandwidth

## Problem Statement

`clear_pressure()` allocates a Vec of zeros on CPU and uploads to GPU every frame. For a 162x52x40 grid, this is **1.34 MB allocated and transferred per clear**.

## Findings

**Location:** `crates/game/src/gpu/pressure_3d.rs`, lines 556-557 and 677-679

```rust
pub fn clear_pressure(&self, queue: &wgpu::Queue) {
    let cell_count = (self.width * self.height * self.depth) as usize;
    queue.write_buffer(&self.pressure_buffer, 0, &vec![0u8; cell_count * 4]);
}
```

**Also affects:** `upload_cell_types()` at line 557

## Proposed Solutions

### Option A: Use encoder.clear_buffer() (Recommended)
**Pros:** No CPU allocation, no transfer, GPU-native
**Cons:** None
**Effort:** Trivial
**Risk:** None

```rust
pub fn clear_pressure(&self, encoder: &mut wgpu::CommandEncoder) {
    encoder.clear_buffer(&self.pressure_buffer, 0, None);
}
```

## Acceptance Criteria

- [ ] No Vec allocation in hot paths
- [ ] Use encoder.clear_buffer() for GPU buffer clearing

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-19 | Created | Found by performance-oracle agent |
