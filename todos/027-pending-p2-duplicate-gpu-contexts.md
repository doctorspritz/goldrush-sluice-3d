---
status: pending
priority: p2
issue_id: "027"
tags: [code-review, architecture, duplication]
dependencies: []
---

# Duplicate GPU Context Implementations

## Problem Statement

Two nearly identical GPU context structs exist:
- `GpuContext` in `gpu/mod.rs`
- `WgpuContext` in `example_utils/wgpu_context.rs`

Both have similar `async fn new()` and `fn resize()` implementations.

## Findings

### gpu/mod.rs:
```rust
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub size: (u32, u32),
}
```

### example_utils/wgpu_context.rs:
```rust
pub struct WgpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
}
```

**Locations:**
- `crates/game/src/gpu/mod.rs`
- `crates/game/src/example_utils/wgpu_context.rs`

## Proposed Solutions

### Option A: Consolidate into Single GpuContext (Recommended)
**Pros:** Single source of truth, easier maintenance
**Cons:** Need to update all examples
**Effort:** Small (1-2 hours)
**Risk:** Low

```rust
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub size: (u32, u32),
}
```

## Acceptance Criteria

- [ ] Single GpuContext struct
- [ ] All examples updated
- [ ] No duplicate resize() implementations

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-19 | Created | Found by architecture-strategist agent |
