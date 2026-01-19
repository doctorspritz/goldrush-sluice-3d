---
status: pending
priority: p3
issue_id: "035"
tags: [code-review, duplication, gpu]
dependencies: []
---

# Verbose Bind Group Layout Boilerplate

## Problem Statement

Bind group layout definitions are extremely verbose and repetitive. The same pattern appears 100+ times across GPU modules.

## Findings

**Pattern repeated in:**
- `crates/game/src/gpu/p2g_3d.rs` (lines 265-545)
- `crates/game/src/gpu/g2p_3d.rs` (lines 194-388)
- `crates/game/src/gpu/pressure_3d.rs` (lines 152-348)

```rust
wgpu::BindGroupLayoutEntry {
    binding: 0,
    visibility: wgpu::ShaderStages::COMPUTE,
    ty: wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Uniform,
        has_dynamic_offset: false,
        min_binding_size: None,
    },
    count: None,
},
```

## Proposed Solutions

### Option A: Helper functions (Recommended)
**Pros:** Immediate improvement, no macros
**Cons:** Still some verbosity
**Effort:** Small (1-2 hours)
**Risk:** None

```rust
fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry { ... }
fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry { ... }
```

### Option B: Macro
**Pros:** Most concise
**Cons:** Macros can be harder to debug
**Effort:** Small
**Risk:** Low

## Acceptance Criteria

- [ ] Helper functions created
- [ ] Verbose patterns replaced
- [ ] LOC reduced significantly

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-19 | Created | Found by pattern-recognition-specialist and architecture-strategist agents |
