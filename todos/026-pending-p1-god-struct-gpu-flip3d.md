---
status: pending
priority: p1
issue_id: "026"
tags: [code-review, architecture, refactoring]
dependencies: []
---

# GpuFlip3D God Struct Anti-Pattern

## Problem Statement

`GpuFlip3D` has **90+ fields** and the file is **5196 lines**. This "God struct" mixes:
- Grid configuration (5 fields)
- Physics parameters (14 sediment, 10 vorticity, etc.)
- GPU buffers and pipelines
- Readback infrastructure
- Sub-solvers

**Why it matters:** Maintenance nightmare. Adding features requires modifying a 5000+ line file. Hard to reason about state.

## Findings

### Field Categories in GpuFlip3D:
| Category | Field Count |
|----------|-------------|
| Grid configuration | 5 |
| Sediment physics | 14 |
| Particle buffers | 6 |
| Sub-solvers | 5 |
| Gravity | 3 |
| Flow | 3 |
| Vorticity | 10 |
| Sediment fraction | 4 |
| Sediment pressure | 4 |
| Porosity drag | 4 |
| Boundary conditions | 4 |
| Grid velocity backup | 3 |
| Density projection | 15 |
| SDF collision | 5 |
| Gravel obstacles | 5 |
| Particle sorting | 5 |

**Total: 90+ public fields**

**Location:** `crates/game/src/gpu/flip_3d.rs`

## Proposed Solutions

### Option A: Extract into Sub-modules (Recommended)
**Pros:** Incremental, maintains compatibility
**Cons:** Still single struct, just organized better
**Effort:** Medium (1-2 days)
**Risk:** Low

Structure:
```
gpu/flip_3d/
├── mod.rs          # Main GpuFlip3D, step()
├── params.rs       # All #[repr(C)] param structs
├── readback.rs     # ReadbackSlot, async buffer management
├── vorticity.rs    # Vorticity pipeline
├── sediment.rs     # Sediment fraction/pressure
├── density.rs      # Density projection
└── pipelines.rs    # Remaining pipeline creation
```

### Option B: Feature Trait System
**Pros:** Maximum extensibility
**Cons:** Major refactor, API changes
**Effort:** Large (1 week+)
**Risk:** High

### Option C: Configuration Struct Extraction
**Pros:** Separates config from internal state
**Cons:** Partial fix
**Effort:** Small (2-4 hours)
**Risk:** Low

```rust
pub struct FlipConfig {
    pub grid: GridConfig,
    pub physics: FlipPhysicsConfig,
    pub sediment: SedimentConfig,
}
```

## Recommended Action

Start with Option C (quick win), then Option A (full refactor).

## Technical Details

**Affected files:**
- `crates/game/src/gpu/flip_3d.rs` (split into multiple files)

## Acceptance Criteria

- [ ] GpuFlip3D struct has <30 direct fields
- [ ] Configuration is encapsulated in FlipConfig
- [ ] No single file >1500 lines
- [ ] All tests pass

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-19 | Created | Found by architecture-strategist and pattern-recognition agents |

## Resources

- Architecture Strategist Agent report
- Pattern Recognition Agent report
