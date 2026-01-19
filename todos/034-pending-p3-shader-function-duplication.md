---
status: pending
priority: p3
issue_id: "034"
tags: [code-review, duplication, shader]
dependencies: []
---

# Shader Function Duplication (quadratic_bspline_1d, cell_index, etc.)

## Problem Statement

Core shader functions are copy-pasted across 25+ shader files:
- `quadratic_bspline_1d()` - 4 identical copies
- `cell_index()`, `u_index()`, `v_index()`, `w_index()` - 30+ copies
- Cell type constants - 25+ definitions

## Findings

**Duplicated in:**
- `p2g_scatter_3d.wgsl`
- `p2g_scatter_tiled_3d.wgsl`
- `p2g_cell_centric_3d.wgsl`
- `g2p_3d.wgsl`
- And 20+ more shader files

## Proposed Solutions

### Option A: Shader preprocessing (Recommended for scale)
**Pros:** Single source of truth
**Cons:** Requires build step
**Effort:** Medium (1 day)
**Risk:** Low

Use `include!()` or custom preprocessing to inject common code.

### Option B: Document sync requirement
**Pros:** No infrastructure changes
**Cons:** Still manual, error-prone
**Effort:** Trivial
**Risk:** Medium (drift)

## Acceptance Criteria

- [ ] Common functions defined in one place
- [ ] All shaders use shared definitions
- [ ] No duplicate function definitions

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-19 | Created | Found by pattern-recognition-specialist agent |
