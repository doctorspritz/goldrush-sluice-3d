---
status: pending
priority: p1
issue_id: "024"
tags: [code-review, gpu, shader, critical]
dependencies: []
---

# Cell Type Constant Inconsistency in Multigrid Shaders

## Problem Statement

The cell type constants (CELL_AIR, CELL_FLUID, CELL_SOLID) have **different orderings** between regular shaders and multigrid shaders. This could cause subtle physics bugs where cells are misclassified.

**Why it matters:** If a pressure solve uses multigrid and the cell types don't match, fluid cells could be treated as solid or vice versa, causing incorrect pressure gradients.

## Findings

### Regular Shaders (25+ files):
```wgsl
const CELL_AIR: u32 = 0u;
const CELL_FLUID: u32 = 1u;
const CELL_SOLID: u32 = 2u;
```

### Multigrid Shaders (mg_*.wgsl):
```wgsl
const CELL_SOLID: u32 = 0u;
const CELL_FLUID: u32 = 1u;
const CELL_AIR: u32 = 2u;
```

**Affected files:**
- `crates/game/src/gpu/shaders/mg_restrict.wgsl`
- `crates/game/src/gpu/shaders/mg_prolong.wgsl`
- `crates/game/src/gpu/shaders/mg_smooth.wgsl`
- And other mg_*.wgsl files

## Proposed Solutions

### Option A: Unify to Regular Shader Convention (Recommended)
**Pros:** Most shaders use this; minimal changes
**Cons:** Need to update mg_*.wgsl files
**Effort:** Small (1 hour)
**Risk:** Low

Update all mg_*.wgsl files to use:
```wgsl
const CELL_AIR: u32 = 0u;
const CELL_FLUID: u32 = 1u;
const CELL_SOLID: u32 = 2u;
```

### Option B: Create Shared Shader Include
**Pros:** Single source of truth
**Cons:** Requires shader preprocessing pipeline
**Effort:** Medium (1 day)
**Risk:** Medium

## Recommended Action

Option A - Quick fix to unify constants.

## Technical Details

**Affected files:**
- `crates/game/src/gpu/shaders/mg_*.wgsl` (all multigrid shaders)

**Verification:**
- grep for `CELL_SOLID.*0u` to find reversed orderings
- Ensure GPU tests pass after fix

## Acceptance Criteria

- [ ] All shaders use consistent CELL_AIR=0, CELL_FLUID=1, CELL_SOLID=2
- [ ] GPU pressure tests pass
- [ ] No visual regressions in simulation

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-19 | Created | Found by pattern-recognition-specialist agent |

## Resources

- Pattern Recognition Agent report
- Shader files in `crates/game/src/gpu/shaders/`
