---
status: pending
priority: p1
issue_id: "020"
tags: [code-review, physics, flow, critical]
dependencies: []
---

# Flow Shader Skips Faces Adjacent to Riffles

## Problem Statement

The `flow_3d.wgsl` shader does not apply flow acceleration to U-faces adjacent to solid cells (riffles). This creates "flow dead zones" directly upstream of each riffle, causing water to pool instead of being pushed downstream.

**Why it matters:** This is the root cause of the 13x velocity difference between `flow_test` (1.6 m/s, no riffles) and `gold_sluice_3d` (0.12 m/s, with riffles).

## Findings

### Evidence from code review agents:

1. **Architecture Strategist:** "The flow shader skips faces adjacent to solids. In a riffle geometry, when water pools behind riffles, the cells immediately upstream of the riffle are adjacent to CELL_SOLID. This causes flow_accel to NOT be applied to those U-faces."

2. **Pattern Recognition:** "When a U-face is between a FLUID cell and a SOLID cell, the first check returns (skips flow). This means the upstream side of riffles never gets flow acceleration."

3. **Data Integrity Guardian:** "The flow acceleration is not applied to faces adjacent to solid riffles, causing water to pool upstream."

### Location:
`crates/game/src/gpu/shaders/flow_3d.wgsl` lines 60-68:
```wgsl
// Don't apply flow to solid boundary faces
if (left_type == CELL_SOLID || right_type == CELL_SOLID) {
    return;  // <-- PROBLEM: No flow at riffle faces
}

// Only apply flow if at least one adjacent cell is fluid
if (left_type == CELL_FLUID || right_type == CELL_FLUID) {
    let idx = u_index(i, j, k);
    grid_u[idx] += params.flow_accel_dt;
}
```

### Physics explanation:
Flow acceleration represents the gravity component along the slope. Gravity does NOT stop acting on water just because there is a riffle nearby. The riffle should redirect flow (via pressure), but the driving force should still be present.

## Proposed Solutions

### Option A: Apply flow to all fluid cells, skip only boundary solids
**Pros:** Physically correct - riffles deflect flow, don't block gravity
**Cons:** Need to distinguish boundary solids from internal obstacles
**Effort:** Medium
**Risk:** Low - matches real physics

```wgsl
// Only skip flow at DOMAIN BOUNDARY solids, not internal obstacles
let is_boundary = i == 0u || j == 0u || k == 0u ||
                  k == params.depth - 1u;  // inlet, floor, side walls

if (is_boundary && (left_type == CELL_SOLID || right_type == CELL_SOLID)) {
    return;
}

// Apply flow if at least one cell is fluid
if (left_type == CELL_FLUID || right_type == CELL_FLUID) {
    grid_u[idx] += params.flow_accel_dt;
}
```

### Option B: Apply flow to ALL faces with at least one fluid cell
**Pros:** Simplest fix
**Cons:** May apply flow incorrectly at some boundaries
**Effort:** Small
**Risk:** Medium - could cause issues at closed boundaries

```wgsl
// Apply flow anywhere there's fluid
if (left_type == CELL_FLUID || right_type == CELL_FLUID) {
    let idx = u_index(i, j, k);
    grid_u[idx] += params.flow_accel_dt;
}
```

### Option C: Use cell position to determine if face is "above" solid
**Pros:** Most physically accurate for overflow scenarios
**Cons:** Complex, requires knowing riffle top heights
**Effort:** Large
**Risk:** Medium

## Recommended Action

Option A - distinguishes boundary solids from internal obstacles

## Technical Details

**Affected files:**
- `crates/game/src/gpu/shaders/flow_3d.wgsl`

**Components:**
- GPU flow acceleration shader
- Affects all sluice simulations with riffles

## Acceptance Criteria

- [ ] Flow acceleration applied to fluid faces even when adjacent to riffles
- [ ] `riffle_diagnostic` shows avg Vx > 0.5 m/s (vs current 0.12)
- [ ] Particles exit through outlet (currently 0%)
- [ ] `flow_test` (simple slope) still works correctly

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-03 | Identified by multi-agent review | Root cause of riffle pooling |

## Resources

- PR: uncommitted changes
- flow_test.rs - working reference (no riffles)
- riffle_diagnostic.rs - failing case with riffles
