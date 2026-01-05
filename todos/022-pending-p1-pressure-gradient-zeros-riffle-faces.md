---
status: pending
priority: p1
issue_id: "022"
tags: [code-review, physics, pressure, critical]
dependencies: ["020"]
---

# Pressure Gradient Zeros Velocity at ALL Solid Faces

## Problem Statement

The pressure gradient shader zeros U velocity at ALL solid-fluid interfaces, including the faces above riffles where water should be able to flow over.

**Why it matters:** Combined with issue #020 (no flow accel at solid faces), this ensures water has no mechanism to flow over riffles.

## Findings

### From Performance Oracle agent:

"This zeros U velocity at ALL solid-fluid interfaces, including riffles. When water hits a riffle:
1. The riffle's upwind face becomes solid-fluid interface
2. Pressure gradient shader zeros U velocity at this interface
3. Particles lose X-velocity via FLIP blend
4. Water pools behind riffle"

### Location:
`crates/game/src/gpu/shaders/pressure_gradient_3d.wgsl` lines 99-102:
```wgsl
// If either side is solid, zero velocity (no-penetration)
if (left_type == CELL_SOLID || right_type == CELL_SOLID) {
    grid_u[idx] = 0.0;
    return;
}
```

### Physics issue:
This correctly prevents flow THROUGH solids but incorrectly prevents flow OVER them. A riffle is a barrier, not a seal - water should be able to pass above it.

## Proposed Solutions

### Option A: Only zero velocity for faces INTO solids
**Pros:** Allows flow over riffle tops
**Cons:** More complex solid geometry detection needed
**Effort:** Medium
**Risk:** Medium - must be careful about which faces

### Option B: Don't zero at free-surface faces adjacent to solids
**Pros:** Allows overflow behavior
**Cons:** Complex to detect free surface
**Effort:** Large
**Risk:** High

### Option C: Accept current behavior, fix via issues #020 and #021
**Pros:** Simpler - fixes root causes instead
**Cons:** Doesn't address this symptom directly
**Effort:** None (relies on other fixes)
**Risk:** Low

## Recommended Action

Option C - this issue is a symptom. Fixing #020 (flow accel) and #021 (spawn height) should resolve the pooling problem without changing pressure gradient logic.

## Technical Details

**Affected files:**
- `crates/game/src/gpu/shaders/pressure_gradient_3d.wgsl`

**Dependencies:**
- Should be evaluated AFTER fixing issues #020 and #021

## Acceptance Criteria

- [ ] Verify if still needed after #020 and #021 are fixed
- [ ] If still needed: water flows over riffles correctly

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-03 | Identified by code review | Symptom, not root cause |
