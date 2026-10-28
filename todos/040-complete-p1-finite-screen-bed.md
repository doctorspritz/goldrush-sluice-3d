---
status: complete
priority: p1
issue_id: "040"
tags: [physics, rendering, screen]
dependencies: ["039"]
---

# Finite smooth screen-bed implementation stage

## Problem Statement
The opt-in bulk deck used cell-bin forces and point markers, causing discontinuous bed pressure/presentation; wet basal adhesion incorrectly applied to every layer.

## Findings
A conservative positive smooth field permits a coherent finite envelope and reciprocal pressure. Area-limited basal traction lets a deep wet layer yield. Native captures expose remaining mean-surface/art limitations; gravel grading is still uncalibrated and real time fails.

## Proposed Solutions
Retain the detailed default; preserve legacy saved bulk law; implement the smooth opt-in law and derived geometry. Defer default adoption until mixed/dense/wet calibration and actual loaded operation qualify it.

## Recommended Action
This finite-bed implementation and validation stage is complete. Keep menace-94n, hybrid, real-time and the full operation goal in progress.

## Acceptance Criteria
- [x] Conserve finite field material/volume and zero-self/equal-opposite pair pressure.
- [x] Preserve legacy continuation and explicit new model version.
- [x] Render the mean stock once at recorded deck time with bounded surface detail.
- [x] Check actual dug wet feed, river withdrawal, active restore and stopped cleanout.
- [x] Inspect native captures and report separate FPS/physical speed without a real-time claim.
- [x] Retain compact evidence, remove/archive bulky generated output and publish the stage.

## Work Log
2026-10-28: 154 domain CPU, 99 app and three actual Metal checks pass (256 distinct); two profiling checks remain ignored. Source-scoped formatting, strict lint and native build pass. Final native wet capture progresses and sheds finite stock. Loaded 57.26 FPS / 0.464364× physical time fails real time; gravel screen underflow differs -30.451866% from detailed. See docs/playtests/2026-10-28-screen-bed.md/json. Full mine construction, digging/mud, plants, ponds, tailings, floods/failures and gameplay remain active.
