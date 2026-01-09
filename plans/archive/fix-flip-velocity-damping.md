# fix: FLIP Water Simulation Velocity Damping

**Date:** 2025-12-27
**Status:** PENDING USER APPROVAL
**Category:** physics, bug-fix, simulation
**Severity:** CRITICAL - Water unusable

---

## Overview

Water in the FLIP simulation flows like honey/treacle instead of naturally. Diagnostic output shows ~2% momentum loss per frame, compounding to **70% velocity loss per second**. Previous fix attempts (parameter tweaks, pure FLIP, disabled APIC) failed because they didn't identify the actual root causes.

This plan provides a systematic diagnosis and fix strategy based on comprehensive research of:
- The actual codebase state (what's enabled/disabled)
- FLIP best practices (Bridson, Houdini, SPlisHSPlasH)
- Canonical APIC implementation (Jiang et al. 2015)
- Mathematical verification of transfer formulas

---

## Problem Statement

### Current Behavior
```
MOMENTUM: particles=10471 → P2G_grid=8092 → +gravity=582580 → +pressure=224370 → G2P_particles=10280 → advect=10225
```

**Key Observations:**
- P2G transfer: 23% momentum loss (10471 → 8092)
- Pressure correction: 62% reduction (correct - enforcing incompressibility)
- G2P recovery: Most momentum recovered
- **Net loss per frame: ~2%** (10471 → 10225)
- **Compounded over 1 second: 0.98^60 = 30% retained = 70% LOST**

### Expected Behavior
- Net momentum loss should be < 5% per second for inviscid water
- Water should flow dynamically, not sluggishly
- Vortices should form and persist behind obstacles

---

## Root Cause Analysis

### Confirmed Root Causes (from code analysis)

| Priority | Root Cause | Location | Evidence | Impact |
|----------|-----------|----------|----------|--------|
| **P1** | APIC affine term disabled | flip.rs:308-310, 351-353 | Comments say "DISABLED APIC" | Angular momentum loss, excessive diffusion |
| **P1** | Pressure gradient formula | grid.rs:734-739 | `_dt` parameter ignored | May over/under-correct pressure |
| **P2** | P2G weight normalization | flip.rs:365-379 | 23% momentum loss in P2G | Possible partition-of-unity violation |
| **P2** | Pure FLIP ratio (1.0) | flip.rs:525 | Industry standard is 0.95-0.97 | Potential instability/noise |
| **P3** | CFL at limit (1.0) | main.rs:389, dt=1/120 | CFL=v*dt/dx=120*0.0083/1=1.0 | Borderline stability |

### Confirmed NOT Root Causes

| Suspect | Location | Status | Evidence |
|---------|----------|--------|----------|
| `apply_near_pressure()` | flip.rs:984 | **NOT CALLED** | Grep confirms no call in update() |
| `damp_surface_vertical()` | grid.rs:788 | **NOT CALLED** | Grep confirms dead code |
| PIC smoothing | flip.rs:525 | **DISABLED** | FLIP_RATIO = 1.0 (pure FLIP) |
| Insufficient pressure iterations | flip.rs:171 | **ADDRESSED** | Already increased to 40 |
| CFL violation | main.rs:389 | **ADDRESSED** | Changed from 1/60 to 1/120 |

---

## Technical Approach

### Phase 0: Pre-Flight (BEFORE ANY CODE CHANGES)

#### 0.1 Establish Baseline Measurements

Run current simulation for 60 seconds (3600 frames) and record:

```rust
// Metrics to capture (line 194 already has partial diagnostics)
- Momentum loss per second: ___%  (target: current ~70%)
- Grid kinetic energy: ___
- Enstrophy (vorticity²): ___
- CFL number (max velocity * dt / cell_size): ___
- FPS average: ___
```

**Acceptance:** Baseline documented before any changes.

#### 0.2 Verify Dead Code Status

```bash
# Confirm these are NOT called in update()
grep -n "apply_near_pressure" crates/sim/src/flip.rs
grep -n "damp_surface_vertical" crates/sim/src/grid.rs
```

**Acceptance:** Both confirmed as dead code.

---

### Phase 1: Diagnostic Instrumentation

**Goal:** Add diagnostics to verify suspected issues WITHOUT changing behavior.

#### 1.1 P2G Weight Sum Diagnostic

**File:** `crates/sim/src/flip.rs:365-379`

Add weight sum verification in diagnostic mode (every 60 frames):

```rust
// After P2G transfer, verify weights sum correctly
if diagnose {
    let total_u_weight: f32 = self.u_weight.iter().sum();
    let total_v_weight: f32 = self.v_weight.iter().sum();
    let expected_weight = self.particles.len() as f32; // Each particle contributes weight 1.0

    let u_error = (total_u_weight / expected_weight - 1.0).abs();
    let v_error = (total_v_weight / expected_weight - 1.0).abs();

    if u_error > 0.01 || v_error > 0.01 {
        eprintln!("WARNING: P2G weight mismatch! u_error={:.3}, v_error={:.3}", u_error, v_error);
    }
}
```

**Acceptance:** Weights verified to sum to particle count ± 1%.

#### 1.2 APIC Status Diagnostic

**File:** `crates/sim/src/flip.rs:308-310, 351-353`

Verify current APIC state:

```rust
// In P2G, verify affine term is zero
if diagnose && self.frame % 600 == 0 {
    let sample_c = self.particles.list[0].affine_velocity;
    eprintln!("APIC STATUS: C matrix sample = [{:.3}, {:.3}; {:.3}, {:.3}]",
        sample_c.x_axis.x, sample_c.x_axis.y, sample_c.y_axis.x, sample_c.y_axis.y);
    eprintln!("APIC P2G: DISABLED (lines 308-310)");
}
```

**Acceptance:** Diagnostic confirms APIC is disabled in P2G but C matrix is built in G2P.

---

### Phase 2: Incremental Fixes (ONE AT A TIME)

**CRITICAL:** Each fix must be tested in isolation. Commit after each successful fix.

#### 2.1 Fix FLIP Ratio (LOWEST RISK)

**File:** `crates/sim/src/flip.rs:525`

**Current:**
```rust
const FLIP_RATIO: f32 = 1.0; // Pure FLIP - no PIC smoothing
```

**Fix:**
```rust
const FLIP_RATIO: f32 = 0.97; // Industry standard: 97% FLIP, 3% PIC
```

**Rationale:** Industry standard (Blender, Houdini) uses 0.95-0.97 for stability without excessive damping.

**Test:** Run 60 seconds, measure momentum loss.

**Rollback:** Revert to 1.0 if momentum loss increases.

**Acceptance:** Momentum loss per second reduced OR no regression.

---

#### 2.2 Re-enable APIC Affine Velocity (MODERATE RISK)

**File:** `crates/sim/src/flip.rs:308-310, 351-353`

**Current (DISABLED):**
```rust
// DISABLED APIC: affine term may break momentum conservation
// let affine_vel = c_mat * offset;
let _ = c_mat; // suppress warning
...
// Pure FLIP: just velocity, no affine term
self.u_sum[idx] += vel.x * scaled_w;
```

**Fix (RE-ENABLE):**
```rust
// APIC: affine velocity contribution preserves angular momentum
let affine_vel = c_mat * offset;

self.u_sum[idx] += (vel.x + affine_vel.x) * scaled_w;
```

And for V component (lines 351-353):
```rust
let affine_vel = c_mat * offset;
self.v_sum[idx] += (vel.y + affine_vel.y) * scaled_w;
```

**Rationale:** APIC conserves angular momentum. Disabling it causes excessive numerical diffusion. Reference: [Jiang et al. 2015](https://dl.acm.org/doi/10.1145/2766996)

**Test:**
1. Run 60 seconds, measure momentum loss
2. Verify vortex formation behind obstacles
3. Check for instability/noise

**Rollback:** Re-comment if instability occurs.

**Acceptance:** Angular momentum preserved, vortices form and persist.

---

#### 2.3 Verify Pressure Gradient Formula (HIGH RISK)

**File:** `crates/sim/src/grid.rs:734-739`

**Current:**
```rust
pub fn apply_pressure_gradient(&mut self, _dt: f32) {
    // CRITICAL FIX: Remove dt from scale. With dt, only 0.1% of divergence
    // was corrected per frame. Without dt, we get full correction.
    let scale = 1.0 / self.cell_size;
```

**Issue:** The `dt` parameter is ignored. This contradicts standard FLIP formulation.

**Reference Check Needed:**

From Bridson's "Fluid Simulation for Computer Graphics" (2008):
- Pressure Poisson equation: ∇²p = ρ/Δt · ∇·u
- Velocity update: u^{n+1} = u^* - Δt/ρ · ∇p

The pressure `p` already has time baked in from the RHS of Poisson equation.

**Analysis:** The current code may actually be CORRECT because:
1. `divergence` is computed as `(u_right - u_left + v_top - v_bottom) / h` (no dt)
2. `solve_pressure` solves `∇²p = div` (no dt factor)
3. So `p` has units of [velocity × length]
4. Gradient `∇p / h` has units of [velocity]
5. Subtracting directly is dimensionally correct

**Action:** Add diagnostic to verify divergence before/after pressure solve:

```rust
if diagnose {
    let div_before = self.total_divergence();
    self.apply_pressure_gradient(dt);
    let div_after = self.total_divergence_after_gradient();
    eprintln!("PRESSURE: div_before={:.4}, div_after={:.4}, reduction={:.1}%",
        div_before, div_after, 100.0 * (1.0 - div_after / div_before.max(0.0001)));
}
```

**Acceptance:** Divergence reduced to < 0.01 after pressure correction.

---

#### 2.4 Reduce CFL to 0.67 (LAST RESORT)

**File:** `crates/game/src/main.rs:389`

**Current:**
```rust
let dt = 1.0 / 120.0;  // CFL ≈ 1.0
```

**Fix:**
```rust
let dt = 1.0 / 180.0;  // CFL ≈ 0.67
```

**Rationale:** CFL < 1.0 required for stability. Current CFL = 1.0 is at the limit. Reducing to 0.67 gives safety margin.

**Trade-off:** 50% more simulation steps per rendered frame = ~30% FPS drop.

**When to Apply:** Only if fixes 2.1-2.3 are insufficient.

**Acceptance:** CFL < 1.0 and acceptable FPS (>30).

---

### Phase 3: Combined Validation

After individual fixes proven:

1. Apply all successful fixes from Phase 2
2. Run 60-second test
3. Measure:
   - Momentum loss: should be < 10% per second
   - Kinetic energy: should decay slowly
   - Visual: water should flow dynamically
4. Run 10-minute stability test (36000 frames)
5. Verify no explosions, drifts, or artifacts

---

## Acceptance Criteria

### Functional Requirements

- [ ] Water flows dynamically, not like honey
- [ ] Momentum loss < 10% per second (currently 70%)
- [ ] Vortices form behind obstacles and persist
- [ ] No visual artifacts or instabilities
- [ ] Simulation remains stable for 10+ minutes

### Non-Functional Requirements

- [ ] FPS >= 30 (preferably >= 45)
- [ ] CFL number < 1.0 for all typical velocities
- [ ] No new memory allocations in hot path

### Quality Gates

- [ ] Each fix tested in isolation before combining
- [ ] Baseline measurements captured before any changes
- [ ] Diagnostics verify weight normalization
- [ ] Git commit after each successful fix

---

## Risk Analysis & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| APIC re-enable causes instability | Medium | High | Test in isolation, easy to revert |
| Pressure formula change breaks solver | Low | Critical | Verify with divergence diagnostic first |
| CFL reduction tanks FPS | High | Medium | Apply as last resort only |
| Fixes interact badly | Medium | Medium | Test combined after individual validation |
| No single fix sufficient | Medium | High | May need multiple fixes together |

---

## Dependencies & Prerequisites

- [ ] Git working tree clean (no uncommitted changes that could be lost)
- [ ] Baseline measurements captured
- [ ] Reference material accessible (Bridson textbook, APIC paper)
- [ ] Understanding of existing diagnostic output format

---

## File Locations

### Files to Modify

| File | Lines | Change |
|------|-------|--------|
| `crates/sim/src/flip.rs` | 525 | FLIP_RATIO: 1.0 → 0.97 |
| `crates/sim/src/flip.rs` | 308-310, 351-353 | Re-enable APIC affine term |
| `crates/sim/src/flip.rs` | ~365 | Add weight diagnostic |
| `crates/sim/src/grid.rs` | 734-739 | Verify pressure formula |
| `crates/game/src/main.rs` | 389 | dt: 1/120 → 1/180 (if needed) |

### Files to Read (Reference)

| File | Purpose |
|------|---------|
| `docs/APIC_IMPLEMENTATION_GUIDE.md` | APIC formula reference |
| `docs/compound/failure-claude-impulsive-fixes-2025-12-27.md` | Previous failed attempts |
| `todos/003-pending-p1-cfl-violation.md` | CFL analysis |
| `todos/004-pending-p1-pressure-iterations.md` | Pressure solver analysis |

---

## Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Momentum loss/sec | 70% | < 10% | Diagnostic line 194 |
| CFL number | 1.0 | < 1.0 | `v_max * dt / cell_size` |
| Vortex persistence | None | Visible | Visual inspection |
| FPS | ~60 | >= 30 | `get_frame_time()` |

---

## References

### Internal References
- `crates/sim/src/flip.rs:129-204` - Main update loop
- `crates/sim/src/flip.rs:246-380` - P2G transfer (APIC disabled here)
- `crates/sim/src/flip.rs:389-529` - G2P transfer
- `crates/sim/src/grid.rs:621-659` - Pressure solver
- `crates/sim/src/grid.rs:732-779` - Pressure gradient application

### External References
- [Bridson: Fluid Simulation for Computer Graphics](https://www.cs.ubc.ca/~rbridson/fluidsimulation/fluids_notes.pdf) - Canonical FLIP reference
- [Jiang et al. 2015: APIC Paper](https://dl.acm.org/doi/10.1145/2766996) - Affine velocity conservation
- [Ten Minute Physics: FLIP Tutorial](https://matthias-research.github.io/pages/tenMinutePhysics/18-flip.pdf) - Educational reference
- [Houdini FLIP Solver Docs](https://www.sidefx.com/docs/houdini/nodes/dop/flipsolver.html) - Industry parameters
- [Blender FLIP Fluids](https://github.com/rlguy/Blender-FLIP-Fluids/wiki/Domain-Advanced-Settings) - FLIP_RATIO defaults

### Related Work
- Previous failure: `docs/compound/failure-claude-impulsive-fixes-2025-12-27.md`
- CFL analysis: `todos/003-pending-p1-cfl-violation.md`
- Pressure analysis: `todos/004-pending-p1-pressure-iterations.md`

---

## Implementation Order

```
Phase 0: Baseline (NO CODE CHANGES)
    └── Capture measurements
    └── Verify dead code status

Phase 1: Diagnostics (BEHAVIOR UNCHANGED)
    └── 1.1 Weight sum diagnostic
    └── 1.2 APIC status diagnostic

Phase 2: Fixes (ONE AT A TIME)
    └── 2.1 FLIP ratio → 0.97 (lowest risk)
    └── 2.2 Re-enable APIC (moderate risk)
    └── 2.3 Verify pressure formula (high risk)
    └── 2.4 Reduce CFL (last resort)

Phase 3: Validation
    └── Combined testing
    └── Long-term stability
```

---

## Notes

**CRITICAL REMINDER FROM CLAUDE.md:**
- Make ONE small change at a time
- Test after each change
- Commit working states frequently
- NEVER make large refactors without user approval

**Previous failures documented in** `docs/compound/failure-claude-impulsive-fixes-2025-12-27.md` - parameter tweaks without root cause analysis all failed.

**This plan is different because:**
1. Based on comprehensive code analysis (not guessing)
2. Verified what's actually enabled/disabled
3. Compared to canonical FLIP references
4. Incremental with clear rollback points
