# Fix Riffle Flow - Gated Implementation Plan

## Problem Statement

Water pools behind riffles at 0.12 m/s instead of flowing over them at 1.6 m/s.

**Root Causes Identified:**
1. `flow_3d.wgsl` skips flow acceleration at solid-adjacent faces (riffles) - FIXED
2. Water spawns at y=0.44m = exact riffle top height (cannot overflow) - FIXED
3. `pressure_gradient_3d.wgsl` KILLS all velocity at solid faces instead of just preventing penetration - **NEW ISSUE**

## Gates

Each gate MUST pass before proceeding to the next.

---

## GATE 0: Baseline Measurement
**Status:** [x] PASSED

### Objective
Establish current broken behavior with hard numbers.

### Baseline Results (2026-01-03)
- **Avg Vx:** 0.123 m/s
- **Total Spawned:** 1500
- **Total Exited:** 0
- **Result:** FLOW IS BLOCKED

---

## GATE 1: Fix Flow Shader
**Status:** [x] PASSED

### Objective
Apply flow acceleration to fluid cells even when adjacent to internal solids (riffles).

### What Was Done
- Removed early return for solid-adjacent faces in `flow_3d.wgsl`
- Now applies flow to any face with at least one FLUID cell

### Verification
- `flow_test` (simple slope) passes: 100% exit, 1.6 m/s

---

## GATE 2: Fix Spawn Height
**Status:** [x] PASSED

### Objective
Spawn water ABOVE riffle tops, not at them.

### What Was Done
- Changed `spawn_inlet_water` to calculate spawn height from first riffle top + 2 cells
- Spawn height now 0.520m (above riffle top at 0.440m)

### Verification
- riffle_diagnostic shows: "Spawn height: y=0.520m (2 cells above riffle top)"

---

## GATE 2.5: Add Density-Based Pressure Solve
**Status:** [ ] NOT STARTED

### Root Cause Analysis (Updated 2026-01-04)

**Original hypothesis was WRONG**: The directional no-penetration fix (zeroing only velocity INTO solids) would help at boundaries, but it's NOT the root cause of water not rising to overflow.

**The REAL problem**: Standard FLIP incompressibility doesn't handle volume accumulation.

In standard FLIP:
1. Grid sees cells as "fluid" or "not fluid" (binary)
2. Pressure solve enforces ∇·v = 0 (incompressible)
3. 10 particles vs 100 particles in a cell = same pressure behavior
4. Particles can pile up in cells without increasing effective "water level"

**What's missing**: When particles accumulate (pile up behind riffle), there's no mechanism to:
- Recognize the crowding
- Generate extra pressure to push particles apart
- Cause water level to "rise" and overflow

### Research Findings

1. **DFSPH/PBF** (crates/dfsph/src/simulation.rs) - 2D only, different algorithm:
   - Has REST_DENSITY and density tracking per particle
   - Uses SPH kernels to compute density from neighbors
   - Lambda correction pushes particles from crowded areas

2. **Unified Pressure Plan** (plans/unified-pressure-based-physics.md) - NOT implemented:
   - Describes: `pressure = cumulative_mass_above * GRAVITY`
   - Would naturally create hydrostatic pressure from column height
   - Was planned but never built

3. **Houdini Volume Solve** (mentioned by user) - NOT found in codebase:
   - Creates pressure gradient based on particle density
   - May have been removed or was conceptual

### Proposed Solution: Density Correction in Divergence

Add a density term to the divergence calculation:

```wgsl
// divergence_3d.wgsl - MODIFIED
let standard_div = (u_right - u_left + v_top - v_bottom + w_front - w_back) * inv_dx;

// Count particles in this cell (need new buffer)
let particle_count = cell_particle_count[idx];
let rest_particles = 8.0;  // ~8 particles per cell at rest
let density_error = (f32(particle_count) - rest_particles) / rest_particles;

// Add density correction: crowded cells have positive divergence (push apart)
let DENSITY_COEFF = 0.5;  // tune this
divergence[idx] = standard_div + DENSITY_COEFF * density_error;
```

**Why this works**:
- When particles crowd: density_error > 0 → positive divergence
- Pressure solve sees positive divergence → creates outward pressure gradient
- Particles get pushed from crowded to empty areas
- Water level effectively "rises" when particles accumulate

### Implementation Tasks

- [ ] **2.5a:** Add particle count per cell buffer (GPU)
- [ ] **2.5b:** Compute particle counts in P2G pass
- [ ] **2.5c:** Modify divergence shader to include density correction
- [ ] **2.5d:** Tune DENSITY_COEFF parameter
- [ ] **2.5e:** Test: water piles up behind obstacle, then overflows

### Alternative: Hydrostatic Pressure Approach

If density correction doesn't work, try explicit hydrostatic pressure:

1. Compute water column height per (x,z) from particle positions
2. Add hydrostatic pressure: p_hydro = ρ * g * h
3. Include this in pressure gradient application

### Pass Criteria
- [ ] Particles accumulate behind riffle (pooling visible)
- [ ] After accumulation, water level rises
- [ ] Water eventually overflows riffle top
- [ ] riffle_diagnostic: avg Vx > 0.3 m/s, exits > 0

---

## GATE 3: Programmatic Verification
**Status:** [ ] NOT STARTED

### Objective
Run riffle_diagnostic and verify flow actually works with riffles.

### Pass Criteria
- [ ] `cargo run --example riffle_diagnostic --release`
- [ ] avg Vx > 0.3 m/s
- [ ] exit count > 0
- [ ] Particles visibly progress past all riffles

---

## GATE 4: Visual Verification
**Status:** [ ] NOT STARTED

### Objective
User visually confirms flow works.

### Expected Visual Behavior
1. Water enters from left (inlet) at elevated height
2. Water flows OVER first riffle (not pooling behind it forever)
3. Water continues downstream, passing each riffle
4. Particles exit on right side (outlet)
5. Console shows: avg Vx > 0.3, Exit count increasing

### Pass Criteria
- [ ] Run `cargo run --example gold_sluice_3d --release`
- [ ] User confirms behavior matches expectations

---

## Rollback Plan

If any gate fails:
1. DO NOT proceed to next gate
2. Debug the specific gate that failed
3. Re-run gate verification
4. Only proceed when gate passes

## Current Status

**Completed:** GATE 0, 1, 2
**In Progress:** Root cause analysis complete - the real issue is volume accumulation

**Key Finding (2026-01-04):**
The original hypothesis about pressure_gradient_3d.wgsl killing velocity was a red herring.
The test passed because G2P interpolation averages multiple faces, smoothing out any zeroed values.

The REAL issue: Standard FLIP doesn't handle volume accumulation. Particles pile up in the same
cells but the grid doesn't see this as "higher water" because it only tracks fluid/not-fluid.

**Next Action:** GATE 2.5a - Add particle count per cell buffer to enable density-based pressure
