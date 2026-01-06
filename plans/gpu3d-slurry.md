# Plan: GPU 3D Slurry (Water + Sand) with Deposition/Entrainment

## Problem Statement
The GPU 3D FLIP simulation only models water. We need slurry behavior where sand behaves as a suspended solid:
- Sand deposits in slow flow and accumulates behind riffles.
- Sand re-entrains under high flow and washes downstream.
- Water remains incompressible and stable (sediment must not break pressure solve).
- Target performance: 60 FPS with ~200k particles.

## Scope
- GPU 3D pipeline only (industrial_sluice).
- Start with a single sediment material (sand) using density > 1.0.
- Approximate physics is acceptable; prioritize stability and performance.
- Multi-material and full DEM are deferred.

## Core Approach
1) **Water stays FLIP/APIC.**
   - Only water contributes to P2G momentum and cell fluid markers.
2) **Sediment is Lagrangian solid.**
   - Sediment follows grid velocity via drag + settling (PIC-like).
   - No FLIP delta for sediment.
3) **Two-way coupling via porosity drag.**
   - Sediment volume fraction damps grid velocities (Darcy-like).
4) **Continuous bed evolution (no switchy thresholds).**
   - Maintain a bed heightfield per (x,z) column with smooth exchange rates.
   - Deposition and entrainment driven by shear and settling (Shields/Ferguson-Church).
   - Bedload transport uses a flux law (rolling as continuous transport, not cell flips).
   - Mark bed cells as solid in `cell_types` each frame.

## Phased Plan

### Phase 1: Sediment Data Plumbing (GPU 3D)
**Goal:** Carry sediment identity/density to shaders without affecting P2G momentum.

- Add per-particle `density` (or `material`) buffer alongside positions/velocities/C.
  - Source: `sim3d::Particle3D.density` already exists.
- Extend `GpuP2g3D` and `p2g_scatter_3d.wgsl`:
  - Skip momentum accumulation for sediment (`density > 1.0`).
  - Accumulate sediment count/volume in a new `sediment_count` atomic buffer.
  - Keep `particle_count` for water only (density projection correctness).
- Add `sediment_fraction` buffer (cell-centered f32) computed from counts.
  - New compute pass: `sediment_fraction_3d.wgsl`.
  - Fraction = clamp(count / REST_PARTICLES_PER_CELL, 0..1).

**Files:**
- `crates/game/src/gpu/p2g_3d.rs`
- `crates/game/src/gpu/shaders/p2g_scatter_3d.wgsl`
- `crates/game/src/gpu/shaders/sediment_fraction_3d.wgsl` (new)

### Phase 2: Sediment Forces (GPU G2P)
**Goal:** Sediment follows flow and settles without breaking water.

- Extend `g2p_3d.wgsl` to branch on `density`:
  - **Water:** existing FLIP/APIC.
  - **Sediment:**
    - Sample grid velocity (PIC).
    - Apply drag: `v = lerp(v, v_grid, drag * dt)`.
    - Apply settling: `v.y += settling_velocity * dt` (negative Y).
    - Optional turbulence suspension: reduce settling when vorticity magnitude is high.
- Add params buffer for sediment tuning:
  - `drag_rate`, `settling_scale`, `vort_lift_scale`, `vort_min_threshold`.
- Reuse existing `vorticity_mag_buffer` for turbulence intensity.

**Files:**
- `crates/game/src/gpu/g2p_3d.rs`
- `crates/game/src/gpu/shaders/g2p_3d.wgsl`
- `crates/game/src/gpu/flip_3d.rs`

### Phase 3: Two-Way Coupling (Porosity Drag)
**Goal:** Water slows in dense sediment regions without pressure instability.

- Add `porosity_drag_3d.wgsl` pass:
  - Use `sediment_fraction` (cell centered).
  - For each face (u/v/w), damp velocity based on avg of adjacent cell fractions.
  - Exponential damping: `v *= exp(-drag * frac^2 * dt)`.
- Insert pass after pressure solve, before G2P.

**Files:**
- `crates/game/src/gpu/shaders/porosity_drag_3d.wgsl` (new)
- `crates/game/src/gpu/flip_3d.rs`

### Phase 4: Continuous Bed Evolution (Shear + Settling)
**Goal:** Sand accumulates and erodes smoothly (no hard thresholds), with rolling/bedload transport.

- Maintain CPU heightfield `bed_height[x,z]` and `bed_mass[x,z]` (continuous).
- Compute near-bed flow per (x,z) bin every N frames:
  - Accumulate water particle velocities in a thin layer above the bed (1-2 cells).
  - Derive shear velocity `u*` from near-bed speed (u* ~= sqrt(Cf) * |U|).
- Deposition rate (smooth):
  - `D = ws * C` where `ws` is Ferguson-Church settling velocity and `C` is near-bed sediment concentration.
  - Remove particle mass probabilistically or via fractional mass bookkeeping (no binary switch).
- Entrainment rate (smooth):
  - Shields parameter: `theta = (rho_w * u*^2) / ((rho_s - rho_w) * g * d)`.
  - `E = k_e * max(0, theta - theta_c)^n` with `theta_c` from sand Shields.
  - Spawn particles proportional to `E` (continuous exchange).
- Bedload (rolling) transport:
  - Compute bedload flux `q_b = k_b * max(0, theta - theta_c)^(1.5) * dir(U)`.
  - Update bed via Exner: `d(eta)/dt = -(div q_b) + (D - E)/rho_s`.
- Update `cell_types` each frame: mark `j <= bed_height[x,z]` as solid.
- Optional: add a simple heightfield collision term to `sdf_collision_3d.wgsl` for bed contact.

**Files:**
- `crates/game/examples/industrial_sluice.rs` (heightfield + deposition/entrainment)
- `crates/game/src/gpu/flip_3d.rs` (cell_type upload includes bed)
- `crates/game/src/gpu/shaders/sdf_collision_3d.wgsl` (optional heightfield collide)

### Phase 5: Diagnostics + Tuning
- Add controls to adjust drag/settling/entrain thresholds in real time.
- Visualize sediment fraction / bed height (color overlay or debug print).
- Add counters: deposited cells, entrained cells, sediment count.

## Performance Strategy (60 FPS @ 200k)
- GPU passes are O(grid) or O(particles) with simple math.
- Avoid neighbor searches or DEM in 3D.
- Update deposition/entrainment every 2-4 frames.
- Use small heightfield (width * depth) buffer for dynamic bed.
- Keep extra GPU buffers to fixed size; avoid per-frame allocation.

## Acceptance Criteria
- Sand particles remain suspended in high-vorticity/high-speed zones.
- Sand deposits behind riffles smoothly (no flicker/step artifacts).
- Increasing flow smoothly re-entrains deposits (no abrupt mass loss).
- Bedload transport shows rolling/downslope movement tied to flow direction.
- Water remains stable (no divergence explosion).
- FPS stays near target with 200k particles in industrial_sluice.

## Risks
- Over-damping from porosity drag can kill flow; cap drag and apply only above threshold.
- Bed exchange rates need careful tuning to avoid runaway erosion or permanent buildup.
- Heightfield collision may introduce artifacts if not blended with SDF.

## Follow-ups (Deferred)
- Multi-material deposition (sand/magnetite/gold) with per-cell composition.
- Rouse-based suspension for better physics and vertical segregation.
- GPU-based deposition/entrainment to remove CPU cost.
