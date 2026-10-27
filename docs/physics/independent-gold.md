# Independent downstream gold

Current operations preserve pure gold discharged by the sluice as saved world particles. This is a dilute, one-way, physical-diameter point-parcel model. It implements a downstream destination and finite custody; it does not establish calibrated placer-gold recovery, flake hydraulics or a complete mining operation. The wider acceptance criteria remain in the [realistic-operation plan](../../plans/2026-10-22-realistic-operation.md).

Geological paydirt retains its existing [terrain-gold provenance model](terrain-gold.md). Its bed/mobile gold tag follows the coarse paydirt class and its actual water/bed events. Independent discharged gold does not enter that tag, acquire a fictitious paydirt carrier, or inherit the coarse class's settling diameter. The original in-sluice FLIP/APIC and DEM calculation continues to determine actual solver-domain exits.

## Motion and flow

`mining-sim::loose_gold::LooseGold` model version 1 retains the complete `Grain`, its diameter, gold mass, rotation, angular velocity and affine matrix. Each world particle has a saved identity and authoritative f64 position/velocity, mirrored into the grain's f32 fields. A weighted parcel can represent multiple physical grains: its physical diameter controls this model's settling and contact, while its mass controls inventory. The packed DEM parcel envelope is not used as a giant downstream gold particle.

The terminal relation uses the same [Ferguson–Church form](https://geoweb.uwyo.edu/geol5330/FergusonChurch_GrainSettling_JSR04.pdf) with `C1 = 18`, `C2 = 1` already implemented in `crates/game/src/gpu/shaders/g2p_3d.wgsl` and `phase_transfer_3d.wgsl`:

```text
R = rho_gold / rho_water - 1
w_s = R g d² / (18 nu + sqrt(0.75 R g d³))
g_eff = g (1 - rho_water / rho_gold)
k = g_eff / w_s
```

Here `g = 9.81 m/s²`, `rho_gold = 19,300 kg/m³`, `rho_water = 1,000 kg/m³`, and `nu = 10^-6 m²/s`. These are the implemented prototype assumptions, not measurements of a particular claim or slurry. Reusing the relation does not qualify its application to gold flakes or dense, clay-bearing feed.

For constant sampled flow `u`, submerged motion solves the linear relaxation law analytically for both velocity and displacement:

```text
v_inf = u - w_s Y
v(t) = v_inf + (v_0 - v_inf) exp(-k t)
x(t) - x_0 = v_inf t + (v_0 - v_inf) (1 - exp(-k t)) / k
```

Dry motion uses ballistic gravity. The environmental solver supplies current depth-averaged horizontal velocity, sampled with wet-depth-weighted bilinear weights; the water surface is sampled bilinearly. It supplies no resolved vertical velocity or turbulent eddies. The gold step neither adds water nor applies an equal-and-opposite drag impulse to the SWE solver. Analytic integration is exact for a constant sampled field, not for arbitrary spatially and temporally varying flow.

Existing world particles consume the fixed timestep before outlet flight advances. At first flight contact, atomic admission advances only the new gold particle through the remaining part of that timestep. An old pending-contact packet receives the current timestep on successful admission. Existing particles are not advanced again by that transfer.

## Bed contact and burial

Bed height and normal follow the actual two-triangle heightfield diagonal used by `soil::substrate_height`, evaluated from the terrain nodes in f64. This differs from using the smoother bilinear terrain query for penetration. Physical radius and plane normal determine clearance. Bracketed impacts are localized within a bounded substep, then the remaining time advances with bed forces.

The saved prototype Coulomb coefficient defaults to `0.6`. Static friction can hold supported gold; sufficiently strong modeled drag or downslope gravity can produce bed motion. Collinear planar drag and friction use an analytic solution, including stopping. Changing two-dimensional slip direction freezes the friction direction within each bounded substep. Internal-edge contact-normal correction, finite flake geometry and grain–grain/moving-clod collisions are not resolved by this downstream model.

Deposited terrain can cover an existing particle. Burial retains its world position rather than lifting it onto each new surface increment; lowering the support can uncover it. Supported, buried and mobile state is saved, with resting-state mirrors checked on restore. This is geometric burial, not a granular packing, pore-pressure or soil-collapse model. Retained spin/orientation does not imply qualified rotational drag, lift or flake torque; the saved affine matrix is not an environmental APIC reconstruction.

## Transfers, limits and saves

Independent custody closes as:

```text
admitted = active particles + exported + excavated + withdrawn
```

Only active particles count as current Site stock. The cumulative export is counted once in Session's gold/material ledger. Admission, excavation and withdrawal receipts are cumulative transfer diagnostics and are not added again as stored material. Claim-boundary exits become finite exported mass; they are not clamped back onto the map or presented as recovered gold.

Finite tool cuts collect only gold overlapping their actual triangle footprint and removed depth, including exposed gold on bare rock. Mixed pump withdrawal samples only local mobile gold by the represented withdrawn-water fraction; it excludes bed-supported and buried particles. These existing bulk boundaries transfer grams and discard individual grading and shape. Modern bulk redeposition reconstructs nominal 100 µm gold separately from mineral deposits; the existing pumped inlet also reconstructs nominal 100 µm gold, while hopper feeding uses its existing prototype grading. They preserve mass but do not preserve the original gold population through a complete dig/haul/reprocess loop. That fidelity remains open.

Hopper handling includes gold's physical solid volume, and finite pending feed remains owned across feed changes. The modern feed path can drain pending gold into bounded batches of physical DEM grains when feed, pump and actual water availability permit. Storage limits are numerical resource limits: the independent owner admits at most 65,536 particles, and motion has bounded substeps/work. Refusal retains the donor packet or the pre-step owner; it does not discard mass, enlarge grains, clip velocity or prescribe a recovery percentage.

Version 13 saves include the independent owner and its model, identity, motion, contact/burial state and cumulative receipts. Legacy version 12 remains the previous spatial-tailings policy; earlier formats retain their established mechanism/version rules. Modern application loading can explicitly enable the new owner for an older shift. Historical pooled gold keeps its existing owner because its former position and particle grading were never recorded. Migration does not fabricate that history. Gold-bearing packets that cannot be admitted retain pending tailings custody.

Component controls target analytic dry/wet motion, parcel-weight independence, static and moving contact, passive slope energy, timestep refinement, burial/uncovering, finite cuts and intake, refusal, exports and saved continuation. Their scope does not establish empirical calibration. Real-GPU integration, native motion/rendering review, throughput measurements, dense-slurry/rough-bed behavior, flake sensitivity and end-to-end operating assays require their own dated evidence. This model document does not claim those gates have passed.
