# Equipment accuracy and validation

Accuracy is a measured property of a specified configuration. A plausible render, conserved material or longer particle residence cannot establish calibrated gold recovery. This project keeps numerical verification, observed qualitative behaviour and external experimental validation separate.

## Current model and the corrections in this milestone

The sluice uses the existing 1 cm FLIP/APIC pressure/advection solver, 120 Hz fluid/contact updates, 80/40 pressure sweeps and CPU rigid-clump contacts. Water/terrain accounting remains complete. Changes target mistakes in the coupling rather than prescribing a recovery percentage:

- New shifts use Coulomb grain/wall and grain/grain impulses with rotational effective mass, static sticking or bounded sliding, and a normal-impulse-based floor rolling resistance. The older externally integrated path omitted tangential grain/grain friction and damped wall sliding by an arbitrary rate.
- CPU SDF interpolation now uses the same cell-centred coordinates as water. Its prior half-cell offset displaced contacts by 5 mm on the default grid. Contact normals are derivatives of the interpolated distance, replacing nearest-node finite differences.
- For the Coulomb path, the GPU advects grains without first removing their incoming normal velocity. CPU DEM then resolves the finite-radius impact every substep. Previously the point collision could erase the normal reaction before DEM computed friction.
- An individual physical grain uses its physical collision radius. The 0.60 packed-volume factor applies to weighted bulk parcels, not individual gold grains.
- Quadratic GPU transfer support is centred on the nearest node. Previously a floored stencil could omit a nonzero kernel weight and generate a false APIC velocity gradient even in uniform water. Scatter, tiled scatter, gather and the archived DEM bridge now use the complete support; cell-centric gathering includes all candidate particle cells.
- Parcel volume can be varied explicitly for convergence studies. New saves record the contact model and parcel override; old saves default to their damping model. This preserves load compatibility, not every earlier numerical defect.
- Pump-off cleanup collects only settled parcels over the mat region. Moving or upstream parcels remain in the physical domain and ledger; they neither block every settled collection nor become free concentrate. Even a single fine quartz grain below the bulk empty-material tolerance is preserved.
- Operator arrows adjust water in 0.25 L/s steps and feed in 0.025 L/s steps. Shift retains the coarser steps. The HUD displays the precision needed to see the change.

The shared impulse implementation follows contact-point velocity and the Coulomb friction cone. It is a sequential, rigid-contact approximation; it is not the complete stacking algorithm in [Guendelman, Bridson and Fedkiw (2003)](https://www.cs.ubc.ca/~rbridson/docs/rigid_bodies.pdf). Its coefficients still need material-specific measurements.

The transfer invariant comes from the complete quadratic basis used by [Jiang et al. (2015), APIC](https://doi.org/10.1145/2766996). A manufactured constant-flow GPU check must return the same velocity with a zero affine gradient, within floating-point roundoff, throughout a cell.

## Reproducible sluice experiments

`mining-physics/examples/sluice_accuracy.rs` runs one case per process. Outputs include the normal complete save and a versioned JSON receipt. Reusing an output filename is rejected.

```bash
cargo +nightly run -p mining-physics --example sluice_accuracy \
  --profile playable --jobs 1 -- /tmp/sluice-study nominal 30
```

Available cases are low/nominal/high water, heavy feed, a bare-channel control, and a coarse/fine parcel pair at identical feed. A trial has 8 seconds of clear-water warmup, 12 seconds of bed loading, a configurable tracer interval, 10 seconds of clear-water flushing and 15 seconds after pump shutdown. Laboratory feed is explicitly specified gold-free quartz gravel. Each release adds 24 physical gold grains and 24 equal-sized quartz grains in each of four diameter bins: 80, 150, 300 and 500 micrometres. This known input is laboratory test material, not manufactured gameplay yield.

Report separately:

1. **Delivered operating conditions:** requested and measured water/solid rates; a capacity-limited run cannot represent the requested setting.
2. **Residence and outlet loss by size:** gold still in the channel is never called recovered. Lost-by-size mass is inferred from the known labelled feed and remaining particles, then checked against actual outlet gold.
3. **Cleanup:** call the actual gameplay cleanup and report its error. If cleanup is unavailable, recovery is undefined; report null, not a successful recovery percentage.
4. **Bed and flow:** settled volume and peak height by riffle, reverse and slow-flow samples. These are pocket/backflow diagnostics, not a claim of resolved turbulent spectra.
5. **Numerics and boundaries:** global water/gold residuals and penetration into the same solid geometry used by water.
6. **Throughput:** simulated versus wall seconds. Rendering FPS is a separate native-app measurement.

Twenty-four deterministic lateral release positions do not justify statistical confidence intervals. Repeat releases, timing, seeds and size/shape distributions before generalising. Thirty seconds improves on the earlier eight-second tracer test, but still does not establish a long-duration equilibrium recovery.

To evaluate the cleanup operation on an existing trial checkpoint without re-running its physics:

```bash
cargo +nightly run -p mining-physics --example sluice_accuracy \
  --profile playable --jobs 1 -- /tmp/cleanup-assays --cleanup /tmp/sluice-study/nominal.save.json
```

This writes a separate receipt and preserves the original save/report. The receipt distinguishes a cleanup reassessment from additional simulated time.

## External calibration

[Poling and Hamilton's UBC pilot study](https://ygsftp.gov.yk.ca/ygsftp/publications/tech/fine_gold_recovery_sluiceboxes.pdf) varies water, solids, slope and riffle/matting configuration. It links recovery to bed scour, documents fine-gold losses at high scour, and describes riffle burial at low flow. Its 12 inch by 8 foot apparatus differs from this game's channel and lacks a direct geometry match. Its recovery percentages are comparison data, not values to hard-code into this solver. Preserve the original units: US gallons/minute, inches/foot, pounds/minute and sieve fractions; separately verify whether a metric flow is normalised by channel width.

The [reference manifest](reference-cases.json) records source provenance and blocks unsupported comparisons. Transcribe numerical recovery tables only after inspecting the original table images. Match channel dimensions, riffle cross-section/spacing, mat construction, slope, water delivery, feed PSD/density, gold shape and the cleanup procedure before comparing absolute recovery.

[Robinson, Luding and Ramaioli (2013)](https://arxiv.org/html/1301.0752v2) use single-grain, porous-block and multi-particle tests to validate fluid/particle coupling. Adopt that hierarchy. A settling-law check and a P2G batching-equivalence test do not establish equal-and-opposite interphase drag. `coupling_balance` explicitly measures a zero-gravity momentum budget at two timesteps.

## Remaining accuracy gates

| Gate | Required evidence |
|---|---|
| Interphase coupling | Equal/opposite local drag reaction; global closed momentum budget; porosity-aware pressure/continuity; single-grain and porous-block validation |
| Contact and packing | Oblique impacts, energy/momentum, static support, wet/dry friction and rolling; dense-bed convergence; no unresolved wall penetration or tunnelling |
| Geometry | Consistent water/grain boundaries; smooth/cut-cell ramp and matched riffles/mats; grid-resolution convergence |
| Bulk representation | Parcel/grid-resolution convergence with delivered operating conditions and no numerical admission starvation; preserved PSD, density, feed and solid volume |
| Fine gold | Measured shape-dependent drag/lift and wetting; the visual flake mesh is not a flake drag law |
| Recovery | Low/normal/high flow across feed, slope, size and shape; true cleanup assay; repeatability; independent experimental holdouts |
| Long operation | Stable scour/loading, surges, clogging, prolonged clear-water flow, live save/restore and finite mass balance |

Current weighted bulk spheres, thresholded bed obstruction and unresolved mat fibres remain approximations. Do not tune friction or drag merely until one recovery target looks correct. Record parameters, source revision/binary hash, conditions, failures and independent validation cases.

## Shaker deck and subsequent components

Reuse the physical grain properties, contact checks and input/resident/product/tailings assay conventions. The archived shaker examples are prototype code, not a validated screen.

For the deck, explicitly specify aperture geometry and wire thickness, stroke amplitude, frequency, direction/phase, slope, spray-water rate and feed PSD. For sinusoidal motion `x=A sin(2πft)`, the peak acceleration ratio is `(2πf)² A/g`; amplitude is the single-sided SI displacement. Validate a single grain on the moving surface before adding beds, then known-size grains at apertures, bed stratification, wet screening, blinding and finite stream transfers. A grain must pass the geometry through physical motion; avoid an arbitrary size-based passing probability.

Full equipment matching remains tracked in `sluice-calibrate`, with conservative phase exchange in `sluice-phase-coupling` and deck work in `sluice-shaker-calibrate`.

## Dilute fine-pulse transport

The [2026-10-27 study](../playtests/2026-10-27-fine-pulse.md) uses fixed-mass 47/97/173 micrometre quartz pulses and actual bounded removal-face observations through six bare/riffled flow conditions. Independent pulse/tailings and water accounting passes, and capture-enabled full game state matches its ordinary continuation exactly. Transport remains unqualified: nominal riffled 97 micrometre discharge differs by 12.20 percentage points between the final two parcel refinements. Postlip lower-Y removals dominate; pure +X is only one buffered-domain exit route. This is weighted-parcel/contact-envelope evidence at fixed water grid/time, not calibrated gold recovery or cohesive slurry. Full reports, source moments, censored residence, covariance, source limitations and reproduction commands are linked from the study.

The subsequent [carrier-support correction](../playtests/2026-10-27-carrier-support.md) fixes an independently reproduced constant-flow defect: absent water nodes no longer dilute the velocity of supported water during ordinary sediment gathering. Supported stationary water remains valid, and absent components use an explicit no-carrier fallback. All six pure-water warm states remain exactly equal; all 18 corrected pulse runs pass custody and water gates. Nominal riffled 97 micrometre discharge is now 75.00/86.97/81.03 percent at 24/192/648 parcels, with a 5.94 percentage-point final-pair difference. The largest final-pair difference is still 9.58 points (low-flow riffled 173 micrometre quartz). These are censored eight-second quartz results, not gold recovery or a convergence qualification. Quantized deposition, contact/source refinement, immersion/connectivity, post-pressure carrier response and conservative two-way exchange remain open. The full operation milestone in the [operation plan](../../plans/2026-10-22-realistic-operation.md) must also advance transport, screens, plant configuration, clogging, ponds and spatial tailings; numerical studies alone are not that playable deliverable.

The [spatial mineral-tailings integration](../playtests/2026-10-27-spatial-tailings.md) preserves full world exit state, resolves first contact and transfers gold-free mineral stock to existing terrain classes with conservative bed/water bookkeeping. Those classes are not a size-resolved downstream grain model. That checkpoint retained gold-bearing contacts pending a separate owner. Modern operations now use the [independent-gold model](independent-gold.md): dilute one-way point-parcel drag/settling, triangle support, geometric burial, local excavation/intake, finite exports and version-13 persistence. Bulk handling still discards grading/history, and downstream flake shape, turbulence/lift, collisions and water reaction are unresolved. No gold carrier is invented, and retained or resident gold is not recovered until collected. Runout/shape/impact coupling, wet instability and empirical qualification remain open. The same operation plan explicitly includes wet-ground slumps, bogging, deep-cut collapse, uncertain surveys, wet-feed plugging/screen blinding, pond berm breach, plant failure and flood/bridge damage; the current dry mechanics and partial tailings do not implement those systems.
