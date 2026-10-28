# Loaded plant throughput: measured limits and rejected shortcuts

28 October 2026. Baseline `e80296f9a2e0`. This stage ships evidence and implementation priorities, **no production physics or speed change**. All trials were removed, the probe was rebuilt from reference source, and its complete state matches the original after sorting only unordered contact-cache keys. Native/installed apps are unchanged. [Machine-readable receipt](2026-10-28-loaded-throughput.json).

## The late-load limit changes the next implementation

The actual Metal early profile finishes with116,044 water markers and606 grain parcels. A60-second native reference from the same finite dug-pay/shaker/sluice fixture finishes with115,590 channel water markers,10,814 outflow markers and3,075 grain parcels. It renders57.15FPS but advances at **0.37905x physical time**: real-time gate FAILS. Water, gold, bulk and dry-mass budgets remain finite, with no worker/render failure.

Continuing that late saved state headlessly for300 ticks ends with115,690 water markers and3,643 grain parcels. Means per fixed1/60-second domain tick are:

| Work | Early case | Late case |
|---|---:|---:|
| Session/plant/world CPU | 9.442ms | 13.880ms |
| Fluid and readback | 14.105ms | 16.588ms |
| DEM total | 0.433ms | 10.438ms |
| DEM pair work | 0.247ms | 9.028ms |
| Complete Session + hydraulics | 26.822ms | 45.069ms |

Under the **current serial schedule with other costs fixed**, removing every millisecond of late fluid/readback work still leaves **28.480ms** per16.667ms domain tick: at most **0.585x physical time before rendering**. This is a scoped Amdahl bound, not a limit on a redesigned/overlapped simulator. A faster periodic water strip alone cannot make this measured loaded plant real time. CPU material/bed and screen processing must be addressed too.

A bounded task-owned CPU sample identifies `Screen::refine_wire` and DEM penetration/pair resolution as substantial paths. The sample also contains process setup/serialization; it is used to locate work, not to infer exact percentages. Component timings above come from the real step timers. Native FPS, physical progress and headless component timings remain distinct measurements.

## Trials rejected after checking the actual plant

The early device profile records G2P2.437ms, velocity pressure2.211ms, P2G2.013ms and density pressure1.213ms per average domain tick. This motivated transfer experiments first.

1. **Cache three one-dimensional weights per axis.** [Taichi's MPM example](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/mpm128.py) uses per-axis quadratic weights. Reusing our existing scalar formulas passed22 actual Metal transfer/settling/carrier/phase/periodic checks, but measured2.484ms G2P and changed evolving water state. The old-binary repeat is physically exact after contact-key normalization, so the difference is not merely serialization. No useful gain was established; the change was removed.
2. **Change gather launch width.**32/64/128/256-thread launches all preserve the complete canonical plant state. The repeated128-thread G2P times are2.496/2.535ms;64 gives2.721/2.496ms, while single32/256 windows give2.518/2.566ms. This is exploratory timing, not sustained proof. No repeatable gain justified adoption. The original128-thread layout remains; the temporary override/getters were removed.
3. **Certify separating shaker-wire intervals.** [FCL's CCD discussion](https://gamma.cs.unc.edu/FCL/fcl_docs/webpage/pdfs/fcl_icra2012.pdf) motivates conservative motion bounds. A specialized translating-wire derivative bound, then a tighter phase/jerk/normal-rotation version, retained the original exact hit times/normals/velocities across2,048 reference cases and a delayed-reversal negative control. All133 selected CPU checks passed. Whole ordinary/conservative/late physical states remain exact when excluding only actual query-count telemetry. But the tighter version reduces examined end-step queries from258,974 to258,506 and376,898 to376,266—less than0.2%. This does not justify the new code or establish meaningful plant speedup. Both variants were removed.

Some trial windows slowed unrelated CPU and GPU stages by over2x while another task browser was active. Those samples do not support causal performance comparisons. No unrelated process was stopped. No further candidate native run was justified once the actual loaded cases rejected the changes.

State comparison sorts only the three unordered sluice contact caches by unique physical pair keys, retaining every payload and other array order. Wire trials additionally exclude `plant.screen.last_step.contact_queries`, a work counter. The restored reference comparison includes that counter again. The old-binary repeat is a positive control; the cached-weight trial remains a negative control after normalization. No force, water, gold, geometry, impulse, velocity, grading or stored material value is omitted.

## Next: integrate material scaling, not another isolated water-only result

Keep the original detailed mode as reference. The next hybrid implementation needs finite full-width material/bed ownership, size-resolved active/cohort inventory and wake on changing shear, impacts, undercutting and cleanup, while retaining displacement/porosity and local flow diversion. A resting bed cannot simply disappear from pressure/contact geometry or become permanently immune to washout.

The present shaker is a deliberately dilute weighted-parcel prototype; it does not provide dense packing, cohesion or realistic wet blinding. Develop finite bulk/representative screen processing with physical aperture/grade/density/moisture/vibration and local obstruction consequences. Keep a detailed contact patch/reference where needed, preserve residence and finite output custody, and calibrate its operating envelope. Reducing contact tests without changing the representation is insufficient for the examined heavy load.

Couple those CPU models to the full-width lane/head/bed state and the already qualified periodic vertical-fluid operators. Water, dirt and gold each need one saved owner; visual lateral copies cannot create material. A local clog must divert flow and affect transport. Pump surges, graded gold washout, burial/clogging, maintenance and saved continuation remain required before adoption. Every unfinished excavation, machinery, pads, mud, plant, pond, tailings, flood/failure and gameplay requirement remains active in the [operation plan](../../plans/2026-10-22-realistic-operation.md) and [hybrid specification](../physics/hybrid-sluice.md).

Compact local evidence: `/Users/simonheikkila/.codex/visualizations/2026/10/18/01a14ce3-309f-7782-9a7c-ccd0efaf3227/hydraulic-throughput-20261028`. All newly generated replay saves and the temporary baseline executable link are removed after hashing. No app bundles or source archives are retained from this stage.
