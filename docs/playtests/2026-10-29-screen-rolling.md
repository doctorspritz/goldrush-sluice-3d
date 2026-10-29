# Screen rolling and grading study — 29 October 2026

**Not adopted.** Simon's live feedback prioritises performance, an actual DEM-style shaker and tailings piling up in the world. This mean-plane rolling correction is preserved as a study; the playable development branch returns to its validated detailed-screen baseline before the next implementation. The full placer-mine objective remains active.

The original bulk contact removed linear motion without creating spin. On a stopped 12° deck, it falsely held dry grains whose detailed spherical reference rolled toward apertures or the reject end. This candidate couples basal friction and spin using spherical inertia I/m=2r²/5. The tangential contact mobility is 7/2; the limited impulse changes centre velocity and angular velocity together. A new landing is not advanced through gravity/translation twice. New rolling state updates the authoritative spin, its float boundary mirror, orientation and resting flag.

159 domain checks pass:103 library,15 bulk-screen,10 intake,11 plant,11 screen and9 screen-feed. Independent controls verify the5/7 downhill acceleration, no-slip relation, Coulomb impulse bounds, pivot angular momentum and nonincreasing contact-frame kinetic energy. A real stopped-deck negative control confirms the old false static arrest. Scoped formatting and three strict compile/lint checks (domain/example, app/tests, actual circuit target) pass. Six full-state dry/wet/coarse continuations for old saved bulk models2/3 are exact, excluding only the explicit legacy contact tag. New contact is explicitly model4; older saved laws stay point friction.

The 32-case finite-cohort diagnostic uses the same64 physical-diameter parcels and low-discrepancy mesh phases in both models. It compares dilute/weighted stock, stopped/running motor and0.1/1.5/2.8/3.2mm diameters with a3mm aperture. All cases retain inventory and complete without runtime refusal; detailed-reference tallies are unchanged. No timing comparison is claimed while Simon's preview was running.

| Stopped dilute feed | Previous bulk under/reject/retained | Rolling candidate under/reject/retained | Detailed reference |
| --- | --- | --- | --- |
|0.1mm|44 /0 /20|64 /0 /0|64 under|
|1.5mm|9 /0 /55|62 /1 /0, plus1 spill|64 under|
|2.8mm|0 /0 /64|3 /59 /0, plus2 spills|64 under|
|3.2mm|0 /0 /64|0 /64 /0|64 reject|

The new rolling response fixes a mechanics defect; it **does not qualify grading**. Running2.8mm cohorts also pass only3/64 versus64/64 in the dilute detailed reference, and running oversize produces8–9 spills instead of all material reaching the catch bin. The stochastic passage/contact/bin model requires geometric investigation in `menace-94n.1`, not a scalar recovery multiplier. Detailed wires are themselves an uncalibrated dilute reference, and weighted/packed/wet loads must not be forced to match an inapplicable dilute result.

[Dong et al.,2017](https://research.monash.edu/en/publications/discrete-particle-simulation-of-particle-flow-and-separation-on-a/) distinguishes per-attempt passage from particle/screen contact opportunities. That supports investigating the mechanism; it does not validate this candidate's hazard, cohesion, mean normal attachment, vibration friction, grain shape, rolling resistance or energy/work closure. Cohesive arrest and wet drag remain assumed.

App test execution, actual Metal wet-sluice/river/wet-stock checks, native model4 build, movement inspection and loaded FPS/physical-time measurement were **not run**. No new native/performance/installed-app/default-screen result is claimed. The user inspected the previously validated553b4dd model3 executable, not this candidate. It closed normally after feedback; player saves are preserved outside the artefact directory.

See the [machine-readable checkpoint](2026-10-29-screen-rolling.json). This study's source is kept on a separate branch so it cannot replace the requested DEM implementation. Continue full finite digging/machines/logistics/pads/surveys/wet ground/mud/bogging/slump/collapse, real plant processing/upgrades/maintenance, sluice buildup/washout, settling ponds/berm breaches, tailings, floods/bridge failures and complete saved gameplay. No parent issue or overall goal is complete.
