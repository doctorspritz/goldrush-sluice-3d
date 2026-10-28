# Finite clod and hollow-bucket contact checkpoint

The new finite pressure-field component qualifies stored energy and instantaneous wrenches on complete registered clods and the hollow bucket shell. It is a checked reference for the next solver stage. The playable app keeps its existing physics policy.

| Saved geometry and pair coverage | Result |
|---|---:|
| Clods |1,040|
| Clod vertices / faces / pressure tetrahedra |5,868 / 5,072 / 7,576|
| Steel parts / vertices / faces / tetrahedra |19 / 198 / 137 / 320|
| Clod/clod pairs; positive finite overlaps |16,632; 2,701|
| Clod/steel pairs; positive finite overlaps |560; 4|
| Outside qualified scope |1,052|
| Evaluation failures |0|

All represented vertices and ordered faces are retained. Captured coefficients are promoted without quaternion normalization, shape replacement or material-inventory changes. Both the decoded input round trip and complete owner encoding remain byte-identical. The remaining scope contains 1,040 clod/heightfield pairs and 12 other machine contacts; none is silently represented by an infinite plane.

Clod-pair force closure is below 3.38e-12 N, moment closure below 1.28e-13 Nm and instantaneous power closure below 1.78e-15 W. Four distinct saved clod pairs and one steel pair pass all 60 pose-energy derivative checks. Worst normalized derivative discrepancies are below 2.80e-7 for clods and 2.58e-7 for steel, against the unchanged 1e-5 limit. Refinement retains the same material field and domain; it does not establish convergence under changing physical clod size.

The rigid union keeps cavity air open and counts overlapping steel once. In the manufactured overlap control, adding parts separately gives 50 J, whereas the union and independent whole-solid control give 37.5 J. Floating construction slivers found during development are retained as reproducers. Rational compound clipping and positive tetrahedral integration resolve them without discarding thin geometry, snapping source vertices or correcting velocities/energy afterwards.

## Work across a finite edge

The separate analytic fixture translates a held-depth punch across a finite compliant block. It is prescribed motion, not a live mining replay. Its 513 measured positions match independently integrated pressure formulas, and the edge force recovers the initial 61.03515625 J as 61.03515625000001 J of work. The difference is 7.11e-15 J; maximum analytic energy/force errors are 1.07e-13 J and 1.28e-11 N.

| Punch centre x, m | Stored energy, J | Normal force, N | Edge force, N |
|---|---:|---:|---:|
|1.50|61.035156|7,812.500|0|
|1.75|59.763590|7,568.359|122.070|
|2.00|29.246012|3,662.109|122.070|
|2.25|0|0|0|
|2.50|0|0|0|

The [complete measured/analytic dataset](2026-10-25-finite-edge.json) preserves every sample. This demonstrates boundary work for the defined elastic law, not calibrated real soil or a solved coupled timestep.

## Cost and verification

Exact-geometry controls initially took 218.48 seconds. Component-wise inclusion-exclusion and compensated moment integration reduce the same 18 controls to 41.27 seconds. Rejecting provably disjoint rigid cells before exact construction reduces the saved 560-pair steel check from 33.48 seconds to 4.53 seconds: 179,200 possible rigid-cell visits become 1,031 candidates. All reported physical values, pose/partition qualifications and custody fields match the pre-culling results exactly. This is a reference-workload improvement, not a game FPS or real-time claim. Positive-pair latency still reaches 0.369 seconds.

Full default and audit commands pass 65 and 107 checks. The added finite-edge regression passes separately. The saved-bank and legacy-baseline checks also ran explicitly; 21 other opt-in audit checks remain unrun. Strict scoped all-target Clippy, formatting and whitespace pass. The full app, native FPS and long soil integration replay were not rerun in this backend stage. All 726 other initial Rust/WGSL sources, water/riffle/terrain stepping, original dirty checkout, user stash and installed signed app hash remain unchanged.

## Live integration remains open

The supplied pressure scale is an uncalibrated diagnostic value, not Young's modulus or a soil default. No force is applied to the observed bank, no save/runtime policy selects the model, and strain-energy credit remains zero. The old saved-bank baseline still gains **57.216923396 J** of mechanical energy and **fails** its unchanged **0.01 J** gate. Its complete after-state still matches the reference bytes. A passing compatibility check confirms preservation of that failure; it does not qualify the old dynamics.

Next work must supply a physical closed terrain domain, coupled translation/rotation and motor/joint/friction work, damping/yield, CCD, reference persistence, material calibration, timestep/clod-size convergence and usable performance. These remain part of the original realistic-game objective.

Tracked follow-ups: `menace-6um` (terrain ownership), `menace-ea8` (prepared geometry and live coupling). The parent coupling issue remains in progress. [Model notes](../physics/finite-pressure.md) and the [receipt](2026-10-25-finite-contact.json) contain source/input hashes, exact commands, intermediate failures and full measurements.

Jobs ran sequentially. No new panic-full or WindowServer diagnostic appeared. Scoped cleanup leaves no owned test/browser process; 16 shared Playwright MCP and 64 Node processes remain untouched. Installed Beads rejects `bd sync`; only the parent issue and two new follow-ups are merged into the 230-line export, preserving 227 unrelated literal rows. The broad game goal remains active.
