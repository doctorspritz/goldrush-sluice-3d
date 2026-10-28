# Contact strain, numerical conditioning and coupled witness motion

This read-only Stage 2 investigation qualifies the scalar normal-row interpretation and captures a remaining coupled-contact discrepancy. It adds no physical contact policy, material law, save setting or gameplay adoption. Both complete saved-bank replays still fail the unchanged 0.01 J mechanical-energy gate. The installed rendering build is preserved.

The actual scalar normal row matches an independently derived implicit spring/dashpot solution in 108 uncapped active-contact cases: masses 0.1, 2.5 and 250 kg; frequencies 15, 30 and 60 Hz; damping ratios 0, 1 and 10; two timesteps and two approach velocities. Worst relative discrete energy error is 2.704e-7. A compressed 250 kg scalar contact gains 0.077482 J of kinetic energy while its spring strain falls 0.459984 J. Thus positive kinetic work alone does not establish creation of total energy in a compliant scalar model. This control does not qualify the retained bank as a compliant soil model.

For fixed normal, effective mass m, substep h, stored factor a and bias gain beta, the actual row's fixed point gives gamma=(1/a-1)/m, k=beta/(h*gamma), and d=(1/h-beta)/gamma. The historical observer field `normal_inverse_effective_mass` contains `np.r`, which is effective mass in kg; the new report names that quantity correctly. The actual impulse/velocity controls independently verify its unit. These are mass-scaled regularization coefficients, following the [soft-constraint formulation](https://box2d.org/posts/2024/02/solver2d/), rather than a calibrated soil modulus.

The retained clod preset uses 1,000,000 Hz to approach rigid contact. At the observed substep its stored factor is 0.9999998807907104, just two float32 representable steps below one. Inverting that rounded factor changes inferred stiffness/damping by about 21.717% from their nominal values. The first observer rejected that mismatch and preserved the exact row. The corrected observer verifies the configured float32 factor/gain computation exactly and reports the conditioning difference. It never substitutes the nominal spring or adds inferred strain to the physical gate.

Opening unilateral contact also needs special care. The actual row clamps tensile impulse to zero. An ordinary d*v² dashpot loss then invents dissipation; a new negative control reproduces that accounting error. The corrected diagnostic caps separating dashpot traction at the available spring traction and uses the convex unilateral spring remainder. Signed reconstruction roundoff near zero damping remains visible; admitted bank spring mappings require nonnegative damping.

Every observed rigid contact point is paired across its fixed-manifold substep. Normal impulse work is partitioned before integration and during relaxation; actual material-witness gaps, tracking gaps, capped bias, restitution and prescribed work are retained. Only four full operation blocks for the worst interval are exported for reproduction. The complete raw WorkReport is never written. Generic rows remain outside this normal-spring hypothesis and stay inside the existing complete mechanical observer.

| Complete saved-boundary replay | Existing saved policy | Registered convex geometry |
|---|---:|---:|
| Rigid block observations |1,010,800|987,012|
| Point snapshots |3,195,952|2,191,460|
| Paired fixed-contact intervals |168,208|115,340|
| Finite spring mappings |27,204|22,200|
| Bias-capped intervals |18|4|
| Generic blocks outside spring model |5,244|5,244|
| Mechanical energy gain |57.216923 J|42.599000 J|
| Unchanged 0.01 J physical gate |Fail|Fail|

Each replay advances one saved outer step, 1/480 s. These are not fresh 300-step trajectories, sustained settling, loaded digging or native performance measurements. All 1,040 clods, 1,045 dynamic bodies and 9,752.505223 kg remain; engine, motors and actuator pump/elastic work are zero. The original saved-policy after-state remains byte-exact. Existing complete work, body/phase and feedback checks pass. Other fresh-bank and operation requirements remain open.

The counterfactual strain proxies have enormous negative aggregate balances (-3.422e9 J / -2.132e7 J), alongside millions of joules of positive interval residuals. Those values cannot certify physical soil energy: the regularization is uncalibrated, caps and contact topology matter, generic rows are unmodelled, and the contact operators/geometry evolve. A favourable aggregate would conceal rather than resolve the 57/43 J mechanical failures. No proxy term changes the physical acceptance gate.

A useful witness survives refreshed geometry on body pair 653/654. In the existing policy, the selected point leaves its local block with normal velocity +0.465555 m/s, while its observed gap increment gives -0.934058 m/s. In the registered-geometry trajectory the corresponding values are +0.452002 and -0.819145 m/s. Tracking/material gap-increment differences are only 38.76 and 18.78 nm. Both selected intervals are uncapped and have zero restitution seed. One body's translation-derived speed also rises after the selected local solve, from 0.34956 to 0.48542 m/s in the baseline and 0.36312 to 0.47750 m/s with registered geometry.

These are witness-motion discrepancies consistent with later coupled updates undoing local corrections. They are not independently re-queried post-integration overlap certificates. The registered selected point performs approximately -1.07e-6 J of normal work, so it is not established as the source of the macro energy gain. The next required measurement is the complete coupled residual immediately before integration, together with current support geometry, normal/friction/limit feasibility and retained history. A local block residual or extra iteration count alone cannot establish convergence.

Validation: 64 audit-library tests pass, with 19 ordinary diagnostics ignored; the final two complete saved replays run explicitly. All 115 vendor tests and 23 default simulation tests pass; three vendor diagnostics remain ignored. Strict scoped all-target simulation and vendor Clippy pass with the existing vendor `let_and_return` exception. Maintained simulation and actual Rust 2024 vendor formatting pass. Failed fixture, coefficient and wrong-edition format attempts are retained; preliminary loss formulas are explicitly superseded. No native, GPU, manual gameplay, full fresh-bank or real-time claim is made for this test-only stage.

All 713 other initial Rust/WGSL sources, original nine dirty checkout files, input boundaries, user stash and installed signed app remain unchanged. Heavy checks ran serially with one compiler/test worker, bounded 2 MiB logs and 16 MiB summaries. The [receipt](../playtests/2026-10-25-normal-contact-strain.json) includes hashes, exact selected intervals and results. Stage 2 contact/frame/history, physical soil calibration, loaded refinement, and all unfinished mud, pad, shaker, buildup, settling/recirculation, graphics, gameplay and real-time requirements remain open.

Scoped cleanup leaves no owned test/browser/build process; 16 existing shared Playwright MCP and 64 Node processes remain untouched. No new panic-full or WindowServer diagnostic appeared. Installed Beads rejects `bd sync`; only the one touched issue row is merged into the project export, preserving all other lines.
