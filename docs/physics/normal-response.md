# Normal-compliance response and material-calibration prerequisite

This Stage 2 experiment makes a specified single-contact spring behave correctly through the complete rigid-body pipeline. It is default-off, available only with `contact-work-audit`, and explicitly refused by both saved-world wrappers. It does not supply a calibrated soil model or change the installed game. Existing saved numerical continuation remains byte-exact.

The rigid-contact relaxation pass removes normal positional bias and CFM after positions integrate. This is deliberate numerical stabilization: [Solver2D describes relaxation as removing unwanted spring-like motion](https://box2d.org/posts/2024/02/solver2d/). Treating the frequency setting alone as a material spring therefore changes the transient velocity law. That conclusion is independently reproduced in the actual pinned engine, not inferred solely from external documentation.

The new diagnostic flag retains the solved normal response through that pass. Joint, attachment and friction relaxation still execute; no iteration count, friction coefficient, correction cap, joint limit or timestep is widened. Coulomb rigid/generic contact paths are selected. Twist and soft-body worlds are explicitly refused. Their complete physical response is outside this experiment. The flag is serde-skipped, defaults false, survives runtime clones, and cannot be encoded through ContactHistory or RetainedSoil; the ordinary world can still encode/restore after the trial is deselected.

A manufactured frictionless ball/plane point contact is specified by normal stiffness k=100,000 or 1,000,000 N/m, mass 1 or 50 kg, and critical damping c=2√(km). The frequency input is independently derived as √(k/m)/(2π) for this one translation-only contact. These are test coefficients, not measured soil or Hertz parameters. Sixteen complete pipeline cases cover loaded support and compressed release at 480 and 1,920 physical steps/s for 0.5 s. Actual mass is recomputed from the registered body/collider. Load support matches F/k within the unchanged 3 μm bound; resting speed is below 0.0001 m/s. The specified contact's kinetic, gravitational and stored spring energy does not exceed its initial total beyond the fixed 0.0001 J numerical allowance.

Four additional cases exercise the actual four temporal substeps per outer step with the same independent spring. Maximum errors remain below the fixed 30 μm / 0.002 m/s bounds; loaded F/k support and the fixed energy allowance also pass.

Every trajectory is compared against an independent double-precision unilateral implicit-Euler spring/dashpot calculation. Maximum position and velocity differences are 4.975209 μm and 0.000001502 m/s. The ordinary-relaxation negative control has 0.654920 m/s velocity error in compressed release; the new normal option gives 0.000000739 m/s. That supports the specified point-contact law only, not an entire granular bed, current rotating contact frames, tangential history or loaded excavation.

The loaded cases also refine toward the independent continuous critically damped solution at common 480 Hz observation times. Physical integration changes from 480 to 1,920 Hz. All four position and velocity error ratios stay below the unchanged 0.65 bound. Comparing maxima sampled on unrelated time grids initially failed for the fastest transient because the coarse grid skipped a peak; that attempt is retained, and the final comparison explicitly uses the same observation times. It is not a dense-time or fully converged continuum certificate.

| Mass kg | k N/m | Position error μm, coarse → fine | Velocity error m/s, coarse → fine |
|---:|---:|---:|---:|
|1|100000|7.967 → 2.338|0.003148 → 0.001040|
|1|1e+06|1.642 → 0.579|0.000761 → 0.000304|
|50|100000|68.026 → 17.504|0.004383 → 0.001155|
|50|1e+06|20.009 → 5.413|0.003837 → 0.001114|

The complete saved-bank check retains all 1,040 clods, 1,045 dynamic bodies and 9,752.505223 kg, with zero engine/motor/actuator work. Each run advances one 1/480 s outer step with four 1/1920 s integrations, sixteen biased iterations and the existing joint/friction relaxation count. Existing policy and registered-geometry controls remain. Candidate runs use registered geometry and retain normal response while sweeping mass-scaled frequency. All observed/unobserved candidate canonical/body states and complete in-memory world serializations match exactly; candidate bytes are compared in memory only, never saved. The original default after-boundary is byte-exact against the saved reference.

| Saved bank variant | Frequency Hz | Mechanical energy gain J | Max normal response m/s | Unchanged 0.01 J mechanical gate |
|---|---:|---:|---:|---|
|baseline|1e+06|57.216923|3.132013|Fail|
|registered|1e+06|42.599000|3.022651|Fail|
|preserve|1e+06|43.508303|3.022651|Fail|
|preserve|120|-0.891266|0.006518|Pass|
|preserve|30|-1.182755|0.006395|Pass|

These frequency sweeps cannot select a physical soil model. A constant frequency implies k=m_eff·ω², so stiffness changes with contact mass, rotational response, manifold point count and refinement. [LAMMPS distinguishes dimensional Hooke stiffness from material/curvature-dependent Hertz models](https://docs.lammps.org/pair_granular.html). For our polyhedral clods, a physical normal law still needs a justified contact-area/length or other appropriate constitutive relation, consistent damping/yield, state/history and independent support/strain calibration. No inferred or unqualified spring energy is credited against the bank's unchanged mechanical gate. A favourable one-step energy number alone cannot qualify the material.

Validation: 123 vendor tests, 65 audit-library tests and 23 default simulation tests pass; three vendor and twenty-two ordinary audit diagnostics remain ignored. The five full saved-step variants and the fresh 300-step bank execute explicitly. Strict scoped all-target simulation/vendor Clippy pass with the existing vendor `let_and_return` exception; simulation and Rust 2024 vendor formatting plus whitespace pass. The earlier unsupported-model, missing-floor fixture and sampling-clock failures remain recorded rather than counted as passes. 632 other initial Rust/WGSL sources, original nine dirty paths, user stash, input boundaries and signed installed rendering app are preserved. Heavy checks run serially with one compiler/test worker. No native/GPU/FPS or loaded-bank gameplay claim is made.

The fresh 120 Hz normal-response trial advances all 300 outer steps (0.625 physical seconds), preserves the complete inventory and parameters, and refuses unqualified saving. It records a peak step mechanical-energy gain of 17.857651 J, 128 violations of the unchanged 0.01 J mechanical gate, and -377.187641 J net change. Per-step residual reports are dropped after bounded aggregation; no raw trajectory trace is retained. This is a passive-bank check, not loaded support, resolution convergence or calibrated strain accounting.

Next qualify dimensional material response across contact area, particle resolution, rotating frames and loaded support before retaining soil strain or adopting this mode. Full Stages 2–6 remain open: wet/cohesive mud, continuous activation, compacted pads, shaker geometry, sluice buildup/scour, constructed settling/recirculation, graphics, controls, persistence and real-time operation. See the [machine-readable receipt](../playtests/2026-10-25-normal-response.json).
