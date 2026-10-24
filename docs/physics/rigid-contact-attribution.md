The rigid Coulomb work observer now reproduces the complete saved bank after-state exactly and closes all measured rigid/generic contact stages against independent physical energy. It exposes an actual cached-feature defect: one overlapping clod pair retains a roughly 92 mm contact depth although a fresh query of the same registered shapes gives roughly 1.14 mm. Rebuilding every cached patch reduces the saved-step gain, but the full fresh bank still violates the unchanged physical limit. This is qualified diagnostic code and an unqualified, test-only physical experiment; ordinary gameplay and saved solver policies retain their previous behavior.

The [compact receipt](../playtests/2026-10-24-rigid-attribution.json) records the measured values, original input hashes, artifact hashes, commands and final checks. Full bounded reports remain in the named local artifact directory. The original [unapplied checkpoint](rigid-work-observer-checkpoint.md) remains a reproducible historical archive; its read-only observer has now been completed and verified.

The observer requires explicit `contact-work-audit` and a single worker. `record()` retains its existing generic-contact/phase scope; the new `record_with_colliders()` explicitly selects rigid Coulomb observations. Collider parent and local transforms are captured once; current collider poses are composed from the actual float32 solver CoM, body local CoM and collider attachment before promotion to float64. Physical inertia uses current orientation. Solver weighted response, material witnesses, frozen application points, dominance, padding and handles remain separate. Inactive collectors return before taking references to worker-owned storage. No impulse, equation, contact order, coefficient, cadence or save field is changed.

Normal/tangent work is a fixed-block midpoint partition including cross terms, not sequential work per contact row or an independently closed causal bias budget. Colored and overflow warmstart, local projection and biased/unbiased Coulomb solves are observed after RHS refresh. Twist and final restitution remain outside scope; this bank selects no twist and has zero rigid restitution seeds. Arbitrary scenarios with these features require additional observers.

The saved baseline contains all 1,040 clods and 1,045 dynamic bodies, physical mass 9,752.505223 kg, at 480 outer Hz with four temporal substeps. There are no actuator motor rows, engine work, pump work, stiffness, damping or elastic energy. The final complete encoded world matches the saved reference byte for byte. All 1,010,800 rigid blocks and 5,244 generic blocks contribute to 12 aggregate work closures, with a maximum difference below 6.54e-13 J. The 220 phase samples and independent body-energy change agree within 3.063e-6 J, below the unchanged 0.0001 J observer limit. The independent physical gain is still **57.216923 J**, far above its separate **0.01 J** acceptance limit.

| Rigid contact stage | Physical kinetic change, J |
|---|---:|
| Warmstart | +5885.088144 |
| Local projection | -5649.073694 |
| Biased solves | -61.319620 |
| Unbiased solves | -2.514475 |

Warmstart plus projection adds 236.014450 J in rigid contacts; generic warmstart/projection removes 66.809806 J. Rigid biased normal work is +29.209951 J and tangent work -90.530297 J. Their physical-versus-impulse aggregate difference is about 0.000726 J, too small to explain the 57 J gain. Individual work/response residuals are reported rather than declared zero. The full phase table includes force/gyroscopic increments, integration/gravity and relaxation; this is a measured partition, not proof that any single stage is the sole cause.

All 16 largest positive biased blocks belong to primitive manifold [2242,0], registered colliders [898,0]/[900,0], bodies [895,0]/[897,0]. Each clod is 1.328132 kg. The original material-witness gaps are about -90 to -92 mm while frozen impulse points are near touching. The normal correction reaches its existing -3 m/s saturation. The largest positive block adds 1.45958 J, mostly normal work. These are actual registered clods; no reconstructed ideal hull is substituted.

An independent float64 supporting-face/edge-cross SAT calculation on the registered vertices finds about 1.137 mm overlap at entry. A different minimum-translation direction alone does not disprove an old contact. The decisive additional evidence is 27 actual persistent-dispatcher queries: three policies, cached/fresh/marked-deformed starts, and three captured pose sets. Default, TOUCH_ONLY and VERIFIED all accept the stale cached patch at the entry pose (minimum -92.0647 mm); fresh/marked-deformed queries retain five contacts with a minimum near -1.1358 mm. The portable serialized-shape/manifold fixture reproduces this without a large saved bank. It retains exact registered topology because rebuilding a convex hull can renumber cached features.

The direct query does not replay Rapier's earlier outer `OUTCOME_RECYCLED` decision. Both that outer reuse path and Parry's inner feature reuse must be considered in a future correction. The existing VERIFIED check repairs independently separated pairs and coincident touch geometry; this genuinely overlapping pair exposes a different invalid feature/normal case. A support-plane analysis finds the old closest witness misses its claimed support by 111.8 mm. Fresh EPA closest witnesses also have combined residuals around 0.43–0.44 mm, so a 10 micrometre support certificate cannot yet certify the fresh result. No tolerance was widened, no automatic certificate was adopted, and no EPA root cause is claimed. The bounded design and measurements are preserved locally as `convex-cache-certificate.{md,json,py}`.

Registered compound/triangle ownership and primitive SAT have independent controls for rotation, thin overlap, separation, actual heightfield triangles, compound exterior versus internal seams, and off-surface witnesses. The geometry report interprets only the ranked subset: 32 ranked blocks reduce to 16 pair/pose/normal cases, 12 applicable to primitive SAT. Ambiguous seams, terrain/union and unsupported shapes remain explicitly unproved. This does not certify all 374 previously flagged clustered manifolds or the entire contact equation.

The test-only full-regeneration trial marks persistent patches deformed and sets only the existing `normalized_contact_recycle_distance` from 0.05 to 0, preventing the outer skip. It retains all other parameters, biased friction, warm history and physical geometry. These switches are compiled only in tests; there is no runtime flag, dispatcher enum or new saved numerical policy.

| Trial | Saved-step gain, J | Full fresh 300-step maximum gain, J | Steps above 0.01 J |
|---|---:|---:|---:|
| Existing checked policy, earlier full-bank reference | 57.216923 | 57.216923 | 56 |
| Test-only full regeneration | 40.889740 | 34.897176 | 61 |

The fresh trial completes all 300 requested ticks (0.625 physical seconds) with all stock, bodies and mass preserved at every tick. Net energy falls 289.137276 J, but 61 local increases still fail passivity. Its feedback/independent energy discrepancy stays below 0.000793 J, within the existing 0.05 J feedback check. The trial's recorder is disabled over the trajectory; it is not a performance benchmark or new native qualification. Regeneration is rejected as a complete fix and remains outside Session. An in-memory encode is attempted for diagnostic admission; decode correctly refuses the changed integration parameters. No candidate boundary file is written, and no saved candidate continuation is claimed. This is decode refusal, not a general candidate-encode guard.

The actual Coulomb builder also still reproduces two anchor objectivity failures: a common 0.025 rad rotation invents 2.3997502 m/s of tangential target despite coincident material witnesses, and genuine slip receives the wrong target. Two unrotated/unbiased controls pass. The two known failures are explicitly ignored in the ordinary vendor suite and were run as negative controls, failing as expected. They are not counted as passes, not corrected here, and not yet attributed across the full bank.

The first raw-trace attempt failed with `No space left on device`: a 9,315,594,225-byte incomplete generated JSON exhausted available disk. That is a failed diagnostic run, not a physical test result or macOS crash. The test had already exited when checked. Only its owned incomplete trace was removed; saved inputs and source hashes remained intact, recovering about 8.8 GiB. A later JSON-macro compilation failure was fixed by splitting object construction, without raising recursion limits.

The revised harness folds every row into exact aggregates and bounded top-16 sets and writes only a summary with a hard 16 MiB cap. Partial owned outputs are removed on serialization/write failure; existing files are preserved via `create_new`. A 64-byte-budget control verifies overflow, partial serializer failure, sentinel preservation and success. Final baseline/fresh summaries are about 1.2 MiB each. The explicit one-step collector still retains rows in memory (about 1.5 GiB in this bank); this does not establish arbitrary-duration bounded recording. Never serialize the complete rigid `WorkReport`, or record the 300-step trajectory.

To reproduce the compact fixture independently of saved bank files:

```bash
cargo +nightly test -p mining-sim --features contact-work-audit --lib \
  cached_oblique_clod_manifold_requires_geometric_refresh \
  --profile playable --locked --jobs 1 -- --test-threads=1
```

The saved bank generation procedure remains in [the work-ledger guide](contact-work-audit.md). With its exact before/after boundaries, use fresh output directories:

```bash
SLUICE_ENERGY_BOUNDARY=/tmp/new-bank-witness/worst-before.bin \
SLUICE_ENERGY_REFERENCE=/tmp/new-bank-witness/worst-after.bin \
SLUICE_RIGID_WORK=/tmp/new-rigid-baseline \
  cargo +nightly test -p mining-sim --features contact-work-audit --lib \
  saved_checked_bank_rigid_work --profile playable --locked --jobs 1 \
  -- --ignored --test-threads=1 --nocapture
```

`saved_bank_fresh_manifold_work` uses the same inputs with a different fresh output directory. `complete_bank_fresh_manifold_trajectory` needs only `SLUICE_RIGID_WORK` as a fresh directory and starts a complete new bank. `saved_rigid_work_primitive_pair_geometry` additionally consumes `SLUICE_RIGID_WORK_SUMMARY` and writes a fresh `SLUICE_RIGID_GEOMETRY` directory. `saved_pair2242_convex_queries` consumes the saved boundary and baseline summary and writes `SLUICE_PAIR2242_QUERY_REPORT`. These explicit diagnostic tests can pass while recording a failed physical gate; inspect the output fields.

Continue by qualifying current support features and accurate fresh closest/penetration witnesses, including outer reuse and terrain/cluster ownership. Separate true persistent friction displacement from invented anchor drift before measuring its full-bank contribution. Retained-world passivity, loaded retention and timestep refinement must precede Session adoption. No new rendered capture, app build or FPS result was made in this diagnostic stage; the verified crest/shoreline bundle remains installed. Stage 2, all remaining Stages 3–6 and the full real-time operation remain open.

Final quality checks: 39 audit-enabled simulation library tests pass with 11 explicit diagnostics ignored; 23 default simulation library tests pass; 106 vendor library tests pass with three ignored (two known anchor failures and one timing benchmark). The two known failures were separately executed and remain failures. Explicit saved-bank, direct-query, primitive geometry, fresh-saved and 300-tick diagnostic runs pass their measurement checks while recording failed physical gates. Both scoped Clippy commands pass after the recorded warning fixes, with only the existing vendor `let_and_return` exception. Simulation and all eight changed vendor Rust files pass formatting; whitespace checks pass. No full integration suite or new native run is claimed.

Final preservation checks verify all 700 other protected sources, saved inputs, the original dirty files and stash, and both copies of the signed installed rendering app. No new panic/WindowServer report appeared. Scoped cleanup leaves zero game/compiler/encoder or owned test-browser jobs. Shared services remain untouched: 18 Playwright MCP and 72 Node processes. The scoped Beads update keeps the parent work in progress and files convex-feature and rigid-anchor defects for follow-up.

The installed `bd` does not implement `bd sync`; its failure is logged. Only the four touched project issue rows are exported to tracked JSONL, preserving other projects. The installed pre-commit shim delegates only to Beads; the scoped export and source/evidence gates above are used for this atomic commit to avoid its broad shared-database export. No safety hook configuration is modified.
