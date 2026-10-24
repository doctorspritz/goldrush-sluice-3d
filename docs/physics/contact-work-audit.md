The optional `contact-work-audit` feature observes the retained soil-machine solver without changing its impulses, geometry, physical cadence or saved numerical policy. It locates the largest positive kinetic increment in contact warm starting/projection, particularly the rigid bank, without proving a single cause for the entire failure. It is a diagnostic foundation, not a physical correction or Session admission.

The saved checked-geometry worst step still gains 57.217 J with no actuator motor rows, pump work or elastic energy. Its existing 0.01 J passivity gate remains failed. A passing observer test means that its measurements agree and the complete numerical after-boundary is unchanged; it never means the physical gate passed.

`rapier3d::contact_work_audit::record` collects generic contact blocks and stage energies during one explicit single-threaded mechanical replay. It refuses parallel builds and nested recording, and clears its collector after a panic. An inactive collector returns before phase hooks form references to worker-owned buffers. The ordinary game does not enable this feature. No world field, physical coefficient, save encoding or solver equation is added or replaced.

For a fixed-pose contact block, the observer pairs each net normal/tangent impulse with the block's midpoint relative velocity. The summed pairing equals kinetic change for a consistent fixed mass map. It independently assembles physical multibody mass from current link Jacobians and local mass/inertia, then measures kinetic change from the live generalized solver velocities. Regular bodies use current solver pose orientation. Current multibody inertia uses `link.local_to_world.rotation`: stored rigid-body rotations/velocities can remain at the previous substep while `next_position` advances. The initial incomplete observer that used the stale rotation is superseded.

The normal/tangent partition includes block cross terms and is not the sequential work of individual rows. `-rhs_bias * delta_impulse` is diagnostic pairing, not an independently closed causal energy budget. Warm-start rows use their applied impulse; projection/solve rows use the actual float32 impulse difference. The ledger also compares stored weighted-Jacobian impulse response with the measured velocity change. It does not change a warm impulse, reproject a body, or substitute its calculated mass into the solver.

Phase snapshots independently sum each awake dynamic body's physical kinetic energy and mass-weighted centre position. Multibody velocities are reconstructed from current `J*q`; stale stored link velocities are not used. Island identity includes the actual handle set. Phase ordering, island identity and the exact float32 gravity promoted to float64 matter: decimal 9.81 would produce a spurious roughly 5.8 mJ absolute potential offset for this bank. Fixed-body work, restitution, damping, CCD and writeback require separate attention outside the sampled stages in other scenarios.

The final observation covers **220 phase samples, all 1,045 dynamic bodies /1,040 clods, one identified island and 5,244 generic contact blocks**. The complete encoded after-world equals the saved reference byte for byte. Independent saved-body energy gains 57.216923396 J; the phase ledger gains 57.216926459 J, differing by 3.063 microjoules. The observer agreement limit is 0.0001 J; the separate physical passivity limit stays 0.01 J and fails.

| Stage, summed across four substeps | Kinetic change J | Gravitational change J |
|---|---:|---:|
| Force/gyroscopic increments | -5.998846 | +0.000000 |
| Contact warm start and local projection | +169.204644 | +0.000000 |
| Biased contact passes | -110.538164 | +0.000000 |
| Position integration | +0.026548 | +7.300499 |
| Unbiased contact passes | -2.777754 | +0.000000 |

Joint/soft-stage kinetic changes are zero in this rigid bank, with zero actuator motors. Generic warm start and projection together remove 66.809806 J. The corresponding **rigid-contact warm/projection increment is +236.014450 J**. All observed generic contact blocks together remove 116.291630 J. These are measured stage contributions, not a complete causal decomposition: the matched no-history control still gains 39.382 J, so suppressing warm starting alone is not an energy solution. Increasing relaxation to 64 still gains 17.298 J; increasing temporal substeps to 8/16 worsens this one-step outcome to 180.074/459.219 J. No setting is adopted from those attribution controls.

The largest generic fixed-block kinetic/work discrepancy is about 0.008 J, and the largest impulse-response velocity discrepancy about 1.53e-5 in the recorded coordinate units. These local measurements do not close every rigid-contact row or establish a universally exact mass map. The [compact receipt](../playtests/2026-10-24-contact-work.json) retains the full measured stage table and control outcomes.


The apparent roughly 7 J difference between raw engine-body kinetic energy and final synchronization in the older source report reflects stale link velocities. Current generalized-coordinate energy already includes that kinetic energy before synchronization. This observer does not establish a new energy source in canonical synchronization.

The stage ledger does not yet close every individual rigid contact's impulse work or certify its clustered geometry. The remaining 374 clustered manifolds in the geometric audit need actual union/child ownership when prioritising positive-work contacts. Neither a negative sum in generic contacts nor small local mass-map residuals proves all contact equations correct. Calibration, physical timestep/refinement, actual loaded retention, continuous activation, wet/cohesive mud, pads, shaker decks, sluice buildup, settling/recirculation and real-time integration remain open.

Next investigate an energy-admissible rigid-bank warm-start/projection that preserves the complete normal/friction cone, actual contact count, physical cadence, biased correction and accuracy. A changed numerical method requires explicit policy/version semantics; old saved owners must keep their original continuation. Do not disable bias, clamp final energy, discard clods, increase allowable penetration or widen the passivity gate to hide this failure.

Regenerate a fresh checked witness if the local saved boundaries are unavailable. This deliberately executes the known failing 300-step physical case and preserves its worst boundary:

```bash
SLUICE_ENERGY_CURRENT_POLICY=1 SLUICE_ENERGY_WITNESS=/tmp/new-bank-witness \
  cargo +nightly test -p mining-sim --features contact-history --lib \
  zero_actuator_bank_energy_witness --profile playable --locked --jobs 1 \
  -- --ignored --test-threads=1 --nocapture
```

Observe that worst step, using fresh output names and one process at a time:

```bash
SLUICE_ENERGY_BOUNDARY=/tmp/new-bank-witness/worst-before.bin \
SLUICE_ENERGY_REFERENCE=/tmp/new-bank-witness/worst-after.bin \
SLUICE_CONTACT_WORK=/tmp/contact-work.json \
  cargo +nightly test -p mining-sim --features contact-work-audit --lib \
  saved_checked_bank_contact_work --profile playable --locked --jobs 1 \
  -- --ignored --test-threads=1 --nocapture
```

`--target-dir` or `CARGO_TARGET_DIR` may select an existing shared target; it does not change dependency lockfiles. The application observation uses the root lockfile (`wide 1.7.1`). Vendor unit controls use the standalone tracked vendor lockfile (`wide 1.6.1`), explicitly separate from application replay:

```bash
cargo +nightly test --manifest-path vendor/rapier3d/Cargo.toml \
  --features contact-work-audit,enhanced-determinism --lib contact_work_audit::tests \
  --profile playable --locked --jobs 1 -- --test-threads=1 --nocapture
```

Verification: 23 default application-library checks pass. The explicitly executed final saved-bank observer preserves the complete encoded reference and closes its independent phase/body measurement to 3.1 microjoules; two standalone analytic/collector controls pass on the frozen final observer. The caught panic in the collector control is intentional. Scoped application all-target Clippy uses `--no-deps` with warnings denied; vendor library/test Clippy denies warnings with only the existing `let_and_return` lint allowed. All ten changed Rust files pass scoped formatting and the diff passes whitespace checks. The initial broader lint fails on ten unchanged sim3d dependency findings; a full-workspace clean lint is not claimed. Earlier unsupported commands, observer compile mistakes and the constant-assert lint failure are retained locally and are not passes. The ordinary default library ran before final formatting and the audit-only compile-time guard conversion; no default physical equation changed. Full integration suites, new native visuals and throughput are not rerun for this read-only instrumentation.

Heavy checks run serially, with one compiler job. Scoped cleanup leaves zero owned app/build/test-browser jobs; shared Playwright MCP/Node services remain untouched. No new panic-full or WindowServer report appeared. Three older generated geometry JSON files and four superseded generated ledgers are losslessly compressed with verified hashes/restoration commands. Saved numerical inputs, user saves, original dirty files, stash and current signed render app remain intact. The full operation and physical accuracy gates stay open.
