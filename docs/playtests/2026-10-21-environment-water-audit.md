# Environmental water conservation audit — 21 October 2026

**Historical audit of the pre-fix build.** The subsequent [repair and measured validation](2026-10-21-environment-water-fix.md) address these defects. At the audited revision, environmental water was not verified conservative, and the saved live run showed an unexplained positive volume discrepancy. Previous zero-water-error receipts measured the sluice ledger, not the complete environment. This audit itself changed no solver or renderer and launched no GPU/game/test workload.

## Measured saved-state comparison

Audited source: `b2baac6c4c0423617dc60b2f1f5f9817eb157893`. Inputs are the bundled solver-warmed demo and the retained final shoreline benchmark save. Their SHA-256 hashes and full arithmetic receipt are in [the audit JSON](2026-10-21-environment-water-audit.json). Ground summation and depth subtraction reproduce f32 operations; volume is summed in f64 over 128² cells at 0.25 m spacing.

Stored water = sum(max(surface-ground,0)×0.0625 m²) + channel/airborne particle volume + inlet credit. **Cumulative returned volume is excluded**, because landed water is already in terrain storage.

| Measurement | Bundled demo | Final saved run |
|---|---:|---:|
| Simulated time | 12.016667 s | 56.000003 s |
| Terrain steps | 240 | 1120 |
| Terrain water | 16.482837292 m³ | 21.864021810 m³ |
| Channel + flight + inlet credit | 0.015097252 m³ | 0.015190010 m³ |
| Whole-system stored water | 16.497934544 m³ | 21.879211821 m³ |
| Sluice-only ledger error | 0.0 m³ | 0.0 m³ |

The additional 880 terrain steps inject approximately **5.280 m³** from the 0.12 m³/s river source. Stored water increases by **5.381277277 m³**, leaving **101.277 L of unexplained gain**, about 1.92% of that source input. Source-rate/timestep f32 rounding changes this estimate by less than 1 mL.

There is no cumulative boundary-discharge ledger, so the residual cannot yet be attributed stage by stage. The only draining edge has effectively no stored water and zero last-step outflow in both saves. Any unmeasured outward loss would increase the unexplained positive source required to reconcile storage; it cannot account for the positive residual.

A four-neighbour component check at depth>0.1 mm finds the outlet puddle around x 13.75–15.25 m, z 18.5–20 m disconnected from the main river at the final snapshot. This establishes current separation, not a claim that a puddle can never overflow.

## Confirmed code defects and contributing setup

1. **The reported ledger stops at landing.** `Sluice::accounted_water_m3()` adds live channel/airborne particles, inlet credit and cumulative `water_returned_m3`. Native `water_balance_error_m3` subtracts that from pumped water. It does not audit subsequent environmental storage, external river input, map discharge or numerical changes. See [sluice.rs](../../crates/mining-sim/src/sluice.rs) and [validation.rs](../../crates/mining/src/validation.rs).
2. **The river's downhill edge is closed.** The bed slopes upward with +Z, with source water added near z 30.5 m. The GPU water shader opens only +X; the -Z face is absent from its boundary fluxes and +Z is explicitly closed. The CPU world's `open_boundaries=true` flag does not change that shader path. Water backs up at the downstream end instead of having a through-flow river. See [heightfield_water.wgsl](../../crates/game/src/gpu/shaders/heightfield_water.wgsl) and [terrain adapter](../../crates/mining-physics/src/terrain.rs).
3. **The flux/depth update is not generally conservative.** Each face is limited against a donor's full water volume independently, allowing several faces to overdraw the same cell. Subsequent per-cell depth-change clipping and negative-depth clamping can create or remove volume. Shared face transfers need total donor-budget limiting and a conservative stable timestep, rather than independent depth clipping. The saved residual does not isolate how much comes from this stage.
4. **Dry excavation can manufacture water.** The game represents dry cells as `water_surface=ground`. `World::excavate` lowers the ground but leaves that old surface height unchanged; its `_had_water` flag is unused. The hole acquires positive depth without receiving water. The existing dry-excavation test starts with `World::new` water surfaces at 0 below ground, so it misses the actual game's dry-cell representation. Underwater excavation also needs conserved volume/displacement. See [terrain.rs](../../crates/sim3d/src/world/terrain.rs) and [current tests](../../crates/game/tests/water_conservation.rs).
5. **The demo deliberately pours onto a flat work pad.** The suggested sluice points along +Z and drops onto the service pad, several metres from the river, with no designed tailrace. Some local pooling is physically plausible. The default warm-up also performs three terrain cuts, so phantom excavation water may already contaminate the bundled initial state. Its amount has not been isolated in this read-only audit. Regenerate the demo only after those conservation fixes.
6. **Submillimetre visible water does not all move.** The terrain solver disables face flow when both sides are at or below 1 mm, while rendering shows films above 0.1 mm. Some visible thin water is therefore numerically stationary. Resolving that requires a stable wetting/drying treatment and explicit conservation tests, not simply lowering a rendering cutoff.

## Required correction and validation

Tracked as **sluice-environment-water (P1)**. Preserve the original 3D FLIP/APIC and DEM/riffle solver while repairing the terrain path:

- Record initial terrain storage, actual external sources, actual boundary discharge, active particle/credit storage and numerical residual. Pump intake and outlet impact are internal transfers.
- Add dry/wet dig/deposit regressions using the game's actual surface representation. Terrain edits must neither create water in dry holes nor destroy displaced water.
- Make shared face fluxes respect each donor's aggregate budget, remove nonconservative depth-change clipping and validate closed-domain conservation plus all configured open boundaries on the actual GPU path.
- Provide a real downhill river outlet and an intentional terrain/tailrace route for the demo discharge. Do not delete pooled water or impose cosmetic advection to fake drainage.
- Resolve thin-film wetting/drying consistently, regenerate the warmed demo, verify saved/restored ledgers and run a sustained source-to-discharge balance with measured tolerances and performance.

No new tests or simulations were run in this audit. This is source inspection plus a measurement of retained actual saves, not a claim that any fix or new conservation test has passed. The physics bug remains open.
