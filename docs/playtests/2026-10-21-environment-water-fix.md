# Environmental water conservation and drainage repair

The river now drains downhill through the -Z map boundary, and the default sluice discharge has a graded, excavatable return channel to the river. Shared GPU face transfers conserve water instead of overdrawing donors and clipping each receiver independently. Dry digging no longer creates phantom water; wet excavation and dumping preserve the existing cell volume.

The original 3D FLIP/APIC pressure solver, CPU DEM, riffle geometry, particle volumes, four fluid substeps and outlet flight are unchanged. Terrain water retains the local-inertial shallow-water model and erosion/settling/collapse stages. The adapter uses the site's Manning roughness, hydrostatic wet/dry face reconstruction and CFL-based water substeps. The 1 mm no-flow cutoff and frame-dependent artificial damping are removed. Four suspended-material channels use the same limited face volumes, with old-depth mass and new-depth dilution; material and trace gold crossing the map boundary are recorded.

## Whole-system accounting

Each save now stores initial water, external river input and actual map discharge. Current storage sums terrain water, channel particles, airborne particles and inlet credit. Pumping and landing are internal transfers. The residual is:

`current storage + boundary discharge - initial storage - external input`.

The previous sluice-only metric remains explicitly named `sluice_water_balance_error_m3`. Native `water_balance_error_m3` now measures the whole system. Its gate is 0.0001 m³ (100 mL), alongside the separate sluice and gold checks. Legacy version-2 saves preserve their water and begin a clearly marked baseline at load time; this cannot retrospectively correct or validate their earlier water history.

## Measured results

Apple M2 Max; one task-owned build, GPU test or app process at a time. Full numeric receipts are in [the JSON report](2026-10-21-environment-water-fix.json).

| Measurement | Regenerated demo | Live continuation |
|---|---:|---:|
| Simulation time | 40.0167 s | 58.5167 s total |
| External input | 4.800000 m³ | 7.020001 m³ |
| Map discharge | 5.560008 m³ | 8.039788 m³ |
| Stored water | 13.353109 m³ | 13.093327 m³ |
| Whole-system residual | -7.281 mL | -9.836 mL |
| Mean render FPS | — | 52.29 |
| Physical simulation / wall time | — | 0.790× |

Initial storage is 14.113124 m³. Boundary discharge can exceed new input while that initial river water drains. At the impact cell the warmed demo is 108 mm deep and shares a continuous four-neighbour wet component with the river at a 0.1 mm connectivity threshold. This demonstrates an available drainage route; a shallow local receiving pool remains physically possible.

The live check measured 20.0208 wall seconds after five seconds of warm-up. It intentionally returns **failure** because physical throughput remains below the 0.98× requirement, despite passing the 50 FPS and conservation gates. This repair is not a claim of real-time physics. The existing `sluice-realtime` issue remains open. No fluid resolution or riffle fidelity was reduced.

## Verification

- All 24 mining-sim tests pass, including new dry, wet and 0.5 mm excavation/deposition and legacy/new ledger persistence cases.
- Three explicitly executed native GPU tests pass: aggregate donor conservation, closed and all four open boundaries including corners, four suspended channels and their exports, 0.5 mm runoff and a lake at rest beside dry ground. Closed water/material tolerance is 1e-7 m³; open water tolerance 2e-7 m³; thin-film water and material-export tolerance 1e-8 m³.
- The real FLIP/APIC + DEM + terrain integration passes pumping, discharge, excavation, camera changes and live save/restore, including whole-system water balance and nonzero downhill river drainage.
- The demo is regenerated from three actual finite terrain cuts and 40 seconds of the complete simulation. Gold error is below 1e-7 g. Its saved APIC, contacts, momentum and ledgers use normal gameplay loading.
- All 24 native mining regressions pass. Focused Clippy for mining-sim, mining-physics and mining with all targets and warnings denied passes; the game dependency retains its existing unused-constructor warning.
- All 29 shared world API tests pass, including material addition and water displacement. The final app is ad-hoc signed and its frozen drainage view is captured from the normal save loader. Workspace-wide CI is not claimed green; the existing legacy format and DEM calibration issues remain separate.

The earlier [101 L audit](2026-10-21-environment-water-audit.md) describes the pre-fix revision and remains available for comparison. This is a conservation and drainage repair, not a laboratory validation of the shallow-water approximation or an exterior splash-pressure solver.
